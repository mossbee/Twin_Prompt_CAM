import torch
import os
import torch.nn as nn
from timm.scheduler.cosine_lr import CosineLRScheduler
from collections import OrderedDict
from engine.optimizer import make_optimizer
from utils.misc import AverageMeter, EarlyStop
from utils.setup_logging import get_logger
from utils.twin_metrics import TwinVerificationMetrics, compute_dataset_statistics, AverageMeter
from utils.wandb_tracker import get_tracker
import numpy as np
import json
import time

logger = get_logger("Prompt_CAM")
torch.backends.cudnn.benchmark = False


class TwinTrainer:
    """
    Twin verification trainer with 3-phase training strategy
    """

    def __init__(self, model, tune_parameters, params):
        self.params = params
        self.model = model
        self.device = params.device
        self.verification_criterion = nn.BCELoss()  # Binary cross-entropy for verification
        
        # Training phases configuration
        self.training_phases = getattr(params, 'training_phases', {
            'base': {'epochs': 40, 'lr': 0.005, 'phase': 'base'},
            'twin_focused': {'epochs': 30, 'lr': 0.001, 'phase': 'twin_focused'},
            'attention_refine': {'epochs': 30, 'lr': 0.0001, 'phase': 'attention_refine'}
        })
        
        # Initialize tracking
        tracker_config = {
            'tracking_method': getattr(params, 'tracking_method', 'wandb'),
            'wandb_enabled': getattr(params, 'wandb_enabled', True),
            'wandb_project': getattr(params, 'wandb_project', 'prompt_cam_twin'),
            'wandb_entity': getattr(params, 'wandb_entity', None),
            'wandb_run_name': getattr(params, 'wandb_run_name', None),
            'wandb_tags': getattr(params, 'wandb_tags', ['twin_verification', 'prompt_cam'])
        }
        self.tracker = get_tracker(tracker_config)
        
        # Setup optimizer and scheduler for initial phase
        logger.info("Setting up the optimizer...")
        self.optimizer = make_optimizer(tune_parameters, params)
        self.scheduler = CosineLRScheduler(
            self.optimizer, 
            t_initial=params.epoch,
            warmup_t=params.warmup_epoch, 
            lr_min=params.lr_min,
            warmup_lr_init=params.warmup_lr_init
        )
        
        # Checkpoint settings
        self.checkpoint_dir = getattr(params, 'checkpoint_dir', './checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_every_epoch = getattr(params, 'checkpoint_every_epoch', True)
        
        # Early stopping
        if getattr(params, 'early_patience', 0) > 0:
            self.early_stop_check = EarlyStop(params.early_patience)
        else:
            self.early_stop_check = None
            
        # Current training state
        self.current_phase = 'base'
        self.current_epoch = 0
        self.phase_start_epoch = 0
        
    def forward_one_batch_twin(self, img1, img2, targets, is_train):
        """
        Forward pass for one batch of twin verification
        
        Args:
            img1: First image batch
            img2: Second image batch
            targets: Binary labels (0 or 1)
            is_train: Whether in training mode
            
        Returns:
            loss, verification_scores, attention_maps, metrics
        """
        img1 = img1.to(self.device, non_blocking=True)
        img2 = img2.to(self.device, non_blocking=True)
        targets = targets.float().to(self.device, non_blocking=True)
        
        if is_train:
            # Forward pass
            verification_scores, attn1, attn2 = self.model(img1, img2)
            verification_scores = verification_scores.squeeze()
            
            # Compute loss
            loss = self.verification_criterion(verification_scores, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping if specified
            if hasattr(self.params, 'max_grad_norm'):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm)
            
            self.optimizer.step()
        else:
            with torch.no_grad():
                verification_scores, attn1, attn2 = self.model(img1, img2)
                verification_scores = verification_scores.squeeze()
                loss = self.verification_criterion(verification_scores, targets)
        
        # Compute metrics
        predictions = (verification_scores > 0.5).float()
        
        # Calculate accuracy
        correct = (predictions == targets).float()
        accuracy = correct.mean()
        
        return loss, verification_scores, (attn1, attn2), accuracy
    
    def train_one_epoch_twin(self, epoch, loader, phase):
        """Train one epoch for twin verification"""
        
        # Set model to training mode
        self.model.train()
        
        # Metrics tracking
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        verification_metrics = TwinVerificationMetrics()
        
        logger.info(f"Training epoch {epoch} (Phase: {phase})")
        
        for batch_idx, (img1, img2, targets) in enumerate(loader):
            # Forward pass
            loss, scores, (attn1, attn2), accuracy = self.forward_one_batch_twin(
                img1, img2, targets, is_train=True
            )
            
            # Update metrics
            batch_size = img1.shape[0]
            loss_meter.update(loss.item(), batch_size)
            accuracy_meter.update(accuracy.item(), batch_size)
            
            # Update verification metrics
            predictions = (scores > 0.5).float()
            verification_metrics.update(predictions, targets, scores)
            
            # Log batch-level metrics occasionally
            if batch_idx % 2 == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(loader)}: "
                    f"Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}"
                )
        
        # Compute final metrics for epoch
        epoch_metrics = verification_metrics.compute_metrics()
        epoch_metrics.update({
            'loss': loss_meter.avg,
            'accuracy': accuracy_meter.avg
        })
        
        logger.info(
            f"Epoch {epoch} Training Summary: "
            f"Loss: {loss_meter.avg:.4f}, "
            f"Accuracy: {accuracy_meter.avg:.4f}, "
            f"AUC: {epoch_metrics.get('auc', 0.0):.4f}"
        )
        
        return epoch_metrics
    
    @torch.no_grad()
    def eval_twin_classifier(self, loader, prefix):
        """Evaluate twin classifier"""
        
        # Set model to eval mode
        self.model.eval()
        
        # Metrics tracking
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        verification_metrics = TwinVerificationMetrics()
        
        logger.info(f"Evaluating on {prefix} set...")
        
        for batch_idx, (img1, img2, targets) in enumerate(loader):
            # Forward pass
            loss, scores, (attn1, attn2), accuracy = self.forward_one_batch_twin(
                img1, img2, targets, is_train=False
            )
            
            # Update metrics
            batch_size = img1.shape[0]
            loss_meter.update(loss.item(), batch_size)
            accuracy_meter.update(accuracy.item(), batch_size)
            
            # Update verification metrics
            predictions = (scores > 0.5).float()
            verification_metrics.update(predictions, targets, scores)
            
            # Log attention maps for visualization (sample some batches)
            if batch_idx < 5 and hasattr(self.tracker, 'log_attention_maps'):
                for i in range(min(2, batch_size)):  # Log first 2 samples
                    self.tracker.log_attention_maps(
                        img1[i], img2[i], attn1[i] if attn1 is not None else None, 
                        attn2[i] if attn2 is not None else None,
                        int(targets[i].item()), scores[i].item(),
                        step=self.current_epoch
                    )
        
        # Compute final metrics
        eval_metrics = verification_metrics.compute_metrics()
        eval_metrics.update({
            'loss': loss_meter.avg,
            'accuracy': accuracy_meter.avg
        })
        
        logger.info(
            f"Evaluation ({prefix}) Summary: "
            f"Loss: {loss_meter.avg:.4f}, "
            f"Accuracy: {accuracy_meter.avg:.4f}, "
            f"EER: {eval_metrics.get('eer', 0.0):.4f}, "
            f"AUC: {eval_metrics.get('auc', 0.0):.4f}"
        )
        
        return eval_metrics
    
    def save_checkpoint(self, epoch, phase, metrics):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'phase': phase,
            'current_phase': self.current_phase,
            'phase_start_epoch': self.phase_start_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'params': self.params.__dict__ if hasattr(self.params, '__dict__') else str(self.params)
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, latest_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Log to tracker if available
        if hasattr(self.tracker, 'log_model_checkpoint'):
            self.tracker.log_model_checkpoint(checkpoint_path, epoch, metrics)
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint"""
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.current_phase = checkpoint.get('current_phase', 'base')
        self.phase_start_epoch = checkpoint.get('phase_start_epoch', 0)
        
        logger.info(f"Resumed from epoch {self.current_epoch}, phase {self.current_phase}")
        
        return checkpoint['metrics']
    
    def update_training_phase(self, new_phase, new_lr):
        """Update training phase and learning rate"""
        logger.info(f"Transitioning from {self.current_phase} to {new_phase}")
        
        self.current_phase = new_phase
        self.phase_start_epoch = self.current_epoch
        
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        # Log phase transition
        if hasattr(self.tracker, 'log_training_phase'):
            self.tracker.log_training_phase(new_phase, self.current_epoch)
        
        logger.info(f"Updated learning rate to {new_lr} for phase {new_phase}")
    
    def train_twin_classifier(self, loaders_dict):
        """
        Main training loop with 3-phase strategy
        
        Args:
            loaders_dict: Dictionary containing loaders for each phase
                         {'base': (train, val, test), 'twin_focused': (...), 'attention_refine': (...)}
        """
        logger.info("Starting 3-phase twin verification training")
        
        # Log dataset statistics
        if hasattr(self.tracker, 'log_dataset_statistics'):
            # Load dataset info for statistics
            try:
                with open(os.path.join(self.params.data_path, 'train_dataset_infor.json'), 'r') as f:
                    dataset_info = json.load(f)
                with open(os.path.join(self.params.data_path, 'train_twin_pairs.json'), 'r') as f:
                    twin_pairs = json.load(f)
                
                stats = compute_dataset_statistics(dataset_info, twin_pairs)
                self.tracker.log_dataset_statistics(stats)
            except Exception as e:
                logger.warning(f"Could not log dataset statistics: {e}")
        
        best_metrics = {}
        all_train_metrics = []
        
        # Resume from checkpoint if specified
        if getattr(self.params, 'resume_from_checkpoint', None):
            self.load_checkpoint(self.params.resume_from_checkpoint)
        
        # Training loop for all phases
        for phase_name, phase_config in self.training_phases.items():
            # Skip phases that have already been completed
            if self.current_epoch >= sum(config['epochs'] for config in 
                                       list(self.training_phases.values())[:list(self.training_phases.keys()).index(phase_name)]):
                continue
            
            logger.info(f"\n{'='*50}")
            logger.info(f"Starting Phase: {phase_name}")
            logger.info(f"Epochs: {phase_config['epochs']}, LR: {phase_config['lr']}")
            logger.info(f"{'='*50}")
            
            # Update training phase
            if phase_name != self.current_phase:
                self.update_training_phase(phase_name, phase_config['lr'])
            
            # Get loaders for this phase
            if phase_name in loaders_dict:
                train_loader, val_loader, test_loader = loaders_dict[phase_name]
            else:
                logger.warning(f"No loaders found for phase {phase_name}, using base loaders")
                train_loader, val_loader, test_loader = loaders_dict['base']
            
            # Train for specified epochs in this phase
            phase_start = self.current_epoch
            phase_end = phase_start + phase_config['epochs']
            
            for epoch in range(phase_start, phase_end):
                self.current_epoch = epoch
                
                # Train one epoch
                train_metrics = self.train_one_epoch_twin(epoch, train_loader, phase_name)
                
                # Evaluate
                if (epoch % self.params.eval_freq == 0) or epoch == phase_end - 1:
                    if test_loader is not None:
                        eval_metrics = self.eval_twin_classifier(test_loader, "test")
                    elif val_loader is not None:
                        eval_metrics = self.eval_twin_classifier(val_loader, "val")
                    else:
                        eval_metrics = train_metrics
                    
                    # Log metrics
                    current_lr = self.optimizer.param_groups[0]['lr']
                    if hasattr(self.tracker, 'log_epoch_metrics'):
                        self.tracker.log_epoch_metrics(epoch, train_metrics, eval_metrics, current_lr)
                    
                    if hasattr(self.tracker, 'log_verification_metrics'):
                        self.tracker.log_verification_metrics(eval_metrics, epoch)
                    
                    # Check for best metrics
                    if 'eer' in eval_metrics and (not best_metrics or eval_metrics['eer'] < best_metrics.get('eer', float('inf'))):
                        best_metrics = eval_metrics.copy()
                        best_metrics['epoch'] = epoch
                        best_metrics['phase'] = phase_name
                
                # Save checkpoint
                if self.checkpoint_every_epoch:
                    self.save_checkpoint(epoch, phase_name, eval_metrics if 'eval_metrics' in locals() else train_metrics)
                
                # Early stopping check
                if self.early_stop_check and 'eval_metrics' in locals():
                    stop, save_model = self.early_stop_check.early_stop(eval_metrics)
                    if save_model:
                        best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                        torch.save({'model_state_dict': self.model.state_dict()}, best_model_path)
                    if stop:
                        logger.info(f"Early stopping triggered at epoch {epoch}")
                        break
                
                # Update scheduler
                self.scheduler.step(epoch)
                
                all_train_metrics.append(train_metrics)
        
        # Final evaluation and cleanup
        logger.info("\nTraining completed!")
        logger.info(f"Best metrics: {best_metrics}")
        
        # Save final model
        final_model_path = os.path.join(self.checkpoint_dir, 'final_model.pth')
        torch.save({'model_state_dict': self.model.state_dict()}, final_model_path)
        
        # Finish tracking
        if hasattr(self.tracker, 'finish'):
            self.tracker.finish()
        
        return all_train_metrics, best_metrics 