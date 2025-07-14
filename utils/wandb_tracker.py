import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List
import os
from utils.setup_logging import get_logger

logger = get_logger("Prompt_CAM")


class WandBTracker:
    """
    WandB tracker for twin face verification training
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize WandB tracker
        
        Args:
            config: Configuration dictionary containing wandb settings
        """
        self.config = config
        self.enabled = config.get('wandb_enabled', True)
        
        if self.enabled:
            # Initialize WandB
            wandb.init(
                project=config.get('wandb_project', 'prompt_cam_twin'),
                entity=config.get('wandb_entity', None),
                name=config.get('wandb_run_name', None),
                config=config,
                tags=config.get('wandb_tags', ['twin_verification', 'prompt_cam'])
            )
            logger.info("WandB tracking initialized")
        else:
            logger.info("WandB tracking disabled")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log training metrics to WandB
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Training step/epoch
        """
        if not self.enabled:
            return
            
        wandb.log(metrics, step=step)
    
    def log_epoch_metrics(self, epoch: int, train_metrics: Dict[str, float], 
                         val_metrics: Dict[str, float], lr: float):
        """
        Log epoch-level metrics
        
        Args:
            epoch: Current epoch
            train_metrics: Training metrics
            val_metrics: Validation metrics
            lr: Learning rate
        """
        if not self.enabled:
            return
        
        metrics = {
            'epoch': epoch,
            'learning_rate': lr,
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()}
        }
        
        self.log_metrics(metrics, step=epoch)
    
    def log_verification_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log verification-specific metrics (EER, FAR, etc.)
        
        Args:
            metrics: Dictionary containing verification metrics
            step: Training step/epoch
        """
        if not self.enabled:
            return
        
        verification_metrics = {
            'verification_accuracy': metrics.get('verification_accuracy', 0.0),
            'eer': metrics.get('eer', 0.0),
            'far_1': metrics.get('far_1', 0.0),
            'far_01': metrics.get('far_01', 0.0),
            'far_001': metrics.get('far_001', 0.0),
            'auc': metrics.get('auc', 0.0),
            'precision': metrics.get('precision', 0.0),
            'recall': metrics.get('recall', 0.0),
            'f1_score': metrics.get('f1_score', 0.0)
        }
        
        self.log_metrics(verification_metrics, step=step)
    
    def log_attention_maps(self, img1: torch.Tensor, img2: torch.Tensor, 
                          attn1: torch.Tensor, attn2: torch.Tensor, 
                          label: int, prediction: float, step: Optional[int] = None):
        """
        Log attention maps visualization
        
        Args:
            img1: First image tensor
            img2: Second image tensor
            attn1: Attention maps from first image
            attn2: Attention maps from second image
            label: Ground truth label (0 or 1)
            prediction: Model prediction (probability)
            step: Training step/epoch
        """
        if not self.enabled:
            return
        
        try:
            # Create attention visualization
            fig = self._create_attention_visualization(
                img1, img2, attn1, attn2, label, prediction
            )
            
            # Log to WandB
            wandb.log({
                'attention_maps': wandb.Image(fig),
                'prediction_confidence': prediction,
                'ground_truth': label
            }, step=step)
            
            plt.close(fig)
            
        except Exception as e:
            logger.warning(f"Failed to log attention maps: {e}")
    
    def _create_attention_visualization(self, img1: torch.Tensor, img2: torch.Tensor,
                                      attn1: torch.Tensor, attn2: torch.Tensor,
                                      label: int, prediction: float):
        """
        Create attention map visualization
        
        Args:
            img1: First image tensor (C, H, W)
            img2: Second image tensor (C, H, W)
            attn1: Attention maps from first image
            attn2: Attention maps from second image
            label: Ground truth label
            prediction: Model prediction
            
        Returns:
            matplotlib figure
        """
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Convert tensors to numpy for visualization
        img1_np = self._tensor_to_numpy(img1)
        img2_np = self._tensor_to_numpy(img2)
        
        # Plot original images
        axes[0, 0].imshow(img1_np)
        axes[0, 0].set_title('Image 1')
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(img2_np)
        axes[1, 0].set_title('Image 2')
        axes[1, 0].axis('off')
        
        # Plot top 3 attention maps for each image
        if attn1 is not None and len(attn1.shape) >= 3:
            attn1_np = attn1.detach().cpu().numpy()
            # Take first 3 attention heads
            for i in range(min(3, attn1_np.shape[0])):
                axes[0, i+1].imshow(img1_np)
                axes[0, i+1].imshow(attn1_np[i], alpha=0.6, cmap='jet')
                axes[0, i+1].set_title(f'Attention Head {i+1}')
                axes[0, i+1].axis('off')
        
        if attn2 is not None and len(attn2.shape) >= 3:
            attn2_np = attn2.detach().cpu().numpy()
            # Take first 3 attention heads
            for i in range(min(3, attn2_np.shape[0])):
                axes[1, i+1].imshow(img2_np)
                axes[1, i+1].imshow(attn2_np[i], alpha=0.6, cmap='jet')
                axes[1, i+1].set_title(f'Attention Head {i+1}')
                axes[1, i+1].axis('off')
        
        # Add prediction info
        label_text = "Same Person" if label == 1 else "Different Person"
        pred_text = f"Pred: {prediction:.3f}"
        fig.suptitle(f'Twin Verification - GT: {label_text}, {pred_text}', fontsize=16)
        
        plt.tight_layout()
        return fig
    
    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert tensor to numpy array for visualization
        
        Args:
            tensor: Input tensor (C, H, W)
            
        Returns:
            numpy array (H, W, C)
        """
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # Convert from (C, H, W) to (H, W, C)
        tensor = tensor.permute(1, 2, 0)
        
        # Denormalize if needed (assuming ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        tensor = tensor * std + mean
        
        # Clamp to [0, 1]
        tensor = torch.clamp(tensor, 0, 1)
        
        return tensor.detach().cpu().numpy()
    
    def log_model_checkpoint(self, model_path: str, epoch: int, metrics: Dict[str, float]):
        """
        Log model checkpoint to WandB
        
        Args:
            model_path: Path to model checkpoint
            epoch: Current epoch
            metrics: Performance metrics
        """
        if not self.enabled:
            return
        
        try:
            # Create WandB artifact
            artifact = wandb.Artifact(
                name=f'twin_model_epoch_{epoch}',
                type='model',
                metadata={
                    'epoch': epoch,
                    **metrics
                }
            )
            
            # Add model file
            artifact.add_file(model_path)
            
            # Log artifact
            wandb.log_artifact(artifact)
            
            logger.info(f"Model checkpoint logged to WandB: epoch {epoch}")
            
        except Exception as e:
            logger.warning(f"Failed to log model checkpoint: {e}")
    
    def log_training_phase(self, phase: str, epoch: int):
        """
        Log training phase transition
        
        Args:
            phase: Training phase ('base', 'twin_focused', 'attention_refine')
            epoch: Current epoch
        """
        if not self.enabled:
            return
        
        wandb.log({
            'training_phase': phase,
            'phase_transition_epoch': epoch
        }, step=epoch)
    
    def log_dataset_statistics(self, stats: Dict[str, Any]):
        """
        Log dataset statistics
        
        Args:
            stats: Dataset statistics
        """
        if not self.enabled:
            return
        
        wandb.log({
            'dataset_total_pairs': stats.get('total_pairs', 0),
            'dataset_positive_pairs': stats.get('positive_pairs', 0),
            'dataset_negative_pairs': stats.get('negative_pairs', 0),
            'dataset_twin_pairs': stats.get('twin_pairs', 0),
            'dataset_total_persons': stats.get('total_persons', 0),
            'dataset_images_per_person_avg': stats.get('images_per_person_avg', 0),
            'dataset_images_per_person_min': stats.get('images_per_person_min', 0),
            'dataset_images_per_person_max': stats.get('images_per_person_max', 0)
        })
    
    def finish(self):
        """
        Finish WandB run
        """
        if self.enabled:
            wandb.finish()
            logger.info("WandB run finished")


class NoTracker:
    """
    Dummy tracker for when tracking is disabled
    """
    
    def __init__(self, config: Dict[str, Any]):
        pass
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        pass
    
    def log_epoch_metrics(self, epoch: int, train_metrics: Dict[str, float], 
                         val_metrics: Dict[str, float], lr: float):
        pass
    
    def log_verification_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        pass
    
    def log_attention_maps(self, img1: torch.Tensor, img2: torch.Tensor, 
                          attn1: torch.Tensor, attn2: torch.Tensor, 
                          label: int, prediction: float, step: Optional[int] = None):
        pass
    
    def log_model_checkpoint(self, model_path: str, epoch: int, metrics: Dict[str, float]):
        pass
    
    def log_training_phase(self, phase: str, epoch: int):
        pass
    
    def log_dataset_statistics(self, stats: Dict[str, Any]):
        pass
    
    def finish(self):
        pass


def get_tracker(config: Dict[str, Any]) -> Any:
    """
    Get appropriate tracker based on configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tracker instance
    """
    tracking_method = config.get('tracking_method', 'wandb')
    
    if tracking_method == 'wandb':
        return WandBTracker(config)
    elif tracking_method == 'none':
        return NoTracker(config)
    else:
        logger.warning(f"Unknown tracking method: {tracking_method}. Using NoTracker.")
        return NoTracker(config) 