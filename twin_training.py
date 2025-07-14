import argparse
import time
import os
from experiment.build_model import get_model
from experiment.build_loader_twin import get_twin_loader_for_phase
from engine.trainer_twin import TwinTrainer
from utils.setup_logging import get_logger
from utils.misc import set_seed, load_yaml, override_args_with_yaml

logger = get_logger("Prompt_CAM")


def main():
    args = setup_parser().parse_args()

    if args.config:
        yaml_config = load_yaml(args.config)
        if yaml_config:
            args = override_args_with_yaml(args, yaml_config)

    set_seed(args.random_seed)
    start = time.time()
    
    # Set visualization flag
    args.vis_attn = getattr(args, 'vis_attn', False)
    
    # Run twin verification training
    twin_training_run(args)
    
    end = time.time()
    logger.info(f'----------- Total Run time : {(end - start) / 60:.2f} mins-----------')


def twin_training_run(params):
    """Main twin training run with 3-phase strategy"""
    
    logger.info("Starting Twin Face Verification Training")
    logger.info(f"Task: {getattr(params, 'task_type', 'twin_verification')}")
    logger.info(f"Model: {params.model}")
    logger.info(f"Pretrained weights: {params.pretrained_weights}")
    logger.info(f"Data path: {params.data_path}")
    
    # Build model
    logger.info("Building twin verification model...")
    model, tune_parameters, model_grad_params_no_head = get_model(params)
    
    logger.info(f"Model: {type(model).__name__}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in tune_parameters)}")
    
    # Create trainer
    trainer = TwinTrainer(model, tune_parameters, params)
    
    # Load data for all phases
    logger.info("Loading datasets for all training phases...")
    loaders_dict = {}
    
    # Get training phases configuration
    training_phases = getattr(params, 'training_phases', {
        'base': {'epochs': 40, 'lr': 0.005, 'phase': 'base'},
        'twin_focused': {'epochs': 30, 'lr': 0.001, 'phase': 'twin_focused'},
        'attention_refine': {'epochs': 30, 'lr': 0.0001, 'phase': 'attention_refine'}
    })
    
    # Load data loaders for each phase
    for phase_name, phase_config in training_phases.items():
        logger.info(f"Loading data for phase: {phase_name}")
        
        train_loader, val_loader, test_loader = get_twin_loader_for_phase(
            params, logger, phase_config['phase']
        )
        
        loaders_dict[phase_name] = (train_loader, val_loader, test_loader)
        
        # Log dataset size
        if train_loader:
            logger.info(f"Phase {phase_name} - Train batches: {len(train_loader)}")
        if val_loader:
            logger.info(f"Phase {phase_name} - Val batches: {len(val_loader)}")
        if test_loader:
            logger.info(f"Phase {phase_name} - Test batches: {len(test_loader)}")
    
    # Start 3-phase training
    logger.info("Starting 3-phase training...")
    train_metrics, best_metrics = trainer.train_twin_classifier(loaders_dict)
    
    # Log final results
    logger.info("Training completed successfully!")
    logger.info(f"Best metrics achieved:")
    for key, value in best_metrics.items():
        logger.info(f"  {key}: {value}")
    
    return train_metrics, best_metrics


def setup_parser():
    parser = argparse.ArgumentParser(description='Twin Face Verification with Prompt-CAM')

    ######################## Pretrained Model #########################
    parser.add_argument('--pretrained_weights', type=str, default='vit_base_patch16_dino',
                        choices=['vit_base_patch16_224_in21k', 'vit_base_mae', 'vit_base_patch14_dinov2',
                                 'vit_base_patch16_dino', 'vit_base_patch16_clip_224'],
                        help='pretrained weights name')
    parser.add_argument('--drop_path_rate', default=0.1, type=float,
                        help='Drop Path Rate (default: %(default)s)')
    parser.add_argument('--model', type=str, default='dino', choices=['vit', 'dino', 'dinov2'],
                        help='pretrained model name')
    parser.add_argument('--train_type', type=str, default='prompt_cam', choices=['vpt', 'prompt_cam', 'linear'],
                        help='training type')

    ######################## Twin-specific Settings #########################
    parser.add_argument('--task_type', type=str, default='twin_verification',
                        help='Task type for twin verification')
    parser.add_argument('--data', type=str, default='twin',
                        help='Dataset name')
    parser.add_argument('--data_path', type=str, default='./data',
                        help='Path to dataset')

    ######################## Optimizer Scheduler #########################
    parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam', 'adamw'],
                        help='Optimizer (default: %(default)s)')
    parser.add_argument('--lr', default=0.005, type=float,
                        help='Learning rate (default: %(default)s)')
    parser.add_argument('--epoch', default=100, type=int,
                        help='Total number of epochs')
    parser.add_argument('--warmup_epoch', default=20, type=int,
                        help='Warmup epochs')
    parser.add_argument('--warmup_lr_init', default=0, type=float,
                        help='Initial warmup learning rate')
    parser.add_argument('--lr_min', default=1e-6, type=float,
                        help='Minimum learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum for SGD')
    parser.add_argument('--wd', default=0.001, type=float,
                        help='Weight decay')

    ######################## Training Settings #########################
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Training batch size')
    parser.add_argument('--test_batch_size', default=32, type=int,
                        help='Test batch size')
    parser.add_argument('--crop_size', default=224, type=int,
                        help='Input image size')
    parser.add_argument('--eval_freq', default=10, type=int,
                        help='Evaluation frequency (epochs)')

    ######################## VPT/Prompt Settings #########################
    parser.add_argument('--vpt_num', default=2, type=int,
                        help='Number of prompts (same_person, different_person)')
    parser.add_argument('--vpt_dropout', default=0.1, type=float,
                        help='VPT dropout rate')
    parser.add_argument('--vpt_layer', default=None, type=int,
                        help='VPT layer specification')
    parser.add_argument('--vpt_mode', default=None, type=str,
                        help='VPT mode')

    ######################## System Settings #########################
    parser.add_argument('--random_seed', default=42, type=int,
                        help='Random seed')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--gpu_num', default=1, type=int,
                        help='Number of GPUs')
    parser.add_argument('--class_num', default=2, type=int,
                        help='Number of classes (binary for verification)')

    ######################## Checkpoint and Output #########################
    parser.add_argument('--checkpoint_dir', default='./checkpoints', type=str,
                        help='Checkpoint directory')
    parser.add_argument('--checkpoint_every_epoch', action='store_true', default=True,
                        help='Save checkpoint every epoch')
    parser.add_argument('--resume_from_checkpoint', default=None, type=str,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--store_ckp', action='store_true', default=True,
                        help='Store checkpoints')

    ######################## Tracking Settings #########################
    parser.add_argument('--tracking_method', default='wandb', choices=['wandb', 'mlflow', 'none'],
                        help='Tracking method')
    parser.add_argument('--wandb_enabled', action='store_true', default=True,
                        help='Enable WandB tracking')
    parser.add_argument('--wandb_project', default='prompt_cam_twin', type=str,
                        help='WandB project name')
    parser.add_argument('--wandb_entity', default=None, type=str,
                        help='WandB entity')
    parser.add_argument('--wandb_run_name', default=None, type=str,
                        help='WandB run name')

    ######################## Other Settings #########################
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--final_run', action='store_true', default=True,
                        help='Final run (no validation split)')
    parser.add_argument('--early_patience', default=101, type=int,
                        help='Early stopping patience')
    parser.add_argument('--max_grad_norm', default=1.0, type=float,
                        help='Maximum gradient norm for clipping')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training')

    return parser


if __name__ == '__main__':
    main() 