import torch
from data.dataset.twin import get_twin
from utils.setup_logging import get_logger

logger = get_logger("Prompt_CAM")


def get_twin_dataset(params, phase='base'):
    """
    Get twin datasets for training and testing
    
    Args:
        params: Parameters object
        phase: Training phase ('base', 'twin_focused', 'attention_refine')
    """
    dataset_train, dataset_val, dataset_test = None, None, None
    
    logger.info(f"Loading Twin Face Verification data (phase: {phase})...")
    
    if hasattr(params, 'final_run') and params.final_run:
        logger.info("Loading training data (final training data for twin verification)...")
        dataset_train = get_twin(params, mode='train', phase=phase)
        dataset_test = get_twin(params, mode='test', phase=phase)
    else:
        # For development/debugging, create train/val split
        logger.info("Loading train/val split for twin verification...")
        full_dataset = get_twin(params, mode='train', phase=phase)
        
        # Split dataset into train/val (80/20 split)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        dataset_train, dataset_val = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        dataset_test = get_twin(params, mode='test', phase=phase)
    
    return dataset_train, dataset_val, dataset_test


def get_twin_loader(params, logger, phase='base'):
    """
    Get twin data loaders following the existing pattern
    
    Args:
        params: Parameters object
        logger: Logger instance
        phase: Training phase ('base', 'twin_focused', 'attention_refine')
    """
    # Set data path to the data directory
    if not hasattr(params, 'data_path'):
        params.data_path = './data'
    
    # Get datasets
    dataset_train, dataset_val, dataset_test = get_twin_dataset(params, phase)
    
    # Generate loaders
    train_loader, val_loader, test_loader = gen_twin_loader(
        params, dataset_train, dataset_val, dataset_test
    )
    
    logger.info("Finish setup twin loaders")
    return train_loader, val_loader, test_loader


def gen_twin_loader(params, dataset_train, dataset_val, dataset_test):
    """
    Generate twin data loaders
    
    Args:
        params: Parameters object
        dataset_train: Training dataset
        dataset_val: Validation dataset
        dataset_test: Test dataset
    """
    train_loader, val_loader, test_loader = None, None, None
    
    # Set number of workers
    if hasattr(params, 'debug') and params.debug:
        num_workers = 1
    else:
        num_workers = 4
    
    # Create training loader
    if dataset_train is not None:
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=params.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    # Create validation loader
    if dataset_val is not None:
        val_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=params.test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    # Create test loader
    if dataset_test is not None:
        test_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=params.test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader, test_loader


def get_twin_loader_for_phase(params, logger, phase):
    """
    Convenience function to get loader for specific training phase
    
    Args:
        params: Parameters object
        logger: Logger instance
        phase: Training phase ('base', 'twin_focused', 'attention_refine')
    """
    return get_twin_loader(params, logger, phase) 