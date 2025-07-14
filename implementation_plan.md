# Implementation Plan: Prompt-CAM Twin Face Verification

## Overview
Adapt existing Prompt-CAM codebase for twin face verification using Option B (dual-input) approach. Prioritize accuracy, use WandB tracking, implement 3-phase training strategy.

## Core Architecture Changes

### Model Modifications
- **File**: `model/vision_transformer.py`
- **Purpose**: Extend existing `VisionTransformerPETL` to `VisionTransformerPETLTwin`
- **Changes**: Add dual-input forward pass, verification head, return attention maps

### Data Pipeline
- **File**: `experiment/build_loader_twin.py` (NEW)
- **Purpose**: Create twin pair dataset from existing data files
- **Logic**: Generate positive pairs (same person) and negative pairs (different person + twin pairs as hard negatives)

### Training Infrastructure
- **File**: `twin_training.py` (NEW)
- **Purpose**: Main training script for twin verification
- **Reuse**: Leverage existing `main.py` structure

## Key Components to Implement

### 1. Twin Dataset Class
```python
class TwinPairDataset:
    # Load from train_dataset_infor.json, train_twin_pairs.json
    # Generate balanced positive/negative pairs
    # Include twin pairs as hard negatives
```

### 2. Verification Head
```python
class VerificationHead:
    # Binary classification: same/different person
    # Input: two feature vectors
    # Output: similarity score + attention maps
```

### 3. WandB Integration
- **File**: `utils/wandb_tracker.py` (NEW)
- **Purpose**: Track metrics, attention visualizations, model checkpoints
- **Integration**: Add to existing trainer.py

## Configuration System

### Config Files Structure
```
experiment/config/prompt_cam/dino/twin/
├── base.yaml           # Base twin verification config
├── local_2080ti.yaml   # Local GPU setup
├── kaggle_t4.yaml      # Kaggle T4 single/dual
├── kaggle_p100.yaml    # Kaggle P100 setup
```

### Key Config Parameters
- `task_type: twin_verification`
- `class_num: 2` (same/different)
- `vpt_num: 2` (same_person_prompt, different_person_prompt)
- `batch_size: 16/12/20` (hardware dependent)
- `crop_size: 224`

## Training Strategy (3 Phases)

### Phase 1: Base Training
- **Epochs**: 40
- **Data**: All person pairs (balanced positive/negative)
- **Focus**: Learn basic facial differences
- **LR**: 0.005

### Phase 2: Twin-Focused Training
- **Epochs**: 30
- **Data**: Emphasize twin pairs as hard negatives
- **Focus**: Distinguish between identical twins
- **LR**: 0.001

### Phase 3: Attention Refinement
- **Epochs**: 30
- **Data**: All data with attention consistency loss
- **Focus**: Improve attention map quality
- **LR**: 0.0001

## Files to Create/Modify

### NEW Files
1. `twin_training.py` - Main training script
2. `experiment/build_loader_twin.py` - Twin dataset loader
3. `utils/wandb_tracker.py` - WandB integration
4. `utils/twin_metrics.py` - EER, FAR, verification accuracy
5. `visualize_twin.py` - Twin attention visualization
6. `experiment/config/prompt_cam/dino/twin/*.yaml` - Hardware configs

### MODIFY Files
1. `model/vision_transformer.py` - Add VisionTransformerPETLTwin
2. `experiment/build_model.py` - Add twin model support
3. `engine/trainer.py` - Add verification metrics, checkpointing
4. `experiment/run.py` - Add twin task support

## Checkpoint Strategy
- Save every epoch (Kaggle 12h timeout)
- Resume from checkpoint capability
- WandB artifact storage for checkpoints

## Evaluation Protocol
- Test on `test_dataset_infor.json`
- Metrics: EER, FAR, Verification Accuracy
- Twin pairs from `test_twin_pairs.json`

## Hardware Optimization
- Image size: 224x224
- Batch sizes: 16 (2080Ti), 12 (T4), 20 (P100)
- Distributed training support for multi-GPU
- Mixed precision training

## Implementation Order
1. Create twin dataset loader
2. Modify model architecture
3. Add WandB tracking
4. Implement verification metrics
5. Create training configs
6. Implement 3-phase training
7. Add checkpoint/resume functionality
8. Create visualization tools 