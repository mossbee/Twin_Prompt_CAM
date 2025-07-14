# Twin Face Verification with Prompt-CAM

This repository implements twin face verification using an adapted version of Prompt-CAM for identical twin face recognition. The model learns to distinguish between identical twins by focusing on subtle facial differences through attention mechanisms.

## 🔍 Overview

**Task**: Given two face images, determine whether they are the same person or different people (specifically focusing on identical twins).

**Method**: Dual-input Prompt-CAM with Vision Transformer backbone and 3-phase training strategy.

**Dataset**: ND TWIN 2009-2010 dataset (353 people, 6,182 images, identical twin pairs for hard negative mining).

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n prompt_cam_twin python=3.8
conda activate prompt_cam_twin

# Install dependencies
pip install torch torchvision timm wandb scikit-learn matplotlib scipy pillow

# Clone and setup
git clone <repository>
cd Prompt_CAM
```

### 2. Data Preparation

Ensure your data structure follows:
```
data/
├── train_dataset_infor.json          # Training dataset info
├── test_dataset_infor.json  # Test dataset info
├── train_twin_pairs.json       # Twin pairs for training
├── test_twin_pairs.json     # Twin pairs for testing
└── dataset/                    # Actual image files
    ├── person_id_1/
    │   ├── image1.jpg
    │   └── image2.jpg
    └── person_id_2/
        ├── image1.jpg
        └── image2.jpg
```

### 3. Training

#### Local Training (2x 2080Ti)

```bash
python twin_training.py --config experiment/config/prompt_cam/dino/twin/local_2080ti.yaml
```

#### Kaggle Training (T4)

```bash
python twin_training.py --config experiment/config/prompt_cam/dino/twin/kaggle_t4.yaml
```

#### Kaggle Training (P100)

```bash
python twin_training.py --config experiment/config/prompt_cam/dino/twin/kaggle_p100.yaml
```

### 4. Resume Training

```bash
python twin_training.py --config <config_file> --resume_from_checkpoint <checkpoint_path>
```

### 5. Visualization

```bash
# Visualize multiple pairs
python visualize_twin.py --checkpoint <checkpoint_path> --config <config_file> --num_pairs 10

# Visualize single pair
python visualize_twin.py --checkpoint <checkpoint_path> --config <config_file> --single_pair <img1_path> <img2_path>
```

## 📊 Model Architecture

### Core Components

1. **VisionTransformerPETLTwin**: Dual-input Vision Transformer
   - Processes two images simultaneously
   - Shared backbone with separate attention streams
   - Verification head for binary classification

2. **3-Phase Training Strategy**:
   - **Phase 1 (Base)**: Learn general facial differences (40 epochs, LR=0.005)
   - **Phase 2 (Twin-focused)**: Hard negative mining with twin pairs (30 epochs, LR=0.001)  
   - **Phase 3 (Attention refinement)**: Fine-tune attention maps (30 epochs, LR=0.0001)

3. **Prompt-CAM Adaptation**:
   - 2 prompts: `same_person_prompt`, `different_person_prompt`
   - Binary classification instead of multi-class
   - Attention map comparison between images

## 🔧 Configuration

### Hardware-Specific Settings

| Hardware | Batch Size | Memory | Config File |
|----------|------------|---------|-------------|
| 2x 2080Ti (11GB) | 16 per GPU | Mixed precision | `local_2080ti.yaml` |
| T4 (16GB) | 12 | Mixed precision | `kaggle_t4.yaml` |
| P100 (16GB) | 20 | Mixed precision | `kaggle_p100.yaml` |

### Key Parameters

```yaml
# Twin-specific settings
task_type: twin_verification
class_num: 2
vpt_num: 2  # same_person, different_person prompts

# Training phases
training_phases:
  base:
    epochs: 40
    lr: 0.005
    phase: base
  twin_focused:
    epochs: 30
    lr: 0.001
    phase: twin_focused
  attention_refine:
    epochs: 30
    lr: 0.0001
    phase: attention_refine

# Tracking
tracking_method: wandb  # or mlflow, none
wandb_project: prompt_cam_twin
```

## 📈 Metrics

The implementation tracks comprehensive verification metrics:

- **Verification Accuracy**: Standard accuracy at optimal threshold
- **EER (Equal Error Rate)**: Point where FAR = FRR
- **FAR (False Acceptance Rate)**: At various thresholds (1%, 0.1%, 0.01%)
- **AUC**: Area under ROC curve
- **Precision/Recall/F1**: Standard classification metrics

## 🎯 Usage Examples

### Custom Training Script

```python
from experiment.build_model import get_model
from experiment.build_loader_twin import get_twin_loader_for_phase
from engine.trainer_twin import TwinTrainer

# Load model
params.task_type = 'twin_verification'
model, tune_parameters, _ = get_model(params)

# Create trainer
trainer = TwinTrainer(model, tune_parameters, params)

# Load data for all phases
loaders_dict = {}
for phase in ['base', 'twin_focused', 'attention_refine']:
    train_loader, val_loader, test_loader = get_twin_loader_for_phase(params, logger, phase)
    loaders_dict[phase] = (train_loader, val_loader, test_loader)

# Train
train_metrics, best_metrics = trainer.train_twin_classifier(loaders_dict)
```

### Custom Evaluation

```python
from utils.twin_metrics import TwinVerificationMetrics

# Initialize metrics
metrics = TwinVerificationMetrics()

# Update with predictions
for batch in test_loader:
    predictions, labels, scores = model_inference(batch)
    metrics.update(predictions, labels, scores)

# Compute final metrics
final_metrics = metrics.compute_metrics()
print(f"EER: {final_metrics['eer']:.4f}")
print(f"Verification Accuracy: {final_metrics['verification_accuracy']:.4f}")
```

## 🔬 Visualization Features

### Attention Map Visualization

- **Side-by-side comparison**: Both input images with their attention maps
- **Multiple attention heads**: Visualize different aspects of facial features
- **Verification scores**: Confidence scores for same/different predictions
- **Batch processing**: Visualize multiple pairs at once

### Example Output

```
Twin Pair 1: person_123 vs person_456
Verification Score: 0.1234 (Different Person)
✓ Attention focuses on different facial features

Same Person Pair 1: person_789
Verification Score: 0.8765 (Same Person)  
✓ Attention focuses on consistent facial features
```

## 📝 Training Tips

### Data Optimization

1. **Balanced Sampling**: Ensure 1:1 ratio of positive/negative pairs
2. **Twin Hard Negatives**: Use twin pairs as challenging negative examples
3. **Cross-validation**: Use 5-fold CV to maximize data usage
4. **Augmentation**: Apply face-specific augmentations

### Memory Optimization

1. **Mixed Precision**: Enable for all GPU types
2. **Gradient Accumulation**: Use if batch size is limited
3. **Checkpoint Frequency**: Save every epoch for Kaggle's 12h limit
4. **Attention Visualization**: Disable during training to save memory

### Performance Tuning

1. **Learning Rate**: Start with 0.005, reduce by 5x each phase
2. **Batch Size**: Adjust based on GPU memory (see config files)
3. **Image Size**: 224x224 provides good balance of quality/speed
4. **Early Stopping**: Use patience=101 to train full phases

## 🚨 Common Issues

### Memory Issues
```bash
# Reduce batch size
batch_size: 8  # instead of 16

# Enable gradient accumulation
gradient_accumulation_steps: 2
```

### Checkpoint Issues
```bash
# Resume from latest checkpoint
python twin_training.py --config <config> --resume_from_checkpoint ./checkpoints/checkpoint_latest.pth
```

### Data Loading Issues
```bash
# Check data paths in config
data_path: ./data  # Ensure this points to correct directory
```

## 📊 Expected Results

Based on the dataset characteristics:

- **Training Data**: 353 people, 6,182 images
- **Expected EER**: 10-20% (challenging due to identical twins)
- **Expected Accuracy**: 80-90% at optimal threshold
- **Training Time**: 
  - Local (2x 2080Ti): ~6-8 hours
  - Kaggle (T4): ~8-10 hours
  - Kaggle (P100): ~6-7 hours

## 🔗 References

- [Original Prompt-CAM Paper](https://github.com/Imageomics/Prompt_CAM)
- ND TWIN Dataset: Twins Days Festival 2009-2010
- Vision Transformer: "An Image is Worth 16x16 Words"
- DINO: "Emerging Properties in Self-Supervised Vision Transformers"

## 🤝 Contributing

When modifying the code:

1. Follow the existing code structure
2. Update configuration files accordingly
3. Test on small dataset first
4. Document any new parameters
5. Update this README if needed

## 📄 License

This project follows the same license as the original Prompt-CAM repository. 