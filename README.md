# BCCD Blood Cell Detection

A deep learning system for detecting and counting blood cells (RBCs, WBCs, and Platelets) in microscopy images using Faster R-CNN with morphological augmentation techniques.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Usage](#usage)
  - [Training](#training)
  - [Testing/Inference](#testinginference)
  - [Model Loading](#model-loading)
- [Morphological Augmentation](#morphological-augmentation-techniques)
- [Output Files](#output-files)
- [Configuration](#key-parameters)
- [Performance](#performance-optimization)
- [Results](#results-interpretation)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Overview

This project implements an object detection pipeline for blood cell analysis using:
- **Faster R-CNN** with ResNet50-FPN backbone
- **Morphological augmentation** techniques for enhanced training
- **Mixed training strategy** combining normal and augmented images
- Automated cell counting and analysis

## Architecture

<img width="946" height="451" alt="faster-RCNN" src="https://github.com/user-attachments/assets/a15ccaad-b7d2-4b91-8b34-e011a8c3c872" />

The system uses the Faster R-CNN architecture with the following components:

1. **Convolutional Backbone**: ResNet50 with Feature Pyramid Network (FPN) extracts multi-scale features from input images
2. **Region Proposal Network (RPN)**: Generates candidate bounding boxes (proposals) for potential cell locations
3. **RoI Pooling**: Extracts fixed-size feature maps from each proposed region
4. **Detection Head**: 
   - **Classification layer (cls)**: Predicts cell type (RBC, WBC, Platelets, or Background)
   - **Regression layer (reg)**: Refines bounding box coordinates for precise localization

This two-stage detection approach allows for accurate identification and counting of blood cells in microscopy images.

## Features

- ✅ Detection of 3 blood cell types: RBC, WBC, and Platelets
- ✅ 6 morphological augmentation techniques (watershed, skeletonization, dilation, erosion, opening, closing)
- ✅ Mixed training with 100 images (14 normal + 86 augmented)
- ✅ GPU-accelerated training with mixed precision
- ✅ Automated cell counting and CSV export
- ✅ Visualization of detection results
- ✅ Multiple model save formats for flexibility

## Requirements

### Python Dependencies

```bash
pip install torch torchvision opencv-python numpy pandas matplotlib albumentations scikit-learn tqdm
```

### Detailed Package Versions

```
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
numpy>=1.19.0
pandas>=1.2.0
matplotlib>=3.3.0
albumentations>=1.0.0
scikit-learn>=0.24.0
tqdm>=4.60.0
```

### System Requirements

- Python 3.7+
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM
- ~2GB disk space for dataset and models

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd bccd-blood-cell-detection
```

### Step 2: Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchvision opencv-python numpy pandas matplotlib albumentations scikit-learn tqdm
```

### Step 4: Create Required Directories

```bash
mkdir -p assets
mkdir -p data
mkdir -p outputs
```

### Step 5: Download Dataset

Download the BCCD dataset from [GitHub](https://github.com/Shenggan/BCCD_Dataset) and place it in the `data` directory.

### Step 6: Update Configuration

Edit the paths in `segmentation.py`:

```python
BASE_DIR = Path(r"data/BCCD_Dataset/BCCD")
OUTPUT_DIR = Path(r"outputs/BCCD_detection")
```

## Dataset Structure

The dataset contains three types of blood cells:
- **RBC (Red Blood Cells)**: The most abundant cells, appearing as pink circular cells
- **WBC (White Blood Cells)**: Larger cells with dark purple nuclei
- **Platelets**: Small cell fragments scattered throughout

```
data/
└── BCCD_Dataset/
    └── BCCD/
        ├── JPEGImages/          # Blood cell microscopy images (.jpg)
        └── Annotations/         # XML annotations in PASCAL VOC format
```

The BCCD dataset should contain annotated blood cell images with bounding boxes for RBC, WBC, and Platelets.

## Usage

### Training

Run the complete training pipeline:

```bash
python segmentation.py
```

**Training Pipeline:**
1. Dataset preparation and train/val/test split (70/15/15)
2. Generation of 100 mixed training images (14 normal + 86 augmented)
3. Morphological augmentation examples visualization
4. Model training for 5 epochs with GPU optimization
5. Testing on sample images
6. Cell counting on entire test set

**Training Details:**
- Batch size: 8
- Epochs: 5
- Optimizer: AdamW (lr=0.001)
- Scheduler: OneCycleLR
- Mixed precision training enabled
- Gradient clipping for stability

**Expected Output:**
```
================================================================================
BCCD BLOOD CELL DETECTION - 100 IMAGES MIXED TRAINING
================================================================================

🔍 Step 1: Preparing datasets...
Found 364 annotated images
Split: Train=254, Val=55, Test=55

🔬 Step 2: Generating mixed training set (100 images)...
   Base images: 14
   Target augmented images: ~86

   Generating augmentations...
Augmenting: 100%|██████████| 14/14 [00:05<00:00,  2.45it/s]

✓ Generated training set:
   Normal images: 14
   Augmented images: 84
   Total training images: 98

🚀 Step 6: Training on 100 mixed images...

Epoch 1/5
Training: 100%|██████████| 13/13 [00:45<00:00,  3.51s/it, loss=1.2345]
  Loss: 1.2345
  ✓ New best model! (loss: 1.2345)
    Saved: best_detection_model.pth
    Saved: best_checkpoint.pth
    Saved: complete_model.pth

...

Training completed! Best loss: 0.5678
```

### Testing/Inference

Use the trained model for inference on new images:

```bash
python testing.py
```

Update the image path in `testing.py`:
```python
image_path = r"path/to/your/image.jpg"
```

**Example Output:**
```python
# Detection results displayed in matplotlib window
# Shows bounding boxes with class labels and confidence scores
# Displays counts: RBC: 45, WBC: 3, Platelets: 12
```

### Model Loading

Three model formats are saved for different use cases:

```python
# Format 1: Weights only (recommended - smallest file size)
model = create_detection_model(num_classes=4)
model.load_state_dict(torch.load('best_detection_model.pth'))
model.eval()

# Format 2: Full checkpoint (includes optimizer state for resuming training)
checkpoint = torch.load('best_checkpoint.pth')
model = create_detection_model(num_classes=4)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

# Format 3: Complete model (architecture + weights - largest file)
model = torch.load('complete_model.pth')
model.eval()
```

## Morphological Augmentation Techniques

<img width="392" height="128" alt="a" src="https://github.com/user-attachments/assets/f10188f9-02ce-49e5-af0c-0a80636db346" />

The system applies 6 different morphological operations to enhance training:

| Technique | Description | Effect |
|-----------|-------------|--------|
| **Watershed** | Boundary detection and segmentation | Emphasizes cell boundaries and separation |
| **Skeletonization** | Edge detection and morphological gradients | Highlights cell edges and structures |
| **Dilation** | Expands object boundaries | Makes cells appear larger |
| **Erosion** | Shrinks object boundaries | Makes cells appear smaller |
| **Opening** | Removes small noise (erosion → dilation) | Cleans up small artifacts |
| **Closing** | Fills small gaps (dilation → erosion) | Fills holes in cells |

![b](https://github.com/user-attachments/assets/42d44a11-bd83-43bd-9a14-9db063063693)

Each technique creates unique variations that help the model learn robust features and improves generalization to various imaging conditions.

### Why Morphological Augmentation?

Traditional augmentation (rotation, flip, brightness) modifies the appearance but doesn't change the structural properties of cells. Morphological augmentation:
- Simulates different microscopy conditions
- Helps the model learn invariant features
- Improves robustness to image quality variations
- Increases effective training data by 6x per image

## Output Files

After training, the following files are generated:

```
outputs/
└── BCCD_detection/
    ├── best_detection_model.pth          # Model weights (lightest, ~160MB)
    ├── best_checkpoint.pth               # Full checkpoint with optimizer (~320MB)
    ├── complete_model.pth                # Full model with architecture (~165MB)
    ├── cell_counts.csv                   # Cell counts for test set
    ├── morph_example_1.png               # Morphological operations comparison
    ├── morph_example_2.png
    ├── morph_example_3.png
    ├── detection_result_1.png            # Detection visualizations
    ├── detection_result_2.png
    ├── detection_result_3.png
    ├── detection_result_4.png
    ├── detection_result_5.png
    ├── checkpoints/                      # Training checkpoints
    │   ├── model_epoch_2.pth
    │   ├── model_epoch_4.pth
    │   └── model_epoch_5.pth
    └── morphological_augmentations/      # Augmented images
        ├── watershed/
        ├── skeleton/
        ├── dilation/
        ├── erosion/
        ├── opening/
        └── closing/
```

### CSV Output Format

The `cell_counts.csv` contains:

| Column | Description |
|--------|-------------|
| `image` | Image filename |
| `rbc_count` | Number of detected Red Blood Cells |
| `wbc_count` | Number of detected White Blood Cells |
| `platelet_count` | Number of detected Platelets |
| `total_cells` | Total number of detected cells |

Example:
```csv
image,rbc_count,wbc_count,platelet_count,total_cells
BloodImage_00001,42,3,15,60
BloodImage_00002,38,2,12,52
BloodImage_00003,45,4,18,67
```

## Key Parameters

### Detection Confidence Threshold

```python
confidence_threshold = 0.5  # Adjust for precision/recall tradeoff
# Lower values (0.3-0.4): More detections, but more false positives
# Higher values (0.6-0.7): Fewer detections, but higher precision
```

### Training Configuration

```python
num_epochs = 5              # Number of training epochs
batch_size = 8              # Batch size for training
learning_rate = 0.001       # Learning rate for optimizer
total_images = 100          # Total mixed training images
num_base_images = 14        # Number of original images
```

### Data Split

```python
train_ratio = 0.70          # 70% for training
val_ratio = 0.15            # 15% for validation
test_ratio = 0.15           # 15% for testing
```

### Class Mapping

```python
CLASS_NAMES = {
    0: 'Background',
    1: 'RBC',           # Red Blood Cells
    2: 'WBC',           # White Blood Cells
    3: 'Platelets'
}

CLASS_COLORS = {
    0: (0, 0, 0),       # Black
    1: (255, 0, 0),     # Red
    2: (0, 255, 0),     # Green
    3: (0, 100, 255)    # Orange
}
```

## Performance Optimization

The code includes several optimizations for faster training:

### GPU Acceleration

```python
# Automatic GPU detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Enable cuDNN benchmarking for faster convolutions
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
```

### Mixed Precision Training

```python
# Automatically enabled when using CUDA
# Reduces memory usage and increases training speed
```

### Data Loading Optimization

```python
# Multiple workers for parallel data loading
DataLoader(dataset, batch_size=8, num_workers=4, pin_memory=True)

# Pin memory for faster GPU transfer
pin_memory=True if torch.cuda.is_available() else False
```

### Training Optimizations

- **Backbone Freezing**: Only trains detection heads, not the entire ResNet50
- **Gradient Clipping**: Prevents exploding gradients (max_norm=10.0)
- **OneCycleLR Scheduler**: Improves convergence speed
- **AdamW Optimizer**: Better generalization than standard Adam

### Memory Management

```python
# Efficient gradient zeroing
optimizer.zero_grad(set_to_none=True)

# Non-blocking GPU transfers
images = [img.to(device, non_blocking=True) for img in images]
```

## Results Interpretation

### Detection Visualization

The system provides comprehensive visualizations:

1. **Bounding Boxes**: Color-coded by cell type
   - Red: RBC
   - Green: WBC
   - Orange: Platelets

2. **Confidence Scores**: Displayed above each detection
   - Range: 0.0 to 1.0
   - Higher scores indicate more confident predictions

3. **Cell Counts**: Overlaid on the image
   - Per-class counts
   - Total cell count

### Example Results

```
BloodImage_00003:
  RBC: 45, WBC: 3, Platelets: 12
  Total Cells: 60

Average counts per image:
  RBC: 42.3 ± 8.5
  WBC: 2.8 ± 1.2
  Platelets: 10.5 ± 3.4
```

### Statistical Analysis

The `cell_counts.csv` provides data for:
- Mean cell counts per type
- Standard deviation
- Min/max values
- Distribution analysis

```python
# Load and analyze results
import pandas as pd
df = pd.read_csv('outputs/BCCD_detection/cell_counts.csv')

print(df.describe())
print(f"Total images analyzed: {len(df)}")
print(f"Average RBC per image: {df['rbc_count'].mean():.1f}")
print(f"Average WBC per image: {df['wbc_count'].mean():.1f}")
print(f"Average Platelets per image: {df['platelet_count'].mean():.1f}")
```

## Troubleshooting

### CUDA Out of Memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# Reduce batch size
batch_size = 4  # or even 2

# Reduce number of workers
num_workers = 2  # or 0

# Clear cache between epochs
torch.cuda.empty_cache()
```

### Slow Training on CPU

**Problem**: Training is very slow without GPU

**Solutions**:
- Use Google Colab with free GPU
- Use cloud services (AWS, Azure, GCP)
- Reduce dataset size for testing
- Consider using CPU-optimized smaller models

```python
# Check if GPU is available
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ WARNING: Training on CPU will be slow!")
```

### Invalid Boxes Error

**Problem**: `RuntimeError: boxes must be valid` or boxes with zero width/height

**Solution**: The code automatically filters invalid boxes, but if issues persist:

```python
# Check XML annotations
for box in boxes:
    xmin, ymin, xmax, ymax = box
    if xmax <= xmin or ymax <= ymin:
        print(f"Invalid box detected: {box}")
```

### File Not Found Errors

**Problem**: `FileNotFoundError: Could not load image`

**Solutions**:
```python
# Verify paths
print(f"Images directory exists: {JPEG_IMAGES_DIR.exists()}")
print(f"Annotations directory exists: {ANNOTATIONS_DIR.exists()}")

# List available images
images = list(JPEG_IMAGES_DIR.glob('*.jpg'))
print(f"Found {len(images)} images")
```

### Low Detection Accuracy

**Problem**: Model not detecting cells well

**Solutions**:
1. **Adjust confidence threshold**:
   ```python
   confidence_threshold = 0.3  # Lower for more detections
   ```

2. **Train for more epochs**:
   ```python
   num_epochs = 10  # Increase from 5
   ```

3. **Increase training data**:
   ```python
   total_images = 200  # Double the training set
   ```

4. **Unfreeze backbone** (advanced):
   ```python
   # Comment out freezing in segmentation.py
   # for param in model.backbone.parameters():
   #     param.requires_grad = False
   ```

### Package Version Conflicts

**Problem**: Import errors or version incompatibilities

**Solution**:
```bash
# Create fresh environment
python -m venv fresh_env
source fresh_env/bin/activate  # Windows: fresh_env\Scripts\activate

# Install specific versions
pip install torch==1.12.0 torchvision==0.13.0
pip install opencv-python==4.6.0 albumentations==1.2.0
```

## Citation

If you use this code or the BCCD dataset in your research, please cite:

### BCCD Dataset
```bibtex
@misc{bccd_dataset,
  author = {Shenggan},
  title = {BCCD Dataset},
  year = {2017},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Shenggan/BCCD_Dataset}},
}
```

### Faster R-CNN
```bibtex
@inproceedings{ren2015faster,
  title={Faster r-cnn: Towards real-time object detection with region proposal networks},
  author={Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
  booktitle={Advances in neural information processing systems},
  pages={91--99},
  year={2015}
}
```

### ResNet
```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```

## License

This project is provided for educational and research purposes. Please check the BCCD dataset license for usage restrictions.

### Dataset License
The BCCD dataset is available under the MIT License. Please refer to the [original repository](https://github.com/Shenggan/BCCD_Dataset) for details.

### Code License
MIT License - Feel free to use and modify for your projects.

## Contact

For questions, issues, or contributions:

- **Issues**: Open an issue in the repository
- **Pull Requests**: Contributions are welcome!
- **Email**: [your-email@example.com]

## Acknowledgments

- BCCD Dataset creators for providing annotated blood cell images
- PyTorch team for the deep learning framework
- Torchvision for pre-trained models and utilities
- Albumentations for data augmentation tools

---

**⚠️ Disclaimer**: This is a research/educational implementation. For clinical or diagnostic use, the model must be:
- Thoroughly validated on diverse datasets
- Tested for various blood conditions
- Compliant with medical device regulations (FDA, CE marking, etc.)
- Reviewed and approved by medical professionals

**Never use this model for medical diagnosis without proper validation and regulatory approval.**

---

## Project Structure

```
bccd-blood-cell-detection/
├── assets/
│   ├── faster_rcnn_architecture.png
│   ├── morphological_operations.png
│   └── blood_cells_example.png
├── data/
│   └── BCCD_Dataset/
│       └── BCCD/
│           ├── JPEGImages/
│           └── Annotations/
├── outputs/
│   └── BCCD_detection/
│       ├── best_detection_model.pth
│       ├── cell_counts.csv
│       └── ...
├── segmentation.py           # Main training script
├── testing.py               # Inference script
├── README.md                # This file
├── requirements.txt         # Python dependencies
└── .gitignore              # Git ignore file
```

## Quick Start

```bash
# 1. Clone and setup
git clone <repository-url>
cd bccd-blood-cell-detection
pip install -r requirements.txt

# 2. Download dataset
# Place BCCD dataset in data/BCCD_Dataset/BCCD/

# 3. Train model
python segmentation.py

# 4. Test on new image
python testing.py
```

## Version History

- **v1.0.0** (2024): Initial release
  - Faster R-CNN implementation
  - 6 morphological augmentation techniques
  - Mixed training strategy (100 images)
  - Automated cell counting
  - GPU optimization

---

**Happy Cell Counting! 🔬🩸**

