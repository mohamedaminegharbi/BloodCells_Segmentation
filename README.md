# BCCD Blood Cell Detection

A deep learning system for detecting and counting blood cells (RBCs, WBCs, and Platelets) in microscopy images using Faster R-CNN with morphological augmentation techniques.

## Overview

This project implements an object detection pipeline for blood cell analysis using:
- **Faster R-CNN** with ResNet50-FPN backbone
- **Morphological augmentation** techniques for enhanced training
- **Mixed training strategy** combining normal and augmented images
- Automated cell counting and analysis

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

### System Requirements

- Python 3.7+
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM
- ~2GB disk space for dataset and models

## Dataset Structure

```
BCCD_Dataset/
└── BCCD/
    ├── JPEGImages/          # Blood cell microscopy images (.jpg)
    └── Annotations/         # XML annotations in PASCAL VOC format
```

The BCCD dataset should contain annotated blood cell images with bounding boxes for RBC, WBC, and Platelets.

## Installation

1. Clone or download the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Update paths in `segmentation.py`:
   ```python
   BASE_DIR = Path(r"your/path/to/BCCD_Dataset/BCCD")
   OUTPUT_DIR = Path(r"your/output/directory")
   ```

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

### Testing/Inference

Use the trained model for inference on new images:

```bash
python testing.py
```

Update the image path in `testing.py`:
```python
image_path = r"path/to/your/image.jpg"
```

### Model Loading

Three model formats are saved:

```python
# Format 1: Weights only (recommended)
model.load_state_dict(torch.load('best_detection_model.pth'))

# Format 2: Full checkpoint (includes optimizer state)
checkpoint = torch.load('best_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Format 3: Complete model
model = torch.load('complete_model.pth')
```

## Morphological Augmentation Techniques

The system applies 6 different morphological operations to enhance training:

1. **Watershed**: Boundary detection and segmentation
2. **Skeletonization**: Edge detection and morphological gradients
3. **Dilation**: Expands object boundaries
4. **Erosion**: Shrinks object boundaries
5. **Opening**: Removes small noise (erosion → dilation)
6. **Closing**: Fills small gaps (dilation → erosion)

Each technique creates unique variations that help the model learn robust features.

## Output Files

After training, the following files are generated:

```
BCCD_detection/
├── best_detection_model.pth          # Model weights (lightest)
├── best_checkpoint.pth               # Full checkpoint with optimizer
├── complete_model.pth                # Full model with architecture
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

## Key Parameters

### Detection Confidence Threshold
```python
confidence_threshold = 0.5  # Adjust for precision/recall tradeoff
```

### Training Configuration
```python
num_epochs = 5              # Number of training epochs
batch_size = 8              # Batch size for training
learning_rate = 0.001       # Learning rate
total_images = 100          # Total mixed training images
```

### Class Mapping
```python
CLASS_NAMES = {
    0: 'Background',
    1: 'RBC',           # Red Blood Cells
    2: 'WBC',           # White Blood Cells
    3: 'Platelets'
}
```

## Performance Optimization

The code includes several optimizations for faster training:

- GPU acceleration with CUDA
- Mixed precision training
- cuDNN benchmarking
- Data loading with multiple workers
- Pin memory for faster GPU transfer
- Gradient clipping for stability
- Backbone freezing (only train detection heads)

## Results Interpretation

The system provides:
- **Bounding boxes** around detected cells
- **Confidence scores** for each detection
- **Cell counts** per class (RBC, WBC, Platelets)
- **Statistical summaries** across the test set

Example output:
```
BloodImage_00003:
  RBC: 45, WBC: 3, Platelets: 12
  
Average counts per image:
  RBC: 42.3
  WBC: 2.8
  Platelets: 10.5
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in training
- Reduce image resolution
- Use fewer workers in DataLoader

### Slow Training on CPU
- Training on CPU is significantly slower
- Consider using Google Colab or cloud GPU
- Reduce dataset size for testing

### Invalid Boxes Error
- The code automatically filters invalid boxes
- Check XML annotations for corrupted data
- Ensure boxes have xmax > xmin and ymax > ymin

## Citation

If you use this code or the BCCD dataset, please cite:

```
BCCD Dataset
https://github.com/Shenggan/BCCD_Dataset
```

## License

This project is provided for educational and research purposes. Please check the BCCD dataset license for usage restrictions.

## Contact

For questions or issues, please open an issue in the repository or contact the development team.

---

**Note**: This is a research/educational implementation. For clinical use, the model should be thoroughly validated and comply with relevant medical device regulations.
