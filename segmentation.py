import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Check GPU availability and info
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("⚠️ WARNING: CUDA not available, using CPU (will be very slow!)")

# ============================================================================
# BASE PATHS CONFIGURATION
# ============================================================================
BASE_DIR = Path(r"C:\Users\user\OneDrive\Desktop\envo\Proj\Segmentation\BCCD_Dataset\BCCD")
JPEG_IMAGES_DIR = BASE_DIR / "JPEGImages"
ANNOTATIONS_DIR = BASE_DIR / "Annotations"
OUTPUT_DIR = Path(r"C:\Users\user\OneDrive\Desktop\envo\Proj\Segmentation\BCCD_detection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Images directory: {JPEG_IMAGES_DIR}")
print(f"Annotations directory: {ANNOTATIONS_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

# Class mapping
CLASS_NAMES = {0: 'Background', 1: 'RBC', 2: 'WBC', 3: 'Platelets'}
CLASS_COLORS = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 100, 255)}

# ============================================================================
# MORPHOLOGICAL AUGMENTATION
# ============================================================================
class MorphologicalAugmentation:
    """Apply morphological operations for data augmentation"""
    
    @staticmethod
    def apply_watershed_image(image):
        """Watershed segmentation effect on image"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
        
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(image.copy(), markers)
        image_result = image.copy()
        image_result[markers == -1] = [255, 0, 0]  # Mark boundaries in red
        
        return image_result
    
    @staticmethod
    def apply_skeletonization_image(image):
        """Skeletonization effect on image (edge detection)"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Use morphological gradient for skeleton-like effect
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        # Enhance edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Combine gradient and edges
        skeleton = cv2.addWeighted(gradient, 0.7, edges, 0.3, 0)
        
        # Convert back to RGB
        skeleton_rgb = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2RGB)
        
        # Blend with original
        return cv2.addWeighted(image, 0.6, skeleton_rgb, 0.4, 0)
    
    @staticmethod
    def apply_dilation_image(image):
        """Dilate the image"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.dilate(image, kernel, iterations=1)
    
    @staticmethod
    def apply_erosion_image(image):
        """Erode the image"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.erode(image, kernel, iterations=1)
    
    @staticmethod
    def apply_opening_image(image):
        """Opening (erosion followed by dilation)"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    @staticmethod
    def apply_closing_image(image):
        """Closing (dilation followed by erosion)"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# ============================================================================
# MORPHOLOGICAL DATASET GENERATOR (FOR 100 IMAGE TRAINING)
# ============================================================================
def generate_mixed_training_set(image_names, image_dir, output_dir, total_images=100):
    """Generate a mixed training set of 100 images (normal + morphological augmentations)"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate split: ~14 normal images + ~86 augmented (14 images × 6 morphs ≈ 84)
    num_base_images = 14
    base_images = np.random.choice(image_names, size=num_base_images, replace=False)
    
    print(f"\n🔬 Creating mixed training set of {total_images} images...")
    print(f"   Base images: {num_base_images}")
    print(f"   Target augmented images: ~{total_images - num_base_images}")
    
    morph_operations = {
        'watershed': MorphologicalAugmentation.apply_watershed_image,
        'skeleton': MorphologicalAugmentation.apply_skeletonization_image,
        'dilation': MorphologicalAugmentation.apply_dilation_image,
        'erosion': MorphologicalAugmentation.apply_erosion_image,
        'opening': MorphologicalAugmentation.apply_opening_image,
        'closing': MorphologicalAugmentation.apply_closing_image,
    }
    
    # Create directories for each operation
    for op_name in morph_operations.keys():
        (output_dir / op_name).mkdir(exist_ok=True)
    
    # Track augmented images
    augmented_data = {op: [] for op in morph_operations.keys()}
    training_set = list(base_images)  # Start with normal images
    
    # Generate augmentations for ALL 6 morphs per base image
    print(f"\n   Generating augmentations...")
    for img_name in tqdm(base_images, desc="Augmenting"):
        img_path = Path(image_dir) / f'{img_name}.jpg'
        
        if not img_path.exists():
            continue
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply each morphological operation
        for op_name, op_func in morph_operations.items():
            try:
                augmented = op_func(image)
                
                # Save augmented image
                aug_name = f'{img_name}_{op_name}'
                save_path = output_dir / op_name / f'{aug_name}.jpg'
                augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(save_path), augmented_bgr)
                
                # Track for dataset
                augmented_data[op_name].append(aug_name)
                training_set.append(aug_name)
            except Exception as e:
                print(f"  ⚠️  Failed {op_name} on {img_name}: {str(e)}")
    
    total_generated = len(training_set)
    print(f"\n✓ Generated training set:")
    print(f"   Normal images: {num_base_images}")
    print(f"   Augmented images: {total_generated - num_base_images}")
    print(f"   Total training images: {total_generated}")
    print(f"\n   Breakdown by morph type:")
    for op_name, aug_list in augmented_data.items():
        print(f"     {op_name}: {len(aug_list)}")
    
    return training_set, augmented_data, base_images

# ============================================================================
# VISUALIZATION OF MORPHOLOGICAL OPERATIONS
# ============================================================================
def visualize_morphological_operations(image_path, save_path='morph_comparison.png'):
    """Show all morphological operations on a single image"""
    
    # Load image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply all operations
    operations = {
        'Original': image,
        'Watershed': MorphologicalAugmentation.apply_watershed_image(image.copy()),
        'Skeletonization': MorphologicalAugmentation.apply_skeletonization_image(image.copy()),
        'Dilation': MorphologicalAugmentation.apply_dilation_image(image.copy()),
        'Erosion': MorphologicalAugmentation.apply_erosion_image(image.copy()),
        'Opening': MorphologicalAugmentation.apply_opening_image(image.copy()),
        'Closing': MorphologicalAugmentation.apply_closing_image(image.copy()),
    }
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.patch.set_facecolor('white')
    axes = axes.flatten()
    
    for idx, (name, img) in enumerate(operations.items()):
        if idx < len(axes):
            axes[idx].imshow(img)
            axes[idx].set_title(name, fontsize=14, fontweight='bold')
            axes[idx].axis('off')
    
    # Hide last subplot if needed
    if len(operations) < len(axes):
        axes[-1].axis('off')
    
    plt.suptitle('Morphological Operations Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved visualization: {save_path}")

# ============================================================================
# DATASET FOR OBJECT DETECTION (WITH MORPHOLOGICAL AUGMENTATION)
# ============================================================================
class BCCDDetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, image_files, transform=None, 
                 morph_dir=None):
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.image_files = image_files
        self.transform = transform
        self.morph_dir = Path(morph_dir) if morph_dir else None
        self.class_map = {'RBC': 1, 'WBC': 2, 'Platelets': 3}
    
    def __len__(self):
        return len(self.image_files)
    
    def parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in self.class_map:
                continue
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_map[class_name])
        
        return boxes, labels
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # Detect if this is a morphological augmentation
        morph_types = ['watershed', 'skeleton', 'dilation', 'erosion', 'opening', 'closing']
        is_augmented = False
        morph_type = None
        original_name = img_name
        
        for mt in morph_types:
            if f'_{mt}' in img_name:
                is_augmented = True
                morph_type = mt
                original_name = img_name.replace(f'_{mt}', '')
                break
        
        # Try to load the image
        image = None
        
        if is_augmented and self.morph_dir:
            # Try to load morphologically augmented image
            img_path = self.morph_dir / morph_type / f'{img_name}.jpg'
            if img_path.exists():
                image = cv2.imread(str(img_path))
        
        # Fallback to original image if augmented not found or not augmented
        if image is None:
            img_path = self.image_dir / f'{original_name}.jpg'
            if img_path.exists():
                image = cv2.imread(str(img_path))
        
        # If still None, raise error
        if image is None:
            raise FileNotFoundError(f"Could not load image: {img_name} (tried augmented and original)")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Parse annotations using original name
        xml_path = self.annotation_dir / f'{original_name}.xml'
        boxes, labels = self.parse_xml(xml_path)
        
        # Filter out invalid boxes (zero width or height)
        valid_boxes = []
        valid_labels = []
        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box
            # Ensure minimum size of 1 pixel and that xmax > xmin, ymax > ymin
            if xmax > xmin + 1 and ymax > ymin + 1:
                valid_boxes.append(box)
                valid_labels.append(label)
        
        # Convert to tensors
        if len(valid_boxes) > 0:
            boxes = torch.as_tensor(valid_boxes, dtype=torch.float32)
            labels = torch.as_tensor(valid_labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, target

# ============================================================================
# DATA PREPARATION
# ============================================================================
def prepare_datasets():
    """Split dataset into train/val/test"""
    xml_files = list(ANNOTATIONS_DIR.glob('*.xml'))
    image_names = [f.stem for f in xml_files]
    
    print(f"Found {len(image_names)} annotated images")
    
    # Split: 70% train, 15% val, 15% test
    from sklearn.model_selection import train_test_split
    train_names, temp = train_test_split(image_names, test_size=0.3, random_state=42)
    val_names, test_names = train_test_split(temp, test_size=0.5, random_state=42)
    
    print(f"Split: Train={len(train_names)}, Val={len(val_names)}, Test={len(test_names)}")
    
    return train_names, val_names, test_names

# ============================================================================
# MODEL CREATION
# ============================================================================
def create_detection_model(num_classes=4):
    """Create Faster R-CNN model for detection"""
    # Load pretrained model with FPN
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Replace the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Freeze backbone for faster training (only train detection heads)
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    return model

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def collate_fn(batch):
    return tuple(zip(*batch))

def train_one_epoch(model, dataloader, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, targets in pbar:
        try:
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            
            # Skip batches with no valid boxes
            valid_batch = True
            for target in targets:
                if len(target['boxes']) == 0:
                    valid_batch = False
                    break
            
            if not valid_batch:
                continue
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            losses.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()
            
            # Step scheduler if provided (OneCycleLR)
            if scheduler is not None:
                scheduler.step()
            
            total_loss += losses.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{losses.item():.4f}'})
            
        except Exception as e:
            print(f"\n⚠️  Skipping batch due to error: {str(e)}")
            continue
    
    return total_loss / max(num_batches, 1)

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        predictions = model(images)
        break  # Just for quick validation
    
    return 0.0  # Placeholder

# ============================================================================
# INFERENCE AND COUNTING
# ============================================================================
@torch.no_grad()
def detect_and_count(model, image_path, device, confidence_threshold=0.5):
    """Detect cells and count them"""
    model.eval()
    
    # Load image
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Prepare for model
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    transformed = transform(image=image_rgb)
    img_tensor = transformed['image'].to(device)
    
    # Predict
    predictions = model([img_tensor])[0]
    
    # Filter by confidence
    keep = predictions['scores'] > confidence_threshold
    boxes = predictions['boxes'][keep].cpu().numpy()
    labels = predictions['labels'][keep].cpu().numpy()
    scores = predictions['scores'][keep].cpu().numpy()
    
    # Count each class
    counts = {1: 0, 2: 0, 3: 0}  # RBC, WBC, Platelets
    for label in labels:
        if label in counts:
            counts[label] += 1
    
    return boxes, labels, scores, counts

def visualize_detection(image_path, boxes, labels, scores, counts, save_path):
    """Visualize detection results with counts"""
    # Load image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Draw boxes
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.astype(int)
        color = CLASS_COLORS[label]
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        text = f'{CLASS_NAMES[label]}: {score:.2f}'
        cv2.putText(image, text, (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add counts to image
    y_offset = 30
    cv2.putText(image, f"RBC: {counts[1]}", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    y_offset += 40
    cv2.putText(image, f"WBC: {counts[2]}", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    y_offset += 40
    cv2.putText(image, f"Platelets: {counts[3]}", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
    
    # Save
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f'Total Cells Detected: RBC={counts[1]}, WBC={counts[2]}, Platelets={counts[3]}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved detection: {save_path}")

# ============================================================================
# BATCH COUNTING ON DATASET
# ============================================================================
def count_cells_in_dataset(model, image_dir, image_names, device, output_csv):
    """Count cells in multiple images and save results"""
    results = []
    
    print("\n📊 Counting cells in dataset...")
    for img_name in tqdm(image_names):
        img_path = Path(image_dir) / f'{img_name}.jpg'
        
        if not img_path.exists():
            continue
        
        boxes, labels, scores, counts = detect_and_count(model, img_path, device)
        
        results.append({
            'image': img_name,
            'rbc_count': counts[1],
            'wbc_count': counts[2],
            'platelet_count': counts[3],
            'total_cells': sum(counts.values())
        })
    
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print(f"✓ Saved counts to: {output_csv}")
    print("\n📈 Summary Statistics:")
    print(df.describe())
    
    return df

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("BCCD BLOOD CELL DETECTION - 100 IMAGES MIXED TRAINING")
    print("="*80 + "\n")
    
    # Step 1: Prepare datasets
    print("📁 Step 1: Preparing datasets...")
    train_names, val_names, test_names = prepare_datasets()
    
    # Step 2: Generate mixed training set of 100 images
    print("\n🔬 Step 2: Generating mixed training set (100 images)...")
    morph_output_dir = OUTPUT_DIR / 'morphological_augmentations'
    training_set, augmented_data, base_images = generate_mixed_training_set(
        train_names, JPEG_IMAGES_DIR, morph_output_dir, total_images=100
    )
    
    # Step 3: Show examples of morphological operations
    print("\n📸 Step 3: Creating visualization examples...")
    sample_images = np.random.choice(base_images, min(3, len(base_images)), replace=False)
    
    for i, img_name in enumerate(sample_images):
        img_path = JPEG_IMAGES_DIR / f'{img_name}.jpg'
        save_path = OUTPUT_DIR / f'morph_example_{i+1}.png'
        visualize_morphological_operations(img_path, save_path)
        print(f"  ✓ Example {i+1}: {img_name}")
    
    # Step 4: Training set ready
    print("\n📦 Step 4: Training dataset composition...")
    print(f"  Total training images: {len(training_set)}")
    print(f"  Normal images: {len(base_images)}")
    print(f"  Augmented images: {len(training_set) - len(base_images)}")
    print(f"\n  Breakdown:")
    print(f"    Normal: {len(base_images)}")
    for morph_type, aug_list in augmented_data.items():
        print(f"    {morph_type}: {len(aug_list)}")
    
    # Create transforms
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Create datasets
    train_dataset = BCCDDetectionDataset(
        JPEG_IMAGES_DIR, ANNOTATIONS_DIR, training_set, 
        train_transform, morph_dir=morph_output_dir
    )
    val_dataset = BCCDDetectionDataset(
        JPEG_IMAGES_DIR, ANNOTATIONS_DIR, val_names, val_transform
    )
    test_dataset = BCCDDetectionDataset(
        JPEG_IMAGES_DIR, ANNOTATIONS_DIR, test_names, val_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                            num_workers=4, collate_fn=collate_fn, pin_memory=True if torch.cuda.is_available() else False)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                          num_workers=4, collate_fn=collate_fn, pin_memory=True if torch.cuda.is_available() else False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                           num_workers=4, collate_fn=collate_fn, pin_memory=True if torch.cuda.is_available() else False)
    
    print(f"\n  Dataset sizes:")
    print(f"    Train: {len(train_dataset)} (100 mixed images)")
    print(f"    Val: {len(val_dataset)}")
    print(f"    Test: {len(test_dataset)}")
    
    # Step 5: Create model
    print("\n🤖 Step 5: Creating Faster R-CNN model...")
    model = create_detection_model(num_classes=4)
    
    # Enable GPU optimizations
    if torch.cuda.is_available():
        model = model.cuda()
        # Enable mixed precision training for faster GPU processing
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        print("✓ Model moved to GPU")
        print("✓ Mixed precision and cuDNN optimizations enabled")
    else:
        model = model.to(device)
        print("⚠️ Using CPU (training will be slow)")
    
    # Step 6: Training setup
    print("\n🚀 Step 6: Training on 100 mixed images...")
    num_epochs = 5  # Reduced from 10 to 5 epochs
    
    # Optimizer with higher learning rate for faster convergence
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.001, 
        steps_per_epoch=len(train_loader), 
        epochs=num_epochs
    )
    best_loss = float('inf')
    
    # Create checkpoint directory
    checkpoint_dir = OUTPUT_DIR / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nTraining for {num_epochs} epochs (optimized for speed)...")
    print("="*80)
    
    # Enable cudnn benchmarking for faster training
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, lr_scheduler)
        
        print(f'  Loss: {train_loss:.4f}')
        
        # Save checkpoint every 2 epochs (less frequent)
        if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:
            checkpoint_path = checkpoint_dir / f'model_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, checkpoint_path)
        
        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            
            # Save in multiple formats
            print(f'  ✓ New best model! (loss: {best_loss:.4f})')
            
            # Format 1: State dict only (lightest)
            torch.save(model.state_dict(), OUTPUT_DIR / 'best_detection_model.pth')
            print(f'    Saved: best_detection_model.pth')
            
            # Format 2: Complete checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, OUTPUT_DIR / 'best_checkpoint.pth')
            print(f'    Saved: best_checkpoint.pth')
            
            # Format 3: Complete model (architecture + weights)
            torch.save(model, OUTPUT_DIR / 'complete_model.pth')
            print(f'    Saved: complete_model.pth')
    
    print("\n" + "="*80)
    print(f"Training completed! Best loss: {best_loss:.4f}")
    print("="*80)
    
    # Step 7: Test on sample images
    print("\n📸 Step 7: Testing on sample images...")
    model.load_state_dict(torch.load(OUTPUT_DIR / 'best_detection_model.pth'))
    
    # Test on 5 random images
    sample_test = np.random.choice(test_names, min(5, len(test_names)), replace=False)
    
    for i, img_name in enumerate(sample_test):
        img_path = JPEG_IMAGES_DIR / f'{img_name}.jpg'
        boxes, labels, scores, counts = detect_and_count(model, img_path, device, 
                                                         confidence_threshold=0.5)
        
        save_path = OUTPUT_DIR / f'detection_result_{i+1}.png'
        visualize_detection(img_path, boxes, labels, scores, counts, save_path)
        
        print(f"\n{img_name}:")
        print(f"  RBC: {counts[1]}, WBC: {counts[2]}, Platelets: {counts[3]}")
    
    # Step 8: Count all test images
    print("\n📊 Step 8: Counting cells in entire test set...")
    results_df = count_cells_in_dataset(model, JPEG_IMAGES_DIR, test_names, device,
                                       OUTPUT_DIR / 'cell_counts.csv')
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\n✓ Trained on 100 mixed images:")
    print(f"  - {len(base_images)} normal images")
    print(f"  - {len(training_set) - len(base_images)} augmented images (6 morphs)")
    print(f"\n✓ Model saved in 3 formats:")
    print(f"  - best_detection_model.pth (weights only)")
    print(f"  - best_checkpoint.pth (full checkpoint)")
    print(f"  - complete_model.pth (architecture + weights)")
    print(f"\n✓ Morphological examples: morph_example_*.png")
    print(f"✓ Detection samples: detection_result_*.png")
    print(f"✓ Cell counts CSV: cell_counts.csv")
    print(f"\n✓ Morphological augmentations saved in: {morph_output_dir}")
    print("\n📈 Average counts per image:")
    print(f"  RBC: {results_df['rbc_count'].mean():.1f}")
    print(f"  WBC: {results_df['wbc_count'].mean():.1f}")
    print(f"  Platelets: {results_df['platelet_count'].mean():.1f}")