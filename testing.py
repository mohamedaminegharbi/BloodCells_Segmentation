import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = {0: 'Background', 1: 'RBC', 2: 'WBC', 3: 'Platelets'}
CLASS_COLORS = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 100, 255)}

# Load the model
def create_detection_model(num_classes=4):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

model = create_detection_model(num_classes=4)
model.load_state_dict(torch.load(r"C:\Users\user\OneDrive\Desktop\envo\Proj\Segmentation\BCCD_detection\best_detection_model.pth"))
model.to(device)
model.eval()

# Image path
image_path = r"C:\Users\user\OneDrive\Desktop\envo\Proj\Segmentation\BCCD_Dataset\BCCD\JPEGImages\BloodImage_00003.jpg"

# Preprocess and predict
def detect_and_visualize(model, image_path, device, confidence_threshold=0.5):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Preprocess
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    transformed = transform(image=image)
    img_tensor = transformed['image'].to(device).unsqueeze(0)

    # Predict
    with torch.no_grad():
        predictions = model(img_tensor)[0]

    # Filter by confidence
    keep = predictions['scores'] > confidence_threshold
    boxes = predictions['boxes'][keep].cpu().numpy()
    labels = predictions['labels'][keep].cpu().numpy()
    scores = predictions['scores'][keep].cpu().numpy()

    # Count cells
    counts = {1: 0, 2: 0, 3: 0}
    for label in labels:
        counts[label] += 1

    # Visualize
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.astype(int)
        color = CLASS_COLORS[label]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f'{CLASS_NAMES[label]}: {score:.2f}'
        cv2.putText(image, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Add counts
    y_offset = 30
    for class_id, count in counts.items():
        cv2.putText(image, f"{CLASS_NAMES[class_id]}: {count}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, CLASS_COLORS[class_id], 2)
        y_offset += 30

    # Show result
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f'Detection Results: RBC={counts[1]}, WBC={counts[2]}, Platelets={counts[3]}')
    plt.show()

# Run detection
detect_and_visualize(model, image_path, device)

