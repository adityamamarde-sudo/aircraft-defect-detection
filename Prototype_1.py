import os
import sys
import tkinter as tk
from tkinter import filedialog
from PIL import Image

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from roboflow import Roboflow


# ---------------------------------------------------------------------
# 1. Custom Dataset Wrapper for Extracted COCO JSON Dataset
# ---------------------------------------------------------------------
class CocoDetectionDataset(Dataset):
    """
    Custom Dataset wrapper for COCO JSON annotations from extracted zip.
    Converts COCO [x, y, width, height] format into PyTorch [x1, y1, x2, y2].
    """
    def __init__(self, root_dir, annotation_file, transforms=None):
        from pycocotools.coco import COCO
        
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]

        # Load image file
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")

        # Load annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            # Ignore invalid zero-area boxes
            if w <= 0 or h <= 0:
                continue

            # Convert COCO [x, y, width, height] -> PyTorch [x1, y1, x2, y2]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            areas.append(ann.get('area', w * h))
            iscrowd.append(ann.get('iscrowd', 0))

        # Handle images without bounding boxes
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.ids)


def collate_fn(batch):
    return tuple(zip(*batch))


def get_transforms():
    return T.Compose([
        T.ToTensor()
    ])


# ---------------------------------------------------------------------
# 2. Model Initialization
# ---------------------------------------------------------------------
def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ---------------------------------------------------------------------
# 3. Training & Evaluation Loops
# ---------------------------------------------------------------------
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    running_loss = 0.0

    for step, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

        if (step + 1) % 5 == 0 or (step + 1) == len(data_loader):
            print(f"Epoch [{epoch + 1}] Step [{step + 1}/{len(data_loader)}] Loss: {losses.item():.4f}")

    avg_loss = running_loss / len(data_loader)
    print(f"--> Epoch [{epoch + 1}] Average Training Loss: {avg_loss:.4f}\n")


@torch.no_grad()
def evaluate_model(model, data_loader, device):
    model.eval()
    print("Evaluating on Test Set...")
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        predictions = model(images)
        
        sample_boxes = predictions[0]['boxes']
        sample_scores = predictions[0]['scores']
        
        high_score_indices = sample_scores > 0.5
        filtered_boxes = sample_boxes[high_score_indices]
        print(f"Test Image Sample: Detected {len(filtered_boxes)} defect objects with confidence > 50%.")
        break


# ---------------------------------------------------------------------
# 4. Main Execution Window
# ---------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using compute device: {device}")

    # Authenticate with Roboflow
    API_KEY = "rf_4zbo47ynUacfYgSIpv1qgqGYCjm1"
    try:
        rf = Roboflow(api_key=API_KEY)
        print("Successfully authenticated with Roboflow API.")
    except Exception as e:
        print(f"Roboflow Auth Warning: {e}")

    # File dialog to select extracted dataset folder
    print("\nSelect your extracted dataset folder in the popup window...")
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    dataset_folder = filedialog.askdirectory(title="Select Extracted Roboflow Dataset Folder")

    if not dataset_folder:
        print("No folder selected. Exiting script.")
        sys.exit()

    print(f"Dataset location selected: {dataset_folder}")

    # Verify paths for extracted COCO dataset
    TRAIN_IMG_DIR = os.path.join(dataset_folder, "train")
    TRAIN_ANN_FILE = os.path.join(dataset_folder, "train", "_annotations.coco.json")
    
    TEST_IMG_DIR = os.path.join(dataset_folder, "test")
    TEST_ANN_FILE = os.path.join(dataset_folder, "test", "_annotations.coco.json")

    # Check that annotations exist
    if not os.path.exists(TRAIN_ANN_FILE):
        print(f"\nError: Could not find '{TRAIN_ANN_FILE}' inside the selected folder.")
        print("Please make sure you selected the main extracted dataset folder containing the 'train' and 'test' subfolders.")
        sys.exit()

    # Load Data
    print("Loading dataset annotations into memory...")
    train_dataset = CocoDetectionDataset(TRAIN_IMG_DIR, TRAIN_ANN_FILE, transforms=get_transforms())
    test_dataset = CocoDetectionDataset(TEST_IMG_DIR, TEST_ANN_FILE, transforms=get_transforms())

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # Initialize PyTorch Faster R-CNN Model
    # NUM_CLASSES = count of defect categories + 1 for background
    NUM_CLASSES = 6
    model = get_model_instance_segmentation(num_classes=NUM_CLASSES)
    model.to(device)

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Train Model
    NUM_EPOCHS = 5
    print(f"\nStarting model training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        train_one_epoch(model, optimizer, train_loader, device, epoch)

    # Evaluate Model
    evaluate_model(model, test_loader, device)

    # Save Model Weights
    output_model_path = os.path.join(dataset_folder, "aircraft_defect_model.pth")
    torch.save(model.state_dict(), output_model_path)
    print(f"\nTraining Complete! Model weights successfully saved to: {output_model_path}")