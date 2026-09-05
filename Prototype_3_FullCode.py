import os
import sys
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ---------------------------------------------------------------------
# 1. Custom Dataset Wrapper for COCO JSON Format
# ---------------------------------------------------------------------
class CocoDetectionDataset(Dataset):
    """
    Custom Dataset wrapper for COCO JSON annotations.
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

        if (step + 1) % 10 == 0 or (step + 1) == len(data_loader):
            print(f"Epoch [{epoch + 1}] Step [{step + 1}/{len(data_loader)}] Loss: {losses.item():.4f}")

    avg_loss = running_loss / len(data_loader)
    print(f"--> Epoch [{epoch + 1}] Average Training Loss: {avg_loss:.4f}\n")


@torch.no_grad()
def evaluate_model(model, data_loader, device):
    model.eval()
    print("Evaluating on Merged Test Set...")
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
# 4. Main Execution
# ---------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using compute device: {device}")

    # EXACT DATASET PATHS RESOLVED FROM YOUR SYSTEM
    dataset_paths = [
        r"D:\Course_Projects_CP\Object_Oriented_Programmin_OOP\Dataset_Manual_Extraction\Dataset_1",
        r"D:\Course_Projects_CP\Object_Oriented_Programmin_OOP\Dataset_Manual_Extraction\Dataset_2",
        r"D:\Course_Projects_CP\Object_Oriented_Programmin_OOP\Dataset_Manual_Extraction\Dataset_3"
    ]

    train_datasets = []
    test_datasets = []

    print("\nLoading and validating COCO datasets...")
    for i, path in enumerate(dataset_paths, 1):
        train_ann = None
        train_img_dir = None
        test_ann = None
        test_img_dir = None

        # Search for _annotations.coco.json inside train/test folders
        for root, dirs, files in os.walk(path):
            if "_annotations.coco.json" in files:
                if "train" in root.lower():
                    train_ann = os.path.join(root, "_annotations.coco.json")
                    train_img_dir = root
                elif "test" in root.lower():
                    test_ann = os.path.join(root, "_annotations.coco.json")
                    test_img_dir = root

        # Validate path discovery
        if not train_ann or not os.path.exists(train_ann):
            print(f"ERROR: Could not locate train '_annotations.coco.json' inside Dataset {i}: '{path}'")
            sys.exit()

        if not test_ann or not os.path.exists(test_ann):
            print(f"ERROR: Could not locate test '_annotations.coco.json' inside Dataset {i}: '{path}'")
            sys.exit()

        print(f"Dataset {i} resolved successfully:")
        print(f"  - Train Dir: {train_img_dir}")
        print(f"  - Train Ann: {train_ann}")
        print(f"  - Test Dir:  {test_img_dir}")
        print(f"  - Test Ann:  {test_ann}\n")

        train_datasets.append(CocoDetectionDataset(train_img_dir, train_ann, transforms=get_transforms()))
        test_datasets.append(CocoDetectionDataset(test_img_dir, test_ann, transforms=get_transforms()))

    # Combine all 3 datasets together in memory
    combined_train_dataset = ConcatDataset(train_datasets)
    combined_test_dataset = ConcatDataset(test_datasets)

    print(f"Total combined training images: {len(combined_train_dataset)}")
    print(f"Total combined test images: {len(combined_test_dataset)}")

    train_loader = DataLoader(combined_train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(combined_test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # Initialize PyTorch Faster R-CNN Model
    NUM_CLASSES = 6
    model = get_model_instance_segmentation(num_classes=NUM_CLASSES)
    model.to(device)

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Train Model
    NUM_EPOCHS = 5
    print(f"\nStarting multi-dataset training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        train_one_epoch(model, optimizer, train_loader, device, epoch)

    # Evaluate Model
    evaluate_model(model, test_loader, device)

    # Save Multi-Dataset Model Weights
    output_model_path = os.path.join(os.path.dirname(__file__), "aircraft_defect_model_3datasets.pth")
    torch.save(model.state_dict(), output_model_path)
    print(f"\nTraining Complete! Multi-dataset model saved to: {output_model_path}")
    