import os
import sys
import torch
import torchvision
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader

# Define paths for your 3 datasets
dataset_paths = [
    r"D:\Course_Projects_CP\Object_Oriented_Programmin_OOP\Dataset_Manual_Extraction\Dataset_1",
    r"D:\Course_Projects_CP\Object_Oriented_Programmin_OOP\Dataset_Manual_Extraction\Dataset_2",
    r"D:\Course_Projects_CP\Object_Oriented_Programmin_OOP\Dataset_Manual_Extraction\Dataset_3"
]

train_datasets = []
test_datasets = []

# Load COCO datasets dynamically from each path
for path in dataset_paths:
    train_img = os.path.join(path, "train")
    train_ann = os.path.join(path, "train", "_annotations.coco.json")
    
    test_img = os.path.join(path, "test")
    test_ann = os.path.join(path, "test", "_annotations.coco.json")
    
    train_datasets.append(CocoDetectionDataset(train_img, train_ann, transforms=get_transforms()))
    test_datasets.append(CocoDetectionDataset(test_img, test_ann, transforms=get_transforms()))

# Combine all 3 datasets together
combined_train_dataset = ConcatDataset(train_datasets)
combined_test_dataset = ConcatDataset(test_datasets)

# Pass combined datasets into DataLoaders
train_loader = DataLoader(combined_train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(combined_test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)