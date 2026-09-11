import os
import tkinter as tk
from tkinter import filedialog
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

NUM_CLASSES = 6

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_path = r"D:\Course_Projects_CP\Object_Oriented_Programmin_OOP\Dataset_Manual_Extraction\aircraft_defect_model.pth"
model = get_model(NUM_CLASSES)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Open file dialog to choose ANY image
root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)

image_path = filedialog.askopenfilename(
    title="Select an Image for Defect Detection",
    filetypes=[("Image files", "*.jpg *.jpeg *.png")]
)

if not image_path:
    print("No image selected. Exiting.")
    exit()

image = Image.open(image_path).convert("RGB")
transform = torchvision.transforms.ToTensor()
img_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    predictions = model(img_tensor)

boxes = predictions[0]['boxes'].cpu()
scores = predictions[0]['scores'].cpu()
labels = predictions[0]['labels'].cpu()

# Confidence Threshold (> 50%)
high_conf_indices = scores > 0.50
filtered_boxes = boxes[high_conf_indices]
filtered_scores = scores[high_conf_indices]
filtered_labels = labels[high_conf_indices]

fig, ax = plt.subplots(1, figsize=(10, 8))
ax.imshow(image)

for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
    x1, y1, x2, y2 = box.numpy()
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.text(x1, y1 - 5, f"Class {label.item()} ({score.item():.2f})", 
            bbox=dict(facecolor='red', alpha=0.5), color='white', fontsize=10, weight='bold')

plt.axis("off")
plt.title(f"Inference: {os.path.basename(image_path)}")
plt.show()