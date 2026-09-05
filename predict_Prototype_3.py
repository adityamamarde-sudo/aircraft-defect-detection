import os
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Set Device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load Trained Weights
model_path = r"D:\Course_Projects_CP\Object_Oriented_Programmin_OOP\MODEL_1_CODE\aircraft_defect_model_3datasets.pth"

# Model Configuration
NUM_CLASSES = 6  # Background (0) + 5 Defect Classes
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Run Inference on a Test Image
test_image_path = r"D:\Course_Projects_CP\Object_Oriented_Programmin_OOP\Dataset_Manual_Extraction\Dataset_1\test"

# Grab the first image found in the test folder
image_files = [f for f in os.listdir(test_image_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
if not image_files:
    print("No images found in the test folder.")
    exit()

img_full_path = os.path.join(test_image_path, image_files[0])
image = Image.open(img_full_path).convert("RGB")

transform = T.ToTensor()
img_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    predictions = model(img_tensor)

boxes = predictions[0]['boxes'].cpu()
scores = predictions[0]['scores'].cpu()
labels = predictions[0]['labels'].cpu()

# Display Bounding Boxes (> 50% Confidence)
fig, ax = plt.subplots(1, figsize=(10, 8))
ax.imshow(image)

CONFIDENCE_THRESHOLD = 0.5
detected_count = 0

for box, score, label in zip(boxes, scores, labels):
    if score >= CONFIDENCE_THRESHOLD:
        x1, y1, x2, y2 = box.numpy()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"Class {label.item()} ({score.item()*100:.1f}%)", bbox=dict(facecolor='red', alpha=0.5), color='white', weight='bold')
        detected_count += 1

print(f"Detected {detected_count} defects with > {int(CONFIDENCE_THRESHOLD*100)}% confidence.")
plt.axis("off")
plt.show()