import os
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

# Load model weights
model_path = r"D:\Course_Projects_CP\Object_Oriented_Programmin_OOP\Dataset_Manual_Extraction\aircraft_defect_model.pth"
model = get_model(NUM_CLASSES)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Input directory and Output directory
test_dir = r"D:\Course_Projects_CP\Object_Oriented_Programmin_OOP\Dataset_Manual_Extraction\test"
output_dir = os.path.join(test_dir, "output_predictions")
os.makedirs(output_dir, exist_ok=True)

test_images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

print(f"Found {len(test_images)} images. Processing and saving predictions...")

transform = torchvision.transforms.ToTensor()

for img_name in test_images:
    image_path = os.path.join(test_dir, img_name)
    image = Image.open(image_path).convert("RGB")
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

    # Plot figure
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(image)

    for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
        x1, y1, x2, y2 = box.numpy()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"Class {label.item()} ({score.item():.2f})", 
                bbox=dict(facecolor='red', alpha=0.5), color='white', fontsize=10, weight='bold')

    plt.axis("off")
    plt.title(f"Detections for {img_name}")
    
    # Save image with predictions
    save_path = os.path.join(output_dir, f"pred_{img_name}")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

print(f"Done! All prediction results saved to: {output_dir}")