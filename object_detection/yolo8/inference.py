from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser(description="Configuration for training process")
parser.add_argument("img", type=str)
parser.add_argument("--pt", type=str, default="yolov8n")
args = parser.parse_args()

# Load model
model = YOLO(f"{args.pt}.pt")

# Inference model
results = model(f"{args.img}")

for result in results:
    boxes = result.boxes  # Boxes object
    masks = result.masks  # Masks object
    keypoints = result.keypoints # Keypoints object
    probs = result.probs # Probs object