from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser(description="Configuration for training process")
parser.add_argument("img", type=str)
parser.add_argument("--pt", type=str, default="yolo12n")
args = parser.parse_args()

# Load pre-trained model
model = YOLO(f"{args.pt}.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model(f"{args.img}")

for result in results:
    boxes = result.boxes  # Boxes object
    masks = result.masks  # Masks object
    keypoints = result.keypoints # Keypoints object
    probs = result.probs # Probs object