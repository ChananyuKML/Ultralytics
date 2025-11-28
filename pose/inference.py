from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser(description="Configuration for training process")
parser.add_argument("--pt", type=str, default="yolo12n")
parser.add_argument("--img", type=str)
args = parser.parse_args()

# Load pre-trained model
model = YOLO(f"{args.pt}-pose.yaml").load(f"{args.pt}.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model(f"{args.img}")

# Access the results
for result in results:
    xy = result.keypoints.xy  # x and y coordinates
    xyn = result.keypoints.xyn  # normalized
    kpts = result.keypoints.data  # x, y, visibility (if available)