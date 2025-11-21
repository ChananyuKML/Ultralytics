from ultralytics import RTDETR
import argparse

parser = argparse.ArgumentParser(description="Configuration for training process")
parser.add_argument("img", type=str)
parser.add_argument("--pt", type=str, default="rtdetr-l")
args = parser.parse_args()

# Load pre-trained model
model = RTDETR(f"{args.pt}.pt")

# Inference Model
results = model(f"{args.img}")

# Accessing and iterating through the results
for result in results:
    boxes = result.boxes.xyxy 
    scores = result.boxes.conf 
    classes = result.boxes.cls  