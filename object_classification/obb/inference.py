from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser(description="Configuration for training process")
parser.add_argument("--pt", type=str, default="rtdetr-l")
parser.add_argument("--img", type=str)
args = parser.parse_args()

# Load pre-trained model
model = YOLO(f"{args.pt}-obb.yaml").load(f"{args.pt}.pt")
# Train the model on the COCO8 example dataset for 100 epochs
results = model(f"{args.img}")

# Access the results
for result in results:
    xywhr = result.obb.xywhr  # center-x, center-y, width, height, angle (radians)
    xyxyxyxy = result.obb.xyxyxyxy  # polygon format with 4-points
    names = [result.names[cls.item()] for cls in result.obb.cls.int()]  # class name of each box
    confs = result.obb.conf  # confidence score of each 