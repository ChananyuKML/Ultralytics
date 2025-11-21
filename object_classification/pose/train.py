from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser(description="Configuration for training process")
parser.add_argument("--pt", type=str, default="yolo12n")
parser.add_argument("--dataset", type=str, default="coco8")
parser.add_argument("--epochs", type=int, default="100")
args = parser.parse_args()

# Load pre-trained model
model = YOLO(f"{args.pt}-pose.yaml").load(f"{args.pt}.pt")

# Train the model on the COCO8 example dataset for 100 epochs
model.train(data=f"datasets/{args.dataset}.yaml", epochs=args.epochs, imgsz=640)