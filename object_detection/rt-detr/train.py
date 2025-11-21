from ultralytics import RTDETR
import argparse
import torch

parser = argparse.ArgumentParser(description="Configuration for training process")
parser.add_argument("--pt", type=str, default="rtdetr-l")
parser.add_argument("--dataset", type=str, default="coco8")
parser.add_argument("--epochs", type=int, default="100")
args = parser.parse_args()

# Load pre-trained model
model = RTDETR(f"checkpoints/{args.pt}.pt")
model.to("cuda")
# Train the model on the COCO8 example dataset for 100 epochs
model.train(data=f"datasets/{args.dataset}.yaml", epochs=args.epochs, imgsz=640)