from ultralytics import YOLO
import argparse

def train(pt="yolo11n", dataset="coco8", epochs=100):
    model = YOLO(f"{pt}-pose.yaml").load(f"{pt}-pose.pt")
    model.train(data=f"datasets/{dataset}-pose.yaml", epochs=epochs, imgsz=640)

def run(pt="yolo11n", img="img\car.png", prompt="car"):
    model = YOLO(f"{pt}-pose.pt")
    results = model(f"{img}", save=True)
    json_results = results[0].to_json()
    return json_results