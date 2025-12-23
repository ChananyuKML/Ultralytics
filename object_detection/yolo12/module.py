from ultralytics import YOLO
import argparse

def train(pt="yolo12n", dataset="coco8", epochs=100):
    model = YOLO(f"checkpoints/{pt}.pt")
    model.train(data=f"datasets/{dataset}.yaml", epochs=epochs, imgsz=640)

def run(pt="yolo12n", img="img\car.png", prompt="car"):
    model = YOLO(f"{pt}.pt")
    results = model(f"{img}", save=True)
    json_results = results[0].to_json()
    return json_results

def get_model(pt="yolo12n"):
    model = YOLO(f"{pt}.pt")
    return model