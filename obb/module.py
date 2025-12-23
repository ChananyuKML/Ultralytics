from ultralytics import YOLO
import argparse

def train(pt="yolo12n", dataset="dota8", epochs=100):
    model = YOLO(f"{pt}-obb.yaml").load(f"{pt}.pt")
    model.train(data=f"datasets/{dataset}.yaml", epochs=epochs, imgsz=640)

def run(pt="yolo12n", img="img\car.png", prompt="car"):
    model = YOLO(f"{pt}-obb.yaml").load(f"{pt}.pt")
    results = model(f"{img}", save=True)
    json_results = results[0].to_json()
    return json_results

def get_model(size="n"):
    model = YOLO(f"yolo11{size}-obb.yaml").load(f"yolo11{size}.pt")
    return model