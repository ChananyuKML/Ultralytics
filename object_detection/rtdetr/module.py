from ultralytics import RTDETR
import argparse

def train(pt="rtdetr-l", dataset="coco8", epochs=100):
    model = RTDETR(f"checkpoints/{pt}.pt")
    model.train(data=f"datasets/{dataset}.yaml", epochs=epochs, imgsz=640)

def run(pt="rtdetr-l", img="img\car.png", prompt="car"):
    model = RTDETR(f"{pt}.pt")
    results = model(f"{img}", save=True)
    json_results = results[0].to_json()
    return json_results

def get_model(size="l"):
    model = RTDETR(f"rtsetr-{size}.pt")
    return model