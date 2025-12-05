from ultralytics import YOLO

def train(pt="yolo11n", dataset="coco8", epochs=100):
    model = YOLO(f"{pt}-pose.yaml").load(f"{pt}-pose.pt")
    model.train(data=f"datasets/{dataset}-pose.yaml", epochs=epochs, imgsz=640)

def get_model(pt="yolo12n"):
    model = YOLO(f"{pt}-seg.yaml").load(f"{pt}.pt")
    return model