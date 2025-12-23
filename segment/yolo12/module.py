from ultralytics import YOLO

def train(pt="yolo11n", dataset="coco8", epochs=100):
    model = YOLO(f"{pt}-pose.yaml").load(f"{pt}-pose.pt")
    model.train(data=f"datasets/{dataset}-pose.yaml", epochs=epochs, imgsz=640)

def get_model(size="n"):
    model = YOLO(f"yolo12{size}-seg.yaml").load(f"yolo12{size}.pt")
    return model