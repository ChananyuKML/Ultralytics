from fastapi import FastAPI
from pydantic import BaseModel
import obb.module as obb
import pose.module as pose
import object_detection.yolo12.module as yolo12
import object_detection.yolo8.module as yolo8
import object_detection.rtdetr.module as rtdetr
import segment.sam2.module as sam2
import segment.sam3.module as sam3



app = FastAPI()

class ModelInput(BaseModel):
    task: str = "obb"
    mode: str = "run"
    pt: str = "yolo12n"
    dataset: str = "coco8"
    epochs: int = 100
    prompt: str = "A red car"
    image: str = "img/car.png"

@app.post("/run")
def run(data: ModelInput):
    match data.task:
        case "obb":
            module = obb
        case "pose":
            module = pose
        case "yolo12":
            module = yolo12
        case "yolo8":
            module = yolo8
        case "rtdetr":
            module = rtdetr
        case "sam2":
            module = sam2 
        case "sam3":
            module = sam3 

    match data.mode:
        case "train":
            process = module.train
            result = process(data.pt, data.image, data.epochs)
        case "run":
            process = module.run
            result = process(data.pt, data.image, data.prompt)
            
    return {"result": result}