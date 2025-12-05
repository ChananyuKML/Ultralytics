from fastapi import FastAPI
from pydantic import BaseModel
import obb.module as obb
import pose.module as pose
import object_detection.yolo12.module as yolo12
import object_detection.yolo8.module as yolo8
import object_detection.rtdetr.module as rtdetr
import segment.sam2.module as sam2
import segment.sam3.module as sam3
from utils.training_utils import TrainerConfig, TrainingManager


app = FastAPI()

# Create global training manager
training_manager = TrainingManager()
train_var = TrainerConfig

class runOptions(BaseModel):
    task: str = "obb"
    mode: str = "run"
    pt: str = "yolo12n"
    dataset: str = "coco8"
    epochs: int = 100
    prompt: str = "A red car"
    image: str = "img/car.png"

@app.post("/training")
def start_training(item: train_var):
    if training_manager.status not in ['FINISHED', 'ERROR', 'CANCELLED']:
        return {"message": "Training already in progress"}
    
    training_manager.start_training(item)
    return {"message": "Training started"}

@app.get("/status")
def get_status():
    return {
        "status":training_manager.status,
        "current_epoch":training_manager.current_epoch,
        "total_epochs":training_manager.total_epochs,
        "eta_epoch":training_manager.eta_epoch,
        "metrics":training_manager.metrics,
        "loss":training_manager.loss,
        "error":training_manager.error_msg
    }

@app.post("/shutdown")
def shutdown_event():
    if training_manager.status in ['TRAINING', 'VALIDATING']:
        training_manager.cleanup_current_process()
        return {"message": "Shutdown completed"}
    elif training_manager.status in ['FINISHED', 'ERROR']:
        training_manager.cleanup_current_process()
        return {"message": "No active training found. Shutdown completed"}
    else:
        return {"message": "No active training session found"}

@app.post("/run")
def run(data: runOptions):
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
    
    result = module.run(data.pt, data.image, data.prompt)
            
    return {"result": result}