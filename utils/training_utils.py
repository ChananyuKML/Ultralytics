from ultralytics import YOLO
import os
import yaml
import psutil
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List
from pathlib import Path
from datetime import datetime
import onnx
import gc
import torch
from multiprocessing import Process, Queue
import signal
from threading import Thread
import json
import math
from pydantic import BaseModel, Field
import obb.module as obb
import pose.module as pose
import object_detection.yolo12.module as yolo12
import object_detection.yolo8.module as yolo8
import object_detection.rtdetr.module as rtdetr
import segment.sam2.module as sam2
import segment.sam3.module as sam3


class TrainerConfig(BaseModel):
    path: str
    dataroot: str = "datasets/coco8"
    train_path: str = f"{dataroot}/images/train"
    val_path: str = f"{dataroot}/images/val"
    test_path: str = f"{dataroot}/images/test"
    train_label_path: str = f"{dataroot}/labels/train"
    val_label_path: str = f"{dataroot}/labels/val"
    epochs: int = 100
    model: str = "sam2"
    size: str = "t"
    device: str = "cuda:0"
    batch: int = 1
    workers: int = 1

class TrainingManager:
    def __init__(self):
        self.current_process = None
        self.status_queue = Queue()
        self.status = 'FINISHED'
        self.metrics = {}
        self.loss = 0
        self.current_epoch = 0
        self.eta_epoch = None
        self.total_epochs = 0
        self.error_msg = ''
        self._start_status_monitor()
    
    def start_training(self):
        return 
    
    def get_status(self):
        return
    
    def terminate_process(self):
        return