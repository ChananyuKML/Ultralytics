from ultralytics import YOLO
import os
import sys
import yaml
import psutil
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List
from pathlib import Path
from datetime import datetime
# import onnx
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
import segment.yolo12.module as yolos



class TrainerConfig(BaseModel):
    path: str
    dataroot: str = "datasets/coco8"
    epochs: int = 100
    img_size: int = 640
    task: str = "obb"
    model: str = "yolo12"
    size: str = "n"
    device: str = "cuda:0"
    batch: int = 1
    workers: int = 1

class TrainingManager:
    def __init__(self):
        self.task = "obb"
        self.current_process = None
        self.status_queue = Queue()
        self.status = 'IDLE'
        self.metrics = {}
        self.loss = 0
        self.current_epoch = 0
        self.eta_epoch = None
        self.total_epochs = 0
        self.error_msg = ''
        self._start_status_monitor()

    def _start_status_monitor(self):
        def monitor():
            while True:
                try:
                    status_data = self.status_queue.get_nowait()
                except:
                    continue

                self.status = status_data.get('status', self.status)

                metrics = status_data.get('metrics', self.metrics)
                if metrics:  # ถ้ามี metrics
                    for key, value in metrics.items():
                        if isinstance(value, float) and math.isnan(value):
                            metrics[key] = None

                self.metrics = metrics
                self.loss = status_data.get('loss', self.loss)
                self.current_epoch = status_data.get('epoch', self.current_epoch)
                self.eta_epoch = status_data.get('eta_epoch', self.eta_epoch)
                self.error_msg = status_data.get('error', self.error_msg)

        self.monitor_thread = Thread(target=monitor, daemon=True)
        self.monitor_thread.start()

    def get_child_processes(self, pid):
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            return children
        except psutil.NoSuchProcess:
            return []

    def kill_process_tree(self, pid):
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)

        # 1. Kill all children FIRST
            for child in children:
                try:
                    print(f"Killing child PID={child.pid}")
                    child.kill()  # SIGKILL immediately
                except psutil.NoSuchProcess:
                    pass

            # Wait for children to actually die
            gone, alive = psutil.wait_procs(children, timeout=2)
            for p in alive:
                try:
                    print(f"Force killing stubborn child PID={p.pid}")
                    p.kill()
                except:
                    pass

        # 2. Kill parent
            try:
                print(f"Killing parent PID={parent.pid}")
                parent.kill()
            except psutil.NoSuchProcess:
                pass

        except Exception as e:
            print(f"Error killing process tree: {e}")


    def cleanup_current_process(self):
        if self.current_process and self.current_process.is_alive():

            pid = self.current_process.pid
            print(f"Cleanup starting for PID={pid}")

            try:
                self.kill_process_tree(pid)
            except Exception as e:
                print(f"Error cleaning up: {e}")

        # Ensure process is removed
            self.current_process.join(timeout=2)

            if self.current_process.is_alive():
                print("Process still alive - forcing terminate()")
                self.current_process.terminate()

            self.current_process = None
            self.status_queue.put({"status": "CANCELLED"})

        # Cleanup stray zombies owned by THIS process
        try:
            parent = psutil.Process()
            children = parent.children(recursive=True)
            for child in children:
                try:
                    print(f"Killing zombie child PID={child.pid}")
                    child.kill()
                except:
                    pass
        except:
            pass


    def start_training(self, item: TrainerConfig):
        self.cleanup_current_process()
        self.total_epochs = item.epochs
        self.current_epoch = 0
        self.status = 'STARTING'
        self.error_msg = ''
        self.metrics = {}
        self.loss = 0
        self.eta_epoch = None
        
        self.current_process = Process(
            target=self._train_worker, 
            args=(item, self.status_queue)
        )
        self.current_process.start()

    @staticmethod
    def _train_worker(item:TrainerConfig, status_queue: Queue): 
        def update_status(status, **kwargs):
            status_data = {'status': status, **kwargs}
            status_queue.put(status_data)

        try:
            # Check available memory
            system_memory = psutil.virtual_memory()
            available_ram = system_memory.available / (1024 ** 3)
            estimated_memory = item.batch * 0.5
            if available_ram < (estimated_memory + 2.0):
                update_status('ERROR', error="Insufficient memory to start training")
                return
        
            def on_train_start(trainer):
                update_status('TRAINING')
        
            def on_train_epoch_start(trainer):
                update_status('TRAINING')

            def on_train_epoch_end(trainer):
                try:
                    loss = trainer.loss.item()
                    if math.isnan(loss) or math.isinf(loss):
                        loss = None
                except:
                    loss = None
                print(f"trainer.metrics: {trainer.metrics}")
                
                update_status('TRAINING', 
                        epoch=trainer.epoch + 1,
                        metrics=trainer.metrics,
                        eta_epoch=trainer.epoch_time,
                        loss=loss)

            def on_val_start(trainer):
                update_status('VALIDATING')
        
            def on_pretrain_routine_start(trainer):
                update_status('PREPROCESSING')
            match item.task:
                case "obb":
                    model = obb.get_model(item.size)
                case "pose":
                    model = pose.get_model(item.size)
                case "segment":
                    model = yolos.get_model(item.size)
                case "detection":
                    if item.model == "yolo12":
                        model = yolo12.get_model(item.size)
                    elif item.model == "yolo8":
                        model = yolo8.get_model(item.size)
                    elif item.model == "rtdetr":
                        model = rtdetr.get_model(item.size)
                    else:
                        print("Error : unknown model")
                        return        
            
            model.add_callback("on_train_start", on_train_start)
            model.add_callback("on_train_epoch_end", on_train_epoch_end)
            model.add_callback("on_val_start", on_val_start)
            model.add_callback("on_pretrain_routine_start", on_pretrain_routine_start)
            model.add_callback("on_train_epoch_start", on_train_epoch_start)
            
            model.train(data=f"{item.dataroot}/dataset.yaml", 
                        epochs=item.epochs, 
                        imgsz=item.img_size) 
            
            update_status('FINISHED')
    
        except Exception as e:
            update_status('ERROR', error=str(e))
            print(f"Training error: {e}")    