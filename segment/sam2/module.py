from ultralytics import SAM
import argparse

def run(pt="sam2.1_t", img="img\car.png", prompt="car"):
    model = SAM(f"{pt}.pt")
    results = model(f"{img}", save=True)
    json_results = results[0].to_json()
    return json_results