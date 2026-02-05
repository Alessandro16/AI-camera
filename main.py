import os
os.environ["YOLO_VERBOSE"] = "False"    # Elimino Data YOLO sulla CLI

from sympy.printing.pytorch import torch
from ultralytics import YOLO
from src.camera_stream import start_stream

def main():
    # Check for graphics card invidia
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO('yolov8s').to(device)

    start_stream(model, device)

if __name__ == "__main__":
    main()