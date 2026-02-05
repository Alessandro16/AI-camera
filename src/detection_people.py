import cv2
import numpy as np
from sympy.printing.pytorch import torch

def yolo_detection(tensor, model, conf=50, iou=0.3):
    result = model(tensor,
                      conf=conf / 100,
                      iou=iou,
                      classes=0)
    return result

def yolo_tracking(tensor, model):
    result = model.track(tensor,
                         persist=True,
                         tracker="bytetrack.yaml",
                         classes=0)
    return result

def make_box_detection(frame, result):
    for result in result[0].boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        # Make boundingbox and txt label fro the frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
        cv2.putText(frame, f"ID: {int(class_id)}  Score: {score:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

def make_box_tracking(frame, result):
    for result in result[0].boxes:
        x1, y1, x2, y2 = result.xyxy[0]
        target_id = int(result.id.item())

        # Make boundingbox and txt label fro the frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
        cv2.putText(frame, f"ID: {target_id}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

def start_detection(frame, model, device, choice = 'detection'):
    # Convert frame OpenCV -> Tensor with pyTorch
    conv = frame[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB
    conv = np.ascontiguousarray(conv)
    tensor = torch.from_numpy(conv).float()
    tensor /= 255.0  # Nomralize the tensor [0:1]
    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)

    # Check type of detection
    if choice == 'detection':
        result = yolo_detection(tensor, model)
        if result[0].boxes is not None:
            frame = make_box_detection(frame, result)

    elif choice == 'tracking':
        result = yolo_tracking(tensor, model)
        if result[0].boxes is not None and result[0].boxes.id is not None:
            frame = make_box_tracking(frame, result)

    return frame

