import cv2
import numpy as np
import torch

from detection_people import start_detection

def start_stream(model, device):
    # Open the default camera
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()

        # Convert frame OpenCV -> Tensor with pyTorch
        conv = frame[:, :, ::-1].transpose(2, 0, 1) # BGR -> RGB
        conv = np.ascontiguousarray(conv)
        tensor = torch.from_numpy(conv).float()
        tensor /= 255.0  # Nomralize the tensor [0:1]
        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(device)

        frame = start_detection(tensor, frame, model)

        # Display the captured frame with tracking
        cv2.imshow('Camera', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the capture and writer objects
    cam.release()
    cv2.destroyAllWindows()