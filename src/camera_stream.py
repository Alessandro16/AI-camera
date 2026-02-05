import cv2
from src.detection_people import start_detection

def start_stream(model, device):
    # Open the default camera
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        # Resize for YOLO
        frame = cv2.resize(frame, (640, 640))

        frame = start_detection(frame, model, device)

        # Display the captured frame with tracking
        cv2.imshow('Camera', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the capture and writer objects
    cam.release()
    cv2.destroyAllWindows()