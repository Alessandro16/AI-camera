import cv2

from detection_people import start_detection

def start_stream(model, device):
    # Open the default camera
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()

        frame = start_detection(frame, model, device)

        # Display the captured frame with tracking
        cv2.imshow('Camera', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the capture and writer objects
    cam.release()
    cv2.destroyAllWindows()