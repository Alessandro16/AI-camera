import cv2

from src.detection_people import start_detection

def prova_img(model, device):
    img = cv2.imread('assets/test_image.jpg')
    input_size = 640
    img = cv2.resize(img, (input_size, input_size))

    #img = start_detection(img, model, device, choice='tracking')

    cv2.imwrite('assets/test_image.jpg', img)

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