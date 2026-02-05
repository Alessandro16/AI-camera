import pytest
import cv2
from src.detection_people import yolo_detection

@pytest.fixture
def mock_model():
    # Load the model once for all tests to save time and memory
    from ultralytics import YOLO
    return YOLO('models/yolov8s.pt')

def test_image(mock_model):
    # Load test image
    img = cv2.imread('assets/test_image.jpg')
    assert img is not None

    # Test the yolo_detection function logic
    results = yolo_detection(img, mock_model)

    # Verify that the output is a list (standard for Ultralytics)
    assert isinstance(results, list), "The output should be a list"
    # Ensure the results list is not empty
    assert len(results) > 0, "The results list is empty"
    # Verify the result object contains the 'boxes' attribute
    assert hasattr(results[0], 'boxes'), "The result object does not contain bounding boxes"
    # Check if the model actually detected at least one person
    assert len(results[0].boxes) > 0, "Model failed to detect any person in the test image"


def test_video(mock_model):
    # Open the test video file
    video_path = 'assets/test_video.mp4'
    cap = cv2.VideoCapture(video_path)

    # Check if the video file exists and can be opened
    assert cap.isOpened(), f"Error: Could not open video at {video_path}"

    # Read only the first 5 frames to keep the test fast
    for i in range(5):
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection on the frame
        results = yolo_detection(frame, mock_model)

        # Basic assertions for each frame
        # Verify that the output is a list (standard for Ultralytics)
        assert isinstance(results, list), "The output should be a list"
        # Ensure the results list is not empty
        assert len(results) > 0, "The results list is empty"
        # Verify the result object contains the 'boxes' attribute
        assert hasattr(results[0], 'boxes'), "The result object does not contain bounding boxes"
        # Check if the model actually detected at least one person
        assert len(results[0].boxes) > 0, "Model failed to detect any person in the test image"

    cap.release()