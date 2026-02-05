import cv2

def test_media_loading():
    # Test loading image
    img = cv2.imread('assets/test_image.jpg')
    assert img is not None, "Error: test image not found"
    assert img.shape[2] == 3, "Test image must have 3 channels (RGB/BGR)"

    # Test video opening
    cap = cv2.VideoCapture('assets/test_video.mp4')
    assert cap.isOpened(), "Error: Impossible to open test video"
    cap.release()