import cv2

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
        x1, y1, x2, y2, score, id = result

        # Make boundingbox and txt label fro the frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
        cv2.putText(frame, f"ID: {int(id)}  Score: {score:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame

def make_box_tracking(frame, result):
    for result in result[0].boxes:
        x1, y1, x2, y2 = result.xyxy[0]
        id = int(result.id.item())

        # Make boundingbox and txt label fro the frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
        cv2.putText(frame, f"ID: {id}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame

def start_detection(tensor, frame, model, scelta = 'detection'):

    if scelta == 'detection':
        result = yolo_detection(tensor, model)
        if result[0].boxes is not None:
            frame = make_box_detection(frame, result)

    elif scelta == 'tracking':
        result = yolo_tracking(tensor, model)
        if result[0].boxes is not None and result[0].boxes.id is not None:
            frame = make_box_tracking(frame, result)

    return frame

