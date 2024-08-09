import random
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Opening the file in read mode
my_file = open("utils/coco.txt", "r")
# Reading the file
data = my_file.read()
# Replacing end splitting the text | when newline ('\n') is seen.
class_list = data.split("\n")
my_file.close()

# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# Load the pretrained YOLOv8n model
model = YOLO("weights/yolov8n.pt", "v8").to(device)

# Vals to resize video frames | small frame optimize the run
frame_wid = 640
frame_hyt = 480

cap = cv2.VideoCapture("inference/videos/production_id_4791180 (1080p).mp4")

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Resize the frame while maintaining the aspect ratio
    aspect_ratio = frame.shape[1] / frame.shape[0]
    new_height = int(frame_wid / aspect_ratio)
    frame = cv2.resize(frame, (frame_wid, new_height))

    # Convert the frame to a PIL Image
    pil_image = Image.fromarray(frame)

    # Predict on image
    detect_params = model.predict(source=pil_image, conf=0.45, save=False)

    # Filter out only the detections for the "person" class
    person_detections = []
    for i in range(len(detect_params[0])):
        boxes = detect_params[0].boxes
        box = boxes[i]  # Get one box
        clsID = box.cls.cpu().numpy()[0]  # Get the class ID
        if clsID == 0:  # Check if it corresponds to the "person" class
            person_detections.append(box)

    # Draw bounding boxes for person detections
    for box in person_detections:
        clsID = box.cls.cpu().numpy()[0]
        conf = box.conf.cpu().numpy()[0]
        bb = box.xyxy.cpu().numpy()[0]
        cv2.rectangle(
            frame,
            (int(bb[0]), int(bb[1])),
            (int(bb[2]), int(bb[3])),
            detection_colors[int(clsID)],
            3,
        )

        # Display class name and confidence
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(
            frame,
            class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
            (int(bb[0]), int(bb[1]) - 10),
            font,
            1,
            (255, 255, 255),
            2,
        )

    # Display the resulting frame
    cv2.imshow("ObjectDetection", frame)

    # Terminate run when "Q" pressed
    if cv2.waitKey(1) == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()