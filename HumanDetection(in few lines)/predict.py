from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")
model.predict(source='Videos/humanDetection1 (1080p).mp4', save=True, conf=0.5, save_txt=False, show=True)