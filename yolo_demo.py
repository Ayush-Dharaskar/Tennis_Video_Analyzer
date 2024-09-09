from ultralytics import YOLO
model = YOLO("models/ball_best.pt")
result = model.track('input_video.mp4',save = True)
