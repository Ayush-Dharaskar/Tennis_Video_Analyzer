import cv2

def read_vid(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read() #ret tells us if the frame was read successfully
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_vid(frames, path):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(path, fourcc, 24, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(frame)
    out.release()