import cv2
import sys
print(f"OpenCV version: {cv2.__version__}")
cap = cv2.VideoCapture("../debug_video_muestra3.mp4")
if not cap.isOpened():
    print("Could not open video")
    sys.exit(1)
ret, frame = cap.read()
if ret:
    print(f"Read frame: {frame.shape}")
else:
    print("Could not read frame")
cap.release()
