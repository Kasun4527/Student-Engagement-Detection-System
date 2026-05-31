import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import time
from modules.face_detector import FaceDetector
from modules.head_pose import HeadPoseEstimator

face_detector = FaceDetector()
head_pose = HeadPoseEstimator()

cap = cv2.VideoCapture(0)
print("Looking at camera for 3 seconds...")
start_time = time.time()

while time.time() - start_time < 3.0:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    landmarks, _ = face_detector.get_landmarks(frame)
    if landmarks is not None:
        res = head_pose.estimate(landmarks)
        if res:
            print(f"Yaw: {res['yaw']:.2f}, Pitch: {res['pitch']:.2f}, Roll: {res['roll']:.2f}, Dir: {res['direction']}, Looking Away: {res['looking_away']}")
    time.sleep(0.1)

cap.release()
