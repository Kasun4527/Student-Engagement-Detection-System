import cv2
import numpy as np

from modules.face_detector import FaceDetector
from modules.emotion import EmotionRecognizer
from modules.drowsiness import DrowsinessDetector
from modules.head_pose import HeadPoseEstimator
from modules.behavior_detector import BehaviorDetector
from modules.engagement_engine import EngagementEngine

# ---------------- Init Modules ----------------
face_detector = FaceDetector()
emotion_model = EmotionRecognizer()
drowsiness = DrowsinessDetector()
head_pose = HeadPoseEstimator()
behavior = BehaviorDetector()
engine = EngagementEngine()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    # ---------------- Face + Landmarks ----------------
    landmarks, results = face_detector.get_landmarks(frame)

    if landmarks is None:
        cv2.imshow("Engagement System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    h, w = frame.shape[:2]

    # ---------------- Face ROI ----------------
    x_min, y_min = np.min(landmarks, axis=0)
    x_max, y_max = np.max(landmarks, axis=0)

    x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
    x_max, y_max = min(w, int(x_max)), min(h, int(y_max))

    face_roi = frame[y_min:y_max, x_min:x_max]

    # ---------------- Emotion ----------------
    emotion, emo_conf = emotion_model.predict(face_roi)

    # ---------------- Drowsiness + Yawning ----------------
    drowsy_result = drowsiness.analyze(landmarks)

    ear = drowsy_result["ear"]
    mar = drowsy_result["mar"]
    is_drowsy = drowsy_result["drowsy"]
    is_yawning = drowsy_result["yawning"]

    # ---------------- Head Pose ----------------
    head_result = head_pose.estimate(landmarks)

    if head_result is None:
        head_score = 0.5
        head_direction = "Unknown"
        looking_away = False
    else:
        head_score = 1.0 if not head_result["looking_away"] else 0.3
        head_direction = head_result["direction"]
        looking_away = head_result["looking_away"]

    # ---------------- Behavior (Phone detection) ----------------
    behavior_result = behavior.detect(frame)
    phone_detected = behavior_result["phone_detected"]
    person_present = behavior_result["person_present"]

    behavior_score = behavior.get_score(frame)

    # ---------------- Engagement Score ----------------
    score, state = engine.calculate(
        emotion,
        ear,
        mar,
        head_score,
        behavior_score
    )

    # ---------------- UI TEXT BLOCK ----------------
    y = 30

    cv2.putText(frame, f"State: {state}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    y += 30
    cv2.putText(frame, f"Score: {score}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    y += 30
    cv2.putText(frame, f"Emotion: {emotion} ({emo_conf:.2f})", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    y += 30
    cv2.putText(frame, f"EAR: {ear:.3f}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    y += 30
    cv2.putText(frame, f"MAR: {mar:.3f}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    y += 30
    cv2.putText(frame, f"Drowsy: {is_drowsy}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if is_drowsy else (0, 255, 0), 2)

    y += 30
    cv2.putText(frame, f"Yawning: {is_yawning}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if is_yawning else (0, 255, 0), 2)

    y += 30
    cv2.putText(frame, f"Head: {head_direction}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2)

    y += 30
    cv2.putText(frame, f"Phone: {phone_detected}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if phone_detected else (0, 255, 0), 2)

    # ---------------- Face Box ----------------
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # ---------------- Show ----------------
    cv2.imshow("Engagement System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()