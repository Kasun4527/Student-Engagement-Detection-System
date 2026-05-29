import cv2
from modules.behavior_detector import BehaviorDetector

detector = BehaviorDetector()

cap = cv2.VideoCapture(0)

frame_count = 0
result = {"phone_detected": False, "person_present": True}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 🔥 Run YOLO every 3 frames (performance optimization)
    if frame_count % 3 == 0:
        result = detector.detect(frame)

    # -------- Display --------
    if result["phone_detected"]:
        cv2.putText(frame,
                    "PHONE DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255), 3)

    if not result["person_present"]:
        cv2.putText(frame,
                    "ABSENT!",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (128,128,128), 3)

    cv2.imshow("YOLO Behavior Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()