import cv2
from modules.face_detector import FaceDetector
from modules.drowsiness import DrowsinessDetector

face = FaceDetector()
drowsy = DrowsinessDetector()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks, results = face.get_landmarks(frame)

    if landmarks is not None:
        result = drowsy.analyze(landmarks)

        cv2.putText(frame,
                    f"EAR: {result['ear']}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0), 2)

        if result["drowsy"]:
            cv2.putText(frame,
                        "DROWSY ALERT!",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255), 3)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()