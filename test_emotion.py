import cv2
from modules.face_detector import FaceDetector
from modules.emotion import EmotionRecognizer

face = FaceDetector()
emotion = EmotionRecognizer()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks, results = face.get_landmarks(frame)

    if landmarks is not None:

        # Simple face crop (bounding box idea)
        h, w = frame.shape[:2]

        x_min = int(min(landmarks[:,0]))
        y_min = int(min(landmarks[:,1]))
        x_max = int(max(landmarks[:,0]))
        y_max = int(max(landmarks[:,1]))

        face_roi = frame[y_min:y_max, x_min:x_max]

        if face_roi.size > 0:
            emo, conf = emotion.predict(face_roi)

            cv2.putText(frame,
                        f"{emo} ({conf:.2f})",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()