import cv2
from modules.face_detector import FaceDetector

face_detector = FaceDetector()

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    landmarks_px, results = face_detector.get_landmarks(frame)

    if landmarks_px is not None:

        frame = face_detector.draw_landmarks(frame, results)

        cv2.putText(
            frame,
            f"Landmarks: {len(landmarks_px)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

    cv2.imshow("Face Mesh", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()