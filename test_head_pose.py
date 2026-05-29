import cv2
from modules.face_detector import FaceDetector
from modules.head_pose import HeadPoseEstimator

face = FaceDetector()

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
h, w = frame.shape[:2]

head_pose = HeadPoseEstimator(w, h)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks, _ = face.get_landmarks(frame)

    if landmarks is not None:

        result = head_pose.estimate(landmarks)

        if result:
            cv2.putText(frame,
                        f"{result['direction']}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0), 2)

            cv2.putText(frame,
                        f"Yaw:{result['yaw']} Pitch:{result['pitch']}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255,255,255), 2)

    cv2.imshow("Head Pose Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()