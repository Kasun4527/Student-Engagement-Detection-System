import cv2
import mediapipe as mp
import numpy as np


class FaceDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def get_landmarks(self, frame):
        """
        Returns:
            landmarks_px : numpy array (468,2)
            results      : mediapipe result object
        """

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None, results

        face_landmarks = results.multi_face_landmarks[0]

        h, w = frame.shape[:2]

        landmarks_px = []

        for lm in face_landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)

            landmarks_px.append([x, y])

        landmarks_px = np.array(landmarks_px)

        return landmarks_px, results

    def draw_landmarks(self, frame, results):

        if not results.multi_face_landmarks:
            return frame

        for face_landmarks in results.multi_face_landmarks:

            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                face_landmarks,
                self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=
                mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style()
            )

        return frame