import numpy as np


class DrowsinessDetector:
    def __init__(self):

        # MediaPipe Face Mesh landmarks
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        # Mouth landmarks
        self.MOUTH = [61, 291, 13, 14, 78, 308]

        # Thresholds
        self.EAR_THRESHOLD = 0.25
        self.MAR_THRESHOLD = 0.50  # lowered from 0.60 — easier to trigger yawn

        self.CLOSED_FRAMES = 20
        self.YAWN_FRAMES = 10      # lowered from 20 — fewer frames needed

        # Counters
        self.closed_counter = 0
        self.yawn_counter = 0

        self.drowsy = False
        self.yawning = False

    def _euclidean(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    # ---------------- EAR ----------------
    def _ear(self, eye):
        A = self._euclidean(eye[1], eye[5])
        B = self._euclidean(eye[2], eye[4])
        C = self._euclidean(eye[0], eye[3])

        if C == 0:
            return 0

        return (A + B) / (2.0 * C)

    # ---------------- MAR ----------------
    def _mar(self, mouth):
        A = self._euclidean(mouth[2], mouth[3])
        B = self._euclidean(mouth[4], mouth[5])
        C = self._euclidean(mouth[0], mouth[1])

        if C == 0:
            return 0

        return (A + B) / (2.0 * C)

    # ---------------- Public Methods ----------------
    def get_ear(self, landmarks_px):
        left_eye = landmarks_px[self.LEFT_EYE]
        right_eye = landmarks_px[self.RIGHT_EYE]

        ear = (self._ear(left_eye) + self._ear(right_eye)) / 2.0
        return float(ear)

    def get_mar(self, landmarks_px):
        mouth = landmarks_px[self.MOUTH]

        mar = self._mar(mouth)
        return float(mar)

    # ---------------- Full Analysis ----------------
    def analyze(self, landmarks_px):

        ear = self.get_ear(landmarks_px)
        mar = self.get_mar(landmarks_px)

        # -------- Eye Closure Detection --------
        if ear < self.EAR_THRESHOLD:
            self.closed_counter += 1
        else:
            self.closed_counter = 0
            self.drowsy = False

        if self.closed_counter >= self.CLOSED_FRAMES:
            self.drowsy = True

        # -------- Yawning Detection --------
        if mar > self.MAR_THRESHOLD:
            self.yawn_counter += 1
        else:
            self.yawn_counter = 0
            self.yawning = False

        if self.yawn_counter >= self.YAWN_FRAMES:
            self.yawning = True

        return {
            "ear": round(ear, 3),
            "mar": round(mar, 3),
            "drowsy": self.drowsy,
            "yawning": self.yawning,
            "closed_frames": self.closed_counter,
            "yawn_frames": self.yawn_counter
        }