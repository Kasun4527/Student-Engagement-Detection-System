import cv2
import numpy as np

class HeadPoseEstimator:
    def __init__(self, frame_width, frame_height):

        # 3D model points (generic face model)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (0.0, -330.0, -65.0),     # Chin
            (-225.0, 170.0, -135.0),  # Left eye
            (225.0, 170.0, -135.0),   # Right eye
            (-150.0, -150.0, -125.0), # Left mouth
            (150.0, -150.0, -125.0)   # Right mouth
        ])

        # Camera matrix
        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)

        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        self.dist_coeffs = np.zeros((4, 1))

        # Landmark indexes (MediaPipe)
        self.FACE_IDXS = [1, 152, 33, 263, 61, 291]

        # Thresholds
        self.YAW_THRESHOLD = 20
        self.PITCH_THRESHOLD = 15

    def estimate(self, landmarks):

        # 2D image points
        image_points = np.array([
            landmarks[1],    # Nose
            landmarks[152],  # Chin
            landmarks[33],   # Left eye
            landmarks[263],  # Right eye
            landmarks[61],   # Left mouth
            landmarks[291]   # Right mouth
        ], dtype="double")

        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs
        )

        if not success:
            return None

        # Convert rotation to angles
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

        pitch = angles[0]
        yaw = angles[1]
        roll = angles[2]

        # ---------------- Direction Logic ----------------

        if yaw < -self.YAW_THRESHOLD:
            direction = "Looking Left"
        elif yaw > self.YAW_THRESHOLD:
            direction = "Looking Right"
        elif pitch > self.PITCH_THRESHOLD:
            direction = "Looking Up"
        elif pitch < -self.PITCH_THRESHOLD:
            direction = "Looking Down"
        else:
            direction = "Forward"

        looking_away = direction != "Forward"

        return {
            "pitch": round(pitch, 2),
            "yaw": round(yaw, 2),
            "roll": round(roll, 2),
            "direction": direction,
            "looking_away": looking_away
        }