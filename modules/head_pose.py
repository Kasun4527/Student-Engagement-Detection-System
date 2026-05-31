import cv2
import numpy as np

class HeadPoseEstimator:
    def __init__(self, frame_width=640, frame_height=480):

        self.model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ])

        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)

        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        self.dist_coeffs = np.zeros((4, 1))

        self.YAW_THRESHOLD = 20
        self.PITCH_THRESHOLD = 15

    def estimate(self, landmarks):

        image_points = np.array([
            landmarks[1],
            landmarks[152],
            landmarks[33],
            landmarks[263],
            landmarks[61],
            landmarks[291]
        ], dtype="double")

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs
        )

        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

        pitch, yaw, roll = angles

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

        return {
            "pitch": round(pitch, 2),
            "yaw": round(yaw, 2),
            "roll": round(roll, 2),
            "direction": direction,
            "looking_away": direction != "Forward"
        }