class EngagementEngine:
    def __init__(self):
        self.weights = {
            "emotion": 0.25,
            "eye": 0.25,
            "mouth": 0.15,
            "head": 0.20,
            "behavior": 0.15
        }

        self.emotion_map = {
            "happy": 1.0,
            "neutral": 0.7,
            "surprise": 0.6,
            "sad": 0.3,
            "fear": 0.2,
            "angry": 0.2,
            "disgust": 0.1
        }

    def calculate(self, emotion, ear, mar, head_score, behavior_score):

        # ---------------- Emotion ----------------
        emotion_score = self.emotion_map.get(emotion, 0.5)

        # ---------------- Eye (Drowsiness) ----------------
        if ear < 0.20:
            eye_score = 0.0
        elif ear < 0.25:
            eye_score = 0.4
        else:
            eye_score = 1.0

        # ---------------- Mouth (Yawning) ----------------
        mouth_score = 0.0 if mar > 0.6 else 1.0

        # ---------------- Weighted Fusion ----------------
        engagement = (
            emotion_score * self.weights["emotion"] +
            eye_score * self.weights["eye"] +
            mouth_score * self.weights["mouth"] +
            head_score * self.weights["head"] +
            behavior_score * self.weights["behavior"]
        )

        # Convert to 0–100
        engagement_score = round(engagement * 100, 2)

        # ---------------- Final Label ----------------
        if engagement_score > 75:
            state = "Highly Engaged 😊"
        elif engagement_score > 50:
            state = "Moderately Engaged 😐"
        elif engagement_score > 30:
            state = "Low Engagement 😴"
        else:
            state = "Not Engaged ⚠️"

        return engagement_score, state