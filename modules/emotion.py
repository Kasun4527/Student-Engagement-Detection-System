import torch
import torch.nn as nn
import cv2
import numpy as np


# ---------------- MODEL ----------------
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------- EMOTION RECOGNIZER ----------------
class EmotionRecognizer:
    def __init__(self, model_path="models/emotion_model_best.pt"):
        self.model = EmotionCNN()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

        self.labels = [
            "angry",
            "disgust",
            "fear",
            "happy",
            "neutral",
            "sad",
            "surprise"
        ]

    # ---------------- PREPROCESS ----------------
    def preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (48, 48))
        gray = gray / 255.0

        # (1, 1, 48, 48)
        gray = np.expand_dims(gray, axis=0)
        gray = np.expand_dims(gray, axis=0)

        return torch.tensor(gray, dtype=torch.float32)

    # ---------------- PREDICT ----------------
    def predict(self, img):
        x = self.preprocess(img)

        with torch.no_grad():
            out = self.model(x)
            probs = torch.softmax(out, dim=1)

            conf, pred = torch.max(probs, dim=1)

        emotion = self.labels[pred.item()]
        confidence = conf.item()

        return emotion, confidence