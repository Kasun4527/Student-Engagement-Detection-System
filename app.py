import os
import sys
import time
import threading
import json
import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

# Ensure student_engagement directory is on path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.face_detector import FaceDetector
from modules.emotion import EmotionRecognizer
from modules.drowsiness import DrowsinessDetector
from modules.head_pose import HeadPoseEstimator
from modules.behavior_detector import BehaviorDetector
from modules.engagement_engine import EngagementEngine

app = Flask(__name__)

class VideoProcessor:
    def __init__(self):
        self.cap = None
        self.running = False
        self.thread = None
        self.latest_frame = None
        self.latest_stats = {
            "state": "System Idle 💤",
            "score": 0.0,
            "emotion": "Unknown",
            "emo_conf": 0.0,
            "ear": 0.0,
            "mar": 0.0,
            "drowsy": False,
            "yawning": False,
            "head_direction": "Unknown",
            "phone_detected": False,
            "looking_away": False,
            "person_present": False,
            "timestamp": time.time()
        }
        self.history = []
        self.lock = threading.Lock()

    def start(self):
        with self.lock:
            if not self.running:
                self.running = True
                self.thread = threading.Thread(target=self._run, daemon=True)
                self.thread.start()
                print("VideoProcessor background thread started.")

    def stop(self):
        with self.lock:
            self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
        print("VideoProcessor background thread stopped.")

    def get_frame(self):
        with self.lock:
            return self.latest_frame

    def get_stats_and_history(self):
        with self.lock:
            return self.latest_stats.copy(), self.history.copy()

    def _encode_jpeg(self, frame):
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            return jpeg.tobytes()
        return None

    def _add_to_history(self, score):
        self.history.append({
            "time": time.strftime("%H:%M:%S"),
            "score": score
        })
        # Maintain rolling window of last 60 records (~10-15 seconds at regular intervals)
        if len(self.history) > 60:
            self.history.pop(0)

    def _run(self):
        # Initialize modules inside the thread to avoid multi-threading CUDA/context issues
        try:
            face_detector = FaceDetector()
            emotion_model = EmotionRecognizer()
            drowsiness = DrowsinessDetector()
            head_pose = HeadPoseEstimator()
            behavior = BehaviorDetector()
            engine = EngagementEngine()
            print("Successfully loaded ML/CV models on background thread.")
        except Exception as e:
            print(f"Error loading models in VideoProcessor thread: {e}")
            with self.lock:
                self.running = False
            return

        self.cap = cv2.VideoCapture(0)
        
        # Gracefully handle cameras that cannot be opened
        if not self.cap.isOpened():
            print("WARNING: Webcam could not be opened. Running in offline/simulation mode.")

        last_history_update = 0

        while True:
            # Check run flag
            with self.lock:
                if not self.running:
                    break

            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.02)
                    continue
            else:
                # Fallback: No camera available - generate elegant placeholder frame with text
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                # Fill with a subtle dark grid
                frame[:] = (18, 12, 10)  # Dark slate background
                cv2.putText(frame, "Webcam Not Connected or In Use", (80, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
                cv2.putText(frame, "Check system camera permissions", (140, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                
                stats = {
                    "state": "No Camera Detected ⚠️",
                    "score": 0.0,
                    "emotion": "N/A",
                    "emo_conf": 0.0,
                    "ear": 0.0,
                    "mar": 0.0,
                    "drowsy": False,
                    "yawning": False,
                    "head_direction": "N/A",
                    "phone_detected": False,
                    "looking_away": False,
                    "person_present": False,
                    "timestamp": time.time()
                }
                
                with self.lock:
                    self.latest_stats = stats
                    self.latest_frame = self._encode_jpeg(frame)
                    
                time.sleep(0.5)
                continue

            frame = cv2.resize(frame, (640, 480))
            landmarks, results = face_detector.get_landmarks(frame)

            h, w = frame.shape[:2]

            if landmarks is None:
                # No face detected
                stats = {
                    "state": "No Face Detected 🔍",
                    "score": 0.0,
                    "emotion": "Unknown",
                    "emo_conf": 0.0,
                    "ear": 0.0,
                    "mar": 0.0,
                    "drowsy": False,
                    "yawning": False,
                    "head_direction": "Unknown",
                    "phone_detected": False,
                    "looking_away": False,
                    "person_present": False,
                    "timestamp": time.time()
                }

                # Try behavioral detection on frame even if face landmarks are missing
                try:
                    behavior_result = behavior.detect(frame)
                    stats["phone_detected"] = behavior_result["phone_detected"]
                    stats["person_present"] = behavior_result["person_present"]
                    if stats["phone_detected"]:
                        stats["state"] = "Phone Detected 📱"
                except Exception:
                    pass

                cv2.putText(frame, "No Face Detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                with self.lock:
                    self.latest_stats = stats
                    self.latest_frame = self._encode_jpeg(frame)
                    
                    # Update history every 250ms to keep chart clean
                    now = time.time()
                    if now - last_history_update >= 0.25:
                        self._add_to_history(0.0)
                        last_history_update = now

                time.sleep(0.03)
                continue

            # Face ROI
            x_min, y_min = np.min(landmarks, axis=0)
            x_max, y_max = np.max(landmarks, axis=0)
            x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
            x_max, y_max = min(w, int(x_max)), min(h, int(y_max))
            face_roi = frame[y_min:y_max, x_min:x_max]

            # Emotion
            emotion, emo_conf = emotion_model.predict(face_roi)

            # Drowsiness + Yawning
            drowsy_result = drowsiness.analyze(landmarks)
            ear = drowsy_result["ear"]
            mar = drowsy_result["mar"]
            is_drowsy = drowsy_result["drowsy"]
            is_yawning = drowsy_result["yawning"]

            # Head Pose
            head_result = head_pose.estimate(landmarks)
            if head_result is None:
                head_score = 0.5
                head_direction = "Unknown"
                looking_away = False
            else:
                head_score = 1.0 if not head_result["looking_away"] else 0.3
                head_direction = head_result["direction"]
                looking_away = head_result["looking_away"]

            # Behavior (Phone + Person detection)
            behavior_result = behavior.detect(frame)
            phone_detected = behavior_result["phone_detected"]
            person_present = behavior_result["person_present"]
            behavior_score = behavior.get_score(frame)

            # Calculate Engagement Score
            score, state = engine.calculate(
                emotion,
                ear,
                mar,
                head_score,
                behavior_score
            )

            # Draw visual guides (bounding box) on camera feed
            if score > 75:
                bbox_color = (0, 255, 0)      # Green
            elif score > 50:
                bbox_color = (0, 255, 255)    # Yellow
            elif score > 30:
                bbox_color = (0, 165, 255)    # Orange
            else:
                bbox_color = (0, 0, 255)      # Red

            # ---------------- Face Box ----------------
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), bbox_color, 2)

            # Assemble telemetry dictionary
            stats = {
                "state": state,
                "score": score,
                "emotion": emotion,
                "emo_conf": float(emo_conf),
                "ear": float(ear),
                "mar": float(mar),
                "drowsy": bool(is_drowsy),
                "yawning": bool(is_yawning),
                "head_direction": head_direction,
                "phone_detected": bool(phone_detected),
                "looking_away": bool(looking_away),
                "person_present": bool(person_present),
                "timestamp": time.time()
            }

            with self.lock:
                self.latest_stats = stats
                self.latest_frame = self._encode_jpeg(frame)
                
                # Update history every 250ms to keep chart clean
                now = time.time()
                if now - last_history_update >= 0.25:
                    self._add_to_history(score)
                    last_history_update = now

            time.sleep(0.03)

        if self.cap:
            self.cap.release()
            self.cap = None


# Create global processor instance and start it
processor = VideoProcessor()
processor.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            # Throttle stream reading to save bandwidth and processor overhead
            frame = processor.get_frame()
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.04)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_stats():
    stats, history = processor.get_stats_and_history()
    return jsonify({
        "current": stats,
        "history": history
    })

@app.route('/api/stats/stream')
def stats_stream():
    def event_stream():
        while True:
            stats, history = processor.get_stats_and_history()
            data = json.dumps({"current": stats, "history": history})
            yield f"data: {data}\n\n"
            time.sleep(0.2) # send stats 5 times per second
    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/api/toggle_camera', methods=['POST'])
def toggle_camera():
    data = request.get_json() or {}
    action = data.get('action')
    if action == 'start':
        processor.start()
        return jsonify({"status": "started", "active": True})
    elif action == 'stop':
        processor.stop()
        return jsonify({"status": "stopped", "active": False})
    return jsonify({"error": "invalid action"}), 400

if __name__ == '__main__':
    # Run the server on all interfaces, port 5000
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
