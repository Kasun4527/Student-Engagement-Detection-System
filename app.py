import os
import sys
import time
import threading
import json
import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

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
        self.running = False       # True only while thread is alive and looping
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

    # ------------------------------------------------------------------ #
    #  BUG FIX 1: start() was guarded by `if not self.running` but stop()
    #  set self.running = False BEFORE joining — so the thread could still
    #  be alive when start() was called again, and self.running was already
    #  False so the guard passed, spawning a second thread.
    #  Fix: use a separate `_thread_alive` flag checked by the guard, and
    #  only set self.running = False inside the thread when it exits.
    # ------------------------------------------------------------------ #

    def start(self):
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                print("VideoProcessor already running — ignoring duplicate start().")
                return
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            print("VideoProcessor background thread started.")

    def stop(self):
        # Signal the loop to exit
        with self.lock:
            self.running = False
        # Wait for thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        self.thread = None
        print("VideoProcessor background thread stopped.")

    def get_frame(self):
        with self.lock:
            return self.latest_frame

    def get_stats_and_history(self):
        with self.lock:
            return self.latest_stats.copy(), list(self.history)

    def is_active(self):
        with self.lock:
            return self.running

    def _encode_jpeg(self, frame):
        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            return jpeg.tobytes()
        return None

    def _add_to_history(self, score):
        self.history.append({"time": time.strftime("%H:%M:%S"), "score": score})
        if len(self.history) > 60:
            self.history.pop(0)

    def _run(self):
        # Load models inside thread (avoids CUDA context issues)
        try:
            face_detector = FaceDetector()
            emotion_model = EmotionRecognizer()
            drowsiness = DrowsinessDetector()
            head_pose = HeadPoseEstimator()
            behavior = BehaviorDetector()
            engine = EngagementEngine()
            print("ML/CV models loaded successfully.")
        except Exception as e:
            print(f"Error loading models: {e}")
            with self.lock:
                self.running = False
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("WARNING: Webcam not available. Running in placeholder mode.")

        last_history_update = 0

        while True:
            with self.lock:
                if not self.running:
                    break

            # ---- No camera fallback ----
            if not self.cap or not self.cap.isOpened():
                frame = np.full((480, 640, 3), (18, 12, 10), dtype=np.uint8)
                cv2.putText(frame, "Webcam Not Connected or In Use", (70, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
                cv2.putText(frame, "Check system camera permissions", (130, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                stats = {
                    "state": "No Camera Detected ⚠️", "score": 0.0,
                    "emotion": "N/A", "emo_conf": 0.0, "ear": 0.0, "mar": 0.0,
                    "drowsy": False, "yawning": False, "head_direction": "N/A",
                    "phone_detected": False, "looking_away": False,
                    "person_present": False, "timestamp": time.time()
                }
                with self.lock:
                    self.latest_stats = stats
                    self.latest_frame = self._encode_jpeg(frame)
                time.sleep(0.5)
                continue

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            frame = cv2.resize(frame, (640, 480))
            landmarks, results = face_detector.get_landmarks(frame)
            h, w = frame.shape[:2]

            # ---- No face detected ----
            if landmarks is None:
                stats = {
                    "state": "No Face Detected 🔍", "score": 0.0,
                    "emotion": "Unknown", "emo_conf": 0.0, "ear": 0.0, "mar": 0.0,
                    "drowsy": False, "yawning": False, "head_direction": "Unknown",
                    "phone_detected": False, "looking_away": False,
                    "person_present": False, "timestamp": time.time()
                }
                try:
                    br = behavior.detect(frame)
                    stats["phone_detected"] = br["phone_detected"]
                    stats["person_present"] = br["person_present"]
                    if stats["phone_detected"]:
                        stats["state"] = "Phone Detected 📱"
                except Exception:
                    pass

                cv2.putText(frame, "No Face Detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                now = time.time()
                with self.lock:
                    self.latest_stats = stats
                    self.latest_frame = self._encode_jpeg(frame)
                    if now - last_history_update >= 0.25:
                        self._add_to_history(0.0)
                        last_history_update = now
                time.sleep(0.03)
                continue

            # ---- Face detected: run full pipeline ----
            x_min, y_min = np.min(landmarks, axis=0)
            x_max, y_max = np.max(landmarks, axis=0)
            x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
            x_max, y_max = min(w, int(x_max)), min(h, int(y_max))
            face_roi = frame[y_min:y_max, x_min:x_max]

            emotion, emo_conf = emotion_model.predict(face_roi)

            drowsy_result = drowsiness.analyze(landmarks)
            ear = drowsy_result["ear"]
            mar = drowsy_result["mar"]
            is_drowsy = drowsy_result["drowsy"]
            is_yawning = drowsy_result["yawning"]

            head_result = head_pose.estimate(landmarks)
            if head_result is None:
                head_score, head_direction, looking_away = 0.5, "Unknown", False
            else:
                head_score = 1.0 if not head_result["looking_away"] else 0.3
                head_direction = head_result["direction"]
                looking_away = head_result["looking_away"]

            behavior_result = behavior.detect(frame)
            phone_detected = behavior_result["phone_detected"]
            person_present = behavior_result["person_present"]
            behavior_score = behavior.get_score(frame)

            score, state = engine.calculate(emotion, ear, mar, head_score, behavior_score)

            bbox_color = (0, 255, 0) if score > 75 else \
                         (0, 255, 255) if score > 50 else \
                         (0, 165, 255) if score > 30 else (0, 0, 255)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), bbox_color, 2)

            stats = {
                "state": state, "score": score,
                "emotion": emotion, "emo_conf": float(emo_conf),
                "ear": float(ear), "mar": float(mar),
                "drowsy": bool(is_drowsy), "yawning": bool(is_yawning),
                "head_direction": head_direction,
                "phone_detected": bool(phone_detected),
                "looking_away": bool(looking_away),
                "person_present": bool(person_present),
                "timestamp": time.time()
            }

            now = time.time()
            with self.lock:
                self.latest_stats = stats
                self.latest_frame = self._encode_jpeg(frame)
                if now - last_history_update >= 0.25:
                    self._add_to_history(score)
                    last_history_update = now

            time.sleep(0.03)

        # Cleanup on exit
        if self.cap:
            self.cap.release()
            self.cap = None
        print("VideoProcessor thread exited cleanly.")


# ── Global processor ────────────────────────────────────────────────────────
processor = VideoProcessor()
processor.start()


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            # BUG FIX 2: only stream when processor is active
            if not processor.is_active():
                break
            frame = processor.get_frame()
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.04)   # ~25 fps cap
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/stats')
def get_stats():
    stats, history = processor.get_stats_and_history()
    return jsonify({"current": stats, "history": history})


@app.route('/api/stats/stream')
def stats_stream():
    # BUG FIX 3: SSE generator now terminates when processor stops,
    # instead of running forever and leaking server-side resources.
    def event_stream():
        while processor.is_active():
            stats, history = processor.get_stats_and_history()
            data = json.dumps({"current": stats, "history": history})
            yield f"data: {data}\n\n"
            time.sleep(0.2)
        # Send a final "stopped" event so the client knows cleanly
        yield "data: {\"stopped\": true}\n\n"

    return Response(event_stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


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


@app.route('/api/status')
def api_status():
    return jsonify({"active": processor.is_active()})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)