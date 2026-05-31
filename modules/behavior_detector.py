from ultralytics import YOLO
import cv2


class BehaviorDetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.5):

        # Load YOLO model
        self.model = YOLO(model_path)

        self.conf = conf

        # COCO class IDs
        self.PERSON_CLASS = 0
        self.PHONE_CLASS = 67

    def detect(self, frame):

        results = self.model(frame, conf=self.conf, verbose=False)[0]

        phone_detected = False
        person_present = False

        phone_bbox = None
        person_bbox = None

        for box in results.boxes:

            cls = int(box.cls[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int)

            # Phone
            if cls == self.PHONE_CLASS:
                phone_detected = True
                phone_bbox = xyxy

            # Person
            if cls == self.PERSON_CLASS:
                person_present = True
                person_bbox = xyxy

        return {
            "phone_detected": phone_detected,
            "person_present": person_present,
            "phone_bbox": phone_bbox,
            "person_bbox": person_bbox
        }

    def get_score(self, frame):
        """
        Convert behavior observations into a score between 0 and 1.
        """

        result = self.detect(frame)

        # No person detected
        if not result["person_present"]:
            return 0.0

        score = 1.0

        # Penalize phone usage
        if result["phone_detected"]:
            score -= 0.5

        return max(0.0, min(1.0, score))