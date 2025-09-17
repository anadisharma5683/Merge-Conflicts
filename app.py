import cv2
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from flask import Flask, Response, jsonify
from flask_cors import CORS
from ultralytics import YOLO


@dataclass
class TrackedObject:
    """Data class for tracked vehicles"""
    id: int
    class_name: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    center: Tuple[int, int]
    last_seen: int
    confidence: float


class VehicleTracker:
    """Simple centroid-based vehicle tracker"""
    
    def __init__(self, max_disappeared: int = 40, max_distance: int = 1000):
        self.next_id = 0
        self.objects = {}
        self.disappeared = defaultdict(int)
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.counted_ids = set()
        
    def calculate_centroid(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x, y, w, h = bbox
        cx = x + w // 2
        cy = y + h // 2
        return (cx, cy)
    
    def calculate_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def update(self, detections: List[Tuple], frame_num: int) -> Dict[int, TrackedObject]:
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects
        
        input_centroids = []
        input_data = []
        for (bbox, class_name, confidence) in detections:
            centroid = self.calculate_centroid(bbox)
            input_centroids.append(centroid)
            input_data.append((bbox, class_name, confidence, centroid))
        
        if len(self.objects) == 0:
            for data in input_data:
                bbox, class_name, confidence, centroid = data
                self.register(bbox, class_name, confidence, centroid, frame_num)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[obj_id].center for obj_id in object_ids]
            
            distances = np.zeros((len(object_ids), len(input_centroids)))
            for i, obj_centroid in enumerate(object_centroids):
                for j, input_centroid in enumerate(input_centroids):
                    distances[i][j] = self.calculate_distance(obj_centroid, input_centroid)
            
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                obj_id = object_ids[row]
                if distances[row, col] > self.max_distance:
                    self.disappeared[obj_id] += 1
                    if self.disappeared[obj_id] > self.max_disappeared:
                        self.deregister(obj_id)
                else:
                    bbox, class_name, confidence, centroid = input_data[col]
                    self.objects[obj_id].bbox = bbox
                    self.objects[obj_id].center = centroid
                    self.objects[obj_id].last_seen = frame_num
                    self.objects[obj_id].confidence = confidence
                    self.disappeared[obj_id] = 0
                    used_rows.add(row)
                    used_cols.add(col)
            
            unused_rows = set(range(len(object_ids))) - used_rows
            unused_cols = set(range(len(input_centroids))) - used_cols
            
            for row in unused_rows:
                obj_id = object_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            
            for col in unused_cols:
                bbox, class_name, confidence, centroid = input_data[col]
                self.register(bbox, class_name, confidence, centroid, frame_num)
        
        return self.objects
    
    def register(self, bbox, class_name, confidence, centroid, frame_num):
        self.objects[self.next_id] = TrackedObject(
            id=self.next_id,
            class_name=class_name,
            bbox=bbox,
            center=centroid,
            last_seen=frame_num,
            confidence=confidence
        )
        self.disappeared[self.next_id] = 0
        self.next_id += 1
    
    def deregister(self, obj_id):
        del self.objects[obj_id]
        del self.disappeared[obj_id]


class TrafficDetectionSystem:
    """Main traffic detection and tracking system"""
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.tracker = VehicleTracker()

        # ROI region (x1, y1, x2, y2)
        self.roi = (300, 170, 800, 600)  

        self.vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

        self.counters = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0, 'total': 0}

        self.colors = {'car': (0, 255, 0), 'motorcycle': (255, 0, 0),
                       'bus': (0, 255, 255), 'truck': (0, 0, 255)}

        self.use_yolov8 = False
        try:
            self.model = YOLO('yolov8n.pt' if model_path is None else model_path)
            self.use_yolov8 = True
        except ImportError:
            print("YOLOv8 not available, using YOLOv4-tiny")
            self.load_yolov4()

    def load_yolov4(self):
        try:
            self.net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
            with open("coco.names", "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
            self.output_layers = [self.net.getLayerNames()[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        except cv2.error as e:
            print(f"Error loading YOLOv4 files: {e}")
            self.net = None


    def draw_roi(self, frame):
        """Draw translucent ROI area with border"""
        x1, y1, x2, y2 = self.roi
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame

    def is_inside_roi(self, centroid, roi=None):
        if roi is None:
            roi = self.roi
        x, y = centroid
        x1, y1, x2, y2 = roi
        return x1 <= x <= x2 and y1 <= y <= y2

    def detect_vehicles_yolov8(self, frame):
        detections = []
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        for box in results[0].boxes:
            cls = int(box.cls[0])
            if cls in self.vehicle_classes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                confidence = float(box.conf[0])
                class_name = self.vehicle_classes[cls]
                cx, cy = x + w // 2, y + h // 2
                if self.is_inside_roi((cx, cy)):
                    detections.append(((x, y, w, h), class_name, confidence))
        return detections

    def detect_vehicles_yolov4(self, frame):
        if self.net is None:
            return []
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416),
                                     (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        detections = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confidence_threshold:
                    if class_id < len(self.classes):
                        class_name = self.classes[class_id]
                        vehicle_mapping = {'car': 'car', 'motorbike': 'motorcycle',
                                           'bus': 'bus', 'truck': 'truck'}
                        if class_name in vehicle_mapping:
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            mapped_class = vehicle_mapping[class_name]
                            cx, cy = x + w // 2, y + h // 2
                            if self.is_inside_roi((cx, cy)):
                                detections.append(((x, y, w, h),
                                                    mapped_class,
                                                    float(confidence)))
        return detections

    def draw_detections(self, frame, tracked_objects):
        for obj_id, obj in tracked_objects.items():
            x, y, w, h = obj.bbox
            color = self.colors.get(obj.class_name, (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{obj.class_name.capitalize()} #{obj.id}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x, y - label_size[1] - 10),
                          (x + label_size[0], y), color, -1)
            cv2.putText(frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def draw_statistics(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        y_offset = 35
        cv2.putText(frame, "Vehicle Count", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        for vehicle_type in ['car', 'motorcycle', 'bus', 'truck']:
            count = self.counters[vehicle_type]
            text = f"{vehicle_type.capitalize()}: {count}"
            cv2.putText(frame, text, (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[vehicle_type], 2)
            y_offset += 25
        cv2.putText(frame, f"Total: {self.counters['total']}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return frame

    def update_counters(self, tracked_objects):
        for obj_id, obj in tracked_objects.items():
            if obj_id in self.tracker.counted_ids:
                continue
            if self.is_inside_roi(obj.center):
                self.counters[obj.class_name] += 1
                self.counters['total'] += 1
                self.tracker.counted_ids.add(obj_id)

    def generate_frames(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            frame = self.draw_roi(frame)
            if self.use_yolov8:
                detections = self.detect_vehicles_yolov8(frame)
            else:
                detections = self.detect_vehicles_yolov4(frame)
            tracked_objects = self.tracker.update(detections, frame_num)
            self.update_counters(tracked_objects)
            self.draw_detections(frame, tracked_objects)
            frame = self.draw_statistics(frame)
            
            # Encode the processed frame into JPEG format for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            
            # Yield the frame as a byte stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        cap.release()

# ------------------------------------------------------------
# Flask App Setup
# ------------------------------------------------------------
app = Flask(__name__)
CORS(app)
detector = TrafficDetectionSystem(confidence_threshold=0.55)

@app.route('/')
def index():
    return "Backend is running. Go to your frontend URL to view the analysis."

@app.route('/video_feed')
def video_feed():
    VIDEO_PATH = "traffic.mp4"
    return Response(detector.generate_frames(VIDEO_PATH),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_counts')
def get_counts():
    return jsonify(detector.counters)

if __name__ == '__main__':
    print("Starting Flask backend...")
    print("Navigate to http://127.0.0.1:5000 in your browser.")
    app.run(host='0.0.0.0', port=5000, debug=True)