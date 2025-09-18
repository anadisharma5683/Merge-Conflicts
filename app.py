import cv2
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO
import threading
import time


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
    """Optimized centroid-based vehicle tracker"""
    
    def __init__(self, max_disappeared: int = 40, max_distance: int = 1000):
        self.next_id = 0
        self.objects = {}
        self.disappeared = defaultdict(int)
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.counted_ids = set()
        
    def calculate_centroid(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)
    
    def calculate_distance_squared(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        # Use squared distance to avoid expensive sqrt operation
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
    
    def update(self, detections: List[Tuple], frame_num: int) -> Dict[int, TrackedObject]:
        if len(detections) == 0:
            # Mark disappeared objects
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
            
            # Use squared distance for faster computation
            max_distance_squared = self.max_distance ** 2
            distances = np.zeros((len(object_ids), len(input_centroids)))
            
            for i, obj_centroid in enumerate(object_centroids):
                for j, input_centroid in enumerate(input_centroids):
                    distances[i][j] = self.calculate_distance_squared(obj_centroid, input_centroid)
            
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                obj_id = object_ids[row]
                if distances[row, col] > max_distance_squared:
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
    """Optimized traffic detection and tracking system with playback controls"""
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.tracker = VehicleTracker()

        # ROI region (x1, y1, x2, y2)
        self.roi = (270, 170, 720, 600)  

        self.vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

        self.counters = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0, 'total': 0}

        self.colors = {'car': (0, 255, 0), 'motorcycle': (255, 0, 0),
                       'bus': (0, 255, 255), 'truck': (0, 0, 255)}

        # Playback controls
        self.is_playing = True
        self.frame_skip = 1  # Process every nth frame
        self.current_frame = 0
        
        # Performance optimization
        self.detection_interval = 3  # Run detection every 3 frames
        self.last_detections = []
        self.frame_count = 0

        self.use_yolov8 = False
        try:
            self.model = YOLO('yolov8n.pt' if model_path is None else model_path)
            self.use_yolov8 = True
            # Optimize YOLO settings
            self.model.overrides['verbose'] = False
            self.model.overrides['device'] = 'cpu'  # Change to 'cuda' if GPU available
        except ImportError:
            print("YOLOv8 not available, using YOLOv4-tiny")
            self.load_yolov4()

    def load_yolov4(self):
        try:
            self.net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
            with open("coco.names", "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
            self.output_layers = [self.net.getLayerNames()[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
            # Enable GPU if available
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except cv2.error as e:
            print(f"Error loading YOLOv4 files: {e}")
            self.net = None

    def reset_counters(self):
        """Reset all counters and tracking data"""
        self.counters = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0, 'total': 0}
        self.tracker = VehicleTracker()
        self.current_frame = 0
        self.frame_count = 0
        self.last_detections = []

    def play_pause(self):
        """Toggle play/pause state"""
        self.is_playing = not self.is_playing

    def set_frame_skip(self, skip_value: int):
        """Set frame skip value (1 = no skip, 2 = skip every other frame, etc.)"""
        self.frame_skip = max(1, skip_value)

    def draw_roi(self, frame):
        """Draw translucent ROI area with border - optimized"""
        x1, y1, x2, y2 = self.roi
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
        # Resize frame for faster inference if needed
        height, width = frame.shape[:2]
        if width > 1280:
            scale = 1280 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized_frame = cv2.resize(frame, (new_width, new_height))
        else:
            resized_frame = frame
            scale = 1.0

        results = self.model(resized_frame, conf=self.confidence_threshold, verbose=False)
        
        for box in results[0].boxes:
            cls = int(box.cls[0])
            if cls in self.vehicle_classes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # Scale back to original size
                if scale != 1.0:
                    x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
                
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
        # Use smaller blob size for faster inference
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320),
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
            cv2.putText(frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def draw_statistics(self, frame):
        # Simplified statistics drawing for better performance
        y_offset = 30
        cv2.putText(frame, "Vehicle Count", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        for vehicle_type in ['car', 'motorcycle', 'bus', 'truck']:
            count = self.counters[vehicle_type]
            text = f"{vehicle_type.capitalize()}: {count}"
            cv2.putText(frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors[vehicle_type], 1)
            y_offset += 20
            
        cv2.putText(frame, f"Total: {self.counters['total']}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Show playback status
        status = "PLAYING" if self.is_playing else "PAUSED"
        cv2.putText(frame, f"Status: {status}", (10, frame.shape[0] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if self.is_playing else (0, 0, 255), 2)
        cv2.putText(frame, f"Frame Skip: {self.frame_skip}", (10, frame.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Frame: {self.current_frame}", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
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
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30  # Default to 30 FPS
        
        while True:
            if not self.is_playing:
                time.sleep(0.1)  # Small delay when paused
                continue
                
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                self.reset_counters()
                continue
            
            self.current_frame += 1
            self.frame_count += 1
            
            # Skip frames for performance
            if (self.current_frame - 1) % self.frame_skip != 0:
                continue
            
            frame = self.draw_roi(frame)
            
            # Run detection only every few frames
            if self.frame_count % self.detection_interval == 0:
                if self.use_yolov8:
                    self.last_detections = self.detect_vehicles_yolov8(frame)
                else:
                    self.last_detections = self.detect_vehicles_yolov4(frame)
            
            # Always update tracking with last detections
            tracked_objects = self.tracker.update(self.last_detections, self.current_frame)
            self.update_counters(tracked_objects)
            self.draw_detections(frame, tracked_objects)
            frame = self.draw_statistics(frame)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # Control frame rate
            time.sleep(frame_delay / self.frame_skip)
        
        cap.release()

# ------------------------------------------------------------
# Flask App Setup
# ------------------------------------------------------------
app = Flask(__name__)
CORS(app)
detector = TrafficDetectionSystem(confidence_threshold=0.59)

@app.route('/')
def index():
    return """
    <h1>Traffic Detection System</h1>
    <p>Backend is running. Available endpoints:</p>
    <ul>
        <li><a href="/video_feed">/video_feed</a> - Video stream</li>
        <li><a href="/get_counts">/get_counts</a> - Get current counts</li>
        <li>/play_pause (POST) - Toggle play/pause</li>
        <li>/reset_counters (POST) - Reset all counters</li>
        <li>/set_frame_skip (POST) - Set frame skip value</li>
    </ul>
    """

@app.route('/video_feed')
def video_feed():
    VIDEO_PATH = "traffic.mp4"
    return Response(detector.generate_frames(VIDEO_PATH),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_counts')
def get_counts():
    return jsonify({
        'counters': detector.counters,
        'is_playing': detector.is_playing,
        'frame_skip': detector.frame_skip,
        'current_frame': detector.current_frame
    })

@app.route('/play_pause', methods=['POST'])
def play_pause():
    detector.play_pause()
    return jsonify({
        'status': 'playing' if detector.is_playing else 'paused',
        'is_playing': detector.is_playing
    })

@app.route('/reset_counters', methods=['POST'])
def reset_counters():
    detector.reset_counters()
    return jsonify({
        'message': 'Counters reset successfully',
        'counters': detector.counters
    })

@app.route('/set_frame_skip', methods=['POST'])
def set_frame_skip():
    try:
        skip_value = int(request.json.get('skip_value', 1))
        detector.set_frame_skip(skip_value)
        return jsonify({
            'message': f'Frame skip set to {detector.frame_skip}',
            'frame_skip': detector.frame_skip
        })
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid skip value'}), 400

if __name__ == '__main__':
    print("Starting optimized Flask backend...")
    print("Navigate to http://127.0.0.1:5000 in your browser.")
    print("\nAPI Endpoints:")
    print("- POST /play_pause - Toggle play/pause")
    print("- POST /reset_counters - Reset all counters")
    print("- POST /set_frame_skip - Set frame skip (JSON: {'skip_value': N})")
    print("- GET /get_counts - Get current status and counts")
    app.run(host='0.0.0.0', port=5000, debug=True)