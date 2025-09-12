"""
Real-Time Traffic Object Detection and Tracking System
Uses YOLOv8 for detection and custom tracking logic
"""

import cv2
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
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
    """Simple centroid-based vehicle tracker"""
    
    def __init__(self, max_disappeared: int = 30, max_distance: int = 80):
        self.next_id = 0
        self.objects = {}
        self.disappeared = defaultdict(int)
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.counted_ids = set()
        
    def calculate_centroid(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Calculate centroid of bounding box"""
        x, y, w, h = bbox
        cx = x + w // 2
        cy = y + h // 2
        return (cx, cy)
    
    def calculate_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def update(self, detections: List[Tuple], frame_num: int) -> Dict[int, TrackedObject]:
        """Update tracking with new detections"""
        
        # If no detections, mark existing objects as disappeared
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects
        
        # Get current centroids
        input_centroids = []
        input_data = []
        for (bbox, class_name, confidence) in detections:
            centroid = self.calculate_centroid(bbox)
            input_centroids.append(centroid)
            input_data.append((bbox, class_name, confidence, centroid))
        
        # If no existing objects, register all detections
        if len(self.objects) == 0:
            for data in input_data:
                bbox, class_name, confidence, centroid = data
                self.register(bbox, class_name, confidence, centroid, frame_num)
        else:
            # Match existing objects to new detections
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[obj_id].center for obj_id in object_ids]
            
            # Calculate distance matrix
            distances = np.zeros((len(object_ids), len(input_centroids)))
            for i, obj_centroid in enumerate(object_centroids):
                for j, input_centroid in enumerate(input_centroids):
                    distances[i][j] = self.calculate_distance(obj_centroid, input_centroid)
            
            # Find minimum distances for matching
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            # Update matched objects
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                obj_id = object_ids[row]
                
                # Check if distance is within threshold
                if distances[row, col] > self.max_distance:
                    self.disappeared[obj_id] += 1
                    if self.disappeared[obj_id] > self.max_disappeared:
                        self.deregister(obj_id)
                else:
                    # Update existing object
                    bbox, class_name, confidence, centroid = input_data[col]
                    self.objects[obj_id].bbox = bbox
                    self.objects[obj_id].center = centroid
                    self.objects[obj_id].last_seen = frame_num
                    self.objects[obj_id].confidence = confidence
                    self.disappeared[obj_id] = 0
                    used_rows.add(row)
                    used_cols.add(col)
            
            # Handle unmatched detections and objects
            unused_rows = set(range(len(object_ids))) - used_rows
            unused_cols = set(range(len(input_centroids))) - used_cols
            
            # Mark unmatched objects as disappeared
            for row in unused_rows:
                obj_id = object_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            
            # Register new objects from unmatched detections
            for col in unused_cols:
                bbox, class_name, confidence, centroid = input_data[col]
                self.register(bbox, class_name, confidence, centroid, frame_num)
        
        return self.objects
    
    def register(self, bbox, class_name, confidence, centroid, frame_num):
        """Register a new tracked object"""
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
        """Remove a tracked object"""
        del self.objects[obj_id]
        del self.disappeared[obj_id]


class TrafficDetectionSystem:
    """Main traffic detection and tracking system"""
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.5):
        """
        Initialize the traffic detection system
        
        Args:
            model_path: Path to YOLO weights (if None, uses YOLOv8n)
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.tracker = VehicleTracker()
        
        # Vehicle classes we're interested in (COCO dataset indices)
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        
        # Initialize counters
        self.counters = {
            'car': 0,
            'motorcycle': 0,
            'bus': 0,
            'truck': 0,
            'total': 0
        }
        
        # Colors for different vehicle types (BGR format)
        self.colors = {
            'car': (0, 255, 0),        # Green
            'motorcycle': (255, 0, 0),  # Blue
            'bus': (0, 255, 255),       # Yellow
            'truck': (0, 0, 255)        # Red
        }
        
        # Try to use YOLOv8 (ultralytics), fallback to YOLOv4 if not available
        self.use_yolov8 = False
        try:
            from ultralytics import YOLO
            print("Using YOLOv8 for detection")
            self.model = YOLO('yolov8n.pt' if model_path is None else model_path)
            self.use_yolov8 = True
        except ImportError:
            print("YOLOv8 not available, using YOLOv4-tiny")
            self.load_yolov4()
    
    def load_yolov4(self):
        """Load YOLOv4-tiny as fallback"""
        # Download these files if not present:
        # wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
        # wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg
        # wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names
        
        try:
            self.net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
            with open("coco.names", "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        except:
            print("Warning: YOLOv4 files not found. Please download:")
            print("- yolov4-tiny.weights")
            print("- yolov4-tiny.cfg")
            print("- coco.names")
            self.net = None
    
    def detect_vehicles_yolov8(self, frame):
        """Detect vehicles using YOLOv8"""
        detections = []
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls in self.vehicle_classes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                        confidence = float(box.conf[0])
                        class_name = self.vehicle_classes[cls]
                        detections.append(((x, y, w, h), class_name, confidence))
        
        return detections
    
    def detect_vehicles_yolov4(self, frame):
        """Detect vehicles using YOLOv4"""
        if self.net is None:
            return []
        
        height, width, _ = frame.shape
        
        # Prepare input
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        # Process detections
        detections = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    # Map class_id to class name
                    if class_id < len(self.classes):
                        class_name = self.classes[class_id]
                        
                        # Check if it's a vehicle we're interested in
                        vehicle_mapping = {
                            'car': 'car',
                            'motorbike': 'motorcycle',
                            'bus': 'bus',
                            'truck': 'truck'
                        }
                        
                        if class_name in vehicle_mapping:
                            # Get bounding box
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            
                            mapped_class = vehicle_mapping[class_name]
                            detections.append(((x, y, w, h), mapped_class, float(confidence)))
        
        return detections
    
    def draw_detections(self, frame, tracked_objects):
        """Draw bounding boxes and labels on frame"""
        for obj_id, obj in tracked_objects.items():
            x, y, w, h = obj.bbox
            color = self.colors.get(obj.class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label with ID
            label = f"{obj.class_name.capitalize()} #{obj.id}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Draw label background
            cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def draw_statistics(self, frame):
        """Draw statistics overlay on frame"""
        # Create semi-transparent overlay for stats
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        # Stats background
        cv2.rectangle(overlay, (10, 10), (250, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Draw statistics
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
        
        # Draw total
        cv2.putText(frame, f"Total: {self.counters['total']}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame
    
    def update_counters(self, tracked_objects):
        """Update vehicle counters based on tracked objects"""
        for obj_id, obj in tracked_objects.items():
            if obj_id not in self.tracker.counted_ids:
                self.counters[obj.class_name] += 1
                self.counters['total'] += 1
                self.tracker.counted_ids.add(obj_id)
    
    def process_video(self, video_path: str, display: bool = True):
        """
        Process video file for vehicle detection and tracking
        
        Args:
            video_path: Path to input video file
            display: Whether to display live video feed
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}, Total frames: {total_frames}")
        print("Press 'q' to quit, 'p' to pause/resume")
        
        frame_num = 0
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_num += 1
                 # Get frame dimensions
                height, width, _ = frame.shape   # <-- yahan add kiya hai
                
                # Detect vehicles
                if self.use_yolov8:
                    detections = self.detect_vehicles_yolov8(frame)
                else:
                    detections = self.detect_vehicles_yolov4(frame)
                
                # Update tracker
                tracked_objects = self.tracker.update(detections, frame_num)
                
                # Update counters
                self.update_counters(tracked_objects)
                
                # Draw visualizations
                self.draw_detections(frame, tracked_objects)
                frame = self.draw_statistics(frame)
                
                # Add frame info
                cv2.putText(frame, f"Frame: {frame_num}/{total_frames}", 
                           (width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 255), 1)
                
                # Display frame
                if display:
                    cv2.imshow('Traffic Detection System', frame)
                
                # Print progress
                if frame_num % 30 == 0:
                    progress = (frame_num / total_frames) * 100
                    print(f"Progress: {progress:.1f}% - Total vehicles: {self.counters['total']}")
            
            # Handle keyboard input
            if display:
                key = cv2.waitKey(1 if not paused else 0) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    if paused:
                        print("Paused. Press 'p' to resume.")
                    else:
                        print("Resumed.")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        self.print_final_stats()
    
    def print_final_stats(self):
        """Print final counting statistics"""
        print("\n" + "="*50)
        print("FINAL VEHICLE COUNT STATISTICS")
        print("="*50)
        for vehicle_type in ['car', 'motorcycle', 'bus', 'truck']:
            count = self.counters[vehicle_type]
            percentage = (count / max(self.counters['total'], 1)) * 100
            print(f"{vehicle_type.capitalize():15} : {count:5d} ({percentage:5.1f}%)")
        print("-"*50)
        print(f"{'TOTAL':15} : {self.counters['total']:5d}")
        print("="*50)


def main():
    """Main function to run the traffic detection system"""
    
    # Configuration
    VIDEO_PATH = "sa.mp4"  # Change this to your video file path
    CONFIDENCE_THRESHOLD = 0.5
    
    print("="*50)
    print("REAL-TIME TRAFFIC DETECTION & TRACKING SYSTEM")
    print("="*50)
    
    # Create detection system
    detector = TrafficDetectionSystem(confidence_threshold=CONFIDENCE_THRESHOLD)
    
    # Process video
    try:
        detector.process_video(VIDEO_PATH, display=True)
    except Exception as e:
        print(f"Error processing video: {e}")
        print("\nPlease ensure:")
        print("1. The video file 'traffic.mp4' exists in the current directory")
        print("2. You have installed required packages:")
        print("   pip install opencv-python numpy ultralytics")
        print("3. Or for YOLOv4, download the required files:")
        print("   - yolov4-tiny.weights")
        print("   - yolov4-tiny.cfg")
        print("   - coco.names")


if __name__ == "__main__":
    main()