import cv2
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import threading
import time
import os
import random


@dataclass
class TrackedObject:
    """Data class for tracked vehicles"""
    id: int
    class_name: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    center: Tuple[int, int]
    last_seen: int
    confidence: float
    direction: str  # north, south, east, west


class VehicleTracker:
    """Enhanced centroid-based vehicle tracker with direction detection"""
    
    def __init__(self, max_disappeared: int = 40, max_distance: int = 1000):
        self.next_id = 0
        self.objects = {}
        self.disappeared = defaultdict(int)
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.counted_ids = set()
        
        # Direction zones (x1, y1, x2, y2)
        self.direction_zones = {
            'north': (300, 50, 500, 200),
            'south': (300, 350, 500, 500),
            'east': (500, 200, 650, 350),
            'west': (150, 200, 300, 350)
        }
        
    def calculate_centroid(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)
    
    def get_direction_from_position(self, center: Tuple[int, int]) -> str:
        """Determine which direction/zone a vehicle is in"""
        x, y = center
        for direction, (x1, y1, x2, y2) in self.direction_zones.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return direction
        return 'center'  # Default for vehicles in intersection center
    
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
            direction = self.get_direction_from_position(centroid)
            input_centroids.append(centroid)
            input_data.append((bbox, class_name, confidence, centroid, direction))
        
        if len(self.objects) == 0:
            for data in input_data:
                bbox, class_name, confidence, centroid, direction = data
                self.register(bbox, class_name, confidence, centroid, direction, frame_num)
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
                    bbox, class_name, confidence, centroid, direction = input_data[col]
                    self.objects[obj_id].bbox = bbox
                    self.objects[obj_id].center = centroid
                    self.objects[obj_id].last_seen = frame_num
                    self.objects[obj_id].confidence = confidence
                    self.objects[obj_id].direction = direction
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
                bbox, class_name, confidence, centroid, direction = input_data[col]
                self.register(bbox, class_name, confidence, centroid, direction, frame_num)
        
        return self.objects
    
    def register(self, bbox, class_name, confidence, centroid, direction, frame_num):
        self.objects[self.next_id] = TrackedObject(
            id=self.next_id,
            class_name=class_name,
            bbox=bbox,
            center=centroid,
            last_seen=frame_num,
            confidence=confidence,
            direction=direction
        )
        self.disappeared[self.next_id] = 0
        self.next_id += 1
    
    def deregister(self, obj_id):
        del self.objects[obj_id]
        del self.disappeared[obj_id]
    
    def get_direction_counts(self) -> Dict[str, int]:
        """Get vehicle count per direction"""
        counts = {'north': 0, 'south': 0, 'east': 0, 'west': 0, 'center': 0}
        for obj in self.objects.values():
            if obj.direction in counts:
                counts[obj.direction] += 1
        return counts


class TrafficSignalManager:
    """Enhanced traffic signal manager with dynamic timing and sequential control"""
    
    def __init__(self):
        self.signals = {
            'north': {'state': 'red', 'countdown': 45},
            'south': {'state': 'green', 'countdown': 30},
            'east': {'state': 'red', 'countdown': 60},
            'west': {'state': 'red', 'countdown': 75}
        }
        self.override_mode = False
        self.current_active_direction = 'south'
        self.signal_sequence = ['north', 'east', 'south', 'west']
        self.sequence_index = 2  # Starting with south as green
        self.yellow_time = 5
        self.min_green_time = 10
        self.max_green_time = 45
        self.running = False
        self.signal_thread = None
        self.vehicle_counts = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
        
    def calculate_dynamic_timing(self, vehicle_count: int) -> int:
        """Calculate green signal time based on vehicle count"""
        if vehicle_count <= 5:
            return self.min_green_time  # 10 seconds for low traffic
        elif vehicle_count <= 15:
            return 20  # 20 seconds for medium traffic
        elif vehicle_count <= 25:
            return 30  # 30 seconds for high traffic
        else:
            return self.max_green_time  # 45 seconds for very high traffic
    
    def update_vehicle_counts(self, direction_counts: Dict[str, int]):
        """Update vehicle counts for dynamic timing"""
        self.vehicle_counts.update(direction_counts)
    
    def start_signal_control(self):
        """Start the automatic signal control thread"""
        if not self.running:
            self.running = True
            self.signal_thread = threading.Thread(target=self._signal_loop, daemon=True)
            self.signal_thread.start()
    
    def stop_signal_control(self):
        """Stop the automatic signal control"""
        self.running = False
    
    def _signal_loop(self):
        """Main signal control loop"""
        while self.running:
            if not self.override_mode:
                # Update countdown for all signals
                for direction in self.signals:
                    if self.signals[direction]['countdown'] > 0:
                        self.signals[direction]['countdown'] -= 1
                
                # Check if current green signal needs to change
                current_signal = self.signals[self.current_active_direction]
                if current_signal['countdown'] <= 0:
                    if current_signal['state'] == 'green':
                        # Change to yellow
                        self.signals[self.current_active_direction]['state'] = 'yellow'
                        self.signals[self.current_active_direction]['countdown'] = self.yellow_time
                    elif current_signal['state'] == 'yellow':
                        # Change to red and move to next direction
                        self.signals[self.current_active_direction]['state'] = 'red'
                        # Calculate red time based on total cycle time for other directions
                        total_cycle_time = sum(
                            self.calculate_dynamic_timing(self.vehicle_counts.get(d, 0)) + self.yellow_time
                            for d in self.signal_sequence if d != self.current_active_direction
                        )
                        self.signals[self.current_active_direction]['countdown'] = total_cycle_time
                        
                        # Move to next direction
                        self.sequence_index = (self.sequence_index + 1) % len(self.signal_sequence)
                        self.current_active_direction = self.signal_sequence[self.sequence_index]
                        
                        # Set new green signal
                        green_time = self.calculate_dynamic_timing(
                            self.vehicle_counts.get(self.current_active_direction, 0)
                        )
                        self.signals[self.current_active_direction]['state'] = 'green'
                        self.signals[self.current_active_direction]['countdown'] = green_time
                        
                        # Update other red signals countdown
                        for direction in self.signal_sequence:
                            if direction != self.current_active_direction:
                                remaining_time = 0
                                # Calculate remaining time in cycle
                                for i in range(1, len(self.signal_sequence)):
                                    next_dir_idx = (self.sequence_index + i) % len(self.signal_sequence)
                                    next_dir = self.signal_sequence[next_dir_idx]
                                    if next_dir == direction:
                                        break
                                    remaining_time += (
                                        self.calculate_dynamic_timing(self.vehicle_counts.get(next_dir, 0)) + 
                                        self.yellow_time
                                    )
                                remaining_time += green_time + self.yellow_time
                                self.signals[direction]['countdown'] = remaining_time
            
            time.sleep(1)
    
    def manual_override(self, direction: str, state: str):
        """Manual override for signal control"""
        if direction in self.signals and state in ['red', 'yellow', 'green']:
            self.override_mode = True
            
            if state == 'green':
                # Set all other signals to red
                for d in self.signals:
                    if d != direction:
                        self.signals[d]['state'] = 'red'
                        self.signals[d]['countdown'] = 60
                
                # Set target signal to green
                green_time = self.calculate_dynamic_timing(self.vehicle_counts.get(direction, 0))
                self.signals[direction]['state'] = 'green'
                self.signals[direction]['countdown'] = green_time
                self.current_active_direction = direction
                
                # Update sequence index
                if direction in self.signal_sequence:
                    self.sequence_index = self.signal_sequence.index(direction)
            else:
                self.signals[direction]['state'] = state
                countdown = self.yellow_time if state == 'yellow' else 60
                self.signals[direction]['countdown'] = countdown
    
    def exit_override(self):
        """Exit manual override and return to automatic control"""
        self.override_mode = False
    
    def get_signals(self):
        """Get current signal states"""
        return self.signals.copy()
    
    def get_timing_recommendations(self):
        """Get timing recommendations based on current vehicle counts"""
        recommendations = {}
        for direction, count in self.vehicle_counts.items():
            recommendations[direction] = self.calculate_dynamic_timing(count)
        return recommendations


class TrafficDetectionSystem:
    """Enhanced traffic detection system with dynamic timing and signal management"""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.tracker = VehicleTracker()
        self.signal_manager = TrafficSignalManager()

        # ROI region (x1, y1, x2, y2)
        self.roi = (100, 50, 700, 550)  

        # Simplified vehicle classes for demo purposes
        self.vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

        self.counters = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0, 'total': 0}
        self.direction_counters = {'north': 0, 'south': 0, 'east': 0, 'west': 0}

        self.colors = {'car': (0, 255, 0), 'motorcycle': (255, 0, 0),
                       'bus': (0, 255, 255), 'truck': (0, 0, 255)}

        # Playback controls
        self.is_playing = True
        self.frame_skip = 1
        self.current_frame = 0
        
        # Performance optimization
        self.detection_interval = 3
        self.last_detections = []
        self.frame_count = 0

        # Start signal management
        self.signal_manager.start_signal_control()

        # Load YOLO model - try different options
        self.model = None
        self.use_yolov8 = False
        self.load_detection_model()

    def load_detection_model(self):
        """Try to load YOLO model, fallback to built-in OpenCV DNN if needed"""
        try:
            # Try YOLOv8 first
            from ultralytics import YOLO
            self.model = YOLO('yolov8n.pt')
            self.use_yolov8 = True
            self.model.overrides['verbose'] = False
            print("YOLOv8 loaded successfully")
        except (ImportError, Exception) as e:
            print(f"YOLOv8 not available: {e}")
            try:
                # Try YOLOv4 with OpenCV DNN
                self.load_yolov4()
            except Exception as e2:
                print(f"YOLOv4 not available: {e2}")
                # Use mock detection for demo
                self.use_mock_detection = True
                print("Using mock detection for demo purposes")

    def load_yolov4(self):
        """Load YOLOv4 model files"""
        # Check if YOLO files exist
        weights_path = "yolov4-tiny.weights"
        config_path = "yolov4-tiny.cfg"
        names_path = "coco.names"
        
        if not all(os.path.exists(f) for f in [weights_path, config_path, names_path]):
            raise FileNotFoundError("YOLO model files not found")
            
        self.net = cv2.dnn.readNet(weights_path, config_path)
        with open(names_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.output_layers = [self.net.getLayerNames()[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Enable GPU if available
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def reset_counters(self):
        """Reset all counters and tracking data"""
        self.counters = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0, 'total': 0}
        self.direction_counters = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
        self.tracker = VehicleTracker()
        self.current_frame = 0
        self.frame_count = 0
        self.last_detections = []

    def play_pause(self):
        """Toggle play/pause state"""
        self.is_playing = not self.is_playing

    def set_frame_skip(self, skip_value: int):
        """Set frame skip value"""
        self.frame_skip = max(1, skip_value)

    def draw_roi_and_zones(self, frame):
        """Draw ROI region and direction zones"""
        # Main ROI
        x1, y1, x2, y2 = self.roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Direction zones
        zone_colors = {
            'north': (255, 0, 0),    # Red
            'south': (0, 255, 255),  # Cyan
            'east': (255, 255, 0),   # Yellow
            'west': (255, 0, 255)    # Magenta
        }
        
        for direction, (zx1, zy1, zx2, zy2) in self.tracker.direction_zones.items():
            color = zone_colors.get(direction, (255, 255, 255))
            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), color, 1)
            cv2.putText(frame, direction.upper(), (zx1 + 5, zy1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame

    def is_inside_roi(self, centroid, roi=None):
        if roi is None:
            roi = self.roi
        x, y = centroid
        x1, y1, x2, y2 = roi
        return x1 <= x <= x2 and y1 <= y <= y2

    def detect_vehicles_mock(self, frame):
        """Enhanced mock detection with more realistic vehicle distribution"""
        detections = []
        frame_cycle = (self.current_frame % 300)  # Longer cycle for more variation
        
        # Simulate varying traffic patterns
        time_factor = (frame_cycle % 100) / 100.0
        
        # Different traffic densities for each direction based on time
        traffic_patterns = {
            'north': {
                'base_vehicles': 3 + int(5 * np.sin(time_factor * np.pi)),
                'positions': [(350, 80), (380, 120), (420, 100), (360, 150), (400, 80)]
            },
            'south': {
                'base_vehicles': 4 + int(6 * np.cos(time_factor * np.pi * 0.7)),
                'positions': [(380, 400), (450, 450), (350, 420), (410, 480), (370, 470)]
            },
            'east': {
                'base_vehicles': 2 + int(4 * np.sin(time_factor * np.pi * 1.3)),
                'positions': [(550, 280), (600, 320), (580, 300), (620, 280)]
            },
            'west': {
                'base_vehicles': 3 + int(5 * np.cos(time_factor * np.pi * 0.9)),
                'positions': [(200, 250), (250, 300), (180, 280), (220, 320), (160, 270)]
            }
        }
        
        vehicle_types = ['car', 'car', 'car', 'motorcycle', 'bus', 'truck']  # Car is more common
        
        for direction, pattern in traffic_patterns.items():
            num_vehicles = min(pattern['base_vehicles'], len(pattern['positions']))
            for i in range(num_vehicles):
                if i < len(pattern['positions']):
                    base_x, base_y = pattern['positions'][i]
                    
                    # Add movement and randomness
                    movement_offset = int(10 * np.sin((frame_cycle + i * 20) * 0.05))
                    x = base_x + movement_offset + random.randint(-15, 15)
                    y = base_y + random.randint(-10, 10)
                    
                    # Random vehicle type
                    vehicle_type = random.choice(vehicle_types)
                    
                    # Vehicle size based on type
                    sizes = {'car': (50, 30), 'motorcycle': (30, 20), 'bus': (70, 40), 'truck': (80, 45)}
                    w, h = sizes[vehicle_type]
                    
                    if self.is_inside_roi((x, y)):
                        detections.append(((x-w//2, y-h//2, w, h), vehicle_type, 
                                         0.7 + 0.3 * random.random()))
        
        return detections

    def detect_vehicles_yolov8(self, frame):
        """YOLOv8 detection"""
        detections = []
        height, width = frame.shape[:2]
        
        # Resize for performance
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
        """YOLOv4 detection with OpenCV DNN"""
        if self.net is None:
            return self.detect_vehicles_mock(frame)
        
        height, width, _ = frame.shape
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
            label = f"{obj.class_name.capitalize()} #{obj.id} ({obj.direction})"
            cv2.putText(frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def draw_statistics(self, frame):
        y_offset = 30
        cv2.putText(frame, "Vehicle Count", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        # Vehicle type counts
        for vehicle_type in ['car', 'motorcycle', 'bus', 'truck']:
            count = self.counters[vehicle_type]
            text = f"{vehicle_type.capitalize()}: {count}"
            cv2.putText(frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors[vehicle_type], 1)
            y_offset += 20
            
        cv2.putText(frame, f"Total: {self.counters['total']}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        y_offset += 30
        
        # Direction counts (live)
        cv2.putText(frame, "Live Direction Count", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        direction_colors = {
            'north': (255, 0, 0), 'south': (0, 255, 255),
            'east': (255, 255, 0), 'west': (255, 0, 255)
        }
        
        direction_counts = self.tracker.get_direction_counts()
        for direction in ['north', 'south', 'east', 'west']:
            count = direction_counts.get(direction, 0)
            text = f"{direction.capitalize()}: {count}"
            color = direction_colors.get(direction, (255, 255, 255))
            cv2.putText(frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_offset += 18
        
        # Show signal status
        y_offset += 10
        cv2.putText(frame, "Signal Status", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        signals = self.signal_manager.get_signals()
        for direction in ['north', 'south', 'east', 'west']:
            signal = signals[direction]
            state_colors = {'red': (0, 0, 255), 'yellow': (0, 255, 255), 'green': (0, 255, 0)}
            color = state_colors.get(signal['state'], (255, 255, 255))
            text = f"{direction}: {signal['state'].upper()} ({signal['countdown']}s)"
            cv2.putText(frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            y_offset += 16
        
        # Show playback status
        cv2.putText(frame, f"Status: {'PLAYING' if self.is_playing else 'PAUSED'}", 
                   (10, frame.shape[0] - 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   (0, 255, 0) if self.is_playing else (0, 0, 255), 2)
        cv2.putText(frame, f"Frame: {self.current_frame}", (10, frame.shape[0] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Override: {'ON' if self.signal_manager.override_mode else 'AUTO'}", 
                   (10, frame.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0) if self.signal_manager.override_mode else (0, 255, 0), 1)
        
        return frame

    def update_counters(self, tracked_objects):
        for obj_id, obj in tracked_objects.items():
            if obj_id in self.tracker.counted_ids:
                continue
            if self.is_inside_roi(obj.center):
                self.counters[obj.class_name] += 1
                self.counters['total'] += 1
                if obj.direction in self.direction_counters:
                    self.direction_counters[obj.direction] += 1
                self.tracker.counted_ids.add(obj_id)

    def generate_frames(self, video_path: str):
        """Generate frames from video with proper error handling"""
        # Check if video file exists
        if not os.path.exists(video_path):
            print(f"Error: Video file '{video_path}' not found!")
            # Generate a demo frame instead
            yield from self.generate_demo_frames()
            return
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            # Generate a demo frame instead
            yield from self.generate_demo_frames()
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30
        
        print(f"Video loaded: {total_frames} frames at {fps} FPS")
        
        while True:
            if not self.is_playing:
                time.sleep(0.1)
                continue
                
            ret, frame = cap.read()
            if not ret:
                print("End of video reached, looping...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.reset_counters()
                continue
            
            self.current_frame += 1
            self.frame_count += 1
            
            # Skip frames for performance
            if (self.current_frame - 1) % self.frame_skip != 0:
                continue
            
            try:
                frame = self.draw_roi_and_zones(frame)
                
                # Run detection
                if self.frame_count % self.detection_interval == 0:
                    if self.use_yolov8 and self.model:
                        self.last_detections = self.detect_vehicles_yolov8(frame)
                    elif hasattr(self, 'net') and self.net is not None:
                        self.last_detections = self.detect_vehicles_yolov4(frame)
                    else:
                        self.last_detections = self.detect_vehicles_mock(frame)
                
                # Update tracking
                tracked_objects = self.tracker.update(self.last_detections, self.current_frame)
                self.update_counters(tracked_objects)
                
                # Update signal manager with current vehicle counts
                direction_counts = self.tracker.get_direction_counts()
                self.signal_manager.update_vehicle_counts(direction_counts)
                
                # Draw everything
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
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue
        
        cap.release()

    def generate_demo_frames(self):
        """Generate demo frames when video is not available"""
        print("Generating demo frames...")
        while True:
            if not self.is_playing:
                time.sleep(0.1)
                continue
            
            # Create a demo frame
            frame = np.zeros((600, 800, 3), dtype=np.uint8)
            frame[:] = (40, 40, 40)  # Dark gray background
            
            # Draw intersection layout
            cv2.rectangle(frame, (350, 250), (450, 350), (100, 100, 100), -1)  # Center intersection
            cv2.rectangle(frame, (350, 0), (450, 250), (60, 60, 60), -1)       # North road
            cv2.rectangle(frame, (350, 350), (450, 600), (60, 60, 60), -1)     # South road
            cv2.rectangle(frame, (0, 250), (350, 350), (60, 60, 60), -1)       # West road
            cv2.rectangle(frame, (450, 250), (800, 350), (60, 60, 60), -1)     # East road
            
            # Draw demo content
            cv2.putText(frame, "ENHANCED DEMO MODE", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, "Simulated Traffic Intersection", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            self.current_frame += 1
            self.frame_count += 1
            
            frame = self.draw_roi_and_zones(frame)
            
            # Mock detections for demo
            if self.frame_count % self.detection_interval == 0:
                self.last_detections = self.detect_vehicles_mock(frame)
            
            tracked_objects = self.tracker.update(self.last_detections, self.current_frame)
            self.update_counters(tracked_objects)
            
            # Update signal manager with current vehicle counts
            direction_counts = self.tracker.get_direction_counts()
            self.signal_manager.update_vehicle_counts(direction_counts)
            
            self.draw_detections(frame, tracked_objects)
            frame = self.draw_statistics(frame)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            time.sleep(0.1)  # 10 FPS for demo


# Flask App Setup
app = Flask(__name__)
CORS(app)
detector = TrafficDetectionSystem(confidence_threshold=0.5)

@app.route('/')
def index():
    return """
    <h1>Enhanced Traffic Detection System</h1>
    <p>Backend is running. Available endpoints:</p>
    <ul>
        <li><a href="/video_feed">/video_feed</a> - Video stream with direction tracking</li>
        <li><a href="/get_counts">/get_counts</a> - Get current counts and direction data</li>
        <li><a href="/get_direction_counts">/get_direction_counts</a> - Get vehicle counts per direction</li>
        <li><a href="/get_signal_status">/get_signal_status</a> - Get current signal status</li>
        <li>/play_pause (POST) - Toggle play/pause</li>
        <li>/reset_counters (POST) - Reset all counters</li>
        <li>/set_frame_skip (POST) - Set frame skip value</li>
        <li>/manual_override (POST) - Manual signal override</li>
        <li>/exit_override (POST) - Exit manual override mode</li>
    </ul>
    """

@app.route('/video_feed')
def video_feed():
    VIDEO_PATH = "traffic.mp4"  # Make sure this file exists
    return Response(detector.generate_frames(VIDEO_PATH),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_counts')
def get_counts():
    direction_counts = detector.tracker.get_direction_counts()
    signals = detector.signal_manager.get_signals()
    timing_recommendations = detector.signal_manager.get_timing_recommendations()
    
    return jsonify({
        'counters': detector.counters,
        'direction_counters': direction_counts,
        'live_direction_counts': direction_counts,
        'is_playing': detector.is_playing,
        'frame_skip': detector.frame_skip,
        'current_frame': detector.current_frame,
        'signals': signals,
        'timing_recommendations': timing_recommendations,
        'override_mode': detector.signal_manager.override_mode
    })

@app.route('/get_direction_counts')
def get_direction_counts():
    """Get real-time direction counts for frontend"""
    direction_counts = detector.tracker.get_direction_counts()
    timing_recommendations = detector.signal_manager.get_timing_recommendations()
    
    return jsonify({
        'north': direction_counts.get('north', 0),
        'south': direction_counts.get('south', 0),
        'east': direction_counts.get('east', 0),
        'west': direction_counts.get('west', 0),
        'timing_recommendations': timing_recommendations
    })

@app.route('/get_signal_status')
def get_signal_status():
    """Get current signal status and timing"""
    signals = detector.signal_manager.get_signals()
    direction_counts = detector.tracker.get_direction_counts()
    timing_recommendations = detector.signal_manager.get_timing_recommendations()
    
    return jsonify({
        'signals': signals,
        'direction_counts': direction_counts,
        'timing_recommendations': timing_recommendations,
        'override_mode': detector.signal_manager.override_mode,
        'current_active_direction': detector.signal_manager.current_active_direction,
        'sequence': detector.signal_manager.signal_sequence
    })

@app.route('/manual_override', methods=['POST'])
def manual_override():
    """Manual signal override"""
    try:
        data = request.json
        direction = data.get('direction')
        state = data.get('state')
        
        if direction not in ['north', 'south', 'east', 'west']:
            return jsonify({'error': 'Invalid direction'}), 400
            
        if state not in ['red', 'yellow', 'green']:
            return jsonify({'error': 'Invalid state'}), 400
            
        detector.signal_manager.manual_override(direction, state)
        
        return jsonify({
            'message': f'Signal override applied: {direction} -> {state}',
            'signals': detector.signal_manager.get_signals(),
            'override_mode': detector.signal_manager.override_mode
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/exit_override', methods=['POST'])
def exit_override():
    """Exit manual override mode"""
    detector.signal_manager.exit_override()
    return jsonify({
        'message': 'Exited manual override mode, returning to automatic control',
        'override_mode': detector.signal_manager.override_mode,
        'signals': detector.signal_manager.get_signals()
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
        'counters': detector.counters,
        'direction_counters': detector.direction_counters
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
    print("Starting Enhanced Traffic Detection System...")
    print("Features:")
    print("- Direction-based vehicle tracking")
    print("- Dynamic signal timing based on vehicle count")
    print("- Sequential traffic signal control")
    print("- Manual override capabilities")
    print("- Real-time vehicle count per direction")
    print("\nSignal Timing Logic:")
    print("- Low traffic (<=5 vehicles): 10 seconds green")
    print("- Medium traffic (6-15 vehicles): 20 seconds green")
    print("- High traffic (16-25 vehicles): 30 seconds green") 
    print("- Very high traffic (>25 vehicles): 45 seconds green")
    print("- Yellow signal duration: 5 seconds")
    print("\nMake sure 'traffic.mp4' is in the same directory as this script")
    print("Navigate to http://127.0.0.1:5000 in your browser.")
    print("\nChecking for required files...")
    
    # Check for video file
    if os.path.exists("traffic.mp4"):
        print("✓ traffic.mp4 found")
    else:
        print("✗ traffic.mp4 not found - will use enhanced demo mode")
    
    print("\nStarting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)