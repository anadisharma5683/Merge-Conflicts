const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const WebSocket = require('ws');
const path = require('path');
const fs = require('fs');
const { promisify } = require('util');

const app = express();
const port = 3000;

// WebSocket server for real-time updates
const wss = new WebSocket.Server({ port: 8080 });

// Storage configuration
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + '-' + file.originalname);
  }
});

const upload = multer({ 
  storage: storage,
  limits: {
    fileSize: 50 * 1024 * 1024, // 50MB limit
  }
});

// Ensure directories exist
['uploads', 'temp', 'models'].forEach(dir => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
});

app.use(express.static('public'));
app.use(express.json());

class VehicleCounter {
  constructor() {
    this.classCounts = {};
    this.crossedIds = new Set();
    this.lineY = 430; // Red line position (adjustable)
    this.vehicleClasses = [1, 2, 3, 5, 6, 7]; // car, motorcycle, bus, truck, etc.
    this.classNames = {
      0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 
      5: 'bus', 6: 'train', 7: 'truck', 8: 'boat'
    };
    this.processingQueue = [];
    this.isProcessing = false;
  }

  // Enhanced Python script for YOLO processing without OpenCV Node.js
  async runYOLOProcessing(inputPath, isVideo = false) {
    return new Promise((resolve, reject) => {
      const tempDir = path.join(__dirname, 'temp');
      const outputPath = path.join(tempDir, `output_${Date.now()}.json`);
      
      const pythonScript = `
import cv2
import json
import sys
import os
from ultralytics import YOLO
import numpy as np
from pathlib import Path

def process_media(input_path, output_path, is_video=False, line_y=${this.lineY}):
    try:
        # Load YOLO model (lightweight version)
        model_path = 'yolo11s.pt'  # Change to yolo11n.pt for even faster processing
        if not os.path.exists(model_path):
            print(f"Downloading {model_path}...")
            model = YOLO(model_path)
        else:
            model = YOLO(model_path)
        
        results_data = {
            "success": True,
            "detections": [],
            "frame_count": 0,
            "processing_time": 0,
            "model_info": {
                "model_size": "${model_path}",
                "classes": list(model.names.values())
            }
        }
        
        if is_video:
            cap = cv2.VideoCapture(input_path)
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                
                # Process every 5th frame for performance
                if frame_count % 5 != 0:
                    continue
                
                # Run YOLO tracking
                results = model.track(frame, persist=True, classes=[1,2,3,5,6,7], verbose=False)
                
                frame_detections = []
                if results[0].boxes is not None and results[0].boxes.data is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    if results[0].boxes.id is not None:
                        track_ids = results[0].boxes.id.int().cpu().tolist()
                    else:
                        track_ids = list(range(len(boxes)))
                    
                    class_indices = results[0].boxes.cls.int().cpu().tolist()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    
                    for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
                        x1, y1, x2, y2 = map(int, box)
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        
                        detection = {
                            "frame": frame_count,
                            "bbox": [x1, y1, x2, y2],
                            "center": [cx, cy],
                            "track_id": track_id,
                            "class_id": class_idx,
                            "confidence": float(conf),
                            "class_name": model.names[class_idx],
                            "crossed_line": cy > line_y
                        }
                        frame_detections.append(detection)
                
                if frame_detections:
                    results_data["detections"].extend(frame_detections)
            
            cap.release()
            results_data["frame_count"] = frame_count
            
        else:
            # Single image processing
            frame = cv2.imread(input_path)
            if frame is None:
                results_data["success"] = False
                results_data["error"] = "Could not read image"
            else:
                results = model.track(frame, persist=True, classes=[1,2,3,5,6,7], verbose=False)
                
                if results[0].boxes is not None and results[0].boxes.data is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    if results[0].boxes.id is not None:
                        track_ids = results[0].boxes.id.int().cpu().tolist()
                    else:
                        track_ids = list(range(len(boxes)))
                    
                    class_indices = results[0].boxes.cls.int().cpu().tolist()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    
                    for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
                        x1, y1, x2, y2 = map(int, box)
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        
                        detection = {
                            "frame": 1,
                            "bbox": [x1, y1, x2, y2],
                            "center": [cx, cy],
                            "track_id": track_id,
                            "class_id": class_idx,
                            "confidence": float(conf),
                            "class_name": model.names[class_idx],
                            "crossed_line": cy > line_y
                        }
                        results_data["detections"].append(detection)
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
            
        print(f"SUCCESS: {output_path}")
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "detections": []
        }
        with open(output_path, 'w') as f:
            json.dump(error_result, f, indent=2)
        print(f"ERROR: {str(e)}")

if __name__ == "__main__":
    input_path = "${inputPath.replace(/\\/g, '/')}"
    output_path = "${outputPath.replace(/\\/g, '/')}"
    is_video = ${isVideo}
    
    process_media(input_path, output_path, is_video)
`;

      const python = spawn('python', ['-c', pythonScript], {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: process.cwd()
      });
      
      let output = '';
      let error = '';

      python.stdout.on('data', (data) => {
        output += data.toString();
      });

      python.stderr.on('data', (data) => {
        error += data.toString();
      });

      python.on('close', (code) => {
        if (fs.existsSync(outputPath)) {
          try {
            const result = JSON.parse(fs.readFileSync(outputPath, 'utf8'));
            fs.unlinkSync(outputPath); // Clean up temp file
            resolve(result);
          } catch (e) {
            reject(new Error(`Failed to parse results: ${e.message}`));
          }
        } else {
          reject(new Error(`Processing failed: ${error || 'Unknown error'}`));
        }
      });
    });
  }

  // Process detections and count vehicles
  processDetections(detections) {
    const results = {
      newCrossings: [],
      totalCounts: { ...this.classCounts },
      currentDetections: detections.length,
      processedFrames: 0
    };

    const frameGroups = {};
    detections.forEach(detection => {
      if (!frameGroups[detection.frame]) {
        frameGroups[detection.frame] = [];
      }
      frameGroups[detection.frame].push(detection);
    });

    results.processedFrames = Object.keys(frameGroups).length;

    detections.forEach(detection => {
      const { center, track_id, class_name, confidence, crossed_line } = detection;

      // Check if vehicle crossed the counting line
      if (crossed_line && !this.crossedIds.has(track_id)) {
        this.crossedIds.add(track_id);
        
        if (!this.classCounts[class_name]) {
          this.classCounts[class_name] = 0;
        }
        this.classCounts[class_name]++;
        
        results.newCrossings.push({
          track_id,
          class_name,
          position: center,
          confidence: confidence.toFixed(2),
          frame: detection.frame || 1
        });
        
        results.totalCounts = { ...this.classCounts };
      }
    });

    return results;
  }

  // Reset counters
  reset() {
    this.classCounts = {};
    this.crossedIds.clear();
  }

  // Get current statistics
  getStats() {
    const totalVehicles = Object.values(this.classCounts).reduce((sum, count) => sum + count, 0);
    return {
      classCounts: this.classCounts,
      totalVehicles,
      uniqueVehiclesSeen: this.crossedIds.size,
      countingLineY: this.lineY
    };
  }
}

const vehicleCounter = new VehicleCounter();

// Broadcast updates to connected clients
function broadcastUpdate(data) {
  const message = JSON.stringify({
    timestamp: new Date().toISOString(),
    ...data
  });
  
  wss.clients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(message);
    }
  });
}

// Routes
app.post('/upload-video', upload.single('video'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No video file uploaded' });
  }

  try {
    const videoPath = req.file.path;
    
    // Start processing video asynchronously
    vehicleCounter.runYOLOProcessing(videoPath, true)
      .then(yoloResults => {
        if (!yoloResults.success) {
          throw new Error(yoloResults.error || 'Processing failed');
        }

        const results = vehicleCounter.processDetections(yoloResults.detections);
        
        broadcastUpdate({
          type: 'video_processed',
          filename: req.file.filename,
          ...results,
          stats: vehicleCounter.getStats(),
          processingInfo: {
            totalFrames: yoloResults.frame_count,
            detectionCount: yoloResults.detections.length,
            modelInfo: yoloResults.model_info
          }
        });

        // Clean up video file
        fs.unlinkSync(videoPath);
      })
      .catch(error => {
        broadcastUpdate({
          type: 'processing_error',
          error: error.message,
          filename: req.file.filename
        });
      });
    
    res.json({ 
      message: 'Video uploaded and processing started', 
      videoId: req.file.filename,
      status: 'processing'
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/upload-image', upload.single('image'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No image file uploaded' });
  }

  try {
    const imagePath = req.file.path;
    
    const yoloResults = await vehicleCounter.runYOLOProcessing(imagePath, false);
    
    if (!yoloResults.success) {
      throw new Error(yoloResults.error || 'Processing failed');
    }

    const results = vehicleCounter.processDetections(yoloResults.detections);
    
    broadcastUpdate({
      type: 'image_processed',
      filename: req.file.filename,
      ...results,
      stats: vehicleCounter.getStats(),
      processingInfo: {
        detectionCount: yoloResults.detections.length,
        modelInfo: yoloResults.model_info
      }
    });

    // Clean up image file
    fs.unlinkSync(imagePath);
    
    res.json(results);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/stats', (req, res) => {
  res.json(vehicleCounter.getStats());
});

app.post('/reset', (req, res) => {
  vehicleCounter.reset();
  broadcastUpdate({
    type: 'counters_reset',
    stats: vehicleCounter.getStats()
  });
  res.json({ message: 'Counters reset successfully' });
});

app.post('/set-counting-line', (req, res) => {
  const { lineY } = req.body;
  if (typeof lineY === 'number') {
    vehicleCounter.lineY = lineY;
    res.json({ 
      message: 'Counting line updated', 
      lineY,
      stats: vehicleCounter.getStats()
    });
  } else {
    res.status(400).json({ error: 'Invalid lineY value' });
  }
});

// System health check
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    connectedClients: wss.clients.size,
    stats: vehicleCounter.getStats()
  });
});

// WebSocket connection handling
wss.on('connection', (ws) => {
  console.log('✅ Client connected to WebSocket');
  
  ws.send(JSON.stringify({
    type: 'connected',
    message: 'Connected to vehicle counter',
    stats: vehicleCounter.getStats()
  }));
  
  ws.on('close', () => {
    console.log('❌ Client disconnected from WebSocket');
  });

  ws.on('error', (error) => {
    console.log('🚨 WebSocket error:', error.message);
  });
});

// Enhanced frontend
app.get('/', (req, res) => {
  res.send(`
    <!DOCTYPE html>
    <html>
    <head>
        <title>🚗 Real-Time Vehicle Counter</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container { 
                max-width: 1400px; 
                margin: 0 auto; 
                background: rgba(255,255,255,0.95);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            }
            .header { 
                text-align: center; 
                margin-bottom: 30px;
                background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            .stats-panel { 
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                padding: 25px; 
                margin: 20px 0; 
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
            }
            .upload-area { 
                border: 3px dashed #667eea; 
                padding: 30px; 
                text-align: center; 
                margin: 20px 0;
                border-radius: 15px;
                background: rgba(102, 126, 234, 0.05);
                transition: all 0.3s ease;
            }
            .upload-area:hover {
                border-color: #4ECDC4;
                background: rgba(78, 205, 196, 0.1);
            }
            .upload-area.dragover {
                border-color: #FF6B6B;
                background: rgba(255, 107, 107, 0.1);
                transform: scale(1.02);
            }
            button { 
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white; 
                border: none; 
                padding: 12px 24px; 
                margin: 8px; 
                border-radius: 25px;
                cursor: pointer;
                font-weight: 600;
                transition: all 0.3s ease;
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
            }
            button:hover { 
                transform: translateY(-2px);
                box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
            }
            .counts { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                gap: 20px; 
                margin: 20px 0; 
            }
            .count-card { 
                background: linear-gradient(135deg, #FF6B6B, #4ECDC4);
                color: white;
                padding: 20px; 
                border-radius: 15px; 
                text-align: center;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
            }
            .count-card:hover {
                transform: translateY(-5px) scale(1.02);
            }
            .status-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
            }
            .status-connected { background: #4CAF50; }
            .status-disconnected { background: #F44336; }
            .processing { background: #FF9800; }
            .log-container {
                max-height: 300px;
                overflow-y: auto;
                background: #f8f9fa;
                border-radius: 10px;
                padding: 15px;
                margin: 20px 0;
            }
            .log-entry {
                padding: 8px 12px;
                margin: 5px 0;
                border-radius: 6px;
                border-left: 4px solid;
            }
            .log-success { background: #d4edda; border-left-color: #28a745; }
            .log-error { background: #f8d7da; border-left-color: #dc3545; }
            .log-info { background: #d1ecf1; border-left-color: #17a2b8; }
            .progress-bar {
                width: 100%;
                height: 6px;
                background: #e9ecef;
                border-radius: 3px;
                overflow: hidden;
                margin: 10px 0;
            }
            .progress-fill {
                height: 100%;
                background: linear-gradient(45deg, #667eea, #764ba2);
                width: 0%;
                transition: width 0.3s ease;
            }
            input[type="file"] {
                display: none;
            }
            .file-input-label {
                display: inline-block;
                padding: 12px 24px;
                background: linear-gradient(45deg, #4ECDC4, #44A08D);
                color: white;
                border-radius: 25px;
                cursor: pointer;
                font-weight: 600;
                transition: all 0.3s ease;
                box-shadow: 0 5px 15px rgba(78, 205, 196, 0.3);
            }
            .file-input-label:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 25px rgba(78, 205, 196, 0.4);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚗 Real-Time Vehicle Counter</h1>
                <p><strong>Powered by YOLOv11s/n</strong> • OpenCV-free Node.js implementation</p>
            </div>
            
            <div class="stats-panel">
                <h3>📊 Live Statistics</h3>
                <p>Connection: <span class="status-indicator" id="statusDot"></span><span id="connectionStatus">Connecting...</span></p>
                <p>Total Vehicles Counted: <strong id="totalCount">0</strong></p>
                <p>Counting Line Y Position: <strong id="linePosition">430</strong></p>
                <p>Connected Clients: <strong id="clientCount">0</strong></p>
                <div class="counts" id="classCounts"></div>
            </div>
            
            <div class="upload-area" id="uploadArea">
                <h3>📤 Upload Media</h3>
                <p>Drop files here or click to select</p>
                <input type="file" id="fileInput" accept="video/,image/" multiple>
                <label for="fileInput" class="file-input-label">Choose Files</label>
                <button onclick="processFiles()">🚀 Process</button>
                <div class="progress-bar" id="progressBar" style="display: none;">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
            </div>
            
            <div>
                <button onclick="resetCounters()">🔄 Reset Counters</button>
                <button onclick="setCountingLine()">📏 Set Counting Line</button>
                <button onclick="downloadStats()">📥 Download Stats</button>
                <button onclick="toggleLogs()">📋 Toggle Logs</button>
            </div>
            
            <div class="log-container" id="logContainer" style="display: none;">
                <h4>📝 Processing Logs</h4>
                <div id="logs"></div>
            </div>
        </div>

        <script>
            let ws = null;
            let reconnectInterval = null;
            let uploadedFiles = [];
            
            function connectWebSocket() {
                ws = new WebSocket('ws://localhost:8080');
                
                ws.onopen = () => {
                    updateConnectionStatus(true);
                    clearInterval(reconnectInterval);
                    addLog('Connected to vehicle counter server', 'success');
                };
                
                ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        handleWebSocketMessage(data);
                    } catch (e) {
                        addLog('Failed to parse server message', 'error');
                    }
                };
                
                ws.onclose = () => {
                    updateConnectionStatus(false);
                    addLog('Connection lost. Attempting to reconnect...', 'error');
                    attemptReconnect();
                };
                
                ws.onerror = (error) => {
                    addLog('WebSocket error occurred', 'error');
                };
            }
            
            function attemptReconnect() {
                if (reconnectInterval) return;
                
                reconnectInterval = setInterval(() => {
                    if (ws.readyState === WebSocket.CLOSED) {
                        connectWebSocket();
                    }
                }, 3000);
            }
            
            function updateConnectionStatus(connected) {
                const status = document.getElementById('connectionStatus');
                const dot = document.getElementById('statusDot');
                
                if (connected) {
                    status.textContent = 'Connected';
                    dot.className = 'status-indicator status-connected';
                } else {
                    status.textContent = 'Disconnected';
                    dot.className = 'status-indicator status-disconnected';
                }
            }
            
            function handleWebSocketMessage(data) {
                if (data.stats) {
                    updateStats(data.stats);
                }
                
                switch (data.type) {
                    case 'video_processed':
                    case 'image_processed':
                        addLog('✅ ' + data.filename + ' processed - ' + (data.newCrossings?.length || 0) + ' new vehicles counted', 'success');
                        showNewCrossings(data.newCrossings);
                        break;
                    case 'processing_error':
                        addLog('❌ Error processing ' + data.filename + ': ' + data.error, 'error');
                        break;
                    case 'counters_reset':
                        addLog('🔄 Counters reset successfully', 'info');
                        break;
                    case 'connected':
                        addLog('🎉 Successfully connected to server', 'success');
                        break;
                }
            }
            
            function updateStats(stats) {
                document.getElementById('totalCount').textContent = stats.totalVehicles || 0;
                document.getElementById('linePosition').textContent = stats.countingLineY || 430;
                
                const countsDiv = document.getElementById('classCounts');
                countsDiv.innerHTML = '';
                
                if (stats.classCounts && Object.keys(stats.classCounts).length > 0) {
                    Object.entries(stats.classCounts).forEach(([className, count]) => {
                        const card = document.createElement('div');
                        card.className = 'count-card';
                        card.innerHTML = '<h4>' + className.charAt(0).toUpperCase() + className.slice(1) + '</h4><h2>' + count + '</h2>';
                        countsDiv.appendChild(card);
                    });
                } else {
                    countsDiv.innerHTML = '<div class="count-card"><h4>No vehicles counted yet</h4><h2>0</h2></div>';
                }
            }
            
            function showNewCrossings(crossings) {
                if (!crossings || crossings.length === 0) return;
                
                crossings.forEach(crossing => {
                    addLog(
                        '🚗 New ' + crossing.class_name + ' counted (ID: ' + crossing.track_id + ', Confidence: ' + crossing.confidence + ')',
                        'success'
                    );
                });
            }
            
            function addLog(message, type = 'info') {
                const logsDiv = document.getElementById('logs');
                const logEntry = document.createElement('div');
                logEntry.className = 'log-entry log-' + type;
                logEntry.innerHTML = '<small>' + new Date().toLocaleTimeString() + '</small> ' + message;
                
                logsDiv.insertBefore(logEntry, logsDiv.firstChild);
                
                // Keep only last 50 log entries
                while (logsDiv.children.length > 50) {
                    logsDiv.removeChild(logsDiv.lastChild);
                }
            }
            
            // File handling
            document.getElementById('fileInput').addEventListener('change', (e) => {
                uploadedFiles = Array.from(e.target.files);
                updateUploadArea();
            });
            
            // Drag and drop
            const uploadArea = document.getElementById('uploadArea');
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                uploadArea.addEventListener(eventName, preventDefaults, false);
            });
            
            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }
            
            ['dragenter', 'dragover'].forEach(eventName => {
                uploadArea.addEventListener(eventName, () => uploadArea.classList.add('dragover'), false);
            });
            
            ['dragleave', 'drop'].forEach(eventName => {
                uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('dragover'), false);
            });
            
            uploadArea.addEventListener('drop', (e) => {
                uploadedFiles = Array.from(e.dataTransfer.files);
                updateUploadArea();
            });
            
            function updateUploadArea() {
                const fileCount = uploadedFiles.length;
                const uploadArea = document.getElementById('uploadArea');
                const p = uploadArea.querySelector('p');
                
                if (fileCount === 0) {
                    p.textContent = 'Drop files here or click to select';
                } else {
                    const fileNames = uploadedFiles.map(f => f.name).join(', ');
                    p.textContent = fileCount + ' file(s) selected: ' + (fileNames.length > 50 ? fileNames.substring(0, 50) + '...' : fileNames);
                }
            }
            
            async function processFiles() {
                if (uploadedFiles.length === 0) {
                    addLog('❌ Please select files first', 'error');
                    return;
                }
                
                const progressBar = document.getElementById('progressBar');
                const progressFill = document.getElementById('progressFill');
                progressBar.style.display = 'block';
                
                for (let i = 0; i < uploadedFiles.length; i++) {
                    const file = uploadedFiles[i];
                    const formData = new FormData();
                    const isVideo = file.type.startsWith('video/');
                    const endpoint = isVideo ? '/upload-video' : '/upload-image';
                    
                    formData.append(isVideo ? 'video' : 'image', file);
                    
                    try {
                        addLog('🔄 Processing ' + file.name + '...', 'info');
                        const response = await fetch(endpoint, {
                            method: 'POST',
                            body: formData
                        });
                        
                        const result = await response.json();
                        
                        if (response.ok) {
                            addLog('✅ ' + file.name + ' uploaded successfully', 'success');
                        } else {
                            addLog('❌ Error uploading ' + file.name + ': ' + result.error, 'error');
                        }
                        
                        const progress = ((i + 1) / uploadedFiles.length) * 100;
                        progressFill.style.width = progress + '%';
                        
                    } catch (error) {
                        addLog('❌ Network error processing ' + file.name + ': ' + error.message, 'error');
                    }
                }
                
                setTimeout(() => {
                    progressBar.style.display = 'none';
                    progressFill.style.width = '0%';
                }, 1000);
                
                uploadedFiles = [];
                document.getElementById('fileInput').value = '';
                updateUploadArea();
            }
            
            async function resetCounters() {
                try {
                    const response = await fetch('/reset', { method: 'POST' });
                    const result = await response.json();
                    if (response.ok) {
                        addLog('🔄 Counters reset successfully', 'info');
                    } else {
                        addLog('❌ Error resetting counters: ' + result.error, 'error');
                    }
                } catch (error) {
                    addLog('❌ Network error: ' + error.message, 'error');
                }
            }
            
            async function setCountingLine() {
                const lineY = prompt('Enter Y position for counting line (pixels):', '430');
                if (lineY && !isNaN(lineY)) {
                    try {
                        const response = await fetch('/set-counting-line', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ lineY: parseInt(lineY) })
                        });
                        const result = await response.json();
                        if (response.ok) {
                            addLog('📏 Counting line set to Y=' + lineY, 'success');
                        } else {
                            addLog('❌ Error setting counting line: ' + result.error, 'error');
                        }
                    } catch (error) {
                        addLog('❌ Network error: ' + error.message, 'error');
                    }
                } else if (lineY !== null) {
                    addLog('❌ Invalid Y position. Please enter a number.', 'error');
                }
            }
            
            async function downloadStats() {
                try {
                    const response = await fetch('/stats');
                    const stats = await response.json();
                    
                    const dataStr = JSON.stringify(stats, null, 2);
                    const dataBlob = new Blob([dataStr], { type: 'application/json' });
                    
                    const link = document.createElement('a');
                    link.href = URL.createObjectURL(dataBlob);
                    link.download = 'vehicle-stats-' + new Date().toISOString().split('T')[0] + '.json';
                    link.click();
                    
                    addLog('📥 Statistics downloaded successfully', 'success');
                } catch (error) {
                    addLog('❌ Error downloading stats: ' + error.message, 'error');
                }
            }
            
            function toggleLogs() {
                const logContainer = document.getElementById('logContainer');
                if (logContainer.style.display === 'none') {
                    logContainer.style.display = 'block';
                } else {
                    logContainer.style.display = 'none';
                }
            }
            
            // Initialize
            connectWebSocket();
            
            // Periodic health check
            setInterval(async () => {
                try {
                    const response = await fetch('/health');
                    const health = await response.json();
                    document.getElementById('clientCount').textContent = health.connectedClients || 0;
                } catch (error) {
                    // Health check failed - connection might be down
                }
            }, 10000);
        </script>
    </body>
    </html>
  `);
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('Server error:', error);
  res.status(500).json({ 
    error: 'Internal server error',
    message: error.message 
  });
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('🛑 Received SIGTERM, shutting down gracefully');
  server.close(() => {
    console.log('✅ Process terminated');
    process.exit(0);
  });
});

const server = app.listen(port, () => {
  console.log(`
🚀 Vehicle Counter Server Started!
📡 HTTP Server: http://localhost:${port}
🔌 WebSocket Server: ws://localhost:8080

📋 Setup Instructions:
1. Install Python dependencies: pip install ultralytics opencv-python
2. The system will auto-download YOLOv11s model on first run
3. For faster processing, use YOLOv11n: change 'yolo11s.pt' to 'yolo11n.pt' in the code

🎯 Features:
✅ No OpenCV Node.js dependencies (Python handles CV operations)
✅ Real-time WebSocket updates  
✅ Drag & drop file uploads
✅ Video and image processing
✅ Live statistics dashboard
✅ Downloadable stats export

💡 Performance Tips:
- YOLOv11n: Fastest, good accuracy
- YOLOv11s: Balanced speed/accuracy (recommended)
- Process every 5th frame for videos (configurable)
  `);
});


