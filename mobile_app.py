from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
import queue
from collections import deque
import base64
import io
from PIL import Image
import concurrent.futures
import uuid

app = Flask(__name__)

# Initialize YOLO model
model = YOLO('best.pt')

# Global variables for processing
detection_results = {}
is_detecting = False
fps_counter = deque(maxlen=30)
processing_queue = queue.Queue(maxsize=5)  # Limit queue for mobile
mobile_sessions = {}  # Track mobile sessions

class MobileImageProcessor:
    def __init__(self):
        self.detection_interval = 2  # Process every 2nd frame for mobile
        self.frame_count = {}  # Per-session frame count
        self.last_detections = {}  # Per-session cached detections
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)  # Multi-threading
        
    def process_image_async(self, image_data, session_id):
        """Process image asynchronously"""
        future = self.executor.submit(self._process_image_sync, image_data, session_id)
        return future
    
    def _process_image_sync(self, image_data, session_id):
        """Synchronous image processing"""
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to numpy array
            frame = np.array(image)
            if len(frame.shape) == 3 and frame.shape[2] == 4:  # RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            elif len(frame.shape) == 3:  # RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Resize for optimal mobile processing
            height, width = frame.shape[:2]
            max_size = 640
            if max(height, width) > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Initialize session if needed
            if session_id not in self.frame_count:
                self.frame_count[session_id] = 0
                self.last_detections[session_id] = []
            
            self.frame_count[session_id] += 1
            
            # Smart detection with frame skipping
            if self.frame_count[session_id] % self.detection_interval == 0:
                detections = self._run_detection(frame)
                self.last_detections[session_id] = detections
            else:
                detections = self.last_detections[session_id]
            
            # Encode processed image back to base64
            processed_frame = self._draw_detections(frame, detections)
            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            processed_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                'success': True,
                'processed_image': f"data:image/jpeg;base64,{processed_base64}",
                'detections': detections,
                'session_id': session_id
            }
            
        except Exception as e:
            print(f"❌ Processing error: {e}")
            return {
                'success': False,
                'error': str(e),
                'session_id': session_id
            }
    
    def _run_detection(self, frame):
        """Run YOLO detection on frame"""
        start_time = time.time()
        results = model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]
                    
                    if confidence > 0.3:  # Lower threshold for mobile
                        detections.append({
                            'class': class_name,
                            'confidence': float(confidence),
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        })
        
        # Track FPS
        inference_time = time.time() - start_time
        fps_counter.append(1.0 / max(inference_time, 0.001))
        
        return detections
    
    def _draw_detections(self, frame, detections):
        """Draw detection boxes on frame"""
        colors = {
            'Aphids': (0, 0, 255),      # Red
            'Healthy': (0, 255, 0),     # Green
            'Mosaic virus': (255, 0, 0), # Blue
            'Powdery': (0, 255, 255),   # Yellow
            'Rust': (255, 0, 255),      # Magenta
            'cb lp': (255, 255, 0)      # Cyan
        }
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            class_name = detection['class']
            confidence = detection['confidence']
            color = colors.get(class_name, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 8), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

# Initialize mobile processor
mobile_processor = MobileImageProcessor()

@app.route('/')
def index():
    """Desktop interface"""
    return render_template('index.html')

@app.route('/mobile')
def mobile():
    """Mobile optimized interface"""
    return render_template('mobile.html')

@app.route('/process_mobile_image', methods=['POST'])
def process_mobile_image():
    """Process image from mobile device"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        if not image_data:
            return jsonify({'success': False, 'error': 'No image data provided'})
        
        # Process image asynchronously
        future = mobile_processor.process_image_async(image_data, session_id)
        result = future.result(timeout=10)  # 10 second timeout
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Mobile processing error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_mobile_performance')
def get_mobile_performance():
    """Get performance metrics for mobile"""
    if len(fps_counter) > 0:
        avg_fps = sum(fps_counter) / len(fps_counter)
        return jsonify({
            'fps': round(avg_fps, 1),
            'processing_time': round(1.0/avg_fps, 2) if avg_fps > 0 else 0,
            'queue_size': processing_queue.qsize(),
            'active_sessions': len(mobile_processor.frame_count)
        })
    return jsonify({'fps': 0, 'processing_time': 0, 'queue_size': 0, 'active_sessions': 0})

@app.route('/model_info')
def model_info():
    """Get model information"""
    return jsonify({
        'model_path': 'best.pt',
        'classes': list(model.names.values()),
        'num_classes': len(model.names),
        'mobile_optimized': True
    })

if __name__ == '__main__':
    try:
        print("🚀 Starting Mobile-Optimized YOLOv8 Flask App...")
        print("📱 Mobile interface: http://localhost:5000/mobile")
        print("🖥️  Desktop interface: http://localhost:5000")
        print("Press Ctrl+C to stop the server")
        # Use threading for mobile support
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    finally:
        if 'mobile_processor' in globals():
            mobile_processor.executor.shutdown(wait=True)
