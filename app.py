from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
import base64
import io
from PIL import Image
import concurrent.futures
import uuid

app = Flask(__name__)

# Track application start time for uptime calculation
app.start_time = time.time()

# Initialize YOLO model
model = YOLO('best.pt')

# Global variables for detection tracking
detection_results = []  # Legacy - kept for compatibility
fps_counter = deque(maxlen=30)  # Track FPS over last 30 frames

# Mobile processing variables
mobile_sessions = {}  # Track mobile user sessions
mobile_processor_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)  # Multi-threading for mobile

class CameraDetection:
    def __init__(self):
        self.camera = None
        self.camera_index = None
        self.dummy_mode = False
        self.frame_width = 640
        self.frame_height = 480
        self.target_fps = 30
        self.detection_interval = 3  # Run detection every N frames for better performance
        self.frame_count = 0
        self.last_detection_frame = None
        self.last_detections = []
        self._initialize_camera()
        
    def _initialize_camera(self):
        """Initialize camera with robust fallback options"""
        print("🎥 Initializing camera...")
        
        # Try different camera indices and backends
        camera_options = [
            (0, None),              # Default backend first (most reliable)
            (1, None),              # Second camera default
            (0, cv2.CAP_V4L2),      # USB/Webcam with V4L2  
            (1, cv2.CAP_V4L2),      # Second camera with V4L2
            (0, cv2.CAP_GSTREAMER), # GStreamer backend
            (0, cv2.CAP_ANY),       # Any available backend
        ]
        
        for index, backend in camera_options:
            try:
                print(f"🔍 Trying camera {index} with backend {backend}...")
                
                if backend is not None:
                    camera = cv2.VideoCapture(index, backend)
                else:
                    camera = cv2.VideoCapture(index)
                
                # Give camera time to initialize
                time.sleep(0.3)  # Reduced initialization time
                
                # Test if camera can read frames
                if camera.isOpened():
                    # Try multiple frame reads to ensure stability
                    for _ in range(2):  # Reduced test reads
                        ret, test_frame = camera.read()
                        if ret and test_frame is not None and test_frame.size > 0:
                            self.camera = camera
                            self.camera_index = index
                            print(f"✅ Camera opened successfully: /dev/video{index}")
                            self._configure_camera()
                            return
                        time.sleep(0.05)
                    
                camera.release()
                
            except Exception as e:
                print(f"❌ Failed camera {index}: {e}")
                continue
        
        print("❌ No working camera found! Using dummy mode...")
        self.dummy_mode = True
        
    def _configure_camera(self):
        """Configure camera settings for optimal performance"""
        if self.camera and not self.dummy_mode:
            # Set camera properties for performance
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.camera.set(cv2.CAP_PROP_FPS, self.target_fps)
            # Optimize for real-time streaming
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
            self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            # Additional performance settings
            self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Reduce auto-exposure for consistency
            self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus for speed
    
    def __del__(self):
        if self.camera and hasattr(self.camera, 'isOpened') and self.camera.isOpened():
            self.camera.release()
    
    def get_frame(self):
        if self.dummy_mode or self.camera is None:
            return self._create_dummy_frame()
        
        try:
            # Skip frames if queue is full (real-time performance)
            if hasattr(self, 'camera') and self.camera:
                success, frame = self.camera.read()
                if not success or frame is None:
                    print("⚠️ Failed to read from camera, attempting reconnect...")
                    self._initialize_camera()
                    if not self.dummy_mode and self.camera:
                        success, frame = self.camera.read()
                    
                    if not success:
                        return self._create_dummy_frame()
                
                # Resize frame for consistent performance
                if frame.shape[:2] != (self.frame_height, self.frame_width):
                    frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                
                return frame
            else:
                return self._create_dummy_frame()
                
        except Exception as e:
            print(f"❌ Camera error: {e}")
            return self._create_dummy_frame()
    
    def detect_objects_optimized(self, frame):
        """Optimized object detection with frame skipping"""
        self.frame_count += 1
        
        # Skip detection frames for better performance
        if self.frame_count % self.detection_interval != 0:
            # Return previous detection results with updated frame
            if self.last_detections and self.last_detection_frame is not None:
                return self._draw_cached_detections(frame, self.last_detections)
            return frame, []
        
        # Run actual detection
        start_time = time.time()
        results = model(frame, verbose=False)  # Disable verbose output for speed
        detections = []
        
        # Process results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    class_name = model.names[class_id]
                    
                    # Only show detections with confidence > 0.4 (slightly lower for smoother experience)
                    if confidence > 0.4:
                        detections.append({
                            'class': class_name,
                            'confidence': float(confidence),
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        })
        
        # Cache detection results
        self.last_detections = detections
        self.last_detection_frame = frame.copy()
        
        # Draw detections
        annotated_frame = self._draw_cached_detections(frame, detections)
        
        # Track inference time for performance monitoring
        inference_time = time.time() - start_time
        fps_counter.append(1.0 / max(inference_time, 0.001))
        
        return annotated_frame, detections
    
    def _draw_cached_detections(self, frame, detections):
        """Draw detection boxes on frame"""
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Use different colors for different classes
            colors = {
                'Aphids': (0, 0, 255),      # Red
                'Healthy': (0, 255, 0),     # Green
                'Mosaic virus': (255, 0, 0), # Blue
                'Powdery': (0, 255, 255),   # Yellow
                'Rust': (255, 0, 255),      # Magenta
                'cb lp': (255, 255, 0)      # Cyan
            }
            color = colors.get(class_name, (0, 255, 0))
            
            # Draw bounding box with thicker lines for visibility
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Create label background for better readability
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def _create_dummy_frame(self):
        """Create a dummy frame with test pattern"""
        frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        # Create a test pattern with smooth animation
        cv2.rectangle(frame, (50, 50), (self.frame_width-50, self.frame_height-50), (0, 255, 0), 2)
        cv2.putText(frame, "NO CAMERA DETECTED", (120, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, "Using test pattern", (160, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Connect camera and restart app", (100, 280), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Or run: python fix_camera.py", (120, 320), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add smooth moving elements
        t = time.time() * 2
        x_pos = int(320 + 100 * np.sin(t))
        y_pos = int(360 + 20 * np.cos(t * 2))
        cv2.circle(frame, (x_pos, y_pos), 8, (255, 0, 0), -1)
        cv2.circle(frame, (640 - x_pos, y_pos), 8, (0, 0, 255), -1)
        
        return frame

# Server-side camera class (legacy - keeping for reference)
# The frontend now uses browser camera (WebRTC), not server camera
# class CameraDetection:
#     ... (commented out for browser camera usage)

# Note: Camera detection now happens in browser, processed via /process_mobile_stream

# Mobile Image Processing Functions
def process_mobile_image_async(image_data, session_id):
    """Process mobile image asynchronously"""
    future = mobile_processor_executor.submit(process_mobile_image_sync, image_data, session_id)
    return future

def process_mobile_image_sync(image_data, session_id):
    """Process image from mobile device synchronously"""
    try:
        # Decode base64 image from mobile
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL image to OpenCV format
        frame = np.array(image)
        if len(frame.shape) == 3:
            if frame.shape[2] == 4:  # RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Resize for optimal mobile processing (max 640px)
        height, width = frame.shape[:2]
        max_size = 640
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Run YOLO detection
        start_time = time.time()
        results = model(frame, verbose=False)
        detections = []
        
        # Process detection results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]
                    
                    # Lower confidence threshold for mobile (better user experience)
                    if confidence > 0.25:
                        detections.append({
                            'class': class_name,
                            'confidence': float(confidence),
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        })
        
        # Draw detections on frame
        annotated_frame = draw_mobile_detections(frame, detections)
        
        # Encode processed image back to base64
        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        processed_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Track processing time
        processing_time = time.time() - start_time
        fps_counter.append(1.0 / max(processing_time, 0.001))
        
        return {
            'success': True,
            'processed_image': f"data:image/jpeg;base64,{processed_base64}",
            'detections': detections,
            'processing_time': round(processing_time * 1000, 1),  # ms
            'session_id': session_id
        }
        
    except Exception as e:
        print(f"❌ Mobile processing error: {e}")
        return {
            'success': False,
            'error': str(e),
            'session_id': session_id
        }

def draw_mobile_detections(frame, detections):
    """Draw detection boxes optimized for mobile viewing"""
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
        
        # Draw thicker bounding box for mobile visibility
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Draw label with larger font for mobile
        label = f"{class_name}: {confidence:.2f}"
        font_scale = 0.7  # Larger font for mobile
        thickness = 2
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        
        # Background rectangle for label
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0] + 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    return frame

# Legacy server-side frame generation (no longer needed)
# Frontend now uses browser camera with /process_mobile_stream endpoint
# def generate_frames(): ... (removed for browser camera usage)

@app.route('/')
def index():
    """Unified interface for desktop and mobile using browser camera"""
    return render_template('index.html')

@app.route('/api/features')
def get_features():
    """API endpoint to get application features"""
    features = [
        {
            "icon": "🎥",
            "title": "Real-time Detection",
            "description": "Live camera feed with instant object detection using YOLOv8 model"
        },
        {
            "icon": "📱",
            "title": "Mobile Optimized", 
            "description": "Responsive design that works seamlessly on phones, tablets, and desktops"
        },
        {
            "icon": "⚡",
            "title": "High Performance",
            "description": "Multi-threaded processing with GPU acceleration for smooth performance"
        },
        {
            "icon": "🎯",
            "title": "6 Detection Classes",
            "description": "Specialized model trained for: Aphids, Healthy, Mosaic virus, Powdery, Rust, CB LP"
        },
        {
            "icon": "🔄",
            "title": "Smart Caching",
            "description": "Frame skipping and detection caching for optimal battery life"
        },
        {
            "icon": "📊",
            "title": "Performance Metrics",
            "description": "Real-time FPS monitoring and processing time analytics"
        }
    ]
    return jsonify({"features": features})

@app.route('/api/about')
def get_about():
    """API endpoint to get application information"""
    about_info = {
        "technology_stack": ["YOLOv8", "Flask", "WebRTC", "OpenCV", "JavaScript", "Python"],
        "model_info": {
            "name": "Custom YOLOv8",
            "classes": list(model.names.values()) if hasattr(model, 'names') else ["Aphids", "Healthy", "Mosaic virus", "Powdery", "Rust", "cb lp"],
            "class_descriptions": {
                "Aphids": {
                    "thai_name": "เพลี้ยอ่อน",
                    "description": "แมลงศัตรูพืชขนาดเล็กที่ดูดกินน้ำเลี้ยงจากใบและยอดอ่อน ทำให้ใบเหี่ยวเฟียด เจริญเติบโตผิดปกติ และอาจเป็นพาหะนำโรคไวรัส"
                },
                "Healthy": {
                    "thai_name": "ใบปกติ",
                    "description": "ใบมันสำปะหลังที่มีสุขภาพดี สีเขียวสด ไม่มีการติดเชื้อโรคหรือถูกทำลายจากศัตรูพืช สามารถทำการสังเคราะห์แสงได้ปกติ"
                },
                "Mosaic virus": {
                    "thai_name": "โรคใบด่าง",
                    "description": "โรคไวรัสที่ทำให้ใบมีลายด่างสีเหลือง-เขียว การเจริญเติบโตผิดปกติ ใบย่น และอาจทำให้ผลผลิตลดลงอย่างมาก"
                },
                "Powdery": {
                    "thai_name": "โรคราแป้ง",
                    "description": "โรคเชื้อราที่ปรากฏเป็นผงสีขาวบนผิวใบ ทำให้การสังเคราะห์แสงลดลง ใบเหี่ยวแห้ง และส่งผลต่อการเจริญเติบโต"
                },
                "Rust": {
                    "thai_name": "โรคราสนิม",
                    "description": "โรคเชื้อราที่ทำให้เกิดจุดหรือแผลสีน้ำตาล-ส้มบนใบ คล้ายคราบสนิม ทำให้ใบร่วงและพืชอ่อนแอ"
                },
                "cb lp": {
                    "thai_name": "หนอนเจาะลำต้น",
                    "description": "แมลงศัตรูพืชที่เจาะเข้าไปในลำต้นและกิ่ง ทำลายระบบลำเลียงน้ำและอาหาร ส่งผลให้พืชเหี่ยวแห้งและอาจตายได้"
                }
            },
            "description": "Custom YOLOv8 model trained for plant disease detection with 6 specialized classes. The model achieves high accuracy in real-time scenarios while maintaining efficient processing speeds."
        },
        "requirements": [
            "Modern web browser with WebRTC support",
            "Camera access permissions", 
            "Stable internet connection",
            "Recommended: Mobile device or desktop with camera"
        ],
        "usage_steps": [
            "Allow camera access when prompted",
            "Click 'Start Real-time' to begin detection",
            "Adjust frame rate and quality settings as needed",
            "Use 'Switch Camera' to toggle between front/back cameras",
            "Monitor performance metrics for optimization"
        ]
    }
    return jsonify(about_info)

@app.route('/api/stats')
def get_app_stats():
    """API endpoint to get application statistics"""
    stats = {
        "active_sessions": len(mobile_sessions),
        "total_detections": sum(len(session.get('detections', [])) for session in mobile_sessions.values()),
        "model_classes": len(model.names) if hasattr(model, 'names') else 6,
        "uptime": time.time() - app.start_time if hasattr(app, 'start_time') else 0
    }
    return jsonify(stats)

@app.route('/api/footer')
def get_footer_info():
    """API endpoint to get footer information"""
    footer_info = {
        "company": {
            "name": "Konjac Detecting Assistant",
            "description": "Advanced YOLOv8-powered plant disease detection system with real-time analysis capabilities. Helping farmers and agricultural professionals identify and manage crop health issues efficiently.",
            "year": "2025"
        },
        "social_links": [
            {"name": "GitHub", "url": "https://github.com", "icon": "📚"},
            {"name": "LinkedIn", "url": "https://linkedin.com", "icon": "💼"},
            {"name": "Email", "url": "mailto:contact@example.com", "icon": "📧"},
            {"name": "Twitter", "url": "https://twitter.com", "icon": "🐦"}
        ],
        "quick_links": {
            "features": ["Real-time Detection", "Mobile Optimization", "Multi-threaded Processing", "Smart Caching", "Performance Analytics", "API Documentation"],
            "resources": ["Documentation", "API Reference", "Installation Guide", "System Requirements", "Health Status", "Support"],
            "detection_classes": ["Aphids Detection", "Healthy Plant Analysis", "Mosaic Virus Identification", "Powdery Mildew Detection", "Rust Disease Analysis", "CB LP Detection"]
        },
        "technology_stack": ["YOLOv8", "Flask", "WebRTC", "OpenCV", "Python"]
    }
    return jsonify(footer_info)

@app.route('/api/contact')
def get_contact_info():
    """API endpoint to get contact information"""
    contact_info = {
        "support_email": "support@konjac-detection.com",
        "general_email": "info@konjac-detection.com",
        "technical_email": "tech@konjac-detection.com",
        "documentation_url": "/api/about",
        "health_check_url": "/health",
        "github_url": "https://github.com/your-username/konjac-detection",
        "response_time": "24-48 hours",
        "support_hours": "9 AM - 5 PM UTC"
    }
    return jsonify(contact_info)

# Remove server-side camera routes since frontend uses browser camera
# @app.route('/video_feed') - No longer needed, frontend uses WebRTC
# @app.route('/toggle_detection') - No longer needed, frontend handles detection

# Legacy routes - no longer needed since frontend uses browser camera
@app.route('/get_detections')
def get_detections():
    """Get current detection results - Legacy support"""
    return jsonify({'detections': [], 'message': 'Use browser camera instead'})

@app.route('/get_performance') 
def get_performance():
    """Get performance metrics - Legacy support"""
    return jsonify({
        'fps': 0,
        'detection_interval': 2,
        'frame_size': '640x480',
        'message': 'Using browser camera'
    })

@app.route('/model_info')
def model_info():
    """Get model information"""
    return jsonify({
        'model_path': 'best.pt',
        'classes': list(model.names.values()),
        'num_classes': len(model.names),
        'camera_mode': 'browser_camera'  # Updated to reflect browser camera usage
    })

# Simplified routing - all users get the same unified interface
# No separate mobile routes needed since index.html is now mobile-optimized

@app.route('/process_mobile_image', methods=['POST'])
def process_mobile_image():
    """Process image from mobile device"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        if not image_data:
            return jsonify({'success': False, 'error': 'No image data provided'})
        
        # Process image asynchronously with timeout
        future = process_mobile_image_async(image_data, session_id)
        result = future.result(timeout=15)  # 15 second timeout for mobile
        
        return jsonify(result)
        
    except concurrent.futures.TimeoutError:
        return jsonify({'success': False, 'error': 'Processing timeout - try again'})
    except Exception as e:
        print(f"❌ Mobile processing error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/process_mobile_stream', methods=['POST'])
def process_mobile_stream():
    """Process real-time mobile camera stream"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        session_id = data.get('session_id', str(uuid.uuid4()))
        frame_rate = data.get('frame_rate', 2)  # Process every 2nd frame by default
        
        if not image_data:
            return jsonify({'success': False, 'error': 'No image data provided'})
        
        # Initialize session frame counter
        if session_id not in mobile_sessions:
            mobile_sessions[session_id] = {'frame_count': 0, 'last_detections': []}
        
        mobile_sessions[session_id]['frame_count'] += 1
        
        # Process only every Nth frame for real-time performance
        if mobile_sessions[session_id]['frame_count'] % frame_rate == 0:
            # Process image with shorter timeout for real-time
            future = process_mobile_image_async(image_data, session_id)
            result = future.result(timeout=5)  # Shorter timeout for real-time
            mobile_sessions[session_id]['last_detections'] = result.get('detections', [])
        else:
            # Return cached detections for intermediate frames
            result = {
                'success': True,
                'detections': mobile_sessions[session_id]['last_detections'],
                'cached': True,
                'session_id': session_id
            }
        
        return jsonify(result)
        
    except concurrent.futures.TimeoutError:
        return jsonify({'success': False, 'error': 'Processing timeout', 'cached': True})
    except Exception as e:
        print(f"❌ Mobile stream processing error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/mobile_performance')
def mobile_performance():
    """Get mobile performance metrics"""
    if len(fps_counter) > 0:
        avg_processing_time = 1.0 / (sum(fps_counter) / len(fps_counter))
        return jsonify({
            'avg_processing_time_ms': round(avg_processing_time * 1000, 1),
            'active_sessions': len(mobile_sessions),
            'model_classes': len(model.names),
            'server_status': 'running'
        })
    return jsonify({
        'avg_processing_time_ms': 0,
        'active_sessions': 0,
        'model_classes': len(model.names),
        'server_status': 'running'
    })

if __name__ == '__main__':
    try:
        print("🚀 Starting YOLOv8 Flask App - Browser Camera Mode...")
        print("🌐 Access from any device: http://localhost:5000")
        print("📱 Uses browser camera (WebRTC) for real-time detection")
        print("🔗 Network access: http://[YOUR_IP]:5000")
        print("📝 Allow camera permissions when prompted in browser")
        print("Press Ctrl+C to stop the server")
        # Set debug=False to prevent auto-restart which interferes with camera
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Cleanup mobile processor on shutdown
        if 'mobile_processor_executor' in globals():
            mobile_processor_executor.shutdown(wait=True)
        print("✅ App shutdown complete")
