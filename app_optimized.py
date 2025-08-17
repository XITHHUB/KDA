#!/usr/bin/env python3
"""
YOLOv8 Real-time Object Detection Web Application
=================================================

A high-performance, multitasking Flask web application for real-time object detection
using YOLOv8 with browser camera integration and concurrent processing.

Features:
- Real-time browser camera detection using WebRTC
- Concurrent processing with ThreadPoolExecutor
- Session management for multiple users
- Optimized frame processing with caching
- Cross-platform support (Desktop, Mobile, Tablet)
- Production-ready error handling and logging

Author: GitHub Repository
License: MIT
"""

import os
import sys
import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

# Flask imports
from flask import Flask, render_template, jsonify, request
from flask.logging import default_handler

# Computer Vision and ML imports
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Utilities
import time
import base64
import io
import uuid
from collections import deque, defaultdict
import concurrent.futures
from threading import Lock, Event
import signal
import atexit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration dataclass
@dataclass
class AppConfig:
    """Application configuration"""
    model_path: str = "best.pt"
    max_workers: int = 6  # Increased for better multitasking
    max_sessions: int = 50
    session_timeout: int = 300  # 5 minutes
    max_frame_size: int = 640
    confidence_threshold: float = 0.25
    jpeg_quality: int = 80
    enable_gpu: bool = True
    cache_detections: bool = True
    max_fps: int = 30
    detection_timeout: int = 10
    cleanup_interval: int = 60  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# Performance monitoring
@dataclass
class PerformanceMetrics:
    """Performance tracking metrics"""
    processing_times: deque = None
    fps_counter: deque = None
    session_count: int = 0
    total_processed: int = 0
    errors: int = 0
    last_cleanup: float = 0
    
    def __post_init__(self):
        if self.processing_times is None:
            self.processing_times = deque(maxlen=100)
        if self.fps_counter is None:
            self.fps_counter = deque(maxlen=30)

# Detection result dataclass
@dataclass
class DetectionResult:
    """Object detection result"""
    class_name: str
    confidence: float
    bbox: List[int]
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'class': self.class_name,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'timestamp': self.timestamp
        }

# Session management
class SessionManager:
    """Manage user sessions with automatic cleanup"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.lock = Lock()
        self.cleanup_event = Event()
        
    def create_session(self, session_id: str = None) -> str:
        """Create a new session"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        with self.lock:
            if len(self.sessions) >= self.config.max_sessions:
                self._cleanup_oldest_sessions()
            
            self.sessions[session_id] = {
                'created': time.time(),
                'last_active': time.time(),
                'frame_count': 0,
                'last_detections': [],
                'processing_times': deque(maxlen=10),
                'errors': 0
            }
        
        logger.info(f"Created session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        with self.lock:
            session = self.sessions.get(session_id)
            if session:
                session['last_active'] = time.time()
            return session
    
    def update_session(self, session_id: str, **kwargs) -> None:
        """Update session data"""
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].update(kwargs)
                self.sessions[session_id]['last_active'] = time.time()
    
    def cleanup_sessions(self) -> int:
        """Remove expired sessions"""
        current_time = time.time()
        expired_sessions = []
        
        with self.lock:
            for session_id, session_data in self.sessions.items():
                if current_time - session_data['last_active'] > self.config.session_timeout:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self.sessions[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        
        return len(expired_sessions)
    
    def _cleanup_oldest_sessions(self, count: int = 5) -> None:
        """Remove oldest sessions when limit reached"""
        sorted_sessions = sorted(
            self.sessions.items(),
            key=lambda x: x[1]['last_active']
        )
        
        for session_id, _ in sorted_sessions[:count]:
            del self.sessions[session_id]
            logger.info(f"Removed old session: {session_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        with self.lock:
            active_sessions = len(self.sessions)
            total_frames = sum(s['frame_count'] for s in self.sessions.values())
            total_errors = sum(s['errors'] for s in self.sessions.values())
            
            return {
                'active_sessions': active_sessions,
                'total_frames_processed': total_frames,
                'total_errors': total_errors,
                'max_sessions': self.config.max_sessions
            }

# Object Detection Engine
class YOLODetectionEngine:
    """Optimized YOLO detection engine with caching and performance monitoring"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.model = None
        self.class_colors = {}
        self.detection_cache = {}
        self.cache_lock = Lock()
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        
        self._load_model()
        self._setup_class_colors()
    
    def _load_model(self) -> None:
        """Load YOLO model with GPU support if available"""
        try:
            if not os.path.exists(self.config.model_path):
                raise FileNotFoundError(f"Model file not found: {self.config.model_path}")
            
            logger.info(f"Loading YOLO model: {self.config.model_path}")
            
            # Configure device
            device = 'cuda' if self.config.enable_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
            logger.info(f"Using device: {device}")
            
            self.model = YOLO(self.config.model_path)
            self.model.to(device)
            
            logger.info(f"Model loaded successfully. Classes: {len(self.model.names)}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _setup_class_colors(self) -> None:
        """Setup colors for different object classes"""
        default_colors = {
            'Aphids': (0, 0, 255),      # Red
            'Healthy': (0, 255, 0),     # Green
            'Mosaic virus': (255, 0, 0), # Blue
            'Powdery': (0, 255, 255),   # Yellow
            'Rust': (255, 0, 255),      # Magenta
            'cb lp': (255, 255, 0)      # Cyan
        }
        
        # Generate colors for all model classes
        for class_id, class_name in self.model.names.items():
            if class_name in default_colors:
                self.class_colors[class_name] = default_colors[class_name]
            else:
                # Generate unique color for unknown classes
                np.random.seed(hash(class_name) % 2**32)
                color = tuple(map(int, np.random.randint(0, 255, 3)))
                self.class_colors[class_name] = color
    
    def detect_objects(self, frame: np.ndarray, session_id: str) -> Tuple[List[DetectionResult], float]:
        """Detect objects in frame with performance tracking"""
        start_time = time.time()
        
        try:
            # Preprocess frame
            processed_frame = self._preprocess_frame(frame)
            
            # Run inference
            results = self.model(processed_frame, verbose=False, conf=self.config.confidence_threshold)
            
            # Process results
            detections = self._process_results(results)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics.processing_times.append(processing_time)
            self.metrics.fps_counter.append(1.0 / max(processing_time, 0.001))
            self.metrics.total_processed += 1
            
            logger.debug(f"Detected {len(detections)} objects in {processing_time:.3f}s")
            
            return detections, processing_time
            
        except Exception as e:
            self.metrics.errors += 1
            logger.error(f"Detection error: {e}")
            return [], time.time() - start_time
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for optimal detection"""
        # Resize if too large
        height, width = frame.shape[:2]
        max_size = self.config.max_frame_size
        
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return frame
    
    def _process_results(self, results) -> List[DetectionResult]:
        """Process YOLO results into DetectionResult objects"""
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    detection = DetectionResult(
                        class_name=class_name,
                        confidence=float(confidence),
                        bbox=[int(x1), int(y1), int(x2), int(y2)]
                    )
                    detections.append(detection)
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        """Draw detection boxes on frame"""
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            color = self.class_colors.get(detection.class_name, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            font_scale = 0.7
            thickness = 2
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            
            # Background rectangle for label
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0] + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        return frame
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_path': self.config.model_path,
            'classes': list(self.model.names.values()),
            'num_classes': len(self.model.names),
            'device': str(self.model.device),
            'confidence_threshold': self.config.confidence_threshold
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if len(self.metrics.fps_counter) > 0:
            avg_fps = sum(self.metrics.fps_counter) / len(self.metrics.fps_counter)
            avg_processing_time = sum(self.metrics.processing_times) / len(self.metrics.processing_times)
        else:
            avg_fps = 0
            avg_processing_time = 0
        
        return {
            'avg_fps': round(avg_fps, 2),
            'avg_processing_time_ms': round(avg_processing_time * 1000, 2),
            'total_processed': self.metrics.total_processed,
            'errors': self.metrics.errors,
            'error_rate': round(self.metrics.errors / max(self.metrics.total_processed, 1) * 100, 2)
        }

# Image processing utilities
class ImageProcessor:
    """Utility class for image processing operations"""
    
    @staticmethod
    def decode_base64_image(image_data: str) -> np.ndarray:
        """Decode base64 image to numpy array"""
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to numpy array
        frame = np.array(image)
        
        # Handle different color formats
        if len(frame.shape) == 3:
            if frame.shape[2] == 4:  # RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        return frame
    
    @staticmethod
    def encode_frame_to_base64(frame: np.ndarray, quality: int = 80) -> str:
        """Encode frame to base64 string"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded_image}"

# Flask Application
class OptimizedFlaskApp:
    """Main Flask application with optimized multitasking"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.app = Flask(__name__)
        self.app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
        
        # Initialize components
        self.session_manager = SessionManager(config)
        self.detection_engine = YOLODetectionEngine(config)
        self.image_processor = ImageProcessor()
        
        # Thread pool for concurrent processing
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.max_workers,
            thread_name_prefix="detection"
        )
        
        # Setup routes and error handlers
        self._setup_routes()
        self._setup_error_handlers()
        self._setup_cleanup_tasks()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main interface"""
            return render_template('index.html')
        
        @self.app.route('/api/features')
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
        
        @self.app.route('/api/about')
        def get_about():
            """API endpoint to get application information"""
            model_info = self.detection_engine.get_model_info()
            about_info = {
                "technology_stack": ["YOLOv8", "Flask", "WebRTC", "OpenCV", "JavaScript", "Python"],
                "model_info": {
                    "name": "Custom YOLOv8",
                    "classes": model_info.get('classes', []),
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
        
        @self.app.route('/health')
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'uptime': time.time() - self.start_time,
                'memory_usage': self._get_memory_usage(),
                'sessions': self.session_manager.get_stats(),
                'performance': self.detection_engine.get_performance_stats()
            })
        
        @self.app.route('/model_info')
        def model_info():
            """Get model information"""
            return jsonify(self.detection_engine.get_model_info())
        
        @self.app.route('/process_mobile_stream', methods=['POST'])
        def process_mobile_stream():
            """Process real-time mobile camera stream"""
            return self._process_stream_request()
        
        @self.app.route('/process_mobile_image', methods=['POST'])
        def process_mobile_image():
            """Process single mobile image"""
            return self._process_image_request()
        
        @self.app.route('/stats')
        def get_stats():
            """Get application statistics"""
            return jsonify({
                'app_config': self.config.to_dict(),
                'sessions': self.session_manager.get_stats(),
                'performance': self.detection_engine.get_performance_stats(),
                'uptime': time.time() - self.start_time if hasattr(self, 'start_time') else 0
            })
        
        @self.app.route('/api/footer')
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
        
        @self.app.route('/api/contact')
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
        
        # Legacy compatibility routes
        @self.app.route('/get_detections')
        def get_detections():
            return jsonify({'detections': [], 'message': 'Use /process_mobile_stream endpoint'})
        
        @self.app.route('/get_performance')
        def get_performance():
            return jsonify({
                'fps': 0,
                'detection_interval': 2,
                'frame_size': f"{self.config.max_frame_size}x{self.config.max_frame_size}",
                'message': 'Using browser camera with optimized processing'
            })
    
    def _setup_error_handlers(self):
        """Setup error handlers"""
        
        @self.app.errorhandler(400)
        def bad_request(error):
            return jsonify({'success': False, 'error': 'Bad request'}), 400
        
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({'success': False, 'error': 'Endpoint not found'}), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            logger.error(f"Internal error: {error}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500
        
        @self.app.errorhandler(413)
        def request_entity_too_large(error):
            return jsonify({'success': False, 'error': 'File too large'}), 413
    
    def _setup_cleanup_tasks(self):
        """Setup periodic cleanup tasks"""
        import threading
        
        def cleanup_worker():
            while True:
                try:
                    time.sleep(self.config.cleanup_interval)
                    self.session_manager.cleanup_sessions()
                except Exception as e:
                    logger.error(f"Cleanup error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def _process_stream_request(self):
        """Process streaming request with optimizations"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'error': 'No data provided'})
            
            image_data = data.get('image')
            session_id = data.get('session_id', str(uuid.uuid4()))
            frame_rate = data.get('frame_rate', 2)
            
            if not image_data:
                return jsonify({'success': False, 'error': 'No image data provided'})
            
            # Get or create session
            session = self.session_manager.get_session(session_id)
            if not session:
                session_id = self.session_manager.create_session(session_id)
                session = self.session_manager.get_session(session_id)
            
            # Update frame count
            session['frame_count'] += 1
            
            # Process only every Nth frame for performance
            should_process = session['frame_count'] % frame_rate == 0
            
            if should_process:
                # Submit detection task to thread pool
                future = self.executor.submit(
                    self._detect_objects_from_base64,
                    image_data,
                    session_id
                )
                
                try:
                    detections, processing_time = future.result(timeout=self.config.detection_timeout)
                    
                    # Update session with new detections
                    detections_dict = [d.to_dict() for d in detections]
                    self.session_manager.update_session(
                        session_id,
                        last_detections=detections_dict,
                        processing_times=session['processing_times']
                    )
                    session['processing_times'].append(processing_time)
                    
                    return jsonify({
                        'success': True,
                        'detections': detections_dict,
                        'processing_time': round(processing_time * 1000, 1),
                        'session_id': session_id,
                        'cached': False
                    })
                    
                except concurrent.futures.TimeoutError:
                    logger.warning(f"Detection timeout for session {session_id}")
                    return jsonify({
                        'success': True,
                        'detections': session['last_detections'],
                        'cached': True,
                        'session_id': session_id,
                        'message': 'Using cached results due to timeout'
                    })
            else:
                # Return cached results
                return jsonify({
                    'success': True,
                    'detections': session['last_detections'],
                    'cached': True,
                    'session_id': session_id
                })
        
        except Exception as e:
            logger.error(f"Stream processing error: {e}")
            return jsonify({'success': False, 'error': str(e)})
    
    def _process_image_request(self):
        """Process single image request"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'error': 'No data provided'})
            
            image_data = data.get('image')
            session_id = data.get('session_id', str(uuid.uuid4()))
            
            if not image_data:
                return jsonify({'success': False, 'error': 'No image data provided'})
            
            # Create session if needed
            if not self.session_manager.get_session(session_id):
                session_id = self.session_manager.create_session(session_id)
            
            # Process image
            future = self.executor.submit(
                self._detect_objects_from_base64,
                image_data,
                session_id,
                return_image=True
            )
            
            result = future.result(timeout=15)  # Longer timeout for single image
            
            if len(result) == 3:  # With processed image
                detections, processing_time, processed_image = result
                return jsonify({
                    'success': True,
                    'detections': [d.to_dict() for d in detections],
                    'processed_image': processed_image,
                    'processing_time': round(processing_time * 1000, 1),
                    'session_id': session_id
                })
            else:  # Without processed image
                detections, processing_time = result
                return jsonify({
                    'success': True,
                    'detections': [d.to_dict() for d in detections],
                    'processing_time': round(processing_time * 1000, 1),
                    'session_id': session_id
                })
        
        except concurrent.futures.TimeoutError:
            return jsonify({'success': False, 'error': 'Processing timeout'})
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return jsonify({'success': False, 'error': str(e)})
    
    def _detect_objects_from_base64(self, image_data: str, session_id: str, return_image: bool = False):
        """Detect objects from base64 image data"""
        try:
            # Decode image
            frame = self.image_processor.decode_base64_image(image_data)
            
            # Detect objects
            detections, processing_time = self.detection_engine.detect_objects(frame, session_id)
            
            if return_image:
                # Draw detections and encode
                annotated_frame = self.detection_engine.draw_detections(frame, detections)
                processed_image = self.image_processor.encode_frame_to_base64(
                    annotated_frame, self.config.jpeg_quality
                )
                return detections, processing_time, processed_image
            else:
                return detections, processing_time
        
        except Exception as e:
            logger.error(f"Detection processing error: {e}")
            if return_image:
                return [], 0.0, None
            else:
                return [], 0.0
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask application"""
        self.start_time = time.time()
        
        def signal_handler(signum, frame):
            logger.info("Shutting down gracefully...")
            self.executor.shutdown(wait=True)
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        logger.info("🚀 Starting Optimized YOLOv8 Flask App...")
        logger.info(f"🌐 Access from any device: http://localhost:{port}")
        logger.info(f"📱 Browser camera with real-time detection")
        logger.info(f"🔗 Network access: http://[YOUR_IP]:{port}")
        logger.info(f"⚡ Max workers: {self.config.max_workers}")
        logger.info(f"📊 GPU enabled: {self.config.enable_gpu}")
        logger.info("📝 Allow camera permissions when prompted")
        
        try:
            self.app.run(
                host=host,
                port=port,
                debug=debug,
                threaded=True,
                use_reloader=False  # Prevent issues with threading
            )
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.executor.shutdown(wait=True)
            logger.info("✅ App shutdown complete")

# Configuration loader
def load_config() -> AppConfig:
    """Load configuration from file or environment"""
    config_file = "config.json"
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            logger.info(f"Loaded configuration from {config_file}")
            return AppConfig(**config_data)
        except Exception as e:
            logger.warning(f"Failed to load config file: {e}, using defaults")
    
    return AppConfig()

# Main entry point
def main():
    """Main application entry point"""
    try:
        # Load configuration
        config = load_config()
        
        # Create and run app
        app = OptimizedFlaskApp(config)
        app.run()
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
