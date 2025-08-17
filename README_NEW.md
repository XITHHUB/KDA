# 🎯 YOLOv8 Real-time Object Detection Web App

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-green)](https://flask.palletsprojects.com)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Latest-orange)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-red)](LICENSE)

A high-performance, real-time object detection web application using YOLOv8 and Flask with browser camera integration, concurrent processing, and cross-platform support.

## ✨ Features

### 🎥 **Real-time Detection**

- **Browser Camera Integration**: Uses WebRTC for direct camera access
- **Live Object Detection**: Real-time YOLOv8 inference with your custom model
- **Multi-platform Support**: Works on Desktop, Mobile, and Tablet devices

### ⚡ **Performance Optimized**

- **Concurrent Processing**: Multi-threaded with ThreadPoolExecutor
- **Smart Caching**: Frame-skipping and detection caching for smooth performance
- **Session Management**: Automatic cleanup and resource management
- **GPU Acceleration**: CUDA support when available

### 📱 **Cross-Platform Compatibility**

- **Responsive Design**: Mobile-optimized interface
- **WebRTC Support**: Direct browser camera access
- **Network Access**: Share via WiFi to other devices
- **Touch-Friendly**: Optimized for mobile interaction

### 🛠️ **Production Ready**

- **Error Handling**: Comprehensive error management
- **Logging**: Detailed application logging
- **Health Checks**: Built-in monitoring endpoints
- **Configuration**: JSON-based configuration system

## 🚀 Quick Start

### 1. **Clone Repository**

```bash
git clone https://github.com/your-username/yolov8-web-detection.git
cd yolov8-web-detection
```

### 2. **Setup Environment**

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. **Add Your Model**

Place your YOLOv8 model file as `best.pt` in the project directory.

### 4. **Run Application**

```bash
# Standard version
python app.py

# Optimized version (recommended)
python app_optimized.py
```

### 5. **Access Application**

- **Desktop**: http://localhost:5000
- **Mobile**: http://[YOUR_IP]:5000 (replace with your local IP)

## 📱 Mobile Usage

1. **Connect to WiFi**: Ensure phone and computer are on the same network
2. **Find IP Address**:

   ```bash
   # Linux/Mac
   ip addr show | grep inet

   # Windows
   ipconfig
   ```

3. **Open Browser**: Navigate to `http://[YOUR_IP]:5000`
4. **Allow Camera**: Grant camera permissions when prompted
5. **Start Detection**: Tap "▶️ Start Real-time" for live detection

## 🛠️ Configuration

### Application Settings (`config.json`)

```json
{
  "model_path": "best.pt",
  "max_workers": 8,
  "max_sessions": 100,
  "confidence_threshold": 0.25,
  "max_frame_size": 640,
  "enable_gpu": true
}
```

### Performance Tuning

- **High Performance**: Increase `max_workers`, enable GPU
- **Low Resource**: Reduce `max_frame_size`, lower `max_workers`
- **Battery Saving**: Increase frame skipping, reduce quality

## 📊 API Endpoints

| Endpoint                 | Method | Description                 |
| ------------------------ | ------ | --------------------------- |
| `/`                      | GET    | Main web interface          |
| `/health`                | GET    | Health check and stats      |
| `/model_info`            | GET    | Model information           |
| `/process_mobile_stream` | POST   | Real-time stream processing |
| `/process_mobile_image`  | POST   | Single image processing     |
| `/stats`                 | GET    | Application statistics      |

### Example API Usage

```javascript
// Real-time detection
fetch("/process_mobile_stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    image: base64ImageData,
    session_id: sessionId,
    frame_rate: 2,
  }),
})
  .then((response) => response.json())
  .then((data) => {
    // Handle detection results
    console.log(data.detections);
  });
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Browser       │    │   Flask App      │    │   YOLOv8        │
│   (WebRTC)      │───▶│   (Concurrent)   │───▶│   (GPU/CPU)     │
│                 │    │                  │    │                 │
│ • Camera Access │    │ • Session Mgmt   │    │ • Object Det.   │
│ • Real-time UI  │    │ • Thread Pool    │    │ • Custom Model  │
│ • Frame Capture │    │ • Caching        │    │ • Optimization  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
yolov8-web-detection/
├── app.py                 # Main Flask application
├── app_optimized.py       # Optimized version with multitasking
├── config.json           # Application configuration
├── best.pt               # Your YOLOv8 model (add this)
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html        # Web interface
├── static/              # Static assets (CSS, JS)
├── tests/               # Unit tests
├── docs/                # Documentation
└── README.md            # This file
```

## 🧪 Testing

```bash
# Run tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=app tests/

# Test specific functionality
python test_setup.py
```

## 🔧 Troubleshooting

### Common Issues

**Camera Not Working**

```bash
# Check camera permissions
python fix_camera.py

# Test camera access
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

**Performance Issues**

- Reduce `max_frame_size` in config
- Lower `confidence_threshold`
- Increase frame skipping rate
- Check GPU availability

**Import Errors**

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check PyTorch installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

**Network Access Issues**

- Check firewall settings
- Verify IP address
- Ensure devices on same network
- Try different ports

## 📈 Performance Optimization

### Production Deployment

```bash
# Use production WSGI server
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app_optimized:app

# Or with uWSGI
pip install uwsgi
uwsgi --http :5000 --module app_optimized:app --processes 4
```

### GPU Acceleration

```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify GPU support
python -c "import torch; print('CUDA devices:', torch.cuda.device_count())"
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run code formatting
black app_optimized.py

# Run linting
flake8 app_optimized.py

# Run tests
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection model
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [OpenCV](https://opencv.org/) - Computer vision library
- [WebRTC](https://webrtc.org/) - Real-time communication

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/yolov8-web-detection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/yolov8-web-detection/discussions)
- **Documentation**: [Wiki](https://github.com/your-username/yolov8-web-detection/wiki)

---

**⭐ Star this repository if you find it helpful!**
