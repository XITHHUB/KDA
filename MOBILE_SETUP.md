# 📱 Mobile YOLOv8 Object Detection Setup Guide

## How It Works

Your mobile phone (Android/iOS) uses its camera to capture images, sends them to your computer server for AI processing, and receives detection results back in real-time.

```
📱 Phone Camera → 🌐 WiFi → 💻 Server (YOLOv8) → 📊 Results → 📱 Phone Display
```

## Setup Instructions

### 1. Start the Server

```bash
cd /home/fil/Desktop/KDA
source venv/bin/activate
python app.py
```

### 2. Find Your Server IP Address

```bash
# The server will show your IP, or run:
ip addr show | grep inet
# Look for something like: 192.168.x.x
```

### 3. Connect from Mobile Device

**Option A: Same WiFi Network**

- Connect your phone to the same WiFi as your computer
- Open mobile browser (Chrome/Safari)
- Go to: `http://YOUR_IP_ADDRESS:5000/mobile`
- Example: `http://192.168.152.93:5000/mobile`

**Option B: Local Testing**

- If on the same computer: `http://localhost:5000/mobile`

### 4. Using the Mobile Interface

1. **Allow Camera Access**: Browser will ask for camera permissions
2. **Take Photos**: Tap "📸 Detect Objects" to capture and analyze
3. **Switch Cameras**: Tap "🔄 Switch Camera" for front/back camera
4. **View Results**: See detected objects with confidence scores
5. **Live Processing**: Each photo is processed on your server

## Features

### 📸 **Smart Camera Control**

- **Auto Camera Selection**: Starts with back camera (better for plant detection)
- **Camera Switching**: Easy front/back camera toggle
- **High Quality Capture**: Optimized image capture for detection

### ⚡ **Optimized Processing**

- **Async Processing**: Multi-threaded server processing
- **Mobile Optimization**: Images resized for optimal speed
- **Low Latency**: Typically 1-3 seconds per detection
- **Confidence Threshold**: Lower threshold (25%) for better mobile experience

### 🎯 **Detection Features**

- **6 Plant Classes**: Aphids, Healthy, Mosaic virus, Powdery, Rust, cb lp
- **Color-Coded Results**: Different colors for each detection type
- **Confidence Scores**: Percentage confidence for each detection
- **Visual Feedback**: Processed image shown briefly with bounding boxes

### 📊 **Performance Monitoring**

- **Processing Time**: Real-time processing duration
- **Server Stats**: Performance metrics and server status
- **Session Tracking**: Individual mobile sessions

## Troubleshooting

### Camera Issues

- **Allow Permissions**: Enable camera access in browser settings
- **Try Different Browser**: Chrome/Safari work best
- **Check HTTPS**: Some browsers require HTTPS for camera access

### Connection Issues

- **Same WiFi**: Ensure phone and computer on same network
- **Firewall**: Check if port 5000 is blocked
- **IP Address**: Verify correct IP address from server startup

### Performance Issues

- **Good Lighting**: Ensure adequate lighting for detection
- **Stable Connection**: Strong WiFi signal improves speed
- **Close Apps**: Close other apps for better performance

## Technical Details

### Server Processing

- **YOLOv8 Model**: Uses your custom `best.pt` model
- **Multi-threading**: Concurrent image processing
- **Base64 Encoding**: Images sent as base64 data
- **JSON API**: RESTful API for mobile communication

### Mobile Optimization

- **Responsive Design**: Works on all screen sizes
- **Touch Optimized**: Large buttons and touch-friendly interface
- **Offline Indicators**: Clear status of camera and processing
- **Progressive Enhancement**: Works without JavaScript (basic functionality)

### Network Requirements

- **Local WiFi**: Both devices on same network
- **Port 5000**: Server runs on port 5000
- **HTTP Protocol**: Standard web protocol (no special setup needed)

## Advanced Usage

### Multiple Mobile Devices

- Multiple phones can connect simultaneously
- Each gets its own session tracking
- Server handles concurrent processing

### Custom Configuration

- Modify confidence thresholds in `app.py`
- Adjust image quality in mobile processing
- Change processing timeouts for slower devices

### API Endpoints

- `POST /process_mobile_image`: Process mobile images
- `GET /mobile_performance`: Performance metrics
- `GET /model_info`: Model information
