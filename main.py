#!/usr/bin/env python3
"""
YOLOv8 Flask Object Detection - Main Entry Point
"""

# Import the Flask app from app.py
from app import app

if __name__ == '__main__':
    print("🎯 Starting YOLOv8 Flask Object Detection App...")
    print("📱 Open your browser to: http://localhost:5000")
    print("🔧 Press Ctrl+C to stop the server")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")