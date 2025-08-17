#!/bin/bash

# YOLOv8 Flask Object Detection Startup Script

echo "🚀 Starting YOLOv8 Flask Object Detection App..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Check if best.pt model exists
if [ ! -f "best.pt" ]; then
    echo "❌ Model file 'best.pt' not found!"
    echo "   Please place your YOLOv8 model file in this directory"
    exit 1
fi

# Activate virtual environment and start app
echo "📦 Activating virtual environment..."
source venv/bin/activate

echo "🎯 Loading YOLOv8 model and starting Flask app..."
python app.py
