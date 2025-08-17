#!/usr/bin/env python3
"""
YOLOv8 Web Detection Setup Test
Quick test to verify all dependencies and functionality
"""

import sys
import subprocess
import importlib
import json
from pathlib import Path

def test_python_version():
    """Test Python version compatibility"""
    print("🐍 Testing Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def test_dependencies():
    """Test required dependencies"""
    print("\n📦 Testing dependencies...")
    
    required = [
        'flask', 'ultralytics', 'opencv-python', 'pillow', 
        'numpy', 'torch', 'psutil'
    ]
    
    missing = []
    for package in required:
        try:
            if package == 'opencv-python':
                importlib.import_module('cv2')
            else:
                importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing.append(package)
    
    return len(missing) == 0, missing

def test_gpu_support():
    """Test GPU support"""
    print("\n🚀 Testing GPU support...")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count()
        
        if cuda_available:
            print(f"✅ CUDA available - {device_count} device(s)")
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                print(f"   GPU {i}: {device_name}")
        else:
            print("⚠️  CUDA not available - Will use CPU")
        
        return True
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def test_camera_access():
    """Test camera access"""
    print("\n📷 Testing camera access...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ Camera accessible - Frame size: {frame.shape}")
                cap.release()
                return True
            else:
                print("❌ Camera accessible but can't read frames")
                cap.release()
                return False
        else:
            print("❌ Cannot access camera")
            return False
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_model_file():
    """Test for model file"""
    print("\n🤖 Testing model file...")
    model_path = Path("best.pt")
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Model file found - Size: {size_mb:.1f} MB")
        
        # Test model loading
        try:
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            print(f"✅ Model loads successfully - {len(model.names)} classes")
            print(f"🏷️  Classes: {list(model.names.values())}")
            return True
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
    else:
        print("❌ Model file 'best.pt' not found")
        print("   Please add your YOLOv8 model as 'best.pt'")
        return False

def test_flask_app():
    """Test Flask app import"""
    print("\n🌐 Testing Flask app...")
    try:
        # Test standard app
        import app
        print("✅ Standard app.py imports successfully")
        
        # Test optimized app if exists
        if Path("app_optimized.py").exists():
            import app_optimized
            print("✅ Optimized app_optimized.py imports successfully")
        
        return True
    except Exception as e:
        print(f"❌ Flask app test failed: {e}")
        return False

def test_config():
    """Test configuration"""
    print("\n⚙️ Testing configuration...")
    try:
        if Path("config.json").exists():
            with open("config.json", 'r') as f:
                config = json.load(f)
            print("✅ Configuration file loaded")
            print(f"   Workers: {config.get('max_workers', 'default')}")
            print(f"   Model: {config.get('model_path', 'best.pt')}")
        else:
            print("⚠️  No config.json found - Will use defaults")
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def generate_report(results):
    """Generate test report"""
    print("\n" + "="*50)
    print("🧪 SETUP TEST REPORT")
    print("="*50)
    
    total_tests = len(results)
    passed_tests = sum(1 for _, status in results.items() if status)
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print("\nTest Results:")
    for test_name, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {test_name}")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Your setup is ready.")
        print("Run: python app.py or python app_optimized.py")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Please fix the issues above.")
    
    return passed_tests == total_tests

def install_missing_packages(missing):
    """Install missing packages"""
    if not missing:
        return
    
    print(f"\n� Installing missing packages: {', '.join(missing)}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        print("✅ Packages installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")

def main():
    """Main test function"""
    print("🔍 YOLOv8 Web Detection - Setup Test")
    print("="*50)
    
    results = {}
    
    # Run tests
    results["Python Version"] = test_python_version()
    
    deps_ok, missing = test_dependencies()
    results["Dependencies"] = deps_ok
    
    if missing:
        install_missing_packages(missing)
    
    results["GPU Support"] = test_gpu_support()
    results["Camera Access"] = test_camera_access()
    results["Model File"] = test_model_file()
    results["Flask App"] = test_flask_app()
    results["Configuration"] = test_config()
    
    # Generate report
    success = generate_report(results)
    
    if success:
        print("\n🚀 Quick Start Commands:")
        print("   python app.py              # Standard version")
        print("   python app_optimized.py    # Optimized version")
        print("   Open: http://localhost:5000")
    
    return success

if __name__ == "__main__":
    main()
