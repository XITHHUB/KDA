#!/usr/bin/env python3
"""
Camera Troubleshooting Script for YOLOv8 Flask App
"""

import cv2
import os
import subprocess
import sys

def check_camera_permissions():
    """Check if user has permission to access video devices"""
    print("🔐 Checking camera permissions...")
    
    video_devices = ['/dev/video0', '/dev/video1']
    for device in video_devices:
        if os.path.exists(device):
            if os.access(device, os.R_OK):
                print(f"✅ {device} - Read permission OK")
            else:
                print(f"❌ {device} - No read permission")
                print(f"💡 Try: sudo chmod 666 {device}")
        else:
            print(f"⚠️  {device} - Device not found")

def check_video_group():
    """Check if user is in video group"""
    print("\n👥 Checking user groups...")
    try:
        result = subprocess.run(['groups'], capture_output=True, text=True)
        groups = result.stdout.strip()
        if 'video' in groups:
            print("✅ User is in 'video' group")
        else:
            print("❌ User is NOT in 'video' group")
            print("💡 Add user to video group: sudo usermod -a -G video $USER")
            print("💡 Then logout and login again")
    except Exception as e:
        print(f"❌ Error checking groups: {e}")

def test_opencv_cameras():
    """Test camera access with different OpenCV backends"""
    print("\n📹 Testing camera access...")
    
    backends = [
        ("Default", cv2.CAP_ANY),
        ("V4L2", cv2.CAP_V4L2),
        ("GStreamer", cv2.CAP_GSTREAMER),
    ]
    
    for camera_id in [0, 1, 2]:
        print(f"\n🎥 Testing camera index {camera_id}:")
        
        for name, backend in backends:
            try:
                cap = cv2.VideoCapture(camera_id, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        height, width = frame.shape[:2]
                        print(f"✅ {name} backend - Camera {camera_id} working ({width}x{height})")
                        cap.release()
                        return camera_id, backend  # Return first working camera
                    else:
                        print(f"⚠️  {name} backend - Camera {camera_id} opened but cannot read frames")
                else:
                    print(f"❌ {name} backend - Camera {camera_id} failed to open")
                cap.release()
            except Exception as e:
                print(f"❌ {name} backend - Error: {e}")
    
    return None, None

def check_camera_processes():
    """Check if other processes are using the camera"""
    print("\n🔍 Checking for camera-using processes...")
    try:
        result = subprocess.run(['lsof', '/dev/video0'], capture_output=True, text=True)
        if result.stdout:
            print("⚠️  Processes using /dev/video0:")
            print(result.stdout)
        else:
            print("✅ No processes currently using /dev/video0")
    except Exception:
        # lsof might not be available
        try:
            result = subprocess.run(['fuser', '/dev/video0'], capture_output=True, text=True)
            if result.stdout:
                print("⚠️  Process IDs using /dev/video0:", result.stdout.strip())
            else:
                print("✅ No processes currently using /dev/video0")
        except Exception:
            print("⚠️  Cannot check camera usage (install lsof or fuser)")

def fix_camera_issues():
    """Apply common camera fixes"""
    print("\n🔧 Applying camera fixes...")
    
    fixes = [
        "sudo modprobe uvcvideo",  # Load USB Video Class driver
        "sudo chmod 666 /dev/video*",  # Fix permissions
    ]
    
    for fix in fixes:
        print(f"Running: {fix}")
        try:
            result = subprocess.run(fix.split(), capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Success")
            else:
                print(f"❌ Failed: {result.stderr}")
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    print("🎥 Camera Troubleshooting for YOLOv8 Flask App")
    print("=" * 50)
    
    # Basic checks
    check_camera_permissions()
    check_video_group()
    check_camera_processes()
    
    # Test cameras
    working_camera, working_backend = test_opencv_cameras()
    
    if working_camera is not None:
        print(f"\n🎉 Found working camera: {working_camera} with backend {working_backend}")
        print(f"💡 Update app.py to use camera {working_camera}")
    else:
        print("\n❌ No working cameras found!")
        print("\n🔧 Trying automatic fixes...")
        fix_camera_issues()
        
        # Test again after fixes
        print("\n🔄 Testing cameras again after fixes...")
        working_camera, working_backend = test_opencv_cameras()
        
        if working_camera is not None:
            print(f"🎉 Camera fixed! Using camera {working_camera}")
        else:
            print("\n❌ Still no working cameras. Try:")
            print("1. Reconnect your USB camera")
            print("2. Try a different USB port")
            print("3. Check if camera works in other apps (cheese, vlc)")
            print("4. Restart your computer")

if __name__ == "__main__":
    main()
