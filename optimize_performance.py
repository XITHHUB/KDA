#!/usr/bin/env python3
"""
Performance Optimization Script for YOLOv8 Flask App
"""

import argparse
import json
import os

def update_app_settings(resolution="640x480", detection_interval=3, target_fps=25, jpeg_quality=85):
    """Update app.py with optimized settings"""
    
    width, height = map(int, resolution.split('x'))
    
    print(f"🔧 Optimizing performance settings:")
    print(f"   📐 Resolution: {width}x{height}")
    print(f"   🎯 Detection interval: every {detection_interval} frames")
    print(f"   📹 Target FPS: {target_fps}")
    print(f"   🖼️  JPEG quality: {jpeg_quality}%")
    
    # Read current app.py
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Update settings
    replacements = [
        ('self.frame_width = 640', f'self.frame_width = {width}'),
        ('self.frame_height = 480', f'self.frame_height = {height}'),
        ('self.detection_interval = 3', f'self.detection_interval = {detection_interval}'),
        ('target_frame_time = 1.0 / 25', f'target_frame_time = 1.0 / {target_fps}'),
        ('encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]', 
         f'encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), {jpeg_quality}]')
    ]
    
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            print(f"   ✅ Updated: {old} -> {new}")
        else:
            print(f"   ⚠️  Not found: {old}")
    
    # Backup original
    if not os.path.exists('app.py.backup'):
        with open('app.py.backup', 'w') as f:
            with open('app.py', 'r') as orig:
                f.write(orig.read())
        print("   💾 Created backup: app.py.backup")
    
    # Write updated version
    with open('app.py', 'w') as f:
        f.write(content)
    
    print("✅ Performance optimization complete!")

def create_performance_configs():
    """Create preset performance configurations"""
    configs = {
        "high_quality": {
            "resolution": "1280x720",
            "detection_interval": 5,
            "target_fps": 20,
            "jpeg_quality": 95,
            "description": "High quality for detailed analysis (slower)"
        },
        "balanced": {
            "resolution": "640x480",
            "detection_interval": 3,
            "target_fps": 25,
            "jpeg_quality": 85,
            "description": "Balanced quality and performance (default)"
        },
        "performance": {
            "resolution": "320x240",
            "detection_interval": 2,
            "target_fps": 30,
            "jpeg_quality": 70,
            "description": "High speed for real-time applications"
        },
        "mobile": {
            "resolution": "480x360",
            "detection_interval": 4,
            "target_fps": 20,
            "jpeg_quality": 75,
            "description": "Optimized for mobile/low-power devices"
        }
    }
    
    with open('performance_configs.json', 'w') as f:
        json.dump(configs, f, indent=2)
    
    print("📋 Created performance_configs.json with presets:")
    for name, config in configs.items():
        print(f"   🎛️  {name}: {config['description']}")

def main():
    parser = argparse.ArgumentParser(description='Optimize YOLOv8 Flask App Performance')
    parser.add_argument('--preset', choices=['high_quality', 'balanced', 'performance', 'mobile'],
                        help='Use a performance preset')
    parser.add_argument('--resolution', default='640x480', help='Camera resolution (e.g., 640x480)')
    parser.add_argument('--detection-interval', type=int, default=3, 
                        help='Run detection every N frames (higher = faster)')
    parser.add_argument('--target-fps', type=int, default=25, help='Target streaming FPS')
    parser.add_argument('--jpeg-quality', type=int, default=85, help='JPEG compression quality (1-100)')
    parser.add_argument('--create-configs', action='store_true', help='Create performance config presets')
    parser.add_argument('--restore', action='store_true', help='Restore from backup')
    
    args = parser.parse_args()
    
    if args.create_configs:
        create_performance_configs()
        return
    
    if args.restore:
        if os.path.exists('app.py.backup'):
            os.rename('app.py.backup', 'app.py')
            print("✅ Restored app.py from backup")
        else:
            print("❌ No backup file found")
        return
    
    # Load preset if specified
    if args.preset:
        if os.path.exists('performance_configs.json'):
            with open('performance_configs.json', 'r') as f:
                configs = json.load(f)
            
            if args.preset in configs:
                config = configs[args.preset]
                print(f"🎛️  Using preset: {args.preset}")
                print(f"   {config['description']}")
                
                update_app_settings(
                    resolution=config['resolution'],
                    detection_interval=config['detection_interval'],
                    target_fps=config['target_fps'],
                    jpeg_quality=config['jpeg_quality']
                )
            else:
                print(f"❌ Preset '{args.preset}' not found")
        else:
            print("❌ No performance_configs.json found. Run with --create-configs first.")
    else:
        # Use manual settings
        update_app_settings(
            resolution=args.resolution,
            detection_interval=args.detection_interval,
            target_fps=args.target_fps,
            jpeg_quality=args.jpeg_quality
        )
    
    print("\n🚀 Restart the app to apply changes:")
    print("   python app.py")

if __name__ == "__main__":
    main()
