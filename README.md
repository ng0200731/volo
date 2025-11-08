# Volo Customer Training System

version 1.0.1

YOLO-based customer training image capture system with web interface.

## Quick Start

Double-click START.bat

Or:
```
pip install Flask ultralytics opencv-python pillow
python volo.py
```

Open browser: http://127.0.0.1:8080

## Features

- Web UI interface
- Add customers
- Live webcam with YOLO object detection
- Capture training images per customer
- Images organized by customer ID in training_images folder
- Minimal black and white design

## Usage

1. Add customer (name + email)
2. Select customer from list
3. Click Start Camera
4. Click Capture Image or press SPACE
5. Click Stop Camera when done

## Requirements

Python 3.7+
Webcam
