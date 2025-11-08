from flask import Flask, render_template, request, jsonify, Response
from ultralytics import YOLO
import cv2
import os
import json
import base64
from datetime import datetime
import threading

VERSION = "1.0.2"

app = Flask(__name__)

class CustomerTrainingSystem:
    def __init__(self):
        self.data_file = "customers.json"
        self.upload_folder = "training_images"
        self.model = None
        self.camera = None
        self.current_customer_id = None
        self.customers = self.load_customers()
        self.capturing = False
        
        if not os.path.exists(self.upload_folder):
            os.makedirs(self.upload_folder)
    
    def load_customers(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_customers(self):
        with open(self.data_file, 'w') as f:
            json.dump(self.customers, f, indent=2)
    
    def add_customer(self, name, email):
        customer_id = str(len(self.customers) + 1)
        customer_folder = os.path.join(self.upload_folder, customer_id)
        if not os.path.exists(customer_folder):
            os.makedirs(customer_folder)
        
        self.customers[customer_id] = {
            "id": customer_id,
            "name": name,
            "email": email,
            "created_at": datetime.now().isoformat(),
            "image_count": 0,
            "folder": customer_folder
        }
        self.save_customers()
        return customer_id
    
    def get_customers(self):
        return list(self.customers.values())
    
    def start_camera(self, customer_id):
        if customer_id not in self.customers:
            return False
        
        self.current_customer_id = customer_id
        self.capturing = True
        
        if self.model is None:
            self.model = YOLO("yolov8n.pt")
        
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
        
        return True
    
    def stop_camera(self):
        self.capturing = False
        if self.camera:
            self.camera.release()
            self.camera = None
    
    def generate_frames(self):
        while self.capturing and self.camera:
            success, frame = self.camera.read()
            if not success:
                break
            
            results = self.model(frame, imgsz=320, verbose=False)[0]
            annotated = results.plot()
            
            customer = self.customers.get(self.current_customer_id, {})
            cv2.putText(annotated, f"Customer: {customer.get('name', 'N/A')}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated, f"Images: {customer.get('image_count', 0)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated, f"version {VERSION}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            ret, buffer = cv2.imencode('.jpg', annotated)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    def capture_image(self):
        if not self.camera or not self.current_customer_id:
            return False
        
        success, frame = self.camera.read()
        if not success:
            return False
        
        customer = self.customers[self.current_customer_id]
        image_count = customer["image_count"] + 1
        filename = f"training_{image_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(customer["folder"], filename)
        
        cv2.imwrite(filepath, frame)
        
        customer["image_count"] = image_count
        self.save_customers()
        
        return True

system = CustomerTrainingSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/version')
def get_version():
    return jsonify({'version': VERSION})

@app.route('/api/customers', methods=['GET', 'POST'])
def customers():
    if request.method == 'GET':
        return jsonify(system.get_customers())
    
    if request.method == 'POST':
        data = request.json
        customer_id = system.add_customer(data['name'], data['email'])
        return jsonify(system.customers[customer_id]), 201

@app.route('/api/camera/start/<customer_id>', methods=['POST'])
def start_camera(customer_id):
    if system.start_camera(customer_id):
        return jsonify({'status': 'started'})
    return jsonify({'error': 'Customer not found'}), 404

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    system.stop_camera()
    return jsonify({'status': 'stopped'})

@app.route('/api/camera/capture', methods=['POST'])
def capture():
    if system.capture_image():
        customer = system.customers[system.current_customer_id]
        return jsonify({'status': 'captured', 'count': customer['image_count']})
    return jsonify({'error': 'Failed to capture'}), 400

@app.route('/video_feed')
def video_feed():
    return Response(system.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print(f"Volo Customer Training System v{VERSION}")
    print("Server running at http://127.0.0.1:8080")
    app.run(host='127.0.0.1', port=8080, debug=False, threaded=True)

