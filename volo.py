from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from ultralytics import YOLO
import cv2
import os
import json
import base64
from datetime import datetime
import threading

VERSION = "1.0.28"

app = Flask(__name__)

class CustomerTrainingSystem:
    def __init__(self):
        self.data_file = "customers.json"
        self.upload_folder = "training_images"
        self.model = None
        self.camera = None
        self.current_customer_id = None
        self.customers = self.load_customers()
        self._migrate_customers_schema()
        self.capturing = False
        self.testing = False
        self.face_recognizer = None
        self.label_map = {}
        self.image_cache = {}
        self.recognition_history = []
        
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

    def _migrate_customers_schema(self):
        """Ensure each customer has 'folder' and accurate 'image_count'."""
        changed = False
        for cid, customer in list(self.customers.items()):
            # Ensure 'id' string key
            if 'id' not in customer:
                customer['id'] = str(cid)
                changed = True
            # Ensure folder
            if 'folder' not in customer or not customer['folder']:
                customer_folder = os.path.join(self.upload_folder, str(customer['id']))
                if not os.path.exists(customer_folder):
                    os.makedirs(customer_folder)
                customer['folder'] = customer_folder
                changed = True
            # Recount images on disk
            folder = customer.get('folder', '')
            count = 0
            if folder and os.path.exists(folder):
                count = len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
            if customer.get('image_count') != count:
                customer['image_count'] = count
                changed = True
        if changed:
            self.save_customers()
    
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

    def _train_face_recognizer(self):
        import cv2
        import numpy as np
        if not hasattr(cv2, "face"):
            return False, "OpenCV contrib modules missing (cv2.face). Run: pip uninstall -y opencv-python opencv-python-headless && pip install opencv-contrib-python"
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        images = []
        labels = []
        label_map = {}
        label_id = 0
        for cid, customer in self.customers.items():
            folder = customer["folder"]
            if not os.path.exists(folder):
                continue
            face_found = False
            for fn in os.listdir(folder):
                path = os.path.join(folder, fn)
                if not os.path.isfile(path):
                    continue
                img = cv2.imread(path)
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                for (x, y, w, h) in faces:
                    face = gray[y:y+h, x:x+w]
                    face = cv2.resize(face, (200, 200))
                    images.append(face)
                    labels.append(label_id)
                    face_found = True
            if face_found:
                label_map[label_id] = cid
                label_id += 1
        if images:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(images, np.array(labels))
            self.face_recognizer = recognizer
            self.label_map = label_map
            return True, None
        self.face_recognizer = None
        self.label_map = {}
        return False, "No faces found in training images"

    def _load_image_cache(self):
        import cv2
        cache = {}
        orb = cv2.ORB_create(nfeatures=500)
        for cid, customer in self.customers.items():
            folder = customer["folder"]
            if not os.path.exists(folder):
                continue
            entries = []
            for fn in os.listdir(folder):
                path = os.path.join(folder, fn)
                if not os.path.isfile(path):
                    continue
                img = cv2.imread(path)
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (320, 240))
                kp, des = orb.detectAndCompute(gray, None)
                if des is not None and len(kp) > 0:
                    entries.append((kp, des))
            if entries:
                cache[cid] = entries
        self.image_cache = cache
        return len(cache) > 0

    def _recognize_customer_id(self, frame_bgr):
        import cv2
        # Try face recognition first if available
        if self.face_recognizer:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) > 0:
                best_label = None
                best_conf = float('inf')
                for (x, y, w, h) in faces:
                    face = gray[y:y+h, x:x+w]
                    if face.size == 0:
                        continue
                    face = cv2.resize(face, (200, 200))
                    label, confidence = self.face_recognizer.predict(face)
                    if confidence < best_conf:
                        best_conf = confidence
                        best_label = label
                if best_label is not None and best_label in self.label_map and best_conf <= 70:
                    matched_id = self.label_map[best_label]
                    return matched_id, matched_id, best_conf
        # Fallback: full image matching using ORB (works for signs, objects, etc)
        if not self.image_cache:
            self._load_image_cache()
        if not self.image_cache:
            # Try to train face recognizer as last resort
            if not self.face_recognizer:
                trained, msg = self._train_face_recognizer()
                if not trained:
                    self.last_train_error = msg
            return None, None, None
        orb = cv2.ORB_create(nfeatures=500)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 240))
        kp_frame, des_frame = orb.detectAndCompute(gray, None)
        if des_frame is None:
            return None, None, None
        best_cid = None
        best_score = 0
        best_conf = None
        for cid, entries in self.image_cache.items():
            score_sum = 0
            count = 0
            for kp_train, des_train in entries[:10]:
                matches = bf.match(des_frame, des_train)
                if not matches:
                    continue
                matches = sorted(matches, key=lambda m: m.distance)
                good = [m for m in matches[:50] if m.distance < 60]
                score_sum += len(good)
                count += 1
            if count > 0:
                avg = score_sum / count
                if avg > best_score:
                    best_score = avg
                    best_cid = cid
                    best_conf = 100 - min(100, avg * 2)
        if best_score >= 15:
            return best_cid, best_cid, best_conf
        return None, None, None

    def start_testing(self):
        import cv2
        self.last_train_error = None
        if self.camera is None:
            cam = cv2.VideoCapture(0)
            if not cam.isOpened():
                return False, "Unable to access camera"
            self.camera = cam
        trained, msg = self._train_face_recognizer()
        # Also load image cache for fallback matching (signs, objects, etc)
        has_images = self._load_image_cache()
        if not trained and not has_images:
            # release camera if we are not going to use it
            self.camera.release()
            self.camera = None
            return False, msg or "No training images available"
        self.recognition_history = []
        self.testing = True
        return True, None

    def stop_testing(self):
        self.testing = False
        if self.camera:
            self.camera.release()
            self.camera = None

    def generate_test_frames(self):
        while self.testing and self.camera:
            success, frame = self.camera.read()
            if not success:
                break
            # Run YOLO for visualization boxes
            if self.model is None:
                self.model = YOLO("yolov8n.pt")
            results = self.model(frame, imgsz=320, verbose=False)[0]
            annotated = results.plot()
            # Recognize customer from frame content
            cid, best_id, best_conf = self._recognize_customer_id(frame)
            if cid:
                self.recognition_history.append(cid)
            else:
                self.recognition_history.append(None)
            if len(self.recognition_history) > 10:
                self.recognition_history.pop(0)
            label = None
            if self.recognition_history:
                counts = {}
                for item in self.recognition_history:
                    if item:
                        counts[item] = counts.get(item, 0) + 1
                if counts:
                    top_id = max(counts, key=counts.get)
                    if counts[top_id] >= max(3, len(self.recognition_history)//2):
                        label = self.customers.get(top_id, {}).get("name")
            display_text = "Customer: Unknown"
            if label:
                display_text = f"Customer: {label}"
            elif best_id and best_id in self.customers:
                guess_name = self.customers[best_id]["name"]
                conf_txt = f"{best_conf:.1f}" if best_conf is not None else "-"
                display_text = f"Best Guess: {guess_name} (conf {conf_txt})"
            import cv2
            cv2.putText(annotated, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            import cv2
            cv2.putText(annotated, f"version {VERSION}", (10, annotated.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            ret, buffer = cv2.imencode('.jpg', annotated)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
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

    def list_images(self, customer_id: str):
        customer = self.customers.get(customer_id)
        if not customer:
            return []
        folder = customer["folder"]
        if not os.path.exists(folder):
            return []
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        files.sort()
        return files

    def delete_image(self, customer_id: str, filename: str) -> bool:
        customer = self.customers.get(customer_id)
        if not customer:
            return False
        folder = customer["folder"]
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                return False
            # refresh count from disk to be accurate
            customer["image_count"] = len(self.list_images(customer_id))
            self.save_customers()
            return True
        return False

    def delete_all_images(self, customer_id: str) -> int:
        customer = self.customers.get(customer_id)
        if not customer:
            return 0
        folder = customer["folder"]
        if not os.path.exists(folder):
            return 0
        deleted = 0
        for fn in os.listdir(folder):
            path = os.path.join(folder, fn)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                    deleted += 1
                except OSError:
                    pass
        customer["image_count"] = len(self.list_images(customer_id))
        self.save_customers()
        return deleted

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

@app.route('/api/images/<customer_id>')
def get_images(customer_id):
    return jsonify(system.list_images(customer_id))

@app.route('/images/<customer_id>/<path:filename>')
def serve_image(customer_id, filename):
    customer = system.customers.get(customer_id)
    if not customer:
        return jsonify({"error": "Customer not found"}), 404
    folder = customer["folder"]
    return send_from_directory(folder, filename)

@app.route('/api/images/<customer_id>/<path:filename>', methods=['DELETE'])
def delete_image(customer_id, filename):
    ok = system.delete_image(customer_id, filename)
    if ok:
        return jsonify({"status": "deleted"})
    return jsonify({"error": "Delete failed"}), 400

@app.route('/api/images/<customer_id>', methods=['DELETE'])
def delete_all_images(customer_id):
    n = system.delete_all_images(customer_id)
    return jsonify({"status": "deleted_all", "deleted": n})

@app.route('/api/test/start', methods=['POST'])
def start_test():
    ok, msg = system.start_testing()
    if ok:
        return jsonify({'status': 'started'})
    if not msg and getattr(system, 'last_train_error', None):
        msg = system.last_train_error
    return jsonify({'error': msg or 'failed'}), 400

@app.route('/api/test/stop', methods=['POST'])
def stop_test():
    system.stop_testing()
    return jsonify({'status': 'stopped'})

@app.route('/video_test')
def video_test():
    return Response(system.generate_test_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print(f"Volo Customer Training System v{VERSION}")
    print("Server running at http://127.0.0.1:8080")
    print("Auto-reload enabled - changes will update automatically")
    app.run(host='127.0.0.1', port=8080, debug=True, threaded=True, use_reloader=True)

