"""
🚦 Windows Traffic Sign Detection - Ready to Run!
- Optimized for Windows laptops
- Uses your trained model: best.pt
- Webcam detection with adaptive classification
- Added contour visualization
"""
import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter, deque
import time
from datetime import datetime
import os

class WindowsTrafficDetector:
    def __init__(self):
        print("🚦 Windows AI Traffic Sign Detector")
        print("=" * 50)
        print("🤖 Loading your trained model...")
        
        # Windows-compatible path to your model
        model_path = r"runs\detect\traffic_sign_detector\weights\best.pt"
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found at: {model_path}")
            print("💡 Make sure you're in the correct directory")
            print(f"💡 Current directory: {os.getcwd()}")
            input("Press Enter to exit...")
            return
            
        try:
            self.model = YOLO(model_path)
            print("✅ Model loaded successfully!")
            print(f"📊 Model size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            input("Press Enter to exit...")
            return
        
        # Detection parameters
        self.detection_history = {}
        self.stable_labels = {}
        self.frame_count = 0
        self.fps_list = []

    def calculate_quality(self, image):
        """Simple image quality assessment"""
        try:
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            brightness = np.mean(gray) / 255.0
            contrast = np.std(gray) / 128.0
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var() / 1000.0
            quality = brightness * 0.4 + min(contrast, 1.0) * 0.3 + min(sharpness, 1.0) * 0.3
            return np.clip(quality, 0.0, 1.0)
        except:
            return 0.5

    def classify_shape(self, img):
        """Windows-optimized shape classification"""
        try:
            if img is None or img.size == 0:
                return "Unknown", 0.0, 0.5
                
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
                
            h, w = img.shape[:2]
            if h < 40 or w < 40:
                img = cv2.resize(img, (80, 80))
                h, w = 80, 80
            
            quality = self.calculate_quality(img)
            
            # Edge detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if quality < 0.5:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                gray = clahe.apply(gray)
            
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            edges = cv2.Canny(blurred, 50, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return "Unknown", 0.0, quality
            
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            
            if area < (h * w * 0.08):
                return "Unknown", 0.0, quality
            
            perimeter = cv2.arcLength(largest, True)
            if perimeter == 0:
                return "Unknown", 0.0, quality
            
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            # Vertex counting
            vertex_counts = []
            epsilons = [0.015, 0.02, 0.025, 0.03] if quality > 0.6 else [0.01, 0.015, 0.02, 0.025, 0.03, 0.035]
            for eps in epsilons:
                try:
                    approx = cv2.approxPolyDP(largest, eps * perimeter, True)
                    vertex_counts.append(len(approx))
                except:
                    pass
            
            if not vertex_counts:
                return "Unknown", 0.0, quality
                
            vertices = Counter(vertex_counts).most_common(1)[0][0]
            vertex_freq = Counter(vertex_counts).most_common(1)[0][1] / len(vertex_counts)
            
            # Color analysis with proper data types
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            if hsv.dtype != np.uint8:
                hsv = hsv.astype(np.uint8)
            
            sat_thresh = 100 if quality > 0.6 else 80
            
            # Red detection
            red_mask1 = cv2.inRange(hsv, np.array([0, sat_thresh, 80], dtype=np.uint8), 
                                   np.array([15, 255, 255], dtype=np.uint8))
            red_mask2 = cv2.inRange(hsv, np.array([165, sat_thresh, 80], dtype=np.uint8), 
                                   np.array([180, 255, 255], dtype=np.uint8))
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            red_ratio = cv2.countNonZero(red_mask) / (h * w)
            
            # Yellow detection
            yellow_mask = cv2.inRange(hsv, np.array([15, sat_thresh, 80], dtype=np.uint8), 
                                     np.array([35, 255, 255], dtype=np.uint8))
            yellow_ratio = cv2.countNonZero(yellow_mask) / (h * w)
            
            print(f"🔍 Analysis: {vertices}v (freq:{vertex_freq:.2f}), circ:{circularity:.3f}, red:{red_ratio:.3f}")
            
            # Adaptive classification logic
            circle_thresh = 0.88 if quality > 0.7 else 0.85 if quality > 0.4 else 0.8
            vertex_thresh = 0.4 if quality > 0.7 else 0.3 if quality > 0.4 else 0.25
            color_thresh = 0.1 if quality > 0.7 else 0.08 if quality > 0.4 else 0.05
            
            # Circle detection
            if circularity > circle_thresh:
                conf = 0.85 + (circularity - circle_thresh) * 0.3
                return "Circle", min(conf, 0.95), quality
            
            # Triangle detection
            if vertices == 3 and vertex_freq >= vertex_thresh:
                conf = 0.8 + (vertex_freq - vertex_thresh) * 0.4
                if yellow_ratio > color_thresh:
                    conf += 0.1
                return "Triangle", min(conf, 0.95), quality
            
            # Rectangle detection
            if (vertices == 4 and vertex_freq >= vertex_thresh and 
                circularity < circle_thresh - 0.1 and red_ratio < color_thresh):
                
                x, y, w_box, h_box = cv2.boundingRect(largest)
                aspect = w_box / h_box if h_box != 0 else 1
                
                if aspect > 1.15:  # Must be rectangular
                    conf = 0.75 + (vertex_freq - vertex_thresh) * 0.3
                    if aspect > 1.3:
                        conf += 0.1
                    return "Rectangle", min(conf, 0.9), quality
            
            # Octagon detection
            is_multisided = vertices >= 5 and vertex_freq >= vertex_thresh * 0.8
            is_red = red_ratio > color_thresh
            
            if is_multisided or is_red:
                conf = 0.7
                if vertices == 8:
                    conf += 0.1
                if red_ratio > 2 * color_thresh:
                    conf += 0.15
                return "Octagon", min(conf, 0.95), quality
            
            # Fallbacks
            if circularity > 0.8:
                return "Circle", 0.7, quality
            elif vertices == 4:
                return "Rectangle", 0.65, quality
            elif red_ratio > 0.5 * color_thresh:
                return "Octagon", 0.65, quality
            else:
                return "Unknown", 0.0, quality
                
        except Exception as e:
            print(f"Classification error: {e}")
            return "Unknown", 0.0, 0.5

    def update_history(self, track_id, shape, confidence, quality):
        """Update detection history for stability"""
        current_time = time.time()
        
        if track_id not in self.detection_history:
            self.detection_history[track_id] = deque(maxlen=10)
        
        self.detection_history[track_id].append({
            'shape': shape,
            'confidence': confidence,
            'timestamp': current_time,
            'quality': quality
        })
        
        history = self.detection_history[track_id]
        
        # Remove old entries
        cutoff = current_time - (3.0 if quality > 0.6 else 4.0)
        while history and history[0]['timestamp'] < cutoff:
            history.popleft()
        
        # Need at least 3 predictions
        if len(history) < 3:
            return "Analyzing...", 0.0, False
        
        # Vote for most common shape
        shapes = [pred['shape'] for pred in history]
        votes = Counter(shapes)
        best_shape = votes.most_common(1)[0][0]
        vote_ratio = votes.most_common(1)[0][1] / len(shapes)
        
        # Need 60% agreement
        if vote_ratio >= 0.6 and best_shape not in ["Unknown", "Analyzing..."]:
            avg_conf = np.mean([pred['confidence'] for pred in history if pred['shape'] == best_shape])
            return best_shape, avg_conf, True
        
        return "Analyzing...", 0.0, False

    def run_detection(self):
        """Main Windows detection loop"""
        print("\n🚀 Starting Windows Traffic Sign Detection")
        print("=" * 50)
        print("📹 Initializing webcam...")
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot access webcam!")
            print("💡 Make sure:")
            print("   - Webcam is connected")
            print("   - No other apps are using camera")
            print("   - Try different camera index (1, 2, etc.)")
            input("Press Enter to exit...")
            return
        
        # Set webcam properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Webcam ready!")
        print("\n📱 CONTROLS:")
        print("   Q - Quit detection")
        print("   S - Save screenshot")
        print("   R - Reset detection memory")
        print("\n🎯 Point camera at traffic signs and wait for analysis...")
        
        colors = {
            'Triangle': (0, 255, 255),      # Yellow
            'Rectangle': (255, 0, 0),       # Blue  
            'Circle': (255, 255, 0),        # Cyan
            'Octagon': (0, 0, 255),         # Red
            'Analyzing...': (255, 165, 0),  # Orange
            'Unknown': (128, 128, 128)      # Gray
        }
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Frame read failed")
                    continue
                
                self.frame_count += 1
                
                try:
                    # YOLO detection with tracking
                    results = self.model.track(frame, conf=0.5, persist=True, verbose=False)
                    
                    detections = []
                    
                    if results[0].boxes is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        scores = results[0].boxes.conf.cpu().numpy()
                        
                        # Get tracking IDs
                        if results[0].boxes.id is not None:
                            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                        else:
                            track_ids = list(range(len(boxes)))
                        
                        for box, score, track_id in zip(boxes, scores, track_ids):
                            if score > 0.6:  # High confidence for Windows
                                x1, y1, x2, y2 = map(int, box)
                                
                                # Add padding
                                pad = 10
                                x1, y1 = max(0, x1-pad), max(0, y1-pad)
                                x2 = min(frame.shape[1], x2+pad)
                                y2 = min(frame.shape[0], y2+pad)
                                
                                if x2 > x1 + 30 and y2 > y1 + 30:
                                    # Classify shape
                                    cropped = frame[y1:y2, x1:x2]
                                    shape, confidence, quality = self.classify_shape(cropped)
                                    
                                    # Update history
                                    stable_shape, stable_conf, is_stable = self.update_history(
                                        track_id, shape, confidence, quality
                                    )
                                    
                                    detections.append({
                                        'box': (x1, y1, x2, y2),
                                        'shape': stable_shape,
                                        'confidence': stable_conf,
                                        'is_stable': is_stable,
                                        'track_id': track_id,
                                        'quality': quality
                                    })
                    
                    # Draw results
                    for det in detections:
                        x1, y1, x2, y2 = det['box']
                        shape = det['shape']
                        conf = det['confidence']
                        is_stable = det['is_stable']
                        quality = det['quality']
                        
                        color = colors.get(shape, (0, 255, 0))
                        thickness = 4 if is_stable else 2
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                        
                        # Draw contours within bounding box
                        try:
                            roi = frame[y1:y2, x1:x2].copy()
                            if roi.size > 0:
                                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                                blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
                                edges_roi = cv2.Canny(blurred_roi, 30, 100)
                                contours_roi, _ = cv2.findContours(edges_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                cv2.drawContours(roi, contours_roi, -1, (0, 255, 0), 2)
                                frame[y1:y2, x1:x2] = roi
                        except:
                            pass
                        
                        # Draw label
                        quality_symbol = "★" if quality > 0.7 else "◐" if quality > 0.4 else "○"
                        status = "✓STABLE" if is_stable else "⏳ANALYZING"
                        label = f"{shape} {status} {quality_symbol}"
                        
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        
                        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, 2)
                        cv2.rectangle(frame, (x1, y1-text_h-10), (x1+text_w, y1), color, -1)
                        cv2.putText(frame, label, (x1, y1-5), font, font_scale, (255, 255, 255), 2)
                    
                    # Performance info
                    fps = 1.0 / (time.time() - start_time) if time.time() > start_time else 0
                    self.fps_list.append(fps)
                    if len(self.fps_list) > 30:
                        self.fps_list.pop(0)
                    
                    avg_fps = sum(self.fps_list) / len(self.fps_list) if self.fps_list else 0
                    stable_count = sum(1 for d in detections if d['is_stable'])
                    
                    # Info overlay
                    info_text = f"FPS: {avg_fps:.1f} | STABLE: {stable_count} | TOTAL: {len(detections)} | Frame: {self.frame_count}"
                    cv2.putText(frame, info_text, (10, frame.shape[0]-15), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Show window
                    cv2.imshow('🚦 Windows Traffic Sign Detector - Press Q to Quit', frame)
                    
                    # Print stable detections
                    for det in detections:
                        if det['is_stable']:
                            print(f"🎯 DETECTED: {det['shape']} (confidence: {det['confidence']:.3f}) ID:{det['track_id']}")
                
                except Exception as e:
                    print(f"Detection error: {e}")
                    cv2.imshow('🚦 Windows Traffic Sign Detector - Press Q to Quit', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
                    break
                elif key == ord('s') or key == ord('S'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f'detection_screenshot_{timestamp}.jpg'
                    cv2.imwrite(filename, frame)
                    print(f"📸 Screenshot saved: {filename}")
                elif key == ord('r') or key == ord('R'):
                    self.detection_history.clear()
                    self.stable_labels.clear()
                    print("🔄 Detection memory reset")
        
        except KeyboardInterrupt:
            print("\n⏹️ Detection stopped by user")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\n🎯 Windows Traffic Sign Detection Complete!")
            print("👋 Thanks for testing!")

def main():
    """Main function"""
    print("🚦 WINDOWS AI TRAFFIC SIGN DETECTOR")
    print("=" * 50)
    print("🖥️  Optimized for Windows laptops")
    print("🎯 Uses your trained YOLOv8 model")
    print("📹 Real-time webcam detection")
    print("🧠 Adaptive shape classification")
    print("=" * 50)
    
    try:
        detector = WindowsTrafficDetector()
        detector.run_detection()
    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
    except Exception as e:
        print(f"❌ Startup error: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
