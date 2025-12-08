"""
AUPPBOT FACE TRACKING SYSTEM v3.2
==============================================================
Combined version featuring:
- High-Torque Motor Config (Fixes friction stall)
- Smart Reacquisition (Remembers last position)
- Mixer Control (Smooth curving)
- 320x320 Input (Fixes ONNX crash)
- Cyberpunk HUD (Visual upgrade)
"""

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, Response, render_template_string
import threading
import time
import sys
import os

# Add Documents/Documents to path for auppbot import
sys.path.insert(0, '/home/aupp/Documents/Documents')

# Import robot control
try:
    from auppbot import AUPPBot
    ROBOT_AVAILABLE = True
    print("✓ AUPPBot module loaded")
except ImportError as e:
    ROBOT_AVAILABLE = False
    print(f"⚠ Robot control not available: {e}")
    print("  Running in SIMULATION mode")

# ============================================================================
# CONFIGURATION - HIGH POWER & RESPONSIVE
# ============================================================================

# Motor speed settings (High Power to overcome friction)
BASE_SPEED_FORWARD = 70
BASE_SPEED_TURN = 80 
SPEED_SEARCH = 60

# Proportional control gains
TURN_GAIN = 1.0
DISTANCE_GAIN = 1.0

# Minimum speeds (Prevent stalling)
MIN_TURN_SPEED = 45          
MIN_FORWARD_SPEED = 60       

# Tracking logic
TARGET_LOCK_FRAMES = 15
DETECTION_INTERVAL = 4
SEARCH_THRESHOLD = 30
SEARCH_DIRECTION_SWAP = 60   

# Thresholds
HORIZONTAL_TOLERANCE = 0.20
MIN_FACE_SIZE = 0.15
MAX_FACE_SIZE = 0.35         

# Detection
MIN_CONFIDENCE = 0.5
IOU_THRESHOLD = 0.4

# ============================================================================
# GLOBAL STATE VARIABLES
# ============================================================================
app = Flask(__name__)

session = None
cap = None
current_frame = None
current_detections = []
bot = None

# Tracking State
locked_target_id = None
frames_since_target_seen = 0
next_face_id = 1
previous_faces = []
last_known_position = 0

# Robot State
robot_state = "IDLE"
frames_without_any_face = 0
search_direction = 1  

# ============================================================================
# HARDWARE INIT
# ============================================================================

def init_robot():
    global bot
    if not ROBOT_AVAILABLE: return False
    
    ports = ['/dev/ttyUSB0', '/dev/ttyACM0', '/dev/serial0']
    for port in ports:
        try:
            print(f"Trying {port}...")
            bot = AUPPBot(port=port, baud=115200, auto_safe=True)
            time.sleep(1.0); bot.stop_all(); time.sleep(0.2)
            print(f"✓ Connected on {port}")
            return True
        except: continue
    print("✗ Hardware failed"); return False

# ============================================================================
# MOTOR CONTROL (MIXER LOGIC)
# ============================================================================

def set_motors_smooth(left, right):
    if bot:
        try:
            # Clamp to +/- 100
            l = max(-100, min(100, int(left)))
            r = max(-100, min(100, int(right)))
            bot.motor1.speed(l); bot.motor2.speed(l)
            bot.motor3.speed(r); bot.motor4.speed(r)
        except: pass
    else:
        print(f"[SIM] L:{int(left)} R:{int(right)}")

def stop_robot():
    if bot:
        try: bot.stop_all()
        except: pass

def control_robot_mixer(target, w, h):
    """
    STEP MODE: Moves in tiny 'inches' or 'nudges'.
    Moves for 0.05s then stops immediately.
    """
    if target is None:
        stop_robot()
        return "NO TARGET"
    
    # 1. Calculate Size
    x1, y1, x2, y2 = target['box']
    face_width = x2 - x1
    face_height = y2 - y1
    area_pct = (face_width * face_height) / (w * h)
    
    # 2. Step Logic
    # Only move if face is significantly small (< 25%)
    if area_pct < 0.25:
        # A. Move gently (Speed 45)
        set_motors_smooth(45, 45)
        
        # B. The "Inch" - Move for a tiny moment
        time.sleep(0.05) 
        
        # C. The Brake - Stop immediately
        stop_robot()
        
        return "NUDGE ⬆"
    else:
        # Face is close enough
        stop_robot()
        return "WAITING (CLOSE)"

# ============================================================================
# VISION PIPELINE
# ============================================================================

print("Loading YOLOv8n-face model...")
# Robust path finding
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "yolov8n-face.onnx")

try:
    session = ort.InferenceSession(model_path)
    print(f"✓ Model loaded from {model_path}")
except:
    print("Trying backup path...")
    try:
        session = ort.InferenceSession("/home/aupp/Documents/SE/yolov8n-face.onnx")
    except:
        print("❌ Model not found!"); exit(1)

def detect_faces(frame):
    h, w = frame.shape[:2]
    
    img = cv2.resize(frame, (320, 320))
    img = img[..., ::-1] / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0).astype(np.float32)
    
    output = session.run(None, {session.get_inputs()[0].name: img})[0]
    preds = output[0]
    if preds.shape[1] > preds.shape[0]: preds = preds.T
    
    dets = []
    for p in preds:
        conf = p[4] if len(p) > 4 else p[4:].max()
        if conf < MIN_CONFIDENCE: continue
        
        # Scale back up
        x1 = int((p[0] - p[2]/2) / 320 * w)
        y1 = int((p[1] - p[3]/2) / 320 * h)
        x2 = int((p[0] + p[2]/2) / 320 * w)
        y2 = int((p[1] + p[3]/2) / 320 * h)
        
        dets.append({'box': [x1, y1, x2, y2], 'confidence': float(conf)})
    
    # NMS
    dets = sorted(dets, key=lambda x: x['confidence'], reverse=True)
    final = []
    while dets:
        curr = dets.pop(0)
        final.append(curr)
        dets = [d for d in dets if calculate_iou(curr['box'], d['box']) < IOU_THRESHOLD]
    
    return final[:3]

def calculate_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = ((b1[2]-b1[0])*(b1[3]-b1[1])) + ((b2[2]-b2[0])*(b2[3]-b2[1])) - inter
    return inter/union if union > 0 else 0

# ============================================================================
# LOGIC & TRACKING
# ============================================================================

def assign_ids(detections, previous):
    global next_face_id
    tracked = []
    for det in detections:
        best_id, best_iou = None, 0.3
        for prev in previous:
            iou = calculate_iou(det['box'], prev['box'])
            if iou > best_iou: best_iou = iou; best_id = prev['id']
        
        new_id = best_id if best_id is not None else next_face_id
        if best_id is None: next_face_id += 1
        tracked.append({'id': new_id, 'box': det['box'], 'confidence': det['confidence']})
    return tracked

def update_target_lock(faces, w):
    global locked_target_id, frames_since_target_seen, last_known_position
    
    # AUTO-RELOCK: If searching and see a face, grab the largest one
    if locked_target_id is None and len(faces) > 0:
        best_face = max(faces, key=lambda f: (f['box'][2]-f['box'][0])*(f['box'][3]-f['box'][1]))
        locked_target_id = best_face['id']
        frames_since_target_seen = 0
        return best_face

    if locked_target_id is not None:
        target = next((f for f in faces if f['id'] == locked_target_id), None)
        if target:
            frames_since_target_seen = 0
            # Update memory of where they are (-1 Left, +1 Right)
            cx = (target['box'][0] + target['box'][2]) / 2
            last_known_position = (cx - (w/2)) / (w/2)
            return target
        else:
            frames_since_target_seen += 1
            if frames_since_target_seen >= TARGET_LOCK_FRAMES:
                locked_target_id = None
                frames_since_target_seen = 0
            return None
    return None

def process_frames():
    global current_frame, current_detections, previous_faces
    global robot_state, frames_without_any_face, search_direction
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 640); cap.set(4, 480); cap.set(5, 30)
    
    fc = 0
    tracked_faces = []
    
    while True:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.rotate(frame, cv2.ROTATE_180) # Keep this unless camera is inverted
        h, w = frame.shape[:2]
        
        if fc % DETECTION_INTERVAL == 0:
            dets = detect_faces(frame)
            tracked_faces = assign_ids(dets, previous_faces)
            previous_faces = tracked_faces
            target = update_target_lock(tracked_faces, w)
            
            if target:
                frames_without_any_face = 0
                robot_state = control_robot_mixer(target, w, h)
            elif locked_target_id is None:
                frames_without_any_face += 1
                if frames_without_any_face > SEARCH_THRESHOLD:
                    direction = -1 if last_known_position < 0 else 1
                    if direction == 1: # Search Right
                        set_motors_smooth(SPEED_SEARCH, -SPEED_SEARCH)
                        robot_state = "SEARCHING RIGHT"
                    else: # Search Left
                        set_motors_smooth(-SPEED_SEARCH, SPEED_SEARCH)
                        robot_state = "SEARCHING LEFT"
                else:
                    stop_robot()
                    robot_state = "WAITING..."
            else:
                stop_robot()
                robot_state = "TARGET OCCLUDED"

        fc += 1
        
        # Draw Overlay
        for f in tracked_faces:
            is_lock = (f['id'] == locked_target_id)
            col = (0, 255, 0) if is_lock else (0, 255, 255)
            x1,y1,x2,y2 = f['box']
            cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
            cv2.putText(frame, f"ID:{f['id']}", (x1, y1-10), 0, 0.6, col, 2)
            
        current_frame = frame
        current_detections = tracked_faces
        time.sleep(0.01)

# ============================================================================
# FLASK WEB INTERFACE
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html><html><head><title>AUPPBot HUD v3.2</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;900&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<style>
:root { --p: #00ff41; --s: #008F11; --bg: #050a05; }
body { margin:0; padding:20px; background: #000; color: var(--p); font-family: 'Share Tech Mono', monospace; text-align: center; }
.container { max-width: 800px; margin: 0 auto; }
header { border-bottom: 2px solid var(--s); padding-bottom: 10px; margin-bottom: 20px; display:flex; justify-content:space-between;}
h1 { font-family: 'Orbitron'; margin:0; font-size: 24px; }
.video-frame { border: 1px solid var(--s); box-shadow: 0 0 20px rgba(0,255,65,0.2); position: relative; }
img { width: 100%; display: block; filter: contrast(1.1); }
.dashboard { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-top: 20px; }
.card { background: rgba(0,20,0,0.7); border: 1px solid var(--s); padding: 10px; }
.val { font-size: 20px; font-weight: bold; display:block; margin-top:5px;}
</style>
<script>
setInterval(()=>{fetch('/status').then(r=>r.json()).then(d=>{
document.getElementById('t').innerText=d.target||'SCANNING';
document.getElementById('s').innerText=d.state;
document.getElementById('f').innerText=d.faces;
})},200);
</script>
</head><body>
<div class="container">
<header><h1>AUPPBOT <span style="font-size:0.5em">v3.2</span></h1><span>● REC</span></header>
<div class="video-frame"><img src="/video_feed"></div>
<div class="dashboard">
<div class="card">LOCK<span class="val" id="t">-</span></div>
<div class="card">STATE<span class="val" id="s">-</span></div>
<div class="card">FACES<span class="val" id="f">0</span></div>
</div>
</div></body></html>
"""

@app.route('/')
def index(): return render_template_string(HTML_TEMPLATE)
@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            if current_frame is not None:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + 
                       cv2.imencode('.jpg', current_frame)[1].tobytes() + b'\r\n')
            time.sleep(0.04)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/status')
def status(): return {'target': locked_target_id, 'state': robot_state, 'faces': len(current_detections)}

if __name__ == '__main__':
    print("🤖 SYSTEM v3.2 STARTING...")
    init_robot()
    threading.Thread(target=process_frames, daemon=True).start()
    
    # Motor Test
    if bot:
        print("TEST: Motors ON (0.5s)")
        bot.motor1.speed(60); bot.motor2.speed(60)
        bot.motor3.speed(60); bot.motor4.speed(60)
        time.sleep(0.5); bot.stop_all()

    # Port 5050 (Mac Compatible)
    app.run(host='0.0.0.0', port=5050, debug=False)