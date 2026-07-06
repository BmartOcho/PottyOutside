import cv2
import requests
import keyboard
import threading
import time
import datetime
import os
import json
import urllib.parse
from collections import deque

# --- Optional: YOLOv8 dog detection (Ultralytics) ---
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

# --- Optional: Windows toast (fallback local notification) ---
try:
    from win10toast import ToastNotifier
    TOAST = ToastNotifier()
except Exception:
    TOAST = None

# =========================
# CONFIG (edit these)
# =========================
CAMERA_IP = "192.168.0.101"
HTTP_PORT = 88
USERNAME = "PuppyCam"
PASSWORD = "8616Calvin!"

# RTSP and CGI auth (URL-encoded only for RTSP URL)
USR = urllib.parse.quote(USERNAME)
PWD = urllib.parse.quote(PASSWORD)
BASE_CGI = f"http://{CAMERA_IP}:{HTTP_PORT}/cgi-bin/CGIProxy.fcgi"
RTSP_URL = f"rtsp://{USR}:{PWD}@{CAMERA_IP}:{HTTP_PORT}/videoMain"

# Recording settings
TARGET_FPS = 20.0  # forced write FPS to avoid RTSP PTS weirdness
OUT_DIR = "foscam_output"
SNAPS = os.path.join(OUT_DIR, "snapshots")
RECS = os.path.join(OUT_DIR, "recordings")
os.makedirs(SNAPS, exist_ok=True)
os.makedirs(RECS, exist_ok=True)

# Detection settings
CONF_THRESH = 0.45          # YOLO confidence
PERSIST_FRAMES = int(1.5 * TARGET_FPS)  # how long a dog must be present (~1.5s)
COOLDOWN_SEC = 30           # Reduced from 120 for testing
MODEL_NAME = "yolov8n.pt"   # tiny; good enough
DOG_CLASS = 16              # COCO 'dog'

# Telegram (optional). If not set, falls back to toast/log.
TELEGRAM_BOT_TOKEN = "8338190440:AAHPj9HfF3bDVCFvCl65dzM5S9M1p-8wf5c"     # e.g., "123456:ABC..."
TELEGRAM_CHAT_ID = "-4985758286"       # e.g., "123456789"

# ROI config file (normalized coords)
ROI_CFG = "roi_config.json"
# =========================


# --- Globals (thread state) ---
current_frame = None
stop_thread = False

# Recording state (protected by a lock)
rec_lock = threading.Lock()
is_recording = False
writer = None
last_size = None
next_frame_due = 0.0

# ROI editor state
roi_edit_mode = False
roi_points = []  # each is (x1, y1, x2, y2) in pixels (temp)
dragging = False
start_pt = None

# Loaded ROIs (normalized 0..1): {"inside":[x1,y1,x2,y2], "outside":[x1,y1,x2,y2]}
rois = {"inside": None, "outside": None}

# Detection + alert state
model = None
last_alert_time = {"inside": 0.0, "outside": 0.0}
presence_buffers = {
    "inside": deque(maxlen=PERSIST_FRAMES),
    "outside": deque(maxlen=PERSIST_FRAMES),
}


# ============== Utilities ==============

def test_telegram_connection():
    """Test if Telegram bot token and chat ID are working."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Telegram credentials not configured")
        return False
    
    print(f"🔍 Testing Telegram connection...")
    print(f"   Bot Token: {TELEGRAM_BOT_TOKEN[:20]}...")
    print(f"   Chat ID: {TELEGRAM_CHAT_ID}")
    
    # Test getMe endpoint first
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe"
        r = requests.get(url, timeout=10)
        print(f"   getMe status: {r.status_code}")
        if r.ok:
            data = r.json()
            if data.get("ok"):
                bot_info = data.get("result", {})
                print(f"   ✅ Bot authenticated: {bot_info.get('first_name', 'Unknown')} (@{bot_info.get('username', 'unknown')})")
            else:
                print(f"   ❌ Bot authentication failed: {data}")
                return False
        else:
            print(f"   ❌ HTTP error: {r.status_code} - {r.text}")
            return False
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
        return False
    
    # Test sending a message
    try:
        test_msg = "🤖 PuppyCam Telegram test - connection successful!"
        success = send_telegram_message(test_msg, test_mode=True)
        if success:
            print("   ✅ Test message sent successfully!")
            return True
        else:
            print("   ❌ Test message failed")
            return False
    except Exception as e:
        print(f"   ❌ Test message error: {e}")
        return False


def send_telegram_message(message, test_mode=False):
    """Send message to Telegram with enhanced error handling."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID, 
        "text": message,
        "parse_mode": "HTML"  # Support for basic formatting
    }
    
    # Try different methods
    methods = [
        ("JSON", lambda: requests.post(url, json=payload, timeout=10)),
        ("Form", lambda: requests.post(url, data=payload, timeout=10))
    ]
    
    for method_name, method_func in methods:
        try:
            if test_mode:
                print(f"   Trying {method_name} method...")
            
            r = method_func()
            
            if test_mode:
                print(f"   {method_name} response: {r.status_code}")
                print(f"   Response body: {r.text[:200]}...")
            
            if r.ok:
                data = r.json()
                if data.get("ok"):
                    if not test_mode:
                        print(f"✅ Telegram message sent via {method_name}")
                    return True
                else:
                    error_code = data.get("error_code")
                    description = data.get("description", "Unknown error")
                    print(f"❌ Telegram API error ({method_name}): {error_code} - {description}")
                    
                    # Specific error handling
                    if error_code == 400 and "chat not found" in description.lower():
                        print("   💡 Hint: Make sure the bot has been added to the chat and the chat ID is correct")
                    elif error_code == 401:
                        print("   💡 Hint: Check if the bot token is correct")
                    
            else:
                print(f"❌ HTTP error ({method_name}): {r.status_code} - {r.text[:200]}")
                
        except requests.exceptions.Timeout:
            print(f"❌ Timeout error ({method_name}): Request took too long")
        except requests.exceptions.ConnectionError:
            print(f"❌ Connection error ({method_name}): Could not reach Telegram servers")
        except Exception as e:
            print(f"❌ Unexpected error ({method_name}): {e}")
    
    return False


def notify(msg: str):
    """Send a notification with enhanced debugging."""
    print(f"🔔 Attempting to notify: {msg}")
    
    # Try Telegram first
    telegram_sent = send_telegram_message(msg)
    
    if not telegram_sent:
        print("📱 Telegram failed, trying local notification...")
        # Fallback to local notification
        try:
            if TOAST:
                TOAST.show_toast("PuppyCam", msg, duration=5, threaded=True)
                print("✅ Local toast notification sent")
            else:
                print("❌ No local notification system available")
        except Exception as e:
            print(f"❌ Local notification error: {e}")
        
        # Always log to console as final fallback
        print(f"📝 [ALERT] {msg}")


def load_rois():
    global rois
    if os.path.exists(ROI_CFG):
        try:
            with open(ROI_CFG, "r") as f:
                data = json.load(f)
            rois["inside"] = data.get("inside")
            rois["outside"] = data.get("outside")
            print("Loaded ROIs from", ROI_CFG, rois)
        except Exception as e:
            print("Failed to load ROI config:", e)


def save_rois(frame_w, frame_h):
    """Save current roi_points (two rects) normalized to 0..1."""
    global roi_points, rois
    if len(roi_points) != 2:
        print("Need exactly two rectangles to save.")
        return
    def norm_rect(r):
        x1, y1, x2, y2 = r
        x1, x2 = sorted([max(0, min(frame_w-1, x1)), max(0, min(frame_w-1, x2))])
        y1, y2 = sorted([max(0, min(frame_h-1, y1)), max(0, min(frame_h-1, y2))])
        return [x1/frame_w, y1/frame_h, x2/frame_w, y2/frame_h]

    inside_norm = norm_rect(roi_points[0])
    outside_norm = norm_rect(roi_points[1])
    rois["inside"] = inside_norm
    rois["outside"] = outside_norm

    with open(ROI_CFG, "w") as f:
        json.dump(rois, f, indent=2)
    print("Saved ROIs to", ROI_CFG, rois)


def denorm_rect(norm_rect, w, h):
    """Convert normalized rect to pixel rect."""
    if not norm_rect:
        return None
    x1 = int(norm_rect[0] * w)
    y1 = int(norm_rect[1] * h)
    x2 = int(norm_rect[2] * w)
    y2 = int(norm_rect[3] * h)
    return (x1, y1, x2, y2)


def point_in_rect(cx, cy, rect):
    x1, y1, x2, y2 = rect
    return x1 <= cx <= x2 and y1 <= cy <= y2


# ============== Camera/CGI ==============

def cgi(cmd, extra=None, timeout=5):
    params = {"cmd": cmd, "usr": USERNAME, "pwd": PASSWORD}
    if extra:
        params.update(extra)
    try:
        r = requests.get(BASE_CGI, params=params, timeout=timeout)
        r.raise_for_status()
        print(f"{cmd} -> HTTP {r.status_code}; body: {r.text.strip()[:120]}")
        return r.text
    except Exception as e:
        print(f"CGI error ({cmd}): {e}")
        return None

def ptz_move(direction: str):
    cgi(f"ptzMove{direction}")

def ptz_stop():
    cgi("ptzStopRun")

def zoom(action: str):
    cgi(f"zoom{action}")

def take_snapshot():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = os.path.join(SNAPS, f"snapshot_{ts}.jpg")
    try:
        img = requests.get(
            f"{BASE_CGI}?cmd=snapPicture2&usr={USR}&pwd={PWD}", timeout=10
        )
        img.raise_for_status()
        with open(fn, "wb") as f:
            f.write(img.content)
        print(f"Snapshot saved: {fn}")
    except Exception as e:
        print(f"Snapshot error: {e}")


# ============== Recording ==============

def toggle_record():
    global is_recording, writer, last_size, next_frame_due, current_frame
    with rec_lock:
        if not is_recording:
            if current_frame is None or last_size is None:
                print("No frame yet; cannot start recording.")
                return
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fn = os.path.join(RECS, f"recording_{ts}.avi")
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            w, h = map(int, last_size)
            writer = cv2.VideoWriter(fn, fourcc, TARGET_FPS, (w, h))
            if writer.isOpened():
                is_recording = True
                next_frame_due = time.monotonic()
                print(f"Recording started: {fn} (FPS={TARGET_FPS}, size={w}x{h}, MJPG)")
            else:
                writer = None
                print("Failed to open VideoWriter.")
        else:
            is_recording = False
            if writer:
                writer.release()
                writer = None
            print("Recording stopped.")


# ============== Keyboard & Mouse ==============

def on_key(event):
    global roi_edit_mode, roi_points
    if event.event_type not in (keyboard.KEY_DOWN, keyboard.KEY_UP):
        return
    k = event.name.lower()

    if event.event_type == keyboard.KEY_DOWN:
        if roi_edit_mode:
            # In ROI mode, Enter saves, Esc cancels
            if k == "enter":
                if current_frame is not None:
                    h, w = current_frame.shape[:2]
                    if len(roi_points) == 2:
                        save_rois(w, h)
                    else:
                        print("Need 2 rectangles (inside, then outside).")
                roi_edit_mode = False
                return
            if k == "esc":
                roi_points.clear()
                roi_edit_mode = False
                print("ROI edit cancelled.")
                return
        else:
            # Normal controls
            if k == "w": ptz_move("Up")
            elif k == "s": ptz_move("Down")
            elif k == "a": ptz_move("Left")
            elif k == "d": ptz_move("Right")
            elif k == "q": zoom("Out")
            elif k == "e": zoom("In")
            elif k == "t": take_snapshot()
            elif k == "r": toggle_record()
            elif k == "g":
                roi_points.clear()
                roi_edit_mode = True
                print("ROI edit mode: draw INSIDE rectangle, then OUTSIDE. Press Enter to save, Esc to cancel.")
            elif k == "m":  # New: Manual test notification
                notify("🧪 Manual test notification from PuppyCam!")

    elif event.event_type == keyboard.KEY_UP:
        if not roi_edit_mode:
            if k in ("w", "s", "a", "d"):
                ptz_stop()
            elif k in ("q", "e"):
                zoom("Stop")


def mouse_cb(event, x, y, flags, param):
    global dragging, start_pt, roi_points
    if not roi_edit_mode:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        start_pt = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        pass  # preview drawn in render section
    elif event == cv2.EVENT_LBUTTONUP and dragging:
        dragging = False
        end_pt = (x, y)
        x1, y1 = start_pt
        x2, y2 = end_pt
        roi_points.append((x1, y1, x2, y2))
        idx = len(roi_points)
        label = "INSIDE" if idx == 1 else "OUTSIDE"
        print(f"ROI {idx} ({label}) set:", roi_points[-1])


# ============== Detection ==============

def init_model():
    global model
    if YOLO_AVAILABLE:
        try:
            model = YOLO(MODEL_NAME)
            print("YOLO model loaded:", MODEL_NAME)
        except Exception as e:
            print("Failed to load YOLO model:", e)

def detect_dogs(frame):
    """Return list of (cx, cy, conf) for dog centers."""
    if model is None:
        return []
    try:
        res = model.predict(source=frame, imgsz=640, conf=CONF_THRESH, verbose=False)
        dets = []
        for r in res:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == DOG_CLASS:
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    conf = float(box.conf[0])
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    dets.append((cx, cy, conf))
        return dets
    except Exception as e:
        print("YOLO inference error:", e)
        return []


def update_presence_and_alert(dog_centers, frame_w, frame_h):
    """Update presence buffers per ROI and notify on sustained presence with cooldown."""
    now = time.time()
    results = {}

    # Prepare pixel rects
    inside_rect = denorm_rect(rois["inside"], frame_w, frame_h) if rois["inside"] else None
    outside_rect = denorm_rect(rois["outside"], frame_w, frame_h) if rois["outside"] else None

    for zone, rect in (("inside", inside_rect), ("outside", outside_rect)):
        present = False
        if rect:
            for (cx, cy, conf) in dog_centers:
                if point_in_rect(cx, cy, rect):
                    present = True
                    break
        presence_buffers[zone].append(1 if present else 0)
        
        # Debug presence detection
        if present:
            print(f"🐕 Dog detected in {zone.upper()} zone")
        
        # Decide if persisted
        buffer_sum = sum(presence_buffers[zone])
        buffer_len = len(presence_buffers[zone])
        persistence_threshold = int(0.8 * PERSIST_FRAMES)
        
        if buffer_len == PERSIST_FRAMES and buffer_sum >= persistence_threshold:
            # sustained presence
            time_since_last = now - last_alert_time[zone]
            print(f"🕐 Sustained presence in {zone}: {buffer_sum}/{buffer_len} frames, last alert {time_since_last:.1f}s ago")
            
            if time_since_last >= COOLDOWN_SEC:
                last_alert_time[zone] = now
                direction = "needs to go OUT" if zone == "inside" else "wants to come IN"
                alert_msg = f"🐶 Dog at {zone.upper()} door — {direction}!"
                print(f"🚨 ALERT TRIGGERED: {alert_msg}")
                notify(alert_msg)
            else:
                print(f"⏳ Alert suppressed due to cooldown ({time_since_last:.1f}s < {COOLDOWN_SEC}s)")
        
        results[zone] = present
    return results


# ============== Video loop ==============

def video_loop():
    global current_frame, stop_thread, writer, is_recording, last_size, next_frame_due
    print(f"Connecting RTSP: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Failed to open RTSP stream.")
        return
    print("Video stream connected. Press 'Q' in window to quit.")
    cv2.namedWindow("PuppyCam")
    cv2.setMouseCallback("PuppyCam", mouse_cb)

    while not stop_thread:
        ok, frame = cap.read()
        if not ok:
            print("Frame read failed; reconnecting in 2s...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            continue

        current_frame = frame
        h, w = frame.shape[:2]
        last_size = (w, h)

        # --- Detection
        dog_centers = detect_dogs(frame) if YOLO_AVAILABLE and rois["inside"] and rois["outside"] else []
        if dog_centers:
            # Draw dog centers
            for (cx, cy, conf) in dog_centers:
                cv2.circle(frame, (int(cx), int(cy)), 6, (0, 255, 0), 2)
                cv2.putText(frame, f"dog {conf:.2f}", (int(cx)+6, int(cy)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # --- Update zone presence + alerts
        if rois["inside"] and rois["outside"]:
            update_presence_and_alert(dog_centers, w, h)

        # --- ROI editor overlays
        if rois["inside"]:
            x1,y1,x2,y2 = denorm_rect(rois["inside"], w, h)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,165,255), 2) # orange
            cv2.putText(frame, "INSIDE", (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
        if rois["outside"]:
            x1,y1,x2,y2 = denorm_rect(rois["outside"], w, h)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2) # blue
            cv2.putText(frame, "OUTSIDE", (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        # ROI edit live preview
        if roi_edit_mode and start_pt is not None and dragging:
            x1, y1 = start_pt
            mx, my = cv2.getWindowImageRect("PuppyCam")[:2]  # not used; just to avoid stale var
        if roi_edit_mode and start_pt is not None:
            # draw last temp rect while dragging
            x1, y1 = start_pt
            # get current mouse pos from OpenCV isn't direct; skip live rubber-banding
            pass

        # Draw any confirmed rectangles from this session
        for i, r in enumerate(roi_points):
            x1,y1,x2,y2 = r
            color = (0,165,255) if i == 0 else (255,0,0)
            label = "INSIDE" if i == 0 else "OUTSIDE"
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # --- Recording with pacing
        with rec_lock:
            if is_recording and writer is not None:
                now = time.monotonic()
                if now >= next_frame_due:
                    try:
                        writer.write(frame)
                    except cv2.error as e:
                        print(f"VideoWriter error: {e}. Stopping recording.")
                        is_recording = False
                        writer.release()
                        writer = None
                    next_frame_due = now + (1.0 / max(1.0, TARGET_FPS))

        cv2.imshow("PuppyCam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_thread = True
            break

    cap.release()
    with rec_lock:
        if writer:
            writer.release()
            writer = None
    cv2.destroyAllWindows()
    print("Video loop ended.")


# ============== Main ==============

if __name__ == "__main__":
    print("PuppyCam — PTZ + Recording + ROIs + Dog Alerts")
    print("Keys: W/A/S/D pan/tilt, Q/E zoom out/in, T snapshot, R record, G ROI edit, M test notification, Q (window) to quit")
    
    # Test Telegram connection at startup
    telegram_ok = test_telegram_connection()
    if telegram_ok:
        print("✅ Telegram notifications ready!")
    else:
        print("❌ Telegram notifications not working - check configuration")
    
    load_rois()
    if YOLO_AVAILABLE:
        init_model()
    else:
        print("Ultralytics not installed — dog detection disabled. Install with: pip install ultralytics")

    keyboard.hook(on_key)
    th = threading.Thread(target=video_loop, daemon=True)
    th.start()

    try:
        while not stop_thread:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_thread = True
        keyboard.unhook_all()
        th.join(timeout=5)
        print("Program exited.")