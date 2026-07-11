"""
PuppyCam — dual USB webcam dog-at-the-door detector.

Two local USB webcams, one detection ROI each:
  * Camera A (index 0) watches the INSIDE of the door.
  * Camera B (index 1) watches the OUTSIDE, through the glass.

YOLOv8 finds dogs; a per-zone persistence buffer + cooldown decides when to
fire a Telegram alert. A side-by-side preview window lets you draw/adjust the
ROI on each camera independently.

Keys (focus the preview window):
  G  enter ROI edit mode   (draw a box in either pane; Enter saves, Esc cancels)
  T  save a snapshot of the current side-by-side view
  R  toggle recording of the side-by-side view
  M  send a manual test notification
  Q  quit
"""

import cv2
import numpy as np
import requests
import time
import datetime
import os
import json
from collections import deque


# --- .env loading (python-dotenv if present, tiny fallback otherwise) ---
def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return
    except Exception:
        pass
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_env()

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
# CONFIG
# =========================

# USB webcam device indices. Camera A = INSIDE zone, Camera B = OUTSIDE (glass).
CAM_INSIDE_INDEX = int(os.environ.get("CAM_INSIDE_INDEX", "0"))
CAM_OUTSIDE_INDEX = int(os.environ.get("CAM_OUTSIDE_INDEX", "1"))

# One camera -> one zone -> one ROI.
CAMERAS = [
    {"zone": "inside", "index": CAM_INSIDE_INDEX, "label": "INSIDE", "color": (0, 165, 255)},   # orange
    {"zone": "outside", "index": CAM_OUTSIDE_INDEX, "label": "OUTSIDE (glass)", "color": (255, 0, 0)},  # blue
]
ZONES = [c["zone"] for c in CAMERAS]

# Preview / recording geometry (each pane is resized to this before concat).
DISPLAY_W = 640
DISPLAY_H = 480

# Recording / capture output.
TARGET_FPS = 20.0
OUT_DIR = "puppycam_output"
SNAPS = os.path.join(OUT_DIR, "snapshots")
RECS = os.path.join(OUT_DIR, "recordings")
os.makedirs(SNAPS, exist_ok=True)
os.makedirs(RECS, exist_ok=True)

# Detection settings
CONF_THRESH = 0.45          # YOLO confidence
PERSIST_FRAMES = int(1.5 * TARGET_FPS)  # how long a dog must be present (~1.5s of loop samples)
COOLDOWN_SEC = 30           # min seconds between alerts per zone
MODEL_NAME = "yolov8n.pt"   # tiny; good enough
DOG_CLASS = 16              # COCO 'dog'

# Telegram (from .env). If not set, falls back to toast/log.
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ROI config file (normalized coords, one rect per zone/camera).
ROI_CFG = "roi_config.json"

WINDOW = "PuppyCam"
FONT = cv2.FONT_HERSHEY_SIMPLEX
# =========================


# --- Globals ---
model = None

# Loaded ROIs (normalized 0..1, one per camera): {"inside":[x1,y1,x2,y2], "outside":[...]}
rois = {z: None for z in ZONES}

# Detection + alert state (per zone)
last_alert_time = {z: 0.0 for z in ZONES}
presence_buffers = {z: deque(maxlen=PERSIST_FRAMES) for z in ZONES}

# Recording state
is_recording = False
writer = None
next_frame_due = 0.0

# ROI editor state
roi_edit_mode = False
pending = {z: None for z in ZONES}   # normalized rects drawn during the current edit session
dragging = False
drag_zone = None                     # which pane the current drag started in
drag_start = None                    # (x, y) in local pane display coords
drag_cur = None                      # current mouse pos in local pane display coords


# ============== Telegram / notifications ==============

def test_telegram_connection():
    """Test if Telegram bot token and chat ID are working."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Telegram credentials not configured (set TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in .env)")
        return False

    print("🔍 Testing Telegram connection...")
    print(f"   Bot Token: {TELEGRAM_BOT_TOKEN[:20]}...")
    print(f"   Chat ID: {TELEGRAM_CHAT_ID}")

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

    try:
        success = send_telegram_message("🤖 PuppyCam Telegram test - connection successful!", test_mode=True)
        if success:
            print("   ✅ Test message sent successfully!")
            return True
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
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}

    methods = [
        ("JSON", lambda: requests.post(url, json=payload, timeout=10)),
        ("Form", lambda: requests.post(url, data=payload, timeout=10)),
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
                error_code = data.get("error_code")
                description = data.get("description", "Unknown error")
                print(f"❌ Telegram API error ({method_name}): {error_code} - {description}")
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
    """Send a notification (Telegram first, then local toast, then console)."""
    print(f"🔔 Attempting to notify: {msg}")

    if send_telegram_message(msg):
        return

    print("📱 Telegram failed, trying local notification...")
    try:
        if TOAST:
            TOAST.show_toast("PuppyCam", msg, duration=5, threaded=True)
            print("✅ Local toast notification sent")
        else:
            print("❌ No local notification system available")
    except Exception as e:
        print(f"❌ Local notification error: {e}")

    print(f"📝 [ALERT] {msg}")


# ============== ROI helpers ==============

def load_rois():
    global rois
    if os.path.exists(ROI_CFG):
        try:
            with open(ROI_CFG, "r") as f:
                data = json.load(f)
            for z in ZONES:
                rois[z] = data.get(z)
            print("Loaded ROIs from", ROI_CFG, rois)
        except Exception as e:
            print("Failed to load ROI config:", e)


def save_rois():
    """Persist the pending edit rects (one per camera) to disk."""
    global rois
    for z in ZONES:
        rois[z] = pending[z]
    with open(ROI_CFG, "w") as f:
        json.dump(rois, f, indent=2)
    print("Saved ROIs to", ROI_CFG, rois)


def denorm_rect(norm_rect, w, h):
    """Convert normalized rect to pixel rect."""
    if not norm_rect:
        return None
    return (int(norm_rect[0] * w), int(norm_rect[1] * h),
            int(norm_rect[2] * w), int(norm_rect[3] * h))


def point_in_rect(cx, cy, rect):
    x1, y1, x2, y2 = rect
    return x1 <= cx <= x2 and y1 <= cy <= y2


# ============== ROI editor (mouse) ==============

def _to_local(x, y, zone):
    """Map combined-window coords -> local pane coords for the given zone, clamped."""
    lx = x - DISPLAY_W if zone == "outside" else x
    lx = max(0, min(DISPLAY_W - 1, lx))
    ly = max(0, min(DISPLAY_H - 1, y))
    return lx, ly


def mouse_cb(event, x, y, flags, param):
    global dragging, drag_zone, drag_start, drag_cur, pending
    if not roi_edit_mode:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        drag_zone = "inside" if x < DISPLAY_W else "outside"
        dragging = True
        drag_start = _to_local(x, y, drag_zone)
        drag_cur = drag_start
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        drag_cur = _to_local(x, y, drag_zone)
    elif event == cv2.EVENT_LBUTTONUP and dragging:
        dragging = False
        x1, y1 = drag_start
        x2, y2 = _to_local(x, y, drag_zone)
        nx1, nx2 = sorted([x1 / DISPLAY_W, x2 / DISPLAY_W])
        ny1, ny2 = sorted([y1 / DISPLAY_H, y2 / DISPLAY_H])
        if (nx2 - nx1) > 0.01 and (ny2 - ny1) > 0.01:
            pending[drag_zone] = [nx1, ny1, nx2, ny2]
            print(f"ROI set for {drag_zone.upper()}: {pending[drag_zone]}")
        drag_start = None
        drag_cur = None


def enter_roi_edit():
    global roi_edit_mode, pending
    pending = {z: rois[z] for z in ZONES}
    roi_edit_mode = True
    print("ROI edit mode: draw a box in either pane. Enter=save, Esc=cancel.")


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
    """Return list of (cx, cy, conf) for dog centers (in frame pixel coords)."""
    if model is None:
        return []
    try:
        res = model.predict(source=frame, imgsz=640, conf=CONF_THRESH, verbose=False)
        dets = []
        for r in res:
            for box in r.boxes:
                if int(box.cls[0]) == DOG_CLASS:
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    dets.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0, float(box.conf[0])))
        return dets
    except Exception as e:
        print("YOLO inference error:", e)
        return []


def update_presence_and_alert(zone, dog_centers, frame_w, frame_h):
    """Update one zone's presence buffer and alert on sustained presence (with cooldown)."""
    now = time.time()
    rect = denorm_rect(rois[zone], frame_w, frame_h) if rois[zone] else None

    present = False
    if rect:
        for (cx, cy, _conf) in dog_centers:
            if point_in_rect(cx, cy, rect):
                present = True
                break

    presence_buffers[zone].append(1 if present else 0)
    if present:
        print(f"🐕 Dog detected in {zone.upper()} zone")

    buf = presence_buffers[zone]
    persistence_threshold = int(0.8 * PERSIST_FRAMES)
    if len(buf) == PERSIST_FRAMES and sum(buf) >= persistence_threshold:
        time_since_last = now - last_alert_time[zone]
        print(f"🕐 Sustained presence in {zone}: {sum(buf)}/{len(buf)} frames, last alert {time_since_last:.1f}s ago")
        if time_since_last >= COOLDOWN_SEC:
            last_alert_time[zone] = now
            direction = "needs to go OUT" if zone == "inside" else "wants to come IN"
            alert_msg = f"🐶 Dog at {zone.upper()} door — {direction}!"
            print(f"🚨 ALERT TRIGGERED: {alert_msg}")
            notify(alert_msg)
        else:
            print(f"⏳ Alert suppressed due to cooldown ({time_since_last:.1f}s < {COOLDOWN_SEC}s)")

    return present


# ============== Recording / snapshots ==============

def take_snapshot(frame):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = os.path.join(SNAPS, f"snapshot_{ts}.jpg")
    if cv2.imwrite(fn, frame):
        print(f"Snapshot saved: {fn}")
    else:
        print("Snapshot failed.")


def toggle_record(sample_frame):
    global is_recording, writer, next_frame_due
    if not is_recording:
        h, w = sample_frame.shape[:2]
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = os.path.join(RECS, f"recording_{ts}.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
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


# ============== Rendering ==============

def placeholder_pane(text):
    pane = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    cv2.putText(pane, text, (20, DISPLAY_H // 2), FONT, 0.7, (0, 0, 255), 2)
    return pane


def render_pane(cam, frame, dog_centers):
    """Resize a camera frame to the display pane and draw detections, ROI, and edits."""
    zone, color, label = cam["zone"], cam["color"], cam["label"]
    h, w = frame.shape[:2]
    disp = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
    sx, sy = DISPLAY_W / w, DISPLAY_H / h

    # Detected dog centers
    for (cx, cy, conf) in dog_centers:
        px, py = int(cx * sx), int(cy * sy)
        cv2.circle(disp, (px, py), 6, (0, 255, 0), 2)
        cv2.putText(disp, f"dog {conf:.2f}", (px + 6, py - 6), FONT, 0.5, (0, 255, 0), 1)

    # Saved ROI
    if rois[zone]:
        x1 = int(rois[zone][0] * DISPLAY_W)
        y1 = int(rois[zone][1] * DISPLAY_H)
        x2 = int(rois[zone][2] * DISPLAY_W)
        y2 = int(rois[zone][3] * DISPLAY_H)
        cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)

    # Pending edit rect + live rubber-band
    if roi_edit_mode:
        if pending[zone]:
            x1 = int(pending[zone][0] * DISPLAY_W)
            y1 = int(pending[zone][1] * DISPLAY_H)
            x2 = int(pending[zone][2] * DISPLAY_W)
            y2 = int(pending[zone][3] * DISPLAY_H)
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 1)
        if dragging and drag_zone == zone and drag_start and drag_cur:
            cv2.rectangle(disp, drag_start, drag_cur, (0, 255, 255), 1)

    # Header label
    cv2.rectangle(disp, (0, 0), (DISPLAY_W, 26), (40, 40, 40), -1)
    cv2.putText(disp, label, (8, 19), FONT, 0.6, color, 2)
    return disp


# ============== Main loop ==============

def main():
    global roi_edit_mode, pending, is_recording, writer, next_frame_due

    print("PuppyCam — dual USB webcam + ROIs + dog alerts")
    print("Keys: G ROI edit, T snapshot, R record, M test notification, Q quit")

    if test_telegram_connection():
        print("✅ Telegram notifications ready!")
    else:
        print("❌ Telegram notifications not working - check .env configuration")

    load_rois()
    if YOLO_AVAILABLE:
        init_model()
    else:
        print("Ultralytics not installed — dog detection disabled. Install with: pip install ultralytics")

    caps = {}
    for cam in CAMERAS:
        cap = cv2.VideoCapture(cam["index"])
        if not cap.isOpened():
            print(f"⚠️  Could not open camera index {cam['index']} for {cam['zone'].upper()} zone.")
        else:
            print(f"Camera {cam['index']} opened for {cam['zone'].upper()} zone.")
        caps[cam["zone"]] = cap

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, mouse_cb)

    try:
        while True:
            panes = []
            for cam in CAMERAS:
                zone = cam["zone"]
                ok, frame = caps[zone].read()
                if not ok or frame is None:
                    panes.append(placeholder_pane(f"{cam['label']}: NO SIGNAL"))
                    continue

                dog_centers = detect_dogs(frame) if (YOLO_AVAILABLE and rois[zone]) else []
                if rois[zone]:
                    update_presence_and_alert(zone, dog_centers, frame.shape[1], frame.shape[0])
                panes.append(render_pane(cam, frame, dog_centers))

            combined = cv2.hconcat(panes)

            # Status banner
            if roi_edit_mode:
                cv2.putText(combined, "ROI EDIT — draw in a pane, Enter=save, Esc=cancel",
                            (8, DISPLAY_H - 12), FONT, 0.6, (0, 255, 255), 2)
            if is_recording:
                cv2.circle(combined, (combined.shape[1] - 20, 20), 8, (0, 0, 255), -1)

            # Recording (paced)
            if is_recording and writer is not None:
                now = time.monotonic()
                if now >= next_frame_due:
                    try:
                        writer.write(combined)
                    except cv2.error as e:
                        print(f"VideoWriter error: {e}. Stopping recording.")
                        is_recording = False
                        writer.release()
                        writer = None
                    next_frame_due = now + (1.0 / max(1.0, TARGET_FPS))

            cv2.imshow(WINDOW, combined)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif roi_edit_mode:
                if key == 13:      # Enter
                    save_rois()
                    roi_edit_mode = False
                elif key == 27:    # Esc
                    roi_edit_mode = False
                    print("ROI edit cancelled.")
            else:
                if key == ord("g"):
                    enter_roi_edit()
                elif key == ord("t"):
                    take_snapshot(combined)
                elif key == ord("r"):
                    toggle_record(combined)
                elif key == ord("m"):
                    notify("🧪 Manual test notification from PuppyCam!")
    finally:
        for cap in caps.values():
            cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("Program exited.")


if __name__ == "__main__":
    main()
