import cv2
import requests
import keyboard
import threading
import time
import datetime
import os
import urllib.parse

# --- Camera Configuration ---
CAMERA_IP = "192.168.0.101"
HTTP_PORT = 88
USERNAME = "PuppyCam"
PASSWORD = "8616Calvin!"  # plain; only URL-encode in URLs

USR = urllib.parse.quote(USERNAME)
PWD = urllib.parse.quote(PASSWORD)

BASE_CGI = f"http://{CAMERA_IP}:{HTTP_PORT}/cgi-bin/CGIProxy.fcgi"
RTSP_URL = f"rtsp://{USR}:{PWD}@{CAMERA_IP}:{HTTP_PORT}/videoMain"

# --- Output directories ---
OUT = "foscam_output"
SNAPS = os.path.join(OUT, "snapshots")
RECS = os.path.join(OUT, "recordings")
os.makedirs(SNAPS, exist_ok=True)
os.makedirs(RECS, exist_ok=True)

# --- Globals ---
current_frame = None
stop_thread = False

# Recording state (protected by a lock)
rec_lock = threading.Lock()
is_recording = False
writer = None
last_size = None      # (w, h)
target_fps = 20.0     # we will FORCE this
next_frame_due = 0.0  # monotonic timestamp when next write is due

# ---- CGI helpers ----
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

# ---- PTZ (HD CGI) ----
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

# ---- Recording ----
def toggle_record():
    global is_recording, writer, last_size, target_fps, next_frame_due, current_frame
    with rec_lock:
        if not is_recording:
            if current_frame is None or last_size is None:
                print("No frame yet; cannot start recording.")
                return

            # Force a sane FPS; ignore whatever RTSP reports
            fps = target_fps
            if fps < 5 or fps > 60:
                fps = 20.0
            target_fps = fps

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fn = os.path.join(RECS, f"recording_{ts}.avi")

            # Motion-JPEG avoids B-frame/PTS headaches on Windows
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")

            w, h = map(int, last_size)
            writer = cv2.VideoWriter(fn, fourcc, fps, (w, h))
            if writer.isOpened():
                is_recording = True
                next_frame_due = time.monotonic()
                print(f"Recording started: {fn} (FPS={fps}, size={w}x{h}, MJPG)")
            else:
                writer = None
                print("Failed to open VideoWriter.")
        else:
            # Stop recording safely
            is_recording = False
            if writer:
                writer.release()
                writer = None
            print("Recording stopped.")

# ---- Keyboard handler ----
def on_key(event):
    if event.event_type not in (keyboard.KEY_DOWN, keyboard.KEY_UP):
        return
    k = event.name.lower()

    if event.event_type == keyboard.KEY_DOWN:
        if k == "w":
            ptz_move("Up")
        elif k == "s":
            ptz_move("Down")
        elif k == "a":
            ptz_move("Left")
        elif k == "d":
            ptz_move("Right")
        elif k == "q":
            zoom("Out")
        elif k == "e":
            zoom("In")
        elif k == "t":
            take_snapshot()
        elif k == "r":
            toggle_record()

    elif event.event_type == keyboard.KEY_UP:
        if k in ("w", "s", "a", "d"):
            ptz_stop()
        elif k in ("q", "e"):
            zoom("Stop")

# ---- (Stub) Dog detection hook ----
def detect_dog(frame):
    """
    Replace this with a real model (e.g., ultralytics YOLO).
    For now it returns False; it’s just a placeholder to show where it would run.
    """
    return False

# ---- Video loop ----
def video_loop():
    global current_frame, stop_thread, writer, is_recording, last_size, target_fps, next_frame_due
    print(f"Connecting RTSP: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Failed to open RTSP stream.")
        return
    print("Video stream connected. Press 'Q' in window to quit.")

    # Ignore camera FPS; we will write at forced target_fps

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

        # ---- Dog detection hook (call lightweight model here)
        _dog_here = detect_dog(frame)
        # (Later: draw zones, debounce, and notify)

        # ---- Recording write with monotonic pacing
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
                    # schedule next frame time
                    next_frame_due = now + (1.0 / max(1.0, target_fps))

        cv2.imshow("Foscam Live Feed (press Q to quit)", frame)
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

# ---- Main ----
if __name__ == "__main__":
    print("Foscam HD PTZ Control")
    print("W/A/S/D = pan/tilt, Q/E = zoom out/in, T = snapshot, R = record")

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
