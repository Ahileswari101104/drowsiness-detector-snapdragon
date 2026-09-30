import os
import sys
import time
import math
import threading
import winsound
from collections import deque

import cv2
import requests
import mediapipe as mp

# ===================== CONFIG =====================
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "7217571805:AAGwcHQZsp_80pjeUWvudP2_lXOT-WQeqP0")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "6866858209")

VIDEO_DIR = os.path.join(os.path.expanduser("~"), "Videos", "drowsy_video")
ALERT_WAV = r"C:\DrowsyDetector\alert.wav"   # must be a .wav file

CAM_INDEX = 0
CAP_W, CAP_H = 640, 480
PROC_WIDTH = 480            # frames are downscaled to this width for detection
SHOW_WINDOW = True

EAR_THRESHOLD = 0.21        # tune per person / camera angle
EYE_CLOSED_SECONDS = 2.0
MAR_THRESHOLD = 0.50        # mouth aspect ratio for a yawn
YAWN_SECONDS = 1.5
COOLDOWN_SECONDS = 2.0      # normal behavior needed before stopping the alert
PREROLL_SECONDS = 3.0
MAX_CLIP_SECONDS = 30       # safety cap per clip
SEND_TELEGRAM = True
# ==================================================

os.makedirs(VIDEO_DIR, exist_ok=True)

# Landmarks (MediaPipe Face Mesh)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT = 13, 14, 61, 291


def dist(a, b, w, h):
    return math.hypot((a.x - b.x) * w, (a.y - b.y) * h)


def ear(lm, idx, w, h):
    p = [lm[i] for i in idx]
    horiz = dist(p[0], p[3], w, h)
    if horiz == 0:
        return 0.0
    return (dist(p[1], p[5], w, h) + dist(p[2], p[4], w, h)) / (2.0 * horiz)


def mar(lm, w, h):
    horiz = dist(lm[MOUTH_LEFT], lm[MOUTH_RIGHT], w, h)
    if horiz == 0:
        return 0.0
    return dist(lm[MOUTH_TOP], lm[MOUTH_BOTTOM], w, h) / horiz


# ---------- Threaded camera (always holds the latest frame) ----------
class CameraStream:
    def __init__(self, index, width, height):
        self.cap = cv2.VideoCapture(index, cv2.CAP_MSMF)   # Windows Media Foundation
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # fallback
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.stamp = 0
        self.lock = threading.Lock()
        self.running = self.cap.isOpened()
        if self.running:
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = f
                self.stamp += 1

    def read(self, last_stamp):
        """Returns (frame, stamp) only if a new frame is available."""
        with self.lock:
            if self.frame is None or self.stamp == last_stamp:
                return None, last_stamp
            return self.frame.copy(), self.stamp

    def release(self):
        self.running = False
        time.sleep(0.05)
        self.cap.release()


# ---------- Audio (looping alarm) ----------
def start_alarm():
    try:
        winsound.PlaySound(ALERT_WAV,
                           winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
    except Exception:
        threading.Thread(target=lambda: winsound.Beep(1000, 800), daemon=True).start()


def stop_alarm():
    winsound.PlaySound(None, winsound.SND_PURGE)


# ---------- Telegram ----------
def send_video(path):
    if not (SEND_TELEGRAM and BOT_TOKEN and CHAT_ID):
        print("Telegram not configured, skipping upload.")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendVideo"
    try:
        with open(path, "rb") as f:
            r = requests.post(url, data={"chat_id": CHAT_ID,
                                         "caption": "Drowsiness detected"},
                              files={"video": f}, timeout=120)
        print("Telegram:", "sent" if r.ok else f"failed {r.status_code} {r.text[:100]}")
    except Exception as e:
        print(f"Telegram error: {e}")


# ---------- Recorder ----------
class Recorder:
    def __init__(self):
        self.writer = None
        self.path = ""
        self.start = 0

    @property
    def active(self):
        return self.writer is not None

    def begin(self, size, fps, preroll_frames):
        self.path = os.path.join(VIDEO_DIR, f"drowsy_{int(time.time())}.mp4")
        self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        self.start = time.time()
        for f in preroll_frames:
            self.writer.write(f)
        print(f"Recording started: {self.path}")

    def write(self, frame):
        if self.writer:
            self.writer.write(frame)

    def end(self):
        if not self.writer:
            return
        self.writer.release()
        self.writer = None
        print(f"Recording stopped: {self.path}")
        threading.Thread(target=send_video, args=(self.path,), daemon=True).start()


def main():
    cam = CameraStream(CAM_INDEX, CAP_W, CAP_H)
    if not cam.running:
        print("Unable to open camera.")
        sys.exit(1)

    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                                 refine_landmarks=False,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

    rec = Recorder()
    preroll = deque()               # (timestamp, frame)
    eye_start = yawn_start = None
    alerting = False
    last_alert_time = 0
    last_stamp = 0
    fps_est, last_t = 15.0, time.time()
    status = "Monitoring"

    try:
        while True:
            frame, last_stamp_new = cam.read(last_stamp)
            if frame is None:
                time.sleep(0.003)
                if SHOW_WINDOW and cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            last_stamp = last_stamp_new

            now = time.time()
            fps_est = 0.9 * fps_est + 0.1 * (1.0 / max(now - last_t, 1e-3))
            last_t = now

            h, w = frame.shape[:2]

            # Downscale for faster inference
            scale = PROC_WIDTH / w
            small = cv2.resize(frame, (PROC_WIDTH, int(h * scale)))
            sh, sw = small.shape[:2]
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = face_mesh.process(rgb)

            eyes_drowsy = yawning = False
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                avg_ear = (ear(lm, LEFT_EYE, sw, sh) + ear(lm, RIGHT_EYE, sw, sh)) / 2
                cur_mar = mar(lm, sw, sh)

                eye_start = (eye_start or now) if avg_ear < EAR_THRESHOLD else None
                yawn_start = (yawn_start or now) if cur_mar > MAR_THRESHOLD else None
                eyes_drowsy = eye_start is not None and now - eye_start >= EYE_CLOSED_SECONDS
                yawning = yawn_start is not None and now - yawn_start >= YAWN_SECONDS

                status = f"EAR {avg_ear:.2f}  MAR {cur_mar:.2f}"
            else:
                eye_start = yawn_start = None
                status = "No face detected"

            alert = eyes_drowsy or yawning

            # Keep a rolling pre-roll buffer of clean frames
            preroll.append((now, frame.copy()))
            while preroll and now - preroll[0][0] > PREROLL_SECONDS:
                preroll.popleft()

            # ---- Alert state machine ----
            if alert:
                last_alert_time = now
                if not alerting:
                    alerting = True
                    start_alarm()
                    if not rec.active:
                        rec.begin((w, h), max(5.0, min(fps_est, 30.0)),
                                  [f for _, f in preroll])
            elif alerting and (now - last_alert_time) > COOLDOWN_SECONDS:
                alerting = False
                stop_alarm()
                rec.end()

            if rec.active:
                rec.write(frame)
                if now - rec.start > MAX_CLIP_SECONDS:
                    rec.end()
                    rec.begin((w, h), max(5.0, min(fps_est, 30.0)), [])

            # ---- Display ----
            if SHOW_WINDOW:
                disp = frame
                cv2.putText(disp, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)
                cv2.putText(disp, f"FPS {fps_est:.0f}", (10, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                if eyes_drowsy:
                    cv2.putText(disp, "DROWSY EYES DETECTED!", (30, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                if yawning:
                    cv2.putText(disp, "YAWNING DETECTED!", (30, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                if alerting:  # replaces the GPIO indicator LED
                    cv2.rectangle(disp, (0, 0), (w - 1, h - 1), (0, 0, 255), 12)
                cv2.imshow("Drowsiness Detector", disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        stop_alarm()
        rec.end()
        cam.release()
        face_mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()