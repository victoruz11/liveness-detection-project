"""
Face Liveness Detection — Streamlit App
Motion + Face Focusing pipeline using CNN-LSTM model.

Anti-spoofing is active in BOTH the pre-check (challenge) phase AND the
deep-scan phase. Each active liveness gesture (blink, mouth-open, neck-turn)
has its own snapshot anti-spoof gate: the CNN-LSTM model must classify the
subject as REAL at the exact moment the gesture is detected before the step
is accepted. A video replay of someone blinking will be caught by the model
and the challenge will be blocked / reset.

Three-layer defence during pre-check:
  1. Continuous rolling anti-spoof (pc_verdict) watches every frame.
  2. Snapshot anti-spoof fires the moment each gesture is confirmed — blink,
     mouth-open and neck-turn each get their own independent CNN-LSTM call.
  3. If either layer fires FAKE the challenge resets with a cooldown.

Fake-lock logic (deep scan phase):
  Once "FAKE" triggers, the processor requires 8 consecutive REAL predictions
  (FAKE_LOCK_COUNT = 8) before the lock is released.  A single non-REAL frame
  while locked resets the real-streak counter back to zero.

Face centering / stability requirement:
  The face must be centred within CENTER_TOLERANCE_RATIO and stable for
  STABLE_FACE_MIN_SEC (1.2 s) before challenge step 0 advances.  If the
  face drifts off-centre at any point during the challenge the state resets.

References at the end
"""

import sys
import time
import threading
from collections import deque
from pathlib import Path
from datetime import datetime

import av
import cv2
import numpy as np
import streamlit as st
import torch
import torchvision.transforms as transforms
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# ── Path setup ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))
from model import CNNLSTMModel
from face_utils import MediaPipeFaceDetector, crop_face_square

# Import active liveness helpers — snapshot anti-spoof + head-turn geometry
from active_liveness import (
    run_snapshot_antispoof,
    is_head_turned,
    HEAD_TURN_FRAMES_REQUIRED,
)

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── Constants ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH       = str(BASE_DIR / "models" / "best_model.pth")
DETECTOR_MODEL   = str(BASE_DIR / "models" / "face_detector.task")
LANDMARKER_MODEL = str(BASE_DIR / "models" / "face_landmarker.task")

SEQUENCE_LENGTH  = 10
FRAME_STRIDE     = 1
MISS_TOLERANCE   = 5
SMOOTHING_WINDOW = 8
PRECHECK_SMOOTH  = 5

DEFAULT_REAL_THRESHOLD = 0.65
DEFAULT_FAKE_THRESHOLD = 0.65
DEFAULT_MIN_MARGIN     = 0.10
DEFAULT_PADDING        = 0.30

# Fake-lock: once FAKE triggers, requires this many *consecutive* REAL
# predictions before the lock is released.  A single non-REAL frame resets
# the streak counter back to zero.
FAKE_LOCK_COUNT = 8

# Face must be centred and stable for this many seconds before step 0 passes.
STABLE_FACE_MIN_SEC    = 1.2

BLINK_EAR_THRESHOLD    = 0.21
BLINK_FRAMES_REQUIRED  = 2
MOUTH_OPEN_THRESHOLD   = 0.045
CENTER_TOLERANCE_RATIO = 0.22
MAX_FACE_AREA_RATIO    = 0.28
SPOOF_BLOCK_COOLDOWN   = 2.5

# Colours (BGR for OpenCV)
GREEN  = (0, 220, 0)
RED    = (0, 0, 220)
YELLOW = (0, 200, 220)
WHITE  = (255, 255, 255)
DARK   = (30, 30, 30)
BLUE   = (255, 180, 60)

PHASE_PRECHECK = "precheck"
PHASE_SCAN     = "scan"

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── MediaPipe Face Landmarker ──────────────────────────────────────────────────
class FaceLandmarker:
    def __init__(self, model_path: str):
        opts = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.lm = vision.FaceLandmarker.create_from_options(opts)

    def detect(self, bgr, ts_ms: int):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.lm.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms
        )

    def close(self):
        self.lm.close()


# ── Geometry helpers ───────────────────────────────────────────────────────────
def euclidean(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def get_xy(lm):
    return (lm.x, lm.y)

def eye_aspect_ratio(lms, eye="left"):
    if eye == "left":
        h1, h2, w = (159, 145), (158, 153), (33, 133)
    else:
        h1, h2, w = (386, 374), (385, 380), (362, 263)
    vert  = (euclidean(get_xy(lms[h1[0]]), get_xy(lms[h1[1]])) +
             euclidean(get_xy(lms[h2[0]]), get_xy(lms[h2[1]]))) / 2.0
    horiz = euclidean(get_xy(lms[w[0]]), get_xy(lms[w[1]])) + 1e-6
    return vert / horiz

def mouth_open_ratio(lms):
    v = euclidean(get_xy(lms[13]), get_xy(lms[14]))
    h = euclidean(get_xy(lms[78]), get_xy(lms[308])) + 1e-6
    return v / h

def face_is_centered(meta, frame_shape, tol=CENTER_TOLERANCE_RATIO):
    fh, fw = frame_shape[:2]
    x, y, bw, bh = meta["bbox"]
    dx = abs((x + bw / 2.0) - fw / 2.0) / fw
    dy = abs((y + bh / 2.0) - fh / 2.0) / fh
    return dx <= tol and dy <= tol

def face_signature(bbox):
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0, w * h)

def is_new_face(prev, curr, ct=0.18, at=0.45):
    if prev is None or curr is None:
        return False
    px, py, pa = prev
    cx, cy, ca = curr
    shift = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5 / max(pa ** 0.5, 1.0)
    area  = abs(ca - pa) / max(pa, 1.0)
    return shift > ct or area > at


# ── Drawing helpers ────────────────────────────────────────────────────────────
def draw_confidence_bar(frame, real_p, fake_p, x, y, width=260, height=20):
    cv2.rectangle(frame, (x, y), (x + width, y + height), DARK, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), WHITE, 1)
    rw = int(real_p * width)
    if rw > 0:
        cv2.rectangle(frame, (x, y), (x + rw, y + height), GREEN, -1)
    fw2 = int(fake_p * width)
    if fw2 > 0:
        cv2.rectangle(frame, (x + width - fw2, y), (x + width, y + height), RED, -1)
    cv2.putText(frame, f"REAL {real_p:.0%}", (x + 4, y + height - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1)
    cv2.putText(frame, f"FAKE {fake_p:.0%}", (x + width - 72, y + height - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1)

def draw_top_bar(frame, text, colour=WHITE):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), DARK, -1)
    cv2.putText(frame, text, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, colour, 2)

def draw_precheck_panel(frame, statuses, elapsed,
                        pc_real=None, pc_fake=None, spoof_blocked=False):
    """
    Draw the pre-check challenge status panel.

    Now supports 4 challenge steps (stable → blink → mouth → neck turn).
    Panel height increased to 140 px (was 118) to accommodate the extra row.
    """
    fh, fw = frame.shape[:2]
    extra  = 38 if (pc_real is not None or spoof_blocked) else 0
    # Base height: 140 px accommodates up to 4 status rows at 22 px spacing
    # (rows at y+56, y+78, y+100, y+122 all fit within 140 px).
    bw, bh = 460, 140 + extra
    x1 = (fw - bw) // 2
    y1 = 14

    cv2.rectangle(frame, (x1, y1), (x1 + bw, y1 + bh), DARK, -1)
    cv2.rectangle(frame, (x1, y1), (x1 + bw, y1 + bh), WHITE, 1)
    cv2.putText(frame, f"Liveness Pre-check  {elapsed:.1f}s",
                (x1 + 14, y1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, BLUE, 2)

    for i, (text, ok) in enumerate(statuses):
        col = GREEN if ok else YELLOW
        pre = "OK" if ok else " >"
        cv2.putText(frame, f"{pre}  {text}", (x1 + 14, y1 + 56 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2)

    if spoof_blocked:
        cv2.rectangle(frame, (x1 + 8, y1 + bh - extra + 4),
                      (x1 + bw - 8, y1 + bh - 6), (20, 0, 60), -1)
        cv2.putText(frame, "SPOOF DETECTED — challenge blocked",
                    (x1 + 14, y1 + bh - extra + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, RED, 2)
    elif pc_real is not None:
        bar_x = x1 + 8
        bar_y = y1 + bh - extra + 6
        bar_w = bw - 16
        bar_h = 16
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (20, 20, 30), -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (60, 60, 80), 1)
        rw = int(pc_real * bar_w)
        if rw > 0:
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + rw, bar_y + bar_h), GREEN, -1)
        fkw = int(pc_fake * bar_w)
        if fkw > 0:
            cv2.rectangle(frame, (bar_x + bar_w - fkw, bar_y),
                          (bar_x + bar_w, bar_y + bar_h), RED, -1)
        cv2.putText(frame, f"Anti-spoof: R{pc_real:.0%}  F{pc_fake:.0%}",
                    (bar_x + 4, bar_y + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1)

def draw_fill_bar(frame, filled, total, x, y, width=260, height=20):
    cv2.rectangle(frame, (x, y), (x + width, y + height), DARK, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), WHITE, 1)
    fw = int((filled / total) * width)
    if fw > 0:
        cv2.rectangle(frame, (x, y), (x + fw, y + height), BLUE, -1)
    cv2.putText(frame, f"Collecting  {filled}/{total}",
                (x + 8, y + height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, DARK, 1)


# ── Shared state (thread-safe) ─────────────────────────────────────────────────
class SharedState:
    """
    Thread-safe data bridge between the webrtc background thread (writer)
    and the Streamlit main thread (reader).  Never access st.session_state
    from inside a VideoProcessorBase — use this object instead.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._data = {
            "phase":            PHASE_PRECHECK,
            "label":            "Initialising...",
            "label_colour":     "gray",
            "fps":              0.0,
            "detection_score":  0.0,
            "face_found":       False,
            "log":              [],
            "challenge_step":   0,
            "pc_avg_real":      0.5,
            "pc_avg_fake":      0.5,
            "pc_buf_fill":      0,
            "pc_spoof_blocked": False,
            "avg_real":         0.5,
            "avg_fake":         0.5,
            "buffer_fill":      0,
        }

    def update(self, d: dict):
        # Merge the latest processor outputs into the shared UI snapshot.
        with self._lock:
            self._data.update(d)

    def snapshot(self) -> dict:
        # Return a copy so the Streamlit thread never mutates shared state directly.
        with self._lock:
            return dict(self._data)

    def add_log(self, entry: dict):
        # Keep newest events first so the on-page log reads like a live feed.
        with self._lock:
            self._data["log"].insert(0, entry)
            if len(self._data["log"]) > 50:
                self._data["log"] = self._data["log"][:50]


# ── Video Processor factory ────────────────────────────────────────────────────
def make_processor_factory(shared_state: SharedState, config: dict):
    """
    Returns a VideoProcessorBase *class* (not instance).

    All runtime values are captured in the closure here, in the main thread,
    before webrtc_streamer is called.  The returned class's __init__ never
    touches st.session_state, so it is safe to call from any thread.
    """

    class LivenessProcessor(VideoProcessorBase):

        def __init__(self):
            # ── Closure-injected values (thread-safe) ──────────────────────
            # Read immutable config values from the closure instead of st.session_state.
            self.shared         = shared_state
            self.real_threshold = config.get("real_threshold", DEFAULT_REAL_THRESHOLD)
            self.fake_threshold = config.get("fake_threshold", DEFAULT_FAKE_THRESHOLD)
            self.min_margin     = config.get("min_margin",     DEFAULT_MIN_MARGIN)
            self.padding        = config.get("padding",        DEFAULT_PADDING)
            self.show_debug     = config.get("show_debug",     False)

            # ── CNN-LSTM model ─────────────────────────────────────────────
            self.model = CNNLSTMModel().to(DEVICE)
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            self.model.eval()

            # ── MediaPipe ─────────────────────────────────────────────────
            self.detector = MediaPipeFaceDetector(
                model_path=DETECTOR_MODEL,
                running_mode="video",
                min_detection_confidence=0.6,
            )
            self.landmarker = FaceLandmarker(model_path=LANDMARKER_MODEL)

            # ── Scan-phase buffers ─────────────────────────────────────────
            self.buffer             = deque(maxlen=SEQUENCE_LENGTH)
            self.prediction_history = deque(maxlen=SMOOTHING_WINDOW)
            self.avg_real           = 0.5
            self.avg_fake           = 0.5
            self.miss_counter       = 0
            self.last_bbox          = None
            self.last_label         = "Initialising..."
            self.last_colour        = WHITE

            # ── Fake-lock state ────────────────────────────────────────────
            # Once FAKE triggers, fake_locked=True.  Requires FAKE_LOCK_COUNT
            # (8) consecutive REAL predictions to unlock.  Any non-REAL frame
            # resets real_streak to 0.
            self.fake_locked = False
            self.real_streak = 0

            # ── Pre-check anti-spoof buffers ───────────────────────────────
            self.precheck_buffer = deque(maxlen=SEQUENCE_LENGTH)
            self.pc_pred_history = deque(maxlen=PRECHECK_SMOOTH)
            self.pc_avg_real     = 0.5
            self.pc_avg_fake     = 0.5
            self.pc_verdict      = "uncertain"

            # ── Pre-check challenge state ──────────────────────────────────
            # Challenge steps:
            #   0 – face centred & stable for STABLE_FACE_MIN_SEC (1.2 s)
            #   1 – blink confirmed  + snapshot anti-spoof passes
            #   2 – mouth-open confirmed + snapshot anti-spoof passes
            #   3 – neck-turn confirmed  + snapshot anti-spoof passes
            #   4 – all done → transition to deep scan
            self.phase             = PHASE_PRECHECK
            self.precheck_start    = None
            self.stable_face_start = None
            self.blink_low_counter = 0
            self.head_turn_counter = 0   # consecutive frames head is turned
            self.challenge_step    = 0
            self.last_face_sig     = None
            self.spoof_block_until = 0.0

            # ── Misc ──────────────────────────────────────────────────────
            self.frame_counter = 0
            self._fps_times    = deque(maxlen=30)

        # ── Decision helper ───────────────────────────────────────────────
        def decide(self, avg_real, avg_fake):
            # Require both a strong class score and a clear margin over the other class.
            margin = abs(avg_real - avg_fake)
            if (avg_real >= self.real_threshold and avg_real > avg_fake
                    and margin >= self.min_margin):
                return "real"
            if (avg_fake >= self.fake_threshold and avg_fake > avg_real
                    and margin >= self.min_margin):
                return "fake"
            return "uncertain"

        # ── Reset helpers ─────────────────────────────────────────────────
        def _reset_precheck_state(self):
            # Reset only the active-liveness progress, keeping the loaded models alive.
            self.precheck_start    = None
            self.stable_face_start = None
            self.blink_low_counter = 0
            self.head_turn_counter = 0
            self.challenge_step    = 0

        def _reset_precheck_buffers(self):
            # Clear pre-check history so the next user/session starts fresh.
            self.precheck_buffer.clear()
            self.pc_pred_history.clear()
            self.pc_avg_real = 0.5
            self.pc_avg_fake = 0.5
            self.pc_verdict  = "uncertain"

        def _full_reset(self):
            # Reset scan-phase buffers and the lock state after face loss or user change.
            self.buffer.clear()
            self.prediction_history.clear()
            self.last_bbox   = None
            self.last_label  = "No face detected"
            self.last_colour = WHITE
            self.avg_real    = 0.5
            self.avg_fake    = 0.5
            self.fake_locked = False
            self.real_streak = 0

        # ── CNN-LSTM inference ─────────────────────────────────────────────
        def _run_inference(self, buf, hist):
            """Run one forward pass. Returns (avg_real, avg_fake) or None."""
            if len(buf) < SEQUENCE_LENGTH:
                return None
            # Stack T individual frame tensors into a single [1, T, C, H, W] batch.
            seq = torch.stack(list(buf)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out   = self.model(seq)
                probs = torch.softmax(out, dim=1)
                rp    = probs[0, 0].item()
                fp    = probs[0, 1].item()
            hist.append((rp, fp))
            avg_r = float(np.mean([p[0] for p in hist]))
            avg_f = float(np.mean([p[1] for p in hist]))
            return avg_r, avg_f

        # ── Per-frame callback ────────────────────────────────────────────
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            now = time.time()
            self._fps_times.append(now)
            fps = 0.0
            if len(self._fps_times) >= 2:
                fps = (len(self._fps_times) - 1) / (
                    self._fps_times[-1] - self._fps_times[0] + 1e-6
                )

            self.frame_counter += 1
            ts_ms = int(now * 1000)

            # Face detection
            det             = self.detector.detect(img, timestamp_ms=ts_ms)
            landmark_result = self.landmarker.detect(img, ts_ms)

            face_present  = det is not None and bool(landmark_result.face_landmarks)
            face          = None
            too_close     = False
            face_centered = False
            det_score     = 0.0
            meta          = {}

            if det is not None:
                x, y, w, h, det_score = det
                curr_sig = face_signature((x, y, w, h))

                if is_new_face(self.last_face_sig, curr_sig):
                    self._reset_precheck_state()
                    self._reset_precheck_buffers()
                    self._full_reset()
                    self.phase             = PHASE_PRECHECK
                    self.spoof_block_until = 0.0

                self.last_face_sig = curr_sig
                self.last_bbox     = (x, y, w, h)
                face, meta         = crop_face_square(img, det, padding=self.padding)
                if meta:
                    too_close     = (meta["border_touch"]
                                     and meta["face_area_ratio"] > MAX_FACE_AREA_RATIO)
                    face_centered = face_is_centered(meta, img.shape)
                cv2.rectangle(img, (x, y), (x + w, y + h), self.last_colour, 2)

            # ══════════════════════════════════════════════════════════════
            # PRE-CHECK PHASE
            # ══════════════════════════════════════════════════════════════
            if self.phase == PHASE_PRECHECK:
                # A face is considered valid only when it is visible, centred, and not too close.
                face_ok = face_present and not too_close and face_centered

                # Feed pre-check anti-spoof buffer on every valid frame.
                # This provides continuous (rolling) spoof detection throughout
                # the entire challenge — a second layer on top of per-gesture
                # snapshot checks.
                if face_ok and face is not None:
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    self.precheck_buffer.append(transform(face_rgb))

                    result = self._run_inference(self.precheck_buffer,
                                                 self.pc_pred_history)
                    if result is not None:
                        self.pc_avg_real, self.pc_avg_fake = result
                        self.pc_verdict = self.decide(self.pc_avg_real,
                                                      self.pc_avg_fake)
                        # Rolling spoof detected mid-challenge → reset + cooldown
                        if self.pc_verdict == "fake" and self.challenge_step > 0:
                            # Any spoof signal during the challenge immediately forces a restart.
                            self._reset_precheck_state()
                            self._reset_precheck_buffers()
                            self.spoof_block_until = now + SPOOF_BLOCK_COOLDOWN
                            self.shared.add_log({
                                "time":  datetime.now().strftime("%H:%M:%S"),
                                "label": "SPOOF-PRECHECK",
                                "real":  self.pc_avg_real,
                                "fake":  self.pc_avg_fake,
                            })

                spoof_blocked_now = now < self.spoof_block_until

                if face_ok and not spoof_blocked_now:
                    if self.precheck_start is None:
                        self.precheck_start = now
                    if self.stable_face_start is None:
                        self.stable_face_start = now

                    lms      = landmark_result.face_landmarks[0]
                    mean_ear = (eye_aspect_ratio(lms, "left") +
                                eye_aspect_ratio(lms, "right")) / 2.0
                    mr       = mouth_open_ratio(lms)
                    elapsed  = now - self.precheck_start
                    stable_e = now - self.stable_face_start

                    pc_r = self.pc_avg_real if self.pc_pred_history else None
                    pc_f = self.pc_avg_fake if self.pc_pred_history else None

                    # ── Step 0: face centred & stable for STABLE_FACE_MIN_SEC ──
                    if self.challenge_step == 0:
                        # Gate 0 simply checks for a stable, well-positioned face before gestures.
                        draw_top_bar(img,
                            "Look at the camera — remove glasses if possible",
                            WHITE)
                        statuses = [
                            ("Keep face steady in frame",
                             stable_e >= STABLE_FACE_MIN_SEC),
                            ("Next: blink once",             False),
                            ("Then: open mouth briefly",     False),
                            ("Then: turn head left or right", False),
                        ]
                        draw_precheck_panel(img, statuses, elapsed, pc_r, pc_f)
                        if stable_e >= STABLE_FACE_MIN_SEC:
                            self.challenge_step    = 1
                            self.blink_low_counter = 0

                    # ── Step 1: blink — snapshot anti-spoof at detection moment ─
                    elif self.challenge_step == 1:
                        # Accept the blink only after enough consecutive "closed-eye" frames.
                        draw_top_bar(img,
                            "Blink once — keep looking at camera", WHITE)

                        if mean_ear < BLINK_EAR_THRESHOLD:
                            self.blink_low_counter += 1
                        else:
                            if self.blink_low_counter >= BLINK_FRAMES_REQUIRED:
                                # Rolling guard must also agree
                                if self.pc_verdict != "fake":
                                    # ── Snapshot anti-spoof at blink stage ──
                                    snap = run_snapshot_antispoof(
                                        self.model,
                                        self.precheck_buffer,
                                        DEVICE,
                                        stage=1,
                                    )
                                    if snap.passed:
                                        # Both guards passed — accept blink
                                        self.challenge_step = 2
                                    else:
                                        # Snapshot spoof at blink stage → reset
                                        self._reset_precheck_state()
                                        self._reset_precheck_buffers()
                                        self.spoof_block_until = (
                                            now + SPOOF_BLOCK_COOLDOWN
                                        )
                                        self.shared.add_log({
                                            "time":  datetime.now().strftime(
                                                "%H:%M:%S"
                                            ),
                                            "label": "SPOOF-PRECHECK",
                                            "real":  snap.real_prob or 0.0,
                                            "fake":  snap.fake_prob or 0.0,
                                        })
                            self.blink_low_counter = 0

                        statuses = [
                            ("Face steady in frame",          True),
                            ("Blink once",                    self.challenge_step >= 2),
                            ("Next: open mouth briefly",      False),
                            ("Then: turn head left or right", False),
                        ]
                        draw_precheck_panel(img, statuses, elapsed, pc_r, pc_f)

                    # ── Step 2: mouth-open — snapshot anti-spoof at detection ──
                    elif self.challenge_step == 2:
                        # Mouth opening is validated against the rolling anti-spoof verdict.
                        draw_top_bar(img, "Open your mouth briefly", WHITE)

                        if mr >= MOUTH_OPEN_THRESHOLD and self.pc_verdict != "fake":
                            # ── Snapshot anti-spoof at mouth-open stage ──
                            snap = run_snapshot_antispoof(
                                self.model,
                                self.precheck_buffer,
                                DEVICE,
                                stage=2,
                            )
                            if snap.passed:
                                # Both guards passed — accept mouth-open
                                self.challenge_step = 3
                            else:
                                # Snapshot spoof at mouth stage → reset
                                self._reset_precheck_state()
                                self._reset_precheck_buffers()
                                self.spoof_block_until = now + SPOOF_BLOCK_COOLDOWN
                                self.shared.add_log({
                                    "time":  datetime.now().strftime("%H:%M:%S"),
                                    "label": "SPOOF-PRECHECK",
                                    "real":  snap.real_prob or 0.0,
                                    "fake":  snap.fake_prob or 0.0,
                                })

                        statuses = [
                            ("Face steady in frame",           True),
                            ("Blink once",                     True),
                            ("Open mouth once",                self.challenge_step >= 3),
                            ("Then: turn head left or right",  False),
                        ]
                        draw_precheck_panel(img, statuses, elapsed, pc_r, pc_f)

                    # ── Step 3: neck-turn — snapshot anti-spoof ──────────
                    elif self.challenge_step == 3:
                        draw_top_bar(img,
                            "Turn head left or right — keep face in frame",
                            WHITE)

                        # Count consecutive frames the head is turned
                        if is_head_turned(lms, get_xy):
                            self.head_turn_counter += 1
                        else:
                            self.head_turn_counter = 0

                        if (self.head_turn_counter >= HEAD_TURN_FRAMES_REQUIRED
                                and self.pc_verdict != "fake"):
                            # ── Snapshot anti-spoof at neck-turn stage ──
                            snap = run_snapshot_antispoof(
                                self.model,
                                self.precheck_buffer,
                                DEVICE,
                                stage=3,
                            )
                            if snap.passed:
                                # Both guards passed — accept neck turn
                                self.challenge_step    = 4
                                self.head_turn_counter = 0
                            else:
                                # Snapshot spoof at neck-turn stage → reset
                                self._reset_precheck_state()
                                self._reset_precheck_buffers()
                                self.spoof_block_until = now + SPOOF_BLOCK_COOLDOWN
                                self.shared.add_log({
                                    "time":  datetime.now().strftime("%H:%M:%S"),
                                    "label": "SPOOF-PRECHECK",
                                    "real":  snap.real_prob or 0.0,
                                    "fake":  snap.fake_prob or 0.0,
                                })

                        statuses = [
                            ("Face steady in frame",          True),
                            ("Blink once",                    True),
                            ("Open mouth once",               True),
                            ("Turn head left or right",       self.challenge_step >= 4),
                        ]
                        draw_precheck_panel(img, statuses, elapsed, pc_r, pc_f)

                    # ── Step 4: all challenges passed → seed scan, transition ──
                    elif self.challenge_step == 4:
                        draw_top_bar(img,
                            "Live confirmed! Starting deep scan...", GREEN)
                        statuses = [
                            ("Face steady in frame",    True),
                            ("Blink once",              True),
                            ("Open mouth once",         True),
                            ("Turn head left or right", True),
                        ]
                        draw_precheck_panel(img, statuses, elapsed, pc_r, pc_f)

                        # Seed the scan buffer with frames already collected
                        self.buffer.clear()
                        for t in self.precheck_buffer:
                            self.buffer.append(t)
                        self.prediction_history.clear()
                        self.avg_real    = self.pc_avg_real
                        self.avg_fake    = self.pc_avg_fake
                        self.fake_locked = False
                        self.real_streak = 0
                        self.phase       = PHASE_SCAN
                        self.last_label  = "Scanning..."
                        self.last_colour = BLUE

                else:
                    # Face not OK or spoof blocked — reset stable-face timing
                    if not spoof_blocked_now:
                        self._reset_precheck_state()
                        self.last_face_sig = None
                        self._reset_precheck_buffers()

                    statuses = [
                        ("Stable face in frame",          False),
                        ("Blink once",                    False),
                        ("Open mouth once",               False),
                        ("Turn head left or right",       False),
                    ]
                    draw_precheck_panel(img, statuses, 0.0,
                                        spoof_blocked=spoof_blocked_now)

                    if spoof_blocked_now:
                        remaining = self.spoof_block_until - now
                        draw_top_bar(
                            img,
                            f"Anti-spoof failed — retry in {remaining:.1f}s",
                            RED,
                        )
                        self.last_label  = "SPOOF BLOCKED"
                        self.last_colour = RED
                    elif det is None:
                        draw_top_bar(img, "Position your face to begin", WHITE)
                        self.last_label  = "Awaiting face"
                        self.last_colour = WHITE
                    elif too_close:
                        draw_top_bar(img,
                            "Move back — too close to camera", YELLOW)
                        self.last_label  = "Move back"
                        self.last_colour = YELLOW
                    elif not face_centered:
                        draw_top_bar(img,
                            "Centre your face in the frame", YELLOW)
                        self.last_label  = "Centre face"
                        self.last_colour = YELLOW

            # ══════════════════════════════════════════════════════════════
            # SCAN PHASE
            # ══════════════════════════════════════════════════════════════
            else:
                if det is not None:
                    self.miss_counter = 0
                    if face is not None:
                        too_close2 = (meta.get("border_touch", False) and
                                      meta.get("face_area_ratio", 0) > 0.20)
                        if too_close2:
                            self._full_reset()
                            self.last_label  = "Move back"
                            self.last_colour = YELLOW
                        elif self.frame_counter % FRAME_STRIDE == 0:
                            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                            self.buffer.append(transform(face_rgb))

                elif self.last_bbox is not None:
                    self.miss_counter += 1
                    fallback_face, _ = crop_face_square(
                        img, self.last_bbox + (1.0,), padding=self.padding
                    )
                    if (self.frame_counter % FRAME_STRIDE == 0
                            and fallback_face is not None):
                        self.buffer.append(
                            transform(cv2.cvtColor(fallback_face,
                                                   cv2.COLOR_BGR2RGB))
                        )
                    if self.miss_counter >= MISS_TOLERANCE:
                        self._full_reset()
                        self.miss_counter = 0
                        self.phase        = PHASE_PRECHECK
                        self._reset_precheck_state()
                        self._reset_precheck_buffers()
                        self.last_face_sig     = None
                        self.spoof_block_until = 0.0
                else:
                    self.last_label  = "No face detected"
                    self.last_colour = WHITE

                # CNN-LSTM inference
                result = self._run_inference(self.buffer,
                                             self.prediction_history)
                if result is not None:
                    self.avg_real, self.avg_fake = result
                    raw = self.decide(self.avg_real, self.avg_fake)

                    # ── Fake-lock logic ────────────────────────────────────
                    # Once FAKE triggers, fake_locked stays True until
                    # FAKE_LOCK_COUNT (8) *consecutive* REAL predictions are
                    # seen.  Any non-REAL frame resets real_streak to 0.
                    if raw == "fake":
                        if not self.fake_locked:
                            self.shared.add_log({
                                "time":  datetime.now().strftime("%H:%M:%S"),
                                "label": "FAKE",
                                "real":  self.avg_real,
                                "fake":  self.avg_fake,
                            })
                        self.fake_locked = True
                        self.real_streak = 0   # any FAKE resets streak
                    elif self.fake_locked:
                        # Locked: only count consecutive REAL predictions
                        if raw == "real":
                            self.real_streak += 1
                        else:
                            # UNCERTAIN resets the unlock streak
                            self.real_streak = 0
                        if self.real_streak >= FAKE_LOCK_COUNT:
                            self.fake_locked = False
                            self.real_streak = 0
                    else:
                        if raw == "real":
                            self.shared.add_log({
                                "time":  datetime.now().strftime("%H:%M:%S"),
                                "label": "REAL",
                                "real":  self.avg_real,
                                "fake":  self.avg_fake,
                            })

                    if self.fake_locked:
                        self.last_label, self.last_colour = "FAKE",      RED
                    elif raw == "real":
                        self.last_label, self.last_colour = "REAL",      GREEN
                    elif raw == "uncertain":
                        self.last_label, self.last_colour = "UNCERTAIN", YELLOW
                    else:
                        self.last_label, self.last_colour = "FAKE",      RED

                # Scan-phase HUD
                fh_i, fw_i = img.shape[:2]
                bar_w = 260
                bar_x = (fw_i - bar_w) // 2

                if len(self.buffer) == SEQUENCE_LENGTH:
                    draw_top_bar(img,
                        f"Deep Scan Active  |  {self.last_label}",
                        self.last_colour)
                    draw_confidence_bar(img, self.avg_real, self.avg_fake,
                                        bar_x, fh_i - 46, bar_w)
                else:
                    draw_top_bar(img, "Deep Scan  |  Collecting frames...",
                                 BLUE)
                    draw_fill_bar(img, len(self.buffer), SEQUENCE_LENGTH,
                                  bar_x, fh_i - 46, bar_w)

                # Large centred result label
                fs, tk = 1.1, 2
                (tw, _), _ = cv2.getTextSize(
                    self.last_label, cv2.FONT_HERSHEY_SIMPLEX, fs, tk
                )
                lx = (fw_i - tw) // 2
                ly = fh_i - 60
                cv2.putText(img, self.last_label, (lx + 2, ly + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, DARK, tk + 2)
                cv2.putText(img, self.last_label, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, self.last_colour, tk)

            # Debug overlay
            if self.show_debug:
                lines = [
                    f"FPS: {fps:.1f}",
                    f"Phase: {self.phase}",
                    f"Scan buf:   {len(self.buffer)}/{SEQUENCE_LENGTH}",
                    f"PC buf:     {len(self.precheck_buffer)}/{SEQUENCE_LENGTH}",
                    f"PC verdict: {self.pc_verdict}",
                    f"PC real:    {self.pc_avg_real:.2f}",
                    f"PC fake:    {self.pc_avg_fake:.2f}",
                    f"Det conf:   {det_score:.2f}",
                    f"Head turn:  {self.head_turn_counter}/{HEAD_TURN_FRAMES_REQUIRED}",
                    f"Fake lock:  {self.fake_locked} streak={self.real_streak}/{FAKE_LOCK_COUNT}",
                    f"Device: {DEVICE}",
                ]
                y0 = img.shape[0] - 220
                cv2.rectangle(img, (8, y0 - 14),
                              (240, y0 + len(lines) * 18 + 4), DARK, -1)
                for i, line in enumerate(lines):
                    cv2.putText(img, line, (12, y0 + i * 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1)

            # Push state to Streamlit UI
            cmap = {GREEN: "green", RED: "red", YELLOW: "orange",
                    WHITE: "gray",  BLUE: "blue", DARK: "gray"}
            self.shared.update({
                "phase":            self.phase,
                "label":            self.last_label,
                "label_colour":     cmap.get(self.last_colour, "gray"),
                "avg_real":         self.avg_real,
                "avg_fake":         self.avg_fake,
                "challenge_step":   self.challenge_step,
                "pc_avg_real":      self.pc_avg_real,
                "pc_avg_fake":      self.pc_avg_fake,
                "pc_buf_fill":      len(self.precheck_buffer),
                "pc_spoof_blocked": now < self.spoof_block_until,
                "buffer_fill":      len(self.buffer),
                "fps":              fps,
                "detection_score":  det_score,
                "face_found":       det is not None,
            })

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    return LivenessProcessor


# ═══════════════════════════════════════════════════════════════════════════════
#  Streamlit UI
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Face Liveness Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d0f14;
    color: #e2e8f0;
}
section[data-testid="stSidebar"] {
    background: #05070d;
    border-right: 1px solid #1e2836;
}

section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

section[data-testid="stSidebar"] label {
    color: #ffffff !important;
    font-size: 0.8rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    background: #05070d;
}

section[data-testid="stSidebar"] button {
    background: #111827 !important;
    border: 1px solid #334155 !important;
    color: #ffffff !important;
}
.liveness-card {
    background: #111827;
    border: 1px solid #1e2836;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 14px;
}
.liveness-card h4 {
    margin: 0 0 8px 0;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748b;
}
.result-badge {
    display: inline-block;
    padding: 6px 22px;
    border-radius: 50px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    margin: 4px 0;
}
.badge-real     { background: #052e16; color: #4ade80; border: 1.5px solid #16a34a; }
.badge-fake     { background: #2d0a0a; color: #f87171; border: 1.5px solid #dc2626; }
.badge-uncertain{ background: #2d1f00; color: #fbbf24; border: 1.5px solid #d97706; }
.badge-scanning { background: #0f1d35; color: #60a5fa; border: 1.5px solid #2563eb; }
.badge-waiting  { background: #1a1f2e; color: #94a3b8; border: 1.5px solid #334155; }
.badge-blocked  { background: #2d0a0a; color: #f87171; border: 1.5px dashed #dc2626; }

.prob-bar-wrap { height: 10px; background: #1e2836; border-radius: 6px; overflow: hidden; margin: 6px 0 2px; }
.prob-bar-fill { height: 100%; border-radius: 6px; transition: width 0.3s ease; }
.bar-real { background: linear-gradient(90deg, #16a34a, #4ade80); }
.bar-fake { background: linear-gradient(90deg, #dc2626, #f87171); }

.log-row {
    display: flex; align-items: center; gap: 12px;
    padding: 8px 12px; border-bottom: 1px solid #1e2836;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem;
}
.log-row:hover { background: #1a2030; }
.log-time  { color: #64748b; min-width: 64px; }
.log-real  { color: #4ade80; }
.log-fake  { color: #f87171; }
.log-block { color: #fb923c; }
.log-prob  { color: #94a3b8; }

.phase-pill {
    display: inline-block; padding: 3px 12px; border-radius: 50px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase;
}
.pill-precheck { background: #1e2836; color: #94a3b8; border: 1px solid #334155; }
.pill-scan     { background: #0f1d35; color: #60a5fa; border: 1px solid #1d4ed8; }

.metric-box { text-align: center; }
.metric-val { font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-weight: 700; color: #e2e8f0; }
.metric-lbl { font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase; color: #64748b; margin-top: 2px; }

.step { display: flex; align-items: center; gap: 10px; padding: 7px 0; font-size: 0.88rem; }
.step-done { color: #4ade80; }
.step-todo { color: #475569; }

.section-hdr {
    font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase;
    color: #475569; margin: 18px 0 10px;
    border-bottom: 1px solid #1e2836; padding-bottom: 6px;
}
.notice-box {
    background: #1a0000; border: 1px solid #7f1d1d; border-radius: 8px;
    padding: 10px 16px; margin: 8px 0; font-size: 0.82rem; color: #fca5a5;
}
</style>
""", unsafe_allow_html=True)

# ── Session state — initialise BEFORE webrtc_streamer ─────────────────────────
# This is the only place we create SharedState.  After this point, pass
# st.session_state.shared_state as a plain Python object; never read
# st.session_state from inside a VideoProcessorBase method.
if "shared_state" not in st.session_state:
    st.session_state.shared_state = SharedState()

# ── Sidebar ────────────────────────────────────────────────────────────────────

# Fixed reviewed defaults. Threshold controls removed from UI.
real_threshold = DEFAULT_REAL_THRESHOLD
fake_threshold = DEFAULT_FAKE_THRESHOLD
min_margin     = DEFAULT_MIN_MARGIN
padding        = DEFAULT_PADDING

with st.sidebar:
    st.markdown("## 🔍 Liveness Detection")

    st.markdown("### How it works")
    st.markdown("""
**Pre-check challenge phase**
- Face must be centred & stable for 1.2 s
- CNN-LSTM runs *continuously* on every frame (rolling guard)
- **Blink** → snapshot anti-spoof fires at detection moment
- **Mouth open** → snapshot anti-spoof fires at detection moment
- **Neck turn** → snapshot anti-spoof fires at detection moment
- Either guard firing FAKE → instant reset + cooldown

**Deep scan phase**
- CNN-LSTM runs on 10-frame sequences, smoothed over 8 predictions
- Fake-lock: requires **8 consecutive REAL** predictions to unlock
- UNCERTAIN resets the real-streak counter
    """)
    
    st.markdown("### Debug")
    show_debug = st.checkbox("Show debug overlay", value=False)

    if st.button("Clear log"):
        st.session_state.shared_state.update({"log": []})

# ── Layout ─────────────────────────────────────────────────────────────────────
st.markdown("# Face Liveness Detection")
st.markdown(
    "**Three-layer anti-spoof:** CNN-LSTM is active during **every** stage. "
    "Continuous rolling guard + per-gesture snapshot at blink, mouth-open and "
    "neck-turn, plus a full deep scan."
)

col_cam, col_status = st.columns([3, 2], gap="large")

# ── Build the processor factory in the MAIN THREAD ────────────────────────────
# Capture shared_state and all config values here, before handing off to
# webrtc_streamer.  The background thread never needs st.session_state.
_shared = st.session_state.shared_state
_config = {
    "real_threshold": real_threshold,
    "fake_threshold": fake_threshold,
    "min_margin":     min_margin,
    "padding":        padding,
    "show_debug":     show_debug,
}

with col_cam:
    ctx = webrtc_streamer(
        key="liveness",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=make_processor_factory(_shared, _config),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_status:
    result_ph     = st.empty()
    phase_ph      = st.empty()
    precheck_ph   = st.empty()
    confidence_ph = st.empty()
    metrics_ph    = st.empty()

st.markdown('<div class="section-hdr">Session Detection Log</div>',
            unsafe_allow_html=True)
log_ph = st.empty()
st.markdown('<div class="section-hdr">Session Summary</div>',
            unsafe_allow_html=True)
stats_ph = st.empty()

# Four-step challenge labels (must match challenge_step 0..3 in the processor)
CHALLENGE_LABELS = [
    "Keep face steady in frame",
    "Blink once",
    "Open mouth once",
    "Turn head left or right",
]

# ── Live update loop ───────────────────────────────────────────────────────────
if ctx.state.playing:
    while True:
        snap = st.session_state.shared_state.snapshot()

        phase       = snap["phase"]
        label       = snap["label"]
        avg_real    = snap["avg_real"]
        avg_fake    = snap["avg_fake"]
        c_step      = snap["challenge_step"]
        pc_avg_real = snap["pc_avg_real"]
        pc_avg_fake = snap["pc_avg_fake"]
        pc_buf_fill = snap["pc_buf_fill"]
        pc_blocked  = snap["pc_spoof_blocked"]
        buf_fill    = snap["buffer_fill"]
        fps         = snap["fps"]
        det_score   = snap["detection_score"]
        face_found  = snap["face_found"]
        log         = snap["log"]

        reals  = sum(1 for e in log if e["label"] == "REAL")
        fakes  = sum(1 for e in log if e["label"] == "FAKE")
        blocks = sum(1 for e in log if e["label"] == "SPOOF-PRECHECK")

        # Result badge
        badge_cls = {
            "REAL":          "badge-real",
            "FAKE":          "badge-fake",
            "UNCERTAIN":     "badge-uncertain",
            "SPOOF BLOCKED": "badge-blocked",
        }.get(label, "badge-scanning" if phase == PHASE_SCAN else "badge-waiting")

        with result_ph.container():
            st.markdown(
                f'<div class="liveness-card"><h4>Current Result</h4>'
                f'<span class="result-badge {badge_cls}">{label}</span></div>',
                unsafe_allow_html=True,
            )

        pill_cls  = "pill-scan" if phase == PHASE_SCAN else "pill-precheck"
        phase_lbl = "Deep Scan" if phase == PHASE_SCAN else "Pre-check"
        with phase_ph.container():
            st.markdown(
                f'<span class="phase-pill {pill_cls}">{phase_lbl}</span>'
                f'&nbsp;&nbsp;<span style="color:#64748b;font-size:0.78rem;">'
                f'FPS: {fps:.0f} &nbsp;|&nbsp; '
                f'Face: {"✓" if face_found else "✗"} &nbsp;|&nbsp; '
                f'Det: {det_score:.2f}</span>',
                unsafe_allow_html=True,
            )

        # Pre-check challenge panel
        if phase == PHASE_PRECHECK:
            steps_html = ""
            for i, lbl in enumerate(CHALLENGE_LABELS):
                done  = i < c_step
                cls   = "step-done" if done else "step-todo"
                icon  = "✓" if done else "○"
                color = "#e2e8f0" if done else "#475569"
                steps_html += (
                    f'<div class="step">'
                    f'<span class="{cls}">{icon}</span>'
                    f'<span style="color:{color}">{lbl}</span>'
                    f'</div>'
                )

            pc_bar_html = ""
            if pc_buf_fill > 0:
                pc_bar_html = (
                    f'<div style="margin-top:12px;padding-top:10px;'
                    f'border-top:1px solid #1e2836">'
                    f'<div style="font-size:0.68rem;letter-spacing:0.1em;'
                    f'text-transform:uppercase;color:#64748b;margin-bottom:6px">'
                    f'Anti-spoof guard &nbsp;•&nbsp; '
                    f'buffer {pc_buf_fill}/{SEQUENCE_LENGTH}</div>'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-size:0.78rem;color:#94a3b8">'
                    f'<span>Real</span><span>{pc_avg_real:.1%}</span></div>'
                    f'<div class="prob-bar-wrap"><div class="prob-bar-fill bar-real" '
                    f'style="width:{pc_avg_real*100:.1f}%"></div></div>'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-size:0.78rem;color:#94a3b8;margin-top:6px">'
                    f'<span>Fake</span><span>{pc_avg_fake:.1%}</span></div>'
                    f'<div class="prob-bar-wrap"><div class="prob-bar-fill bar-fake" '
                    f'style="width:{pc_avg_fake*100:.1f}%"></div></div>'
                    f'</div>'
                )

            blocked_html = ""
            if pc_blocked:
                blocked_html = (
                    '<div class="notice-box">⛔ Spoof detected during challenge — '
                    'blocked. Please wait, then try again with a real face.</div>'
                )

            with precheck_ph.container():
                st.markdown(
                    f'<div class="liveness-card"><h4>Liveness Challenge</h4>'
                    f'{steps_html}{blocked_html}{pc_bar_html}</div>',
                    unsafe_allow_html=True,
                )
            confidence_ph.empty()

        # Scan confidence panel
        else:
            precheck_ph.empty()
            buf_pct = int((buf_fill / SEQUENCE_LENGTH) * 100)
            with confidence_ph.container():
                st.markdown(
                    f'<div class="liveness-card"><h4>Confidence Scores</h4>'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-size:0.8rem;color:#94a3b8">'
                    f'<span>Real</span><span>{avg_real:.1%}</span></div>'
                    f'<div class="prob-bar-wrap"><div class="prob-bar-fill bar-real" '
                    f'style="width:{avg_real*100:.1f}%"></div></div>'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-size:0.8rem;color:#94a3b8;margin-top:8px">'
                    f'<span>Fake</span><span>{avg_fake:.1%}</span></div>'
                    f'<div class="prob-bar-wrap"><div class="prob-bar-fill bar-fake" '
                    f'style="width:{avg_fake*100:.1f}%"></div></div>'
                    f'<div style="margin-top:10px;font-size:0.75rem;color:#64748b">'
                    f'Frame buffer: {buf_fill}/{SEQUENCE_LENGTH} ({buf_pct}%)'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

        # Metrics
        total = reals + fakes
        with metrics_ph.container():
            m1, m2, m3, m4 = st.columns(4)
            for col_w, val, lbl_m, color in [
                (m1, total,  "Total",   "#e2e8f0"),
                (m2, reals,  "Real",    "#4ade80"),
                (m3, fakes,  "Fake",    "#f87171"),
                (m4, blocks, "Blocked", "#fb923c"),
            ]:
                with col_w:
                    st.markdown(
                        f'<div class="metric-box">'
                        f'<div class="metric-val" style="color:{color}">{val}</div>'
                        f'<div class="metric-lbl">{lbl_m}</div></div>',
                        unsafe_allow_html=True,
                    )

        # Log
        if log:
            rows = ""
            for e in log[:20]:
                lbl_e   = e["label"]
                badge_e = (
                    "log-real"  if lbl_e == "REAL" else
                    "log-block" if lbl_e == "SPOOF-PRECHECK" else
                    "log-fake"
                )
                rows += (
                    f'<div class="log-row">'
                    f'<span class="log-time">{e["time"]}</span>'
                    f'<span class="{badge_e}">{lbl_e}</span>'
                    f'<span class="log-prob">'
                    f'R:{e["real"]:.1%} F:{e["fake"]:.1%}</span>'
                    f'</div>'
                )
            log_ph.markdown(
                f'<div class="liveness-card" style="padding:0;overflow:hidden">'
                f'{rows}</div>',
                unsafe_allow_html=True,
            )
        else:
            log_ph.markdown(
                '<div class="liveness-card" style="color:#475569;font-size:0.85rem">'
                'No detections yet. Start the webcam and complete the pre-check.'
                '</div>',
                unsafe_allow_html=True,
            )

        # Stats bar
        pct_real = (reals / total * 100) if total > 0 else 0
        with stats_ph.container():
            st.markdown(
                f'<div class="liveness-card" style="display:flex;gap:32px;'
                f'align-items:center">'
                f'<div><span style="color:#64748b;font-size:0.75rem;">'
                f'Session real rate</span><br>'
                f'<span style="font-family:IBM Plex Mono,monospace;'
                f'font-size:1.1rem;color:#4ade80">{pct_real:.0f}%</span></div>'
                f'<div style="flex:1"><div class="prob-bar-wrap" style="height:14px">'
                f'<div class="prob-bar-fill bar-real" '
                f'style="width:{pct_real:.1f}%"></div>'
                f'</div></div>'
                f'<div><span style="color:#64748b;font-size:0.75rem;">'
                f'Thresholds</span><br>'
                f'<span style="font-family:IBM Plex Mono,monospace;'
                f'font-size:0.8rem;color:#94a3b8">'
                f'R:{real_threshold:.2f} F:{fake_threshold:.2f}</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        time.sleep(0.25)

else:
    with result_ph.container():
        st.markdown(
            '<div class="liveness-card"><h4>Current Result</h4>'
            '<span class="result-badge badge-waiting">OFFLINE</span></div>',
            unsafe_allow_html=True,
        )
    st.info(
        "▶ Click **START** above to activate your webcam "
        "and begin liveness detection."
    )
    log_ph.markdown(
        '<div class="liveness-card" style="color:#475569;font-size:0.85rem">'
        'Webcam is not active.</div>',
        unsafe_allow_html=True,
    )

# References:
# - Streamlit Session State Docs: https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state
# - Streamlit Session State Concepts: https://docs.streamlit.io/develop/concepts/architecture/session-state
# - streamlit-webrtc Repository / Docs: https://github.com/whitphx/streamlit-webrtc
# - MediaPipe Face Detector (Python): https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector/python
# - MediaPipe Face Landmarker (Python): https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python
# - Torchvision Models Documentation: https://docs.pytorch.org/vision/main/models.html
# - Torchvision Transforms Documentation: https://docs.pytorch.org/vision/main/transforms.html
# - PyTorch Saving and Loading Models Tutorial: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html