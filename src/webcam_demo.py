"""
webcam_demo.py  -  Liveness Detection with Active Liveness Challenges
======================================================================

Detection flow
--------------

PHASE_PRECHECK  (active liveness — three sequential challenges)
  Step 0  : Face stability gate
              The user must hold their face centred and steady for
              STABLE_FACE_MIN_SEC seconds before any challenge begins.

  Step 1  : Blink challenge  +  anti-spoof gate
              User must blink.  At the moment the blink is confirmed the
              CNN-LSTM model is run on the rolling face buffer.  Both the
              blink AND a real-prob score >= ACTIVE_SPOOF_THRESHOLD are
              required to advance.

  Step 2  : Mouth-open challenge  +  anti-spoof gate
              User must open their mouth.  Same dual-gate logic applies.

  Step 3  : Head-turn challenge  +  anti-spoof gate
              User must turn their head left OR right.  Same dual-gate
              logic applies.  Head turn is measured via nose-tip deviation
              from the inter-ocular midpoint using MediaPipe landmarks.

  If the anti-spoof model returns FAKE at any gate the session is
  immediately reset with a visible alert; the user must restart from Step 0.

PHASE_SCAN  (passive liveness — unchanged from original)
  The CNN-LSTM model runs continuously on a rolling 10-frame buffer and
  outputs a smoothed REAL / FAKE / UNCERTAIN verdict with a confidence bar.

Key design principles
---------------------
* Reuses the existing CNNLSTMModel and MediaPipe pipeline unchanged.
* active_liveness.py owns all gesture geometry and snapshot logic.
* face_utils.py and model.py are 100% unmodified.

References at the end
"""

from collections import deque
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
import torch
import torchvision.transforms as transforms
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from model import CNNLSTMModel
from face_utils import MediaPipeFaceDetector, crop_face_square
from active_liveness import (
    run_snapshot_antispoof,
    is_head_turned,
    head_turn_deviation,
    ACTIVE_SPOOF_THRESHOLD,
    HEAD_TURN_FRAMES_REQUIRED,
)

# ─────────────────────────────────────────────────────────────────
# Paths & device
# ─────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR        = Path(__file__).resolve().parent.parent
MODEL_PATH      = str(BASE_DIR / "models" / "best_model.pth")
DETECTOR_MODEL  = str(BASE_DIR / "models" / "face_detector.task")
LANDMARKER_MODEL = str(BASE_DIR / "models" / "face_landmarker.task")

# ─────────────────────────────────────────────────────────────────
# Scan-phase settings (unchanged)
# ─────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH  = 10
FRAME_STRIDE     = 1
MISS_TOLERANCE   = 5
SMOOTHING_WINDOW = 8

REAL_THRESHOLD  = 0.65
FAKE_THRESHOLD  = 0.65
MIN_MARGIN      = 0.10
FAKE_LOCK_COUNT = 8
PADDING         = 0.30

# ─────────────────────────────────────────────────────────────────
# Pre-check / active liveness settings
# ─────────────────────────────────────────────────────────────────
PRECHECK_REQUIRED_SEC   = 4.5
STABLE_FACE_MIN_SEC     = 1.2

BLINK_EAR_THRESHOLD     = 0.21   # EAR below this -> eye considered closed
BLINK_FRAMES_REQUIRED   = 2      # consecutive frames below threshold to confirm blink

MOUTH_OPEN_THRESHOLD    = 0.045  # mouth-height / mouth-width ratio

CENTER_TOLERANCE_RATIO  = 0.22
MAX_FACE_AREA_RATIO     = 0.28

SPOOF_ALERT_DURATION    = 2.5    # seconds to display spoof-fail alert

# ─────────────────────────────────────────────────────────────────
# Colours
# ─────────────────────────────────────────────────────────────────
GREEN  = (0, 220, 0)
RED    = (0, 0, 220)
YELLOW = (0, 200, 220)
WHITE  = (255, 255, 255)
DARK   = (30, 30, 30)
BLUE   = (255, 180, 60)
ORANGE = (0, 140, 255)

# ─────────────────────────────────────────────────────────────────
# Phase identifiers
# ─────────────────────────────────────────────────────────────────
PHASE_PRECHECK = "precheck"
PHASE_SCAN     = "scan"

# challenge_step:
#   0 -> face stability gate
#   1 -> blink + antispoof
#   2 -> mouth open + antispoof
#   3 -> head turn + antispoof
#   4 -> transition to PHASE_SCAN

# ─────────────────────────────────────────────────────────────────
# Image transform (identical to training pipeline)
# ─────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ═══════════════════════════════════════════════════════════════════
# MediaPipe Face Landmarker wrapper
# ═══════════════════════════════════════════════════════════════════
class MediaPipeFaceLandmarker:
    def __init__(
        self,
        model_path: str,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Face landmarker model not found: {model_path}\n"
                "Download the MediaPipe Face Landmarker task model and place it there."
            )
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, bgr_image, timestamp_ms: int):
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.landmarker.detect_for_video(mp_image, timestamp_ms)

    def close(self):
        self.landmarker.close()


# ═══════════════════════════════════════════════════════════════════
# Drawing helpers
# ═══════════════════════════════════════════════════════════════════

def draw_confidence_bar(frame, real_prob, fake_prob, x, y, width=220, height=18):
    cv2.rectangle(frame, (x, y), (x + width, y + height), DARK, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), WHITE, 1)
    real_w = int(real_prob * width)
    if real_w > 0:
        cv2.rectangle(frame, (x, y), (x + real_w, y + height), GREEN, -1)
    fake_w = int(fake_prob * width)
    if fake_w > 0:
        cv2.rectangle(frame, (x + width - fake_w, y), (x + width, y + height), RED, -1)
    cv2.putText(frame, f"R:{real_prob:.0%}", (x + 3, y + height - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
    cv2.putText(frame, f"F:{fake_prob:.0%}", (x + width - 52, y + height - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)


def draw_top_instruction(frame, text, colour=WHITE):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 54), DARK, -1)
    cv2.putText(frame, text, (20, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)


def draw_precheck_box(frame, status_lines, elapsed_sec):
    """Status panel below the top instruction bar.
    Dynamically sizes to accommodate any number of status_lines items."""
    h, w = frame.shape[:2]
    box_w = 460
    box_h = 52 + len(status_lines) * 28
    x1 = (w - box_w) // 2
    y1 = 62
    x2 = x1 + box_w
    y2 = y1 + box_h

    cv2.rectangle(frame, (x1, y1), (x2, y2), DARK, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), WHITE, 1)

    title = f"Active Liveness Check  {elapsed_sec:.1f}s"
    cv2.putText(frame, title, (x1 + 14, y1 + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, BLUE, 2)

    for i, (text, ok) in enumerate(status_lines):
        colour = GREEN if ok else YELLOW
        prefix = "OK " if ok else "..."
        cv2.putText(frame, f"{prefix}  {text}",
                    (x1 + 14, y1 + 50 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, colour, 2)


def draw_spoof_alert(frame, message):
    """Prominent red banner shown when anti-spoof fails at any gate."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h // 2 - 60), (w, h // 2 + 60), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, _), _ = cv2.getTextSize(message, font, 0.72, 2)
    tx = max(10, (w - tw) // 2)
    cv2.putText(frame, message, (tx, h // 2 - 8), font, 0.72, WHITE, 2)
    sub = "Session reset -- please try again"
    (sw, _), _ = cv2.getTextSize(sub, font, 0.52, 1)
    cv2.putText(frame, sub, ((w - sw) // 2, h // 2 + 30), font, 0.52, YELLOW, 1)


def draw_antispoof_mini_bar(frame, real_prob, fake_prob, label):
    """Small bar shown briefly after a gate passes."""
    h, w = frame.shape[:2]
    bar_w = 200; bar_h = 14
    bx = (w - bar_w) // 2; by = h - 90
    cv2.putText(frame, label, (bx, by - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, ORANGE, 1)
    cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), DARK, -1)
    cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), WHITE, 1)
    real_fill = int(real_prob * bar_w)
    if real_fill > 0:
        cv2.rectangle(frame, (bx, by), (bx + real_fill, by + bar_h), GREEN, -1)
    fake_fill = int(fake_prob * bar_w)
    if fake_fill > 0:
        cv2.rectangle(frame, (bx + bar_w - fake_fill, by),
                      (bx + bar_w, by + bar_h), RED, -1)
    cv2.putText(frame, f"R:{real_prob:.0%}", (bx + 2, by + bar_h - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, WHITE, 1)
    cv2.putText(frame, f"F:{fake_prob:.0%}", (bx + bar_w - 42, by + bar_h - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, WHITE, 1)


def draw_turn_indicator(frame, deviation):
    """Small horizontal bar showing head-turn progress toward the threshold."""
    from active_liveness import HEAD_TURN_THRESHOLD as THR
    h, w = frame.shape[:2]
    bar_w = 180; bar_h = 12
    bx = (w - bar_w) // 2; by = h - 60
    cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), DARK, -1)
    cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), WHITE, 1)
    clamped  = max(-0.5, min(0.5, deviation))
    needle_x = bx + int((clamped + 0.5) * bar_w)
    cv2.line(frame, (needle_x, by), (needle_x, by + bar_h), ORANGE, 2)
    left_t  = bx + int((-THR + 0.5) * bar_w)
    right_t = bx + int(( THR + 0.5) * bar_w)
    cv2.line(frame, (left_t,  by), (left_t,  by + bar_h), GREEN, 1)
    cv2.line(frame, (right_t, by), (right_t, by + bar_h), GREEN, 1)
    cv2.putText(frame, "L", (bx - 12, by + bar_h - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1)
    cv2.putText(frame, "R", (bx + bar_w + 3, by + bar_h - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1)


# ═══════════════════════════════════════════════════════════════════
# Landmark geometry helpers  (unchanged from original)
# ═══════════════════════════════════════════════════════════════════

def euclidean(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def get_xy(landmark):
    return (landmark.x, landmark.y)


def eye_aspect_ratio(face_landmarks, eye="left"):
    if eye == "left":
        h1 = (159, 145); h2 = (158, 153); w = (33, 133)
    else:
        h1 = (386, 374); h2 = (385, 380); w = (362, 263)
    p_h1a = get_xy(face_landmarks[h1[0]]); p_h1b = get_xy(face_landmarks[h1[1]])
    p_h2a = get_xy(face_landmarks[h2[0]]); p_h2b = get_xy(face_landmarks[h2[1]])
    p_wa  = get_xy(face_landmarks[w[0]]);  p_wb  = get_xy(face_landmarks[w[1]])
    eye_height = (euclidean(p_h1a, p_h1b) + euclidean(p_h2a, p_h2b)) / 2.0
    eye_width  = euclidean(p_wa, p_wb) + 1e-6
    return eye_height / eye_width


def mouth_open_ratio(face_landmarks):
    top   = get_xy(face_landmarks[13]); bottom = get_xy(face_landmarks[14])
    left  = get_xy(face_landmarks[78]); right  = get_xy(face_landmarks[308])
    return euclidean(top, bottom) / (euclidean(left, right) + 1e-6)


def face_is_centered(meta, frame_shape):
    img_h, img_w = frame_shape[:2]
    x, y, bw, bh = meta["bbox"]
    dx = abs(x + bw / 2.0 - img_w / 2.0) / img_w
    dy = abs(y + bh / 2.0 - img_h / 2.0) / img_h
    return dx <= CENTER_TOLERANCE_RATIO and dy <= CENTER_TOLERANCE_RATIO


def face_signature_from_bbox(bbox):
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0, w * h)


def is_new_face(prev_sig, curr_sig, center_thresh=0.18, area_thresh=0.45):
    if prev_sig is None or curr_sig is None:
        return False
    px, py, pa = prev_sig; cx, cy, ca = curr_sig
    shift = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
    norm  = max(pa ** 0.5, 1.0)
    return (shift / norm > center_thresh) or (abs(ca - pa) / max(pa, 1.0) > area_thresh)


# ═══════════════════════════════════════════════════════════════════
# State-reset helpers
# ═══════════════════════════════════════════════════════════════════

def full_reset(buffer, prediction_history):
    buffer.clear(); prediction_history.clear()
    return None, "No face detected", WHITE, False, 0, 0.5, 0.5


def reset_precheck():
    # Reset the active liveness state machine without rebuilding the app.
    return (
        None,   # precheck_start_time
        None,   # stable_face_start
        0,      # blink_low_counter
        0,      # challenge_step
        0,      # head_turn_counter
    )


def decide_raw(avg_real, avg_fake):
    # Use threshold + margin logic so borderline predictions stay "uncertain".
    margin = abs(avg_real - avg_fake)
    if avg_real >= REAL_THRESHOLD and avg_real > avg_fake and margin >= MIN_MARGIN:
        return "real"
    if avg_fake >= FAKE_THRESHOLD and avg_fake > avg_real and margin >= MIN_MARGIN:
        return "fake"
    return "uncertain"


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    # ── Load model ───────────────────────────────────────────────
    model = CNNLSTMModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # ── Load MediaPipe ───────────────────────────────────────────
    detector = MediaPipeFaceDetector(
        model_path=DETECTOR_MODEL,
        running_mode="video",
        min_detection_confidence=0.6,
    )
    landmarker = MediaPipeFaceLandmarker(
        model_path=LANDMARKER_MODEL,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    print(f"Model loaded on {DEVICE}. Press Q to quit.")
    print(f"Active liveness anti-spoof threshold: {ACTIVE_SPOOF_THRESHOLD:.0%}")

    # ── Scan-phase state ─────────────────────────────────────────
    buffer             = deque(maxlen=SEQUENCE_LENGTH)
    prediction_history = deque(maxlen=SMOOTHING_WINDOW)
    frame_counter = 0
    miss_counter  = 0
    last_bbox     = None
    last_label    = "Collecting..."
    last_colour   = WHITE
    avg_real = avg_fake = 0.5
    fake_locked = False
    real_streak = 0

    # ── Precheck / active-liveness state ────────────────────────
    phase = PHASE_PRECHECK
    (precheck_start_time, stable_face_start,
     blink_low_counter, challenge_step,
     head_turn_counter) = reset_precheck()
    last_face_signature = None

    # Rolling buffer of TRANSFORMED face tensors collected during precheck.
    # maxlen matches SEQUENCE_LENGTH so padding is minimal.
    precheck_face_buffer = deque(maxlen=SEQUENCE_LENGTH)

    # Spoof-alert state
    spoof_fail_time = None
    spoof_fail_msg  = ""

    # Mini antispoof bar displayed briefly after each passed gate
    last_gate_result          = None
    gate_result_show_until    = 0.0

    # ── Webcam ───────────────────────────────────────────────────
    # Open the default webcam; index 0 is usually the built-in camera.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1
            timestamp_ms  = int(time.time() * 1000)
            now           = time.time()

            # ── Detection ────────────────────────────────────────
            det             = detector.detect(frame, timestamp_ms=timestamp_ms)
            landmark_result = landmarker.detect(frame, timestamp_ms=timestamp_ms)

            face_present = det is not None and landmark_result.face_landmarks
            face         = None
            too_close    = False
            face_centered = False

            if det is not None:
                x, y, w, h, score = det
                curr_face_sig = face_signature_from_bbox((x, y, w, h))

                # New face in frame -> full session reset
                if last_face_signature is not None and is_new_face(
                        last_face_signature, curr_face_sig):
                    (precheck_start_time, stable_face_start,
                     blink_low_counter, challenge_step,
                     head_turn_counter) = reset_precheck()
                    phase = PHASE_PRECHECK
                    precheck_face_buffer.clear()
                    buffer.clear(); prediction_history.clear()
                    avg_real = avg_fake = 0.5
                    fake_locked = False; real_streak = 0
                    spoof_fail_time = None

                last_face_signature = curr_face_sig
                last_bbox = (x, y, w, h)
                face, meta = crop_face_square(frame, det, padding=PADDING)
                too_close    = meta["border_touch"] and meta["face_area_ratio"] > MAX_FACE_AREA_RATIO
                face_centered = face_is_centered(meta, frame.shape)
                cv2.rectangle(frame, (x, y), (x + w, y + h), last_colour, 2)

            # ════════════════════════════════════════════════════
            # PHASE: PRECHECK  (active liveness)
            # ════════════════════════════════════════════════════
            if phase == PHASE_PRECHECK:

                # -- Spoof-alert hold --------------------------------
                # When an antispoof gate fails we freeze on the alert
                # for SPOOF_ALERT_DURATION seconds then wipe everything.
                if spoof_fail_time is not None:
                    draw_top_instruction(frame, "  LIVENESS FAILED  --  Spoof detected", RED)
                    draw_spoof_alert(frame, spoof_fail_msg)

                    if now - spoof_fail_time >= SPOOF_ALERT_DURATION:
                        (precheck_start_time, stable_face_start,
                         blink_low_counter, challenge_step,
                         head_turn_counter) = reset_precheck()
                        precheck_face_buffer.clear()
                        spoof_fail_time = None; spoof_fail_msg = ""

                    cv2.imshow("Liveness Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue

                # -- Normal precheck logic ---------------------------
                face_ok = (face_present and det is not None
                           and not too_close and face_centered)

                if face_ok:
                    if precheck_start_time is None:
                        precheck_start_time = now
                    if stable_face_start is None:
                        stable_face_start = now

                    face_landmarks = landmark_result.face_landmarks[0]
                    left_ear   = eye_aspect_ratio(face_landmarks, "left")
                    right_ear  = eye_aspect_ratio(face_landmarks, "right")
                    mean_ear   = (left_ear + right_ear) / 2.0
                    mouth_ratio = mouth_open_ratio(face_landmarks)
                    turn_dev    = head_turn_deviation(face_landmarks, get_xy)

                    elapsed        = now - precheck_start_time
                    stable_elapsed = now - stable_face_start

                    # Always feed the precheck buffer so snapshots are fresh.
                    if face is not None and frame_counter % FRAME_STRIDE == 0:
                        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        precheck_face_buffer.append(transform(face_rgb))

                    # ── Step 0: Stability gate ───────────────────
                    if challenge_step == 0:
                        draw_top_instruction(
                            frame,
                            "Face detected. Look at the camera and hold still.",
                            WHITE,
                        )
                        statuses = [
                            ("Keep face steady",  stable_elapsed >= STABLE_FACE_MIN_SEC),
                            ("Next: blink once",  False),
                            ("Then: open mouth",  False),
                            ("Then: turn head",   False),
                        ]
                        draw_precheck_box(frame, statuses, elapsed)

                        if stable_elapsed >= STABLE_FACE_MIN_SEC:
                            challenge_step    = 1
                            blink_low_counter = 0

                    # ── Step 1: Blink + antispoof gate ───────────
                    elif challenge_step == 1:
                        draw_top_instruction(frame, "Blink once", WHITE)
                        statuses = [
                            ("Stable face in view", True),
                            ("Blink once",          False),
                            ("Next: open mouth",    False),
                            ("Then: turn head",     False),
                        ]
                        draw_precheck_box(frame, statuses, elapsed)

                        if mean_ear < BLINK_EAR_THRESHOLD:
                            blink_low_counter += 1
                        else:
                            if blink_low_counter >= BLINK_FRAMES_REQUIRED:
                                # Gesture confirmed -- run antispoof snapshot
                                result = run_snapshot_antispoof(
                                    model, precheck_face_buffer, DEVICE, stage=1)
                                if result.passed:
                                    challenge_step         = 2
                                    last_gate_result       = result
                                    gate_result_show_until = now + 1.2
                                else:
                                    spoof_fail_time = now
                                    spoof_fail_msg  = (
                                        f"SPOOF at blink  "
                                        f"real={result.real_prob:.0%}  "
                                        f"fake={result.fake_prob:.0%}"
                                    )
                            blink_low_counter = 0

                    # ── Step 2: Mouth open + antispoof gate ──────
                    elif challenge_step == 2:
                        draw_top_instruction(frame, "Open your mouth wide", WHITE)
                        statuses = [
                            ("Stable face in view", True),
                            ("Blink once",          True),
                            ("Open mouth wide",     False),
                            ("Then: turn head",     False),
                        ]
                        draw_precheck_box(frame, statuses, elapsed)

                        if mouth_ratio >= MOUTH_OPEN_THRESHOLD:
                            result = run_snapshot_antispoof(
                                model, precheck_face_buffer, DEVICE, stage=2)
                            if result.passed:
                                challenge_step         = 3
                                last_gate_result       = result
                                gate_result_show_until = now + 1.2
                            else:
                                spoof_fail_time = now
                                spoof_fail_msg  = (
                                    f"SPOOF at mouth  "
                                    f"real={result.real_prob:.0%}  "
                                    f"fake={result.fake_prob:.0%}"
                                )

                    # ── Step 3: Head turn + antispoof gate ───────
                    elif challenge_step == 3:
                        draw_top_instruction(frame, "Turn your head left or right", WHITE)
                        statuses = [
                            ("Stable face in view",  True),
                            ("Blink once",           True),
                            ("Open mouth wide",      True),
                            ("Turn head left/right", False),
                        ]
                        draw_precheck_box(frame, statuses, elapsed)
                        draw_turn_indicator(frame, turn_dev)

                        if is_head_turned(face_landmarks, get_xy):
                            head_turn_counter += 1
                        else:
                            head_turn_counter = 0  # reset if head straightens

                        if head_turn_counter >= HEAD_TURN_FRAMES_REQUIRED:
                            result = run_snapshot_antispoof(
                                model, precheck_face_buffer, DEVICE, stage=3)
                            if result.passed:
                                challenge_step         = 4
                                last_gate_result       = result
                                gate_result_show_until = now + 0.8
                            else:
                                spoof_fail_time = now
                                spoof_fail_msg  = (
                                    f"SPOOF at head-turn  "
                                    f"real={result.real_prob:.0%}  "
                                    f"fake={result.fake_prob:.0%}"
                                )
                            head_turn_counter = 0

                    # ── Step 4: All gates passed -> PHASE_SCAN ───
                    elif challenge_step == 4:
                        draw_top_instruction(
                            frame, "All checks passed. Starting scan...", GREEN)
                        statuses = [
                            ("Stable face in view", True),
                            ("Blink once",          True),
                            ("Open mouth wide",     True),
                            ("Turn head",           True),
                        ]
                        draw_precheck_box(frame, statuses, elapsed)

                        phase = PHASE_SCAN
                        last_label  = "Live face confirmed"
                        last_colour = GREEN
                        buffer.clear(); prediction_history.clear()
                        precheck_face_buffer.clear()
                        avg_real = avg_fake = 0.5
                        fake_locked = False; real_streak = 0

                    # -- Mini antispoof bar after each passed gate
                    if (last_gate_result is not None
                            and not last_gate_result.insufficient_frames
                            and now < gate_result_show_until):
                        draw_antispoof_mini_bar(
                            frame,
                            last_gate_result.real_prob,
                            last_gate_result.fake_prob,
                            f"Anti-spoof [{last_gate_result.stage_name}]: passed",
                        )

                else:
                    # Face absent / too close / off-centre -> full precheck reset
                    (precheck_start_time, stable_face_start,
                     blink_low_counter, challenge_step,
                     head_turn_counter) = reset_precheck()
                    last_face_signature = None
                    precheck_face_buffer.clear()
                    buffer.clear(); prediction_history.clear()

                    statuses = [
                        ("Stable face in view", False),
                        ("Blink once",          False),
                        ("Open mouth wide",     False),
                        ("Turn head",           False),
                    ]
                    draw_precheck_box(frame, statuses, 0.0)

                    if det is None:
                        draw_top_instruction(
                            frame, "Show your face to begin verification", WHITE)
                        last_label  = "Show your face"
                        last_colour = WHITE
                    elif too_close:
                        draw_top_instruction(frame, "Move back slightly", YELLOW)
                        last_label  = "MOVE BACK SLIGHTLY"
                        last_colour = YELLOW
                    elif not face_centered:
                        draw_top_instruction(frame, "Center your face", YELLOW)
                        last_label  = "Center your face"
                        last_colour = YELLOW

            # ════════════════════════════════════════════════════
            # PHASE: SCAN  (passive liveness — unchanged)
            # ════════════════════════════════════════════════════
            else:
                if det is not None:
                    miss_counter = 0
                    if face is not None:
                        too_close = meta["border_touch"] and meta["face_area_ratio"] > 0.20
                        if too_close:
                            (last_bbox, last_label, last_colour, fake_locked,
                             real_streak, avg_real, avg_fake) = full_reset(
                                buffer, prediction_history)
                            last_label  = "MOVE BACK SLIGHTLY"
                            last_colour = YELLOW
                        elif frame_counter % FRAME_STRIDE == 0:
                            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                            buffer.append(transform(face_rgb))

                elif last_bbox is not None:
                    miss_counter += 1
                    fallback_face, _ = crop_face_square(
                        frame, last_bbox + (1.0,), padding=PADDING)
                    if frame_counter % FRAME_STRIDE == 0 and fallback_face is not None:
                        fallback_face_rgb = cv2.cvtColor(fallback_face, cv2.COLOR_BGR2RGB)
                        buffer.append(transform(fallback_face_rgb))

                    if miss_counter >= MISS_TOLERANCE:
                        (last_bbox, last_label, last_colour, fake_locked,
                         real_streak, avg_real, avg_fake) = full_reset(
                            buffer, prediction_history)
                        miss_counter = 0
                        phase = PHASE_PRECHECK
                        (precheck_start_time, stable_face_start,
                         blink_low_counter, challenge_step,
                         head_turn_counter) = reset_precheck()
                        last_face_signature = None
                        precheck_face_buffer.clear()
                else:
                    last_label  = "No face detected"
                    last_colour = WHITE

                if len(buffer) == SEQUENCE_LENGTH:
                    sequence = torch.stack(list(buffer)).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        output = model(sequence)
                        probs  = torch.softmax(output, dim=1)
                        real_prob = probs[0, 0].item()
                        fake_prob = probs[0, 1].item()

                    prediction_history.append((real_prob, fake_prob))
                    avg_real = float(np.mean([p[0] for p in prediction_history]))
                    avg_fake = float(np.mean([p[1] for p in prediction_history]))
                    raw = decide_raw(avg_real, avg_fake)

                    if raw == "fake":
                        fake_locked = True; real_streak = 0
                    elif fake_locked:
                        real_streak = real_streak + 1 if raw == "real" else 0
                        if real_streak >= FAKE_LOCK_COUNT:
                            fake_locked = False; real_streak = 0

                    if fake_locked:
                        last_label = "FAKE";      last_colour = RED
                    elif raw == "real":
                        last_label = "REAL";      last_colour = GREEN
                    elif raw == "uncertain":
                        last_label = "UNCERTAIN"; last_colour = YELLOW
                    else:
                        last_label = "FAKE";      last_colour = RED

            # ════════════════════════════════════════════════════
            # UI — label + confidence bar  (unchanged)
            # ════════════════════════════════════════════════════
            h_frame, w_frame = frame.shape[:2]
            font_scale = 1.2; thickness = 2
            (tw, th), _ = cv2.getTextSize(
                last_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            label_x = (w_frame - tw) // 2
            label_y = h_frame - 75

            cv2.putText(frame, last_label, (label_x + 2, label_y + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, DARK, thickness + 2)
            cv2.putText(frame, last_label, (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, last_colour, thickness)

            if phase == PHASE_SCAN:
                bar_w = 220
                bar_x = (w_frame - bar_w) // 2
                bar_y = h_frame - 45
                if len(buffer) == SEQUENCE_LENGTH:
                    draw_confidence_bar(frame, avg_real, avg_fake, bar_x, bar_y)
                else:
                    filled = int((len(buffer) / SEQUENCE_LENGTH) * bar_w)
                    cv2.rectangle(frame, (bar_x, bar_y),
                                  (bar_x + bar_w, bar_y + 18), DARK, -1)
                    cv2.rectangle(frame, (bar_x, bar_y),
                                  (bar_x + bar_w, bar_y + 18), WHITE, 1)
                    if filled > 0:
                        cv2.rectangle(frame, (bar_x, bar_y),
                                      (bar_x + filled, bar_y + 18), WHITE, -1)
                    cv2.putText(
                        frame, f"Collecting {len(buffer)}/{SEQUENCE_LENGTH}",
                        (bar_x + 55, bar_y + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, DARK, 1)

            cv2.imshow("Liveness Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        detector.close()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# References:
# - MediaPipe Face Detector (Python): https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector/python
# - MediaPipe Face Landmarker (Python): https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python
# - OpenCV Getting Started with Videos: https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
# - Torchvision Transforms Documentation: https://docs.pytorch.org/vision/main/transforms.html
# - Deep Learning for Face Anti-Spoofing: A Survey: https://oulurepo.oulu.fi/bitstream/handle/10024/45560/nbnfi-fe2023052648600.pdf
# - streamlit-webrtc Repository / Docs: https://github.com/whitphx/streamlit-webrtc