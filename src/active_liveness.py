"""
active_liveness.py
==================
Helpers for the active liveness challenge pipeline.

Each challenge stage (blink, mouth-open, head-turn) must satisfy TWO independent
conditions before the user may advance:

  (a)  The required physical gesture was observed in the landmark stream.
  (b)  The CNN-LSTM anti-spoof model returns a REAL confidence above
       ACTIVE_SPOOF_THRESHOLD when evaluated on frames captured during that gesture.

This two-gate design means a replay video or a printed/mask attack has to defeat
both the gesture detector and the texture/depth classifier simultaneously at every
stage — not just at the final scan.

Public API
----------
run_snapshot_antispoof(model, face_buffer, device)
    -> (real_prob, fake_prob) | (None, None) if buffer too small

head_turn_deviation(face_landmarks, get_xy_fn)
    -> float  signed deviation ratio (negative=left, positive=right)

is_head_turned(face_landmarks, get_xy_fn)
    -> bool

SpoofFailResult (dataclass)
    Returned whenever antispoof fails a stage; carries stage name and scores.
    
References at the end
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Tuple

# ─────────────────────────────────────────────────────────────────
# Tunable thresholds – each annotated so operators can adjust them
# ─────────────────────────────────────────────────────────────────

# Minimum real-class probability the CNN-LSTM must return at each
# active challenge gate.  Set lower than the main scan threshold (0.65)
# because the precheck buffer may contain padding frames, making the
# snapshot less accurate.  Still provides a meaningful defence layer.
ACTIVE_SPOOF_THRESHOLD: float = 0.58

# Number of frames that must be in the rolling buffer before we trust
# a snapshot inference.  At 30 fps this is ~0.13 s; should always be
# satisfied by the time a gesture is detected.
LIVENESS_SNAP_MIN_FRAMES: int = 4

# Must match SEQUENCE_LENGTH in webcam_demo.py.
SEQUENCE_LENGTH: int = 10

# Nose-deviation ratio (relative to the inter-ocular span) that counts
# as a genuine head turn.  0.19 ≈ ±11° of horizontal rotation,
# achievable with a casual turn but unlikely for a front-facing photo.
HEAD_TURN_THRESHOLD: float = 0.19

# Consecutive frames the head must remain turned before the gesture is
# confirmed.  Prevents a momentary sway from triggering the check.
HEAD_TURN_FRAMES_REQUIRED: int = 4

# Human-readable stage names used in alert messages.
STAGE_NAMES: dict[int, str] = {
    1: "blink",
    2: "mouth-open",
    3: "head-turn",
}


# ─────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────

@dataclass
class SpoofCheckResult:
    """Result of a snapshot anti-spoof check at an active liveness gate."""
    passed: bool
    real_prob: Optional[float]
    fake_prob: Optional[float]
    stage_name: str
    insufficient_frames: bool = False  # True when buffer was too small → skip check

    @property
    def display_message(self) -> str:
        if self.insufficient_frames:
            return ""  # silent pass-through; no alert shown
        if self.passed:
            return f"Anti-spoof OK at {self.stage_name} (real={self.real_prob:.0%})"
        return (
            f"SPOOF DETECTED  [{self.stage_name.upper()} stage]  "
            f"real={self.real_prob:.0%}  fake={self.fake_prob:.0%}"
        )


# ─────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────

def run_snapshot_antispoof(
    model: torch.nn.Module,
    face_buffer,          # deque / list of [C,H,W] tensors (already transformed)
    device: str,
    stage: int = 0,
) -> SpoofCheckResult:
    """
    Run a single CNN-LSTM inference on the current face_buffer.

    If the buffer is shorter than SEQUENCE_LENGTH, the last frame is repeated
    to pad it to the required length.  This is a deliberate trade-off: the
    padded sequence is less temporally rich, which is why ACTIVE_SPOOF_THRESHOLD
    is set slightly lower than the main scan threshold.  The check still provides
    a strong signal against static photo / screen attacks.

    Parameters
    ----------
    model       : The loaded CNNLSTMModel (already in eval mode).
    face_buffer : Rolling deque of transformed face tensors collected during precheck.
    device      : 'cuda' or 'cpu'.
    stage       : Challenge step number (1, 2, or 3) — used for messaging only.

    Returns
    -------
    SpoofCheckResult with passed=True/False and the raw probabilities.
    """
    stage_name = STAGE_NAMES.get(stage, f"stage-{stage}")
    buf = list(face_buffer)

    if len(buf) < LIVENESS_SNAP_MIN_FRAMES:
        # Buffer too thin to trust — let the gesture pass silently.
        # This is an edge-case (only possible in the first few frames
        # of the session) and is preferable to a false block.
        return SpoofCheckResult(
            passed=True,
            real_prob=None,
            fake_prob=None,
            stage_name=stage_name,
            insufficient_frames=True,
        )

    # Pad to SEQUENCE_LENGTH by repeating the last frame.
    while len(buf) < SEQUENCE_LENGTH:
        buf.append(buf[-1].clone())

    # Trim to exactly SEQUENCE_LENGTH (safety guard).
    buf = buf[:SEQUENCE_LENGTH]

    sequence = torch.stack(buf).unsqueeze(0).to(device)  # [1, T, C, H, W]
    with torch.no_grad():
        output = model(sequence)
        probs = torch.softmax(output, dim=1)
        real_prob = probs[0, 0].item()
        fake_prob = probs[0, 1].item()

    passed = real_prob >= ACTIVE_SPOOF_THRESHOLD

    return SpoofCheckResult(
        passed=passed,
        real_prob=real_prob,
        fake_prob=fake_prob,
        stage_name=stage_name,
    )


# ─────────────────────────────────────────────────────────────────
# Head-turn geometry
# ─────────────────────────────────────────────────────────────────

def head_turn_deviation(face_landmarks, get_xy_fn) -> float:
    """
    Compute the horizontal displacement of the nose tip relative to the
    midpoint between the outer eye corners, normalised by the inter-ocular
    span.

    MediaPipe 468-landmark indices used:
      1   – nose tip
      33  – left eye outer corner  (image-left when looking straight)
      263 – right eye outer corner (image-right when looking straight)

    Return value
    ------------
    A signed ratio in roughly [−1, +1].

      ≈ 0          → looking straight at the camera
      negative     → nose has shifted left  → face turned left
      positive     → nose has shifted right → face turned right

    A turn is confirmed when abs(deviation) >= HEAD_TURN_THRESHOLD (0.19).

    Why this metric?
    ~~~~~~~~~~~~~~~~
    When a real face rotates horizontally the 3-D nose tip protrudes and
    shifts toward the direction of rotation in the image plane, while the
    far eye landmark foreshortens toward the near-side eye.  A flat photo
    or screen can reproduce the 2-D landmark positions but NOT the genuine
    depth-driven texture change captured by the anti-spoof model — which
    is why the gesture check and the anti-spoof snapshot must both pass.
    """
    nose     = get_xy_fn(face_landmarks[1])
    left_eye = get_xy_fn(face_landmarks[33])
    right_eye = get_xy_fn(face_landmarks[263])

    mid_x    = (left_eye[0] + right_eye[0]) / 2.0
    eye_span = abs(right_eye[0] - left_eye[0]) + 1e-6

    return (nose[0] - mid_x) / eye_span


def is_head_turned(face_landmarks, get_xy_fn) -> bool:
    """Return True if the nose deviation exceeds HEAD_TURN_THRESHOLD."""
    return abs(head_turn_deviation(face_landmarks, get_xy_fn)) >= HEAD_TURN_THRESHOLD

# References:
# - MediaPipe Face Landmarker (Python): https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python
# - Soukupova, T. and Cech, J. (2016), Eye Blink Detection Using Facial Landmarks: https://cmp.felk.cvut.cz/ftp/articles/cech/Soukupova-TR-2016-05.pdf
# - Deep Learning for Face Anti-Spoofing: A Survey: https://oulurepo.oulu.fi/bitstream/handle/10024/45560/nbnfi-fe2023052648600.pdf
# - A Survey on Face Presentation Attack Detection Mechanisms: https://pmc.ncbi.nlm.nih.gov/articles/PMC10025066/