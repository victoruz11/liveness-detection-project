"""Reusable MediaPipe face detection and square-cropping helpers."""
""" References at the end"""

from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

IMG_SIZE = (224, 224)
DEFAULT_PADDING = 0.18


class MediaPipeFaceDetector:
    def __init__(
        self,
        model_path: str = "models/face_detector.task",
        running_mode: str = "image",
        min_detection_confidence: float = 0.6,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Face detector model not found: {self.model_path}\n"
                f"Download the MediaPipe Face Detector task model and place it there."
            )

        # MediaPipe requires different APIs for single images vs video streams.
        if running_mode.lower() == "image":
            mode = vision.RunningMode.IMAGE
        elif running_mode.lower() == "video":
            mode = vision.RunningMode.VIDEO
        else:
            raise ValueError("running_mode must be 'image' or 'video'")

        options = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mode,
            min_detection_confidence=min_detection_confidence,
        )

        self.detector = vision.FaceDetector.create_from_options(options)
        self.mode = mode

    def detect(self, bgr_image, timestamp_ms: int | None = None):
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self.mode == vision.RunningMode.IMAGE:
            result = self.detector.detect(mp_image)
        else:
            if timestamp_ms is None:
                raise ValueError("timestamp_ms is required for video mode")
            result = self.detector.detect_for_video(mp_image, timestamp_ms)

        if not result.detections:
            return None

        # Pick the largest detected face when multiple detections exist.
        best = max(
            result.detections,
            key=lambda d: d.bounding_box.width * d.bounding_box.height
        )

        bbox = best.bounding_box
        score = best.categories[0].score if best.categories else 0.0

        return (
            int(bbox.origin_x),
            int(bbox.origin_y),
            int(bbox.width),
            int(bbox.height),
            float(score),
        )

    def close(self):
        self.detector.close()


def crop_face_square(image, bbox, padding: float = DEFAULT_PADDING, out_size=IMG_SIZE):
    """
    bbox = (x, y, w, h, score?) or (x, y, w, h)
    Returns:
        cropped_face, meta
    meta includes whether crop touched image borders too much.
    """
    img_h, img_w = image.shape[:2]
    x, y, w, h = bbox[:4]

    cx = x + w / 2.0
    cy = y + h / 2.0
    side = max(w, h) * (1.0 + 2.0 * padding)

    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = int(round(cx + side / 2.0))
    y2 = int(round(cy + side / 2.0))

    touched_left = x1 < 0
    touched_top = y1 < 0
    touched_right = x2 > img_w
    touched_bottom = y2 > img_h

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    # Clip the crop to image boundaries; this also lets us report border contact.
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None, {"border_touch": True, "face_area_ratio": 0.0}

    resized = cv2.resize(crop, out_size)
    face_area_ratio = (w * h) / float(img_w * img_h)

    meta = {
        "border_touch": touched_left or touched_top or touched_right or touched_bottom,
        "face_area_ratio": face_area_ratio,
        "bbox": (x, y, w, h),
        "crop_rect": (x1, y1, x2, y2),
    }
    return resized, meta

# References:
# - MediaPipe Solutions Guide: https://ai.google.dev/edge/mediapipe/solutions/guide
# - MediaPipe Face Detector (Python): https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector/python
# - OpenCV Getting Started with Images: https://docs.opencv.org/4.x/db/deb/tutorial_display_image.html
# - Python pathlib Documentation: https://docs.python.org/3/library/pathlib.html