"""Build fixed-length face-only frame sequences from extracted video frames.

This script detects the largest face in each frame, crops it with padding,
and saves 10 evenly spaced cropped faces per video folder.

References at the end
"""

import cv2
import numpy as np
from pathlib import Path

from face_utils import MediaPipeFaceDetector, crop_face_square

INPUT_ROOT  = Path("data/extracted_frames")
OUTPUT_ROOT = Path("data/face_sequences")
TARGET_FRAMES = 10

# Lowered from 10 to 6 — MediaPipe is more accurate than Haar cascade
# but mask/replay videos still have frames where the face is partially
# covered or at an angle. Requiring all 10 is too strict and discards
# too many valid attack videos.
MIN_VALID_FACES = 6

DETECTOR_MODEL = "models/face_detector.task"

# Increased from 0.18 to 0.30 — mask edges, screen bezels, and glare
# around the face are useful cues for detecting attacks. More padding
# means the model sees more context, not just the face itself.
PADDING = 0.30


def select_evenly_spaced(frames, target_count=TARGET_FRAMES):
    """Sample a fixed number of frames while preserving coverage across the clip."""
    if len(frames) < target_count:
        return None
    indices = np.linspace(0, len(frames) - 1, target_count, dtype=int)
    return [frames[i] for i in indices]


def process_video_folder(video_folder: Path, output_folder: Path, detector: MediaPipeFaceDetector):
    """Crop face regions from one extracted-frame folder and save a clean sequence."""
    frame_files  = sorted(video_folder.glob("*.jpg"))
    cropped_faces = []
    missed_frames = 0
    last_det      = None  # fallback: last successful detection bbox

    for frame_file in frame_files:
        image = cv2.imread(str(frame_file))
        if image is None:
            missed_frames += 1
            continue

        det = detector.detect(image)

        if det is not None:
            last_det = det  # update last known good detection
        elif last_det is not None:
            # Fallback — reuse last known bounding box.
            # Face barely moves between consecutive frames so this is safe.
            det = last_det
        else:
            # No detection and no history — skip this frame
            missed_frames += 1
            continue

        face, meta = crop_face_square(image, det, padding=PADDING)

        if face is None:
            missed_frames += 1
            continue

        cropped_faces.append(face)

    if len(cropped_faces) < MIN_VALID_FACES:
        print(f"  Skipped (only {len(cropped_faces)} valid faces, need {MIN_VALID_FACES}): {video_folder.name}")
        return False

    selected_faces = select_evenly_spaced(cropped_faces, TARGET_FRAMES)
    if selected_faces is None:
        print(f"  Skipped (not enough valid faces to build {TARGET_FRAMES} unique frames): {video_folder.name}")
        return False

    output_folder.mkdir(parents=True, exist_ok=True)
    # Write the final fixed-length sequence in frame_01.jpg ... frame_10.jpg format.
    for i, face in enumerate(selected_faces, start=1):
        out_path = output_folder / f"frame_{i:02d}.jpg"
        cv2.imwrite(str(out_path), face)

    print(f"  Processed ({len(cropped_faces)} valid faces, {missed_frames} fallback/missed): {video_folder.name}")
    return True


def process_dataset():
    """Run face cropping over the whole extracted-frame dataset."""
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_ROOT}")

    detector = MediaPipeFaceDetector(
        model_path=DETECTOR_MODEL,
        running_mode="image",
        min_detection_confidence=0.6,
    )

    counts = {"real": 0, "fake": 0, "skipped": 0}

    try:
        for class_dir in sorted(INPUT_ROOT.iterdir()):
            if not class_dir.is_dir():
                continue

            class_skipped = 0
            class_kept    = 0
            print(f"\n--- {class_dir.name.upper()} ---")

            for video_folder in sorted(class_dir.iterdir()):
                if not video_folder.is_dir():
                    continue

                output_folder = OUTPUT_ROOT / class_dir.name / video_folder.name
                success = process_video_folder(video_folder, output_folder, detector)

                if success:
                    class_kept += 1
                    counts[class_dir.name] = counts.get(class_dir.name, 0) + 1
                else:
                    class_skipped += 1
                    counts["skipped"] += 1

            print(f"  -> Kept: {class_kept} | Skipped: {class_skipped}")

    finally:
        detector.close()

    print(f"\n{'='*40}")
    print("FINAL SUMMARY")
    print(f"  Real kept    : {counts.get('real', 0)}")
    print(f"  Fake kept    : {counts.get('fake', 0)}")
    print(f"  Total skipped: {counts['skipped']}")
    print(f"{'='*40}")


if __name__ == "__main__":
    process_dataset()

# References:
# - MediaPipe Face Detector (Python): https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector/python
# - OpenCV Getting Started with Images: https://docs.opencv.org/4.x/db/deb/tutorial_display_image.html
# - OpenCV Getting Started with Videos: https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
# - OpenCV Image Codecs / imread / imwrite: https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html