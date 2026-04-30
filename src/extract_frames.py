"""Extract a fixed number of evenly spaced frames from each raw video.

The output structure mirrors the target class labels so later stages can
build fixed-length face sequences for training and evaluation.

References at the end
"""

import cv2
from pathlib import Path
import re

NUM_FRAMES = 10
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

RAW_ROOT = Path("data/raw_videos")
OUTPUT_ROOT = Path("data/extracted_frames")

# Explicitly map each top-level folder in raw_videos to its class label.
# Any folder not listed here will be skipped with a warning.
FOLDER_LABELS = {
    "real":                                "real",
    "3d_paper_mask_":                      "fake",
    "cutout_attacks":                      "fake",
    "latex_mask":                          "fake",
    "printouts":                           "fake",
    "replay_display_attacks":              "fake",
    "silicone_mask":                       "fake",
    "textile 3d face mask attack sample":  "fake",
    "wrapped_3d_paper_mask":               "fake",
}

# Subfolders inside real/ that contain only images (not videos) — skip entirely
SKIP_SUBFOLDERS = {"selfies"}


def sample_frame_indices(total_frames: int, num_frames: int = NUM_FRAMES):
    """Return evenly spaced frame indices across the full video."""
    if total_frames < num_frames:
        return None
    return [round(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]


def safe_name(path: Path):
    """Convert nested paths into filesystem-safe folder names."""
    text = str(path).replace("\\", "__").replace("/", "__")
    text = re.sub(r"[^A-Za-z0-9_.\-]+", "_", text)
    return text


def extract_10_frames(video_path: Path, output_dir: Path):
    """Read one video and save NUM_FRAMES evenly spaced frames to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    indices = sample_frame_indices(total_frames, NUM_FRAMES)
    if indices is None:
        cap.release()
        print(f"  Skipped (too short): {video_path}")
        return False

    saved = 0
    current_idx = 0
    target_set = set(indices)

    while True:
        # Read frames sequentially and save only the target positions.
        ret, frame = cap.read()
        if not ret:
            break

        if current_idx in target_set:
            frame_name = f"frame_{saved+1:02d}.jpg"
            cv2.imwrite(str(output_dir / frame_name), frame)
            saved += 1

        current_idx += 1

    cap.release()

    if saved != NUM_FRAMES:
        print(f"  Skipped (could not save 10 frames): {video_path}")
        return False

    return True


def get_class_label(video_path: Path):
    """
    Determine class label by looking up the top-level folder name
    inside raw_videos against the explicit FOLDER_LABELS mapping.
    Returns (label, top_folder_name) or (None, top_folder_name) if unknown.
    """
    relative = video_path.relative_to(RAW_ROOT)
    top_folder = relative.parts[0]
    label = FOLDER_LABELS.get(top_folder.lower())
    return label, top_folder


def is_in_skip_subfolder(video_path: Path):
    """Return True if the video is inside a subfolder we want to skip (e.g. Selfies)."""
    relative = video_path.relative_to(RAW_ROOT)
    # Check all intermediate folder names (excluding the top-level and filename)
    for part in relative.parts[1:-1]:
        if part.lower() in SKIP_SUBFOLDERS:
            return True
    return False


def process_dataset():
    """Walk the raw dataset tree and build the extracted-frames dataset."""
    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"Raw dataset folder not found: {RAW_ROOT}")

    skipped_unknown = []
    counts = {"real": 0, "fake": 0, "skipped": 0}

    for video_file in sorted(RAW_ROOT.rglob("*")):
        if video_file.suffix.lower() not in VIDEO_EXTS:
            continue

        # Skip image-only subfolders like Real/Selfies/
        if is_in_skip_subfolder(video_file):
            print(f"  Skipped (image subfolder): {video_file.name}")
            counts["skipped"] += 1
            continue

        class_label, top_folder = get_class_label(video_file)

        if class_label is None:
            if top_folder not in skipped_unknown:
                print(f"  WARNING: Unknown folder '{top_folder}' — not in FOLDER_LABELS, skipping.")
                skipped_unknown.append(top_folder)
            counts["skipped"] += 1
            continue

        relative_path = video_file.relative_to(RAW_ROOT).with_suffix("")
        video_name = safe_name(relative_path)

        out_dir = OUTPUT_ROOT / class_label / video_name
        success = extract_10_frames(video_file, out_dir)
        if success:
            print(f"  [{class_label.upper()}] {video_file.name}")
            counts[class_label] += 1
        else:
            counts["skipped"] += 1

    print(f"\nDone. Real: {counts['real']} | Fake: {counts['fake']} | Skipped: {counts['skipped']}")


if __name__ == "__main__":
    process_dataset()
    
# References:
# - OpenCV Getting Started with Videos: https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
# - OpenCV Image Codecs / imread / imwrite: https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html
# - Python pathlib Documentation: https://docs.python.org/3/library/pathlib.html