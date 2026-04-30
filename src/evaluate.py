"""Evaluate the trained anti-spoof model on the saved test split.

Outputs overall classification metrics plus a per-attack-type breakdown
for fake samples so it is easier to see which attack categories are hardest.

References at the end
"""

import json
from pathlib import Path
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix

from dataset import FaceSequenceDataset
from model import CNNLSTMModel

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR   = "data/face_sequences"
MODEL_PATH = "models/best_model.pth"
SPLIT_PATH = "models/split_indices.json"
BATCH_SIZE = 4


def compute_anti_spoof_metrics(y_true, y_pred):
    """Compute APCER/BPCER/ACER from the confusion matrix.

    Label convention:
    - 0 = real
    - 1 = fake
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    bpcer = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    apcer = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    acer  = (apcer + bpcer) / 2
    return apcer, bpcer, acer, cm


def get_attack_type(sample_path: Path) -> str:
    """
    Extract attack type from the sequence folder name.
    Folder names are safe_name-encoded versions of the original path, e.g.:
      fake/3D_paper_mask___133  → 3D_paper_mask
      fake/Replay_display_attacks___20240823  → Replay_display_attacks
    """
    name = sample_path.name.lower()

    attack_keywords = {
        "3d_paper_mask":    "3D Paper Mask",
        "cutout":           "Cutout Attacks",
        "latex":            "Latex Mask",
        "printout":         "Printouts",
        "replay":           "Replay / Display",
        "silicone":         "Silicone Mask",
        "textile":          "Textile Mask",
        "wrapped":          "Wrapped 3D Paper Mask",
    }
    for key, label in attack_keywords.items():
        if key in name:
            return label
    return "Unknown"


def main():
    # Evaluation uses deterministic transforms so test-time results are repeatable.
    dataset = FaceSequenceDataset(DATA_DIR, train=False)

    if not Path(SPLIT_PATH).exists():
        raise FileNotFoundError(
            f"Split indices not found at {SPLIT_PATH}. Run train.py first."
        )
    with open(SPLIT_PATH) as f:
        split_indices = json.load(f)

    # Reuse the exact test indices saved during training to avoid leakage.
    test_set    = Subset(dataset, split_indices["test"])
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    model = CNNLSTMModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    y_true = []
    y_pred = []

    # Also track per-attack results for fake samples
    # attack_results[attack_type] = {"correct": int, "total": int}
    attack_results = defaultdict(lambda: {"correct": 0, "total": 0})

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs  = inputs.to(DEVICE)
            outputs = model(inputs)
            preds   = outputs.argmax(dim=1).cpu().tolist()
            labs    = labels.tolist()

            # Map each batch prediction back to its original dataset index so we
            # can infer the spoof type from the folder name.
            for i, (pred, true_label) in enumerate(zip(preds, labs)):
                y_pred.append(pred)
                y_true.append(true_label)

                # Per-attack breakdown — only for fake samples (label == 1)
                if true_label == 1:
                    sample_idx   = split_indices["test"][batch_idx * BATCH_SIZE + i]
                    sample_paths, _ = dataset.samples[sample_idx]
                    attack_type  = get_attack_type(sample_paths[0].parent)
                    attack_results[attack_type]["total"]   += 1
                    attack_results[attack_type]["correct"] += int(pred == true_label)

    # --- Overall results ---
    print("=" * 55)
    print("OVERALL RESULTS")
    print("=" * 55)
    print(classification_report(y_true, y_pred, target_names=["real", "fake"]))

    apcer, bpcer, acer, cm = compute_anti_spoof_metrics(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print(f"APCER : {apcer:.4f}  (fake faces wrongly called real)")
    print(f"BPCER : {bpcer:.4f}  (real faces wrongly called fake)")
    print(f"ACER  : {acer:.4f}  (combined error)")

    # --- Per-attack breakdown ---
    print("\n" + "=" * 55)
    print("PER-ATTACK BREAKDOWN (fake samples only)")
    print("=" * 55)
    print(f"{'Attack Type':<30} {'Correct':<10} {'Total':<10} {'Accuracy'}")
    print("-" * 55)
    for attack, counts in sorted(attack_results.items()):
        acc = counts["correct"] / max(counts["total"], 1)
        bar = "█" * int(acc * 20)
        print(f"{attack:<30} {counts['correct']:<10} {counts['total']:<10} {acc:.0%}  {bar}")
    print("=" * 55)


if __name__ == "__main__":
    main()

# References:
# - scikit-learn classification_report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
# - scikit-learn confusion_matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# - Deep Learning for Face Anti-Spoofing: A Survey: https://oulurepo.oulu.fi/bitstream/handle/10024/45560/nbnfi-fe2023052648600.pdf
# - PyTorch Data Loading Documentation: https://docs.pytorch.org/docs/stable/data.html