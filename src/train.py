"""Training script for the CNN-LSTM anti-spoof classifier.

This script builds the train/validation/test split once, saves the split
indices for reproducible evaluation, and trains until validation loss stops
improving.

References at the end
"""

from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

from dataset import FaceSequenceDataset
from model import CNNLSTMModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data/face_sequences"
MODEL_PATH = "models/best_model.pth"
BATCH_SIZE = 4
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-3
PATIENCE = 6


def split_dataset(dataset):
    """
    Split dataset once and return indices + subsets.
    Indices are saved to disk so evaluate.py uses the exact same test set.
    """
    total = len(dataset)
    train_size = int(0.7 * total)
    val_size   = int(0.15 * total)
    test_size  = total - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    # Save the exact indices random_split produced
    split_indices = {
        "train": train_set.indices,
        "val":   val_set.indices,
        "test":  test_set.indices,
    }
    Path("models").mkdir(exist_ok=True)
    with open("models/split_indices.json", "w") as f:
        json.dump(split_indices, f)
    print(f"Split — Train: {train_size} | Val: {val_size} | Test: {test_size}")
    print("Split indices saved to models/split_indices.json")

    return train_set, val_set, test_set


def build_loaders(train_set, val_set):
    """Create training and validation data loaders with the desired shuffle policy."""
    return (
        DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False),
    )


def get_class_weights(dataset):
    """Compute inverse-frequency weights so minority classes matter more."""
    real_count = sum(1 for _, label in dataset.samples if label == 0)
    fake_count = sum(1 for _, label in dataset.samples if label == 1)
    total = real_count + fake_count
    print(f"Class counts — Real: {real_count} | Fake: {fake_count}")
    return torch.tensor([
        total / (2 * max(real_count, 1)),
        total / (2 * max(fake_count, 1)),
    ], dtype=torch.float32)


def evaluate_epoch(model, loader, criterion):
    """Returns (loss, accuracy) on the given loader."""
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total   += labels.size(0)

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def main():
    Path("models").mkdir(exist_ok=True)

    # Training mode enables data augmentation inside FaceSequenceDataset.
    dataset = FaceSequenceDataset(DATA_DIR, train=True)
    if len(dataset) == 0:
        raise ValueError("No valid sequences found. Check data/face_sequences.")
    print(f"Dataset size: {len(dataset)} sequences")

    train_set, val_set, _ = split_dataset(dataset)
    train_loader, val_loader = build_loaders(train_set, val_set)

    model = CNNLSTMModel().to(DEVICE)

    class_weights = get_class_weights(dataset).to(DEVICE)
    print(f"Class weights — Real: {class_weights[0]:.3f} | Fake: {class_weights[1]:.3f}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Only optimise layers that were left trainable in the model definition.
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<10}")
    print("-" * 54)

    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        total_loss = 0.0
        correct    = 0
        total      = 0

        for inputs, labels in train_loader:
            # Move each mini-batch to the selected device before the forward pass.
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total   += labels.size(0)

        train_loss = total_loss / max(len(train_loader), 1)
        train_acc  = correct / max(total, 1)

        # --- Validation ---
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        print(f"{epoch+1:<8} {train_loss:<12.4f} {train_acc:<12.4f} {val_loss:<12.4f} {val_acc:<10.4f}", end="")

        # Save only the best-performing checkpoint on validation loss.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            patience_counter = 0
            print("  ✓ saved")
        else:
            patience_counter += 1
            print()

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"\nBest model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()

# References:
# - torch.utils.data Documentation: https://docs.pytorch.org/docs/stable/data.html
# - torch.nn.CrossEntropyLoss / functional.cross_entropy: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
# - torch.optim Documentation: https://docs.pytorch.org/docs/stable/optim.html
# - PyTorch Saving and Loading Models Tutorial: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
# - PyTorch Tutorials Home: https://docs.pytorch.org/tutorials/