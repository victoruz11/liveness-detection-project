"""Dataset utilities for loading fixed-length face sequences.

Each sample is a folder containing exactly 10 cropped face frames.
The dataset returns a tensor sequence shaped [T, C, H, W] together
with a class label: 0 for real and 1 for fake.

References at the end
"""

from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Standard ImageNet statistics are reused because the ResNet backbone
# was pre-trained on ImageNet and expects inputs normalised this way.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
EXPECTED_FRAMES = 10


class FaceSequenceDataset(Dataset):
    def __init__(self, root_dir: str, train: bool = True):
        self.root_dir = Path(root_dir)
        self.samples = []
        # Keep the label mapping explicit so training/evaluation stay consistent.
        self.class_to_idx = {"real": 0, "fake": 1}

        # Stronger augmentations to prevent the model memorising specific videos.
        # Each epoch the same video looks slightly different, forcing generalisation.
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),              # increased from 5
            transforms.ColorJitter(
                brightness=0.3,                         # increased from 0.2
                contrast=0.3,                           # increased from 0.2
                saturation=0.2,                         # new
                hue=0.05,                               # new — subtle hue shift
            ),
            transforms.RandomGrayscale(p=0.05),         # new — occasional greyscale
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),  # new — slight blur
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),  # new — randomly masks small regions
        ])

        # No augmentations at eval/test time — clean, deterministic transforms only
        self.eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.transform = self.train_transform if train else self.eval_transform

        # Scan the dataset directory once and register only complete sequences.
        for class_name in ["real", "fake"]:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue

            for video_dir in class_dir.iterdir():
                if not video_dir.is_dir():
                    continue

                frames = sorted(video_dir.glob("*.jpg"))
                # Only keep folders with the exact expected frame count so that
                # the model always receives a fixed-length temporal sequence.
                if len(frames) == EXPECTED_FRAMES:
                    self.samples.append((frames, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Load one sequence folder, transform each frame, then stack them."""
        frame_paths, label = self.samples[idx]
        sequence = []

        for frame_path in frame_paths:
            # Force RGB so grayscale or oddly encoded images do not break the pipeline.
            image = Image.open(frame_path).convert("RGB")
            sequence.append(self.transform(image))

        # Final shape becomes [time_steps, channels, height, width].
        sequence = torch.stack(sequence)
        label = torch.tensor(label, dtype=torch.long)
        return sequence, label

# References:
# - PyTorch Data Loading Tutorial: https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html
# - torch.utils.data Documentation: https://docs.pytorch.org/docs/stable/data.html
# - Torchvision Transforms Documentation: https://docs.pytorch.org/vision/main/transforms.html
# - Pillow Documentation: https://pillow.readthedocs.io/en/stable/