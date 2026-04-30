"""CNN-LSTM model for sequence-based face anti-spoofing."""
""" References at the end"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

# Path to the locally saved ResNet18 pretrained weights.
# Download from: https://download.pytorch.org/models/resnet18-f37072fd.pth
# and place it at models/resnet18-f37072fd.pth
LOCAL_WEIGHTS = Path("models/resnet18-f37072fd.pth")


class CNNLSTMModel(nn.Module):
    def __init__(self, hidden_size: int = 128, num_layers: int = 1, num_classes: int = 2):
        super().__init__()

        # Load backbone — use local weights file if available, otherwise try downloading
        if LOCAL_WEIGHTS.exists():
            backbone = models.resnet18(weights=None)
            backbone.load_state_dict(torch.load(LOCAL_WEIGHTS, map_location="cpu"))
            print(f"Loaded ResNet18 weights from {LOCAL_WEIGHTS}")
        else:
            print(f"WARNING: Local weights not found at {LOCAL_WEIGHTS}.")
            print("Attempting to download pretrained weights (requires internet)...")
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Remove the original ImageNet classifier so ResNet outputs features only.
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.feature_extractor = backbone

        # Unfreeze layer3 and layer4 — gives the model capacity to learn
        # texture-level cues (screen pixels, moiré) important for replay attacks.
        # Everything before layer3 stays frozen as a general feature extractor.
        for name, param in self.feature_extractor.named_parameters():
            if "layer3" not in name and "layer4" not in name:
                param.requires_grad = False

        # LSTM with dropout between layers to prevent co-adaptation
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=2,           # increased from 1 to 2 for more capacity
            batch_first=True,
            dropout=0.4,            # dropout between LSTM layers
        )

        # Increased dropout in classifier from 0.3 to 0.5
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        # Flatten the time dimension so the CNN sees one frame at a time.
        batch_size, time_steps, channels, height, width = x.shape
        x = x.view(batch_size * time_steps, channels, height, width)
        
        # Extract a feature vector per frame with the ResNet backbone.
        features = self.feature_extractor(x)
        
        # Restore the time dimension before temporal modelling with the LSTM.
        features = features.view(batch_size, time_steps, -1)
        lstm_out, _ = self.lstm(features)
        
        # Classify using the final LSTM state, which summarises the sequence.
        final_output = lstm_out[:, -1, :]
        return self.classifier(final_output)

# References:
# - Torchvision ResNet Models Documentation: https://docs.pytorch.org/vision/main/models.html
# - torchvision.models.resnet18: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
# - He et al. (2016), Deep Residual Learning for Image Recognition: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
# - torch.nn.LSTM Documentation: https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html