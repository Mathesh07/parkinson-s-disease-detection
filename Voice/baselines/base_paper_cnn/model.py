"""2D Convolutional Neural Network Baseline for Voice STFT Spectrograms.

Inspired by the base-paper architecture:
    "Multimodal assessment of Parkinson's disease: a deep learning approach"
    by J. C. Vásquez-Correa et al., IEEE JBHI, 2019.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn

try:
    from . import config
except ImportError:
    import config


class VoiceBasePaper2DCNN(nn.Module):
    """
    2D CNN architecture for Voice STFT Spectrograms and Speech Transitions.

    Input shape: (B, 1, freq_bins, time_frames)
    Pipeline:
        Conv Block 1: Conv2d(1 -> 32, 3x3) -> BN -> ReLU -> MaxPool(2x2) -> Dropout(0.2)
        Conv Block 2: Conv2d(32 -> 64, 3x3) -> BN -> ReLU -> MaxPool(2x2) -> Dropout(0.2)
        Conv Block 3: Conv2d(64 -> 128, 3x3) -> BN -> ReLU -> MaxPool(2x2) -> Dropout(0.3)
        Adaptive Pooling: AdaptiveAvgPool2d((2, 2)) -> Flatten (512)
        Dense Bottleneck: Linear(512 -> 128) -> BN -> ReLU -> Dropout(0.3)
        Output Head: Linear(128 -> 2) -> Logits [HC, PD]
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: list = config.CONV_CHANNELS,
        dense_units: int = config.DENSE_UNITS,
        num_classes: int = config.NUM_CLASSES,
        dropout_rate: float = config.DROPOUT_RATE
    ):
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.dense_units = dense_units
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        c1, c2, c3 = conv_channels
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate * 0.67)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate * 0.67)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate)
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        flattened_dim = c3 * 2 * 2  # 128 * 4 = 512

        self.dense_block = nn.Sequential(
            nn.Linear(flattened_dim, dense_units, bias=False),
            nn.BatchNorm1d(dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate)
        )

        self.classifier = nn.Linear(dense_units, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 128-dim bottleneck representation."""
        h = self.conv_block1(x)
        h = self.conv_block2(h)
        h = self.conv_block3(h)
        h = self.adaptive_pool(h)
        h = torch.flatten(h, 1)
        features = self.dense_block(h)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning class logits of shape (B, 2)."""
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Softmax class probabilities of shape (B, 2)."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
        return probs

    def get_config(self) -> Dict:
        """Return model architecture configuration dictionary."""
        return {
            "model_type": "VoiceBasePaper2DCNN",
            "in_channels": self.in_channels,
            "conv_channels": self.conv_channels,
            "dense_units": self.dense_units,
            "num_classes": self.num_classes,
            "dropout_rate": self.dropout_rate
        }
