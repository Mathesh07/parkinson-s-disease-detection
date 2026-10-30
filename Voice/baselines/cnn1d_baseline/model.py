"""Conventional 1D CNN Architecture on Raw Audio Waveforms for Voice-Based Parkinson's Detection."""

from typing import Dict, Tuple

import torch
import torch.nn as nn

try:
    from . import config
except ImportError:
    import config


class Voice1DCNN(nn.Module):
    """
    Conventional 1D Convolutional Neural Network operating on raw speech waveforms.
    
    Processes 16 kHz audio without requiring spectrograms or pretrained transformers.
    
    Architecture:
        Raw Waveform (B, 1, 48000)
            ↓
        Conv Block 1: Conv1d(1 -> 32, k=64, s=16, p=32) -> BN -> ReLU -> MaxPool(4)
            ↓
        Conv Block 2: Conv1d(32 -> 64, k=16, s=4, p=8) -> BN -> ReLU -> MaxPool(4)
            ↓
        Conv Block 3: Conv1d(64 -> 128, k=8, s=2, p=4) -> BN -> ReLU -> MaxPool(2)
            ↓
        Conv Block 4: Conv1d(128 -> 256, k=5, s=1, p=2) -> BN -> ReLU -> AdaptiveAvgPool1d(1)
            ↓
        Dense Bottleneck: Linear(256 -> 128) -> BN -> ReLU -> Dropout(0.3)
            ↓
        Classification Head: Linear(128 -> 2) -> Logits [HC, PD]
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: list = config.CONV_CHANNELS,
        dense_dim: int = config.DENSE_DIM,
        num_classes: int = config.NUM_CLASSES,
        dropout: float = config.DROPOUT
    ):
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.dense_dim = dense_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout

        c1, c2, c3, c4 = conv_channels

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=64, stride=16, padding=32, bias=False),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(c1, c2, kernel_size=16, stride=4, padding=8, bias=False),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(c2, c3, kernel_size=8, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.block4 = nn.Sequential(
            nn.Conv1d(c3, c4, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(c4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )

        self.dense_block = nn.Sequential(
            nn.Linear(c4, dense_dim, bias=False),
            nn.BatchNorm1d(dense_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )

        self.classifier = nn.Linear(dense_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 128-dim dense bottleneck representation."""
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, samples) -> (B, 1, samples)

        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h).squeeze(-1)
        features = self.dense_block(h)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits of shape (B, 2)."""
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
            "model_type": "Voice1DCNN_RawWaveform",
            "in_channels": self.in_channels,
            "conv_channels": self.conv_channels,
            "dense_dim": self.dense_dim,
            "num_classes": self.num_classes,
            "dropout": self.dropout_rate,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


if __name__ == "__main__":
    print("Testing Voice1DCNN raw waveform architecture...")
    model = Voice1DCNN()
    dummy = torch.randn(4, 1, config.CHUNK_SAMPLES)
    logits = model(dummy)
    probs = model.predict_proba(dummy)

    print(f"Input Shape:   {dummy.shape} (Expected: [4, 1, 48000])")
    print(f"Logits Shape:  {logits.shape} (Expected: [4, 2])")
    print(f"Probs Shape:   {probs.shape} (Expected: [4, 2])")
    print(f"Config:        {model.get_config()}")
    assert logits.shape == (4, 2)
    assert probs.shape == (4, 2)
    print("Voice1DCNN architecture verified successfully!")
