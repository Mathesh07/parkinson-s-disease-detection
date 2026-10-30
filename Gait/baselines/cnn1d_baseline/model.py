"""Conventional 1D CNN Architecture for Gait VGRF Time-Series.

Serves as the baseline to compare against the proposed 1D CNN + BiLSTM model.
Contains pure 1D convolutional layers with global temporal pooling and a dense head.
Does NOT include recurrent cells (LSTM/BiLSTM) or evidential heads.
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

try:
    from . import config
except ImportError:
    import config


class Gait1DCNN(nn.Module):
    """
    Conventional 1D Convolutional Neural Network for multi-channel VGRF time-series.
    
    Architecture Pipeline:
        Raw VGRF: (B, 500, 16) -> Transpose to (B, 16, 500)
            ↓
        Conv Block 1: Conv1d(16 -> 64, k=5, p=2) -> BN -> ReLU -> MaxPool(2) -> (B, 64, 250)
            ↓
        Conv Block 2: Conv1d(64 -> 128, k=5, p=2) -> BN -> ReLU -> MaxPool(2) -> (B, 128, 125)
            ↓
        Conv Block 3: Conv1d(128 -> 256, k=5, p=2) -> BN -> ReLU -> MaxPool(2) -> (B, 256, 62)
            ↓
        Global Temporal Pooling: AdaptiveAvgPool1d(1) -> (B, 256)
            ↓
        Dense Bottleneck: Linear(256 -> 128) -> BN -> ReLU -> Dropout(0.3) -> (B, 128)
            ↓
        Classification Head: Linear(128 -> 1) -> Binary Logit (B, 1)
    """

    def __init__(
        self,
        in_channels: int = config.NUM_CHANNELS,
        cnn_channels: list = config.CNN_CHANNELS,
        kernel_size: int = config.CNN_KERNEL_SIZE,
        dense_dim: int = config.DENSE_DIM,
        dropout: float = config.DROPOUT
    ):
        super().__init__()
        self.in_channels = in_channels
        self.cnn_channels = cnn_channels
        self.kernel_size = kernel_size
        self.dense_dim = dense_dim
        self.dropout_rate = dropout

        c1, c2, c3 = cnn_channels

        # Conv Block 1
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Conv Block 2
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Conv Block 3
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(c2, c3, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Dimension-safe Global Temporal Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Dense Bottleneck
        self.dense_block = nn.Sequential(
            nn.Linear(c3, dense_dim, bias=False),
            nn.BatchNorm1d(dense_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )

        # Binary Classifier Head (Logits)
        self.classifier = nn.Linear(dense_dim, 1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract 128-dimensional dense embedding for downstream representation analysis.
        
        Args:
            x: Input tensor of shape (B, 500, 16) or (B, 16, 500)
        Returns:
            features: Tensor of shape (B, 128)
        """
        if x.dim() == 3 and x.shape[1] != self.in_channels and x.shape[2] == self.in_channels:
            x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        h = self.conv_block1(x)
        h = self.conv_block2(h)
        h = self.conv_block3(h)
        h = self.global_pool(h).squeeze(-1)
        features = self.dense_block(h)
        return features

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass returning binary logits.
        
        Args:
            x: Tensor of shape (B, 500, 16) or (B, 16, 500)
            return_embedding: If True, returns (logits, embedding)
        Returns:
            logits: Shape (B, 1)
        """
        features = self.extract_features(x)
        logits = self.classifier(features)

        if return_embedding:
            return logits, features
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute sigmoid binary probability P(PD | x)."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs

    def get_config(self) -> Dict:
        """Return model architecture configuration dictionary."""
        return {
            "model_type": "Gait1DCNN",
            "in_channels": self.in_channels,
            "cnn_channels": self.cnn_channels,
            "kernel_size": self.kernel_size,
            "dense_dim": self.dense_dim,
            "dropout": self.dropout_rate,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


if __name__ == "__main__":
    print("Testing Gait1DCNN baseline architecture...")
    model = Gait1DCNN()
    dummy_input = torch.randn(8, config.WINDOW_SIZE, config.NUM_CHANNELS)
    logits, emb = model(dummy_input, return_embedding=True)
    probs = model.predict_proba(dummy_input)

    print(f"Input Shape:       {dummy_input.shape} (Expected: [8, 500, 16])")
    print(f"Logits Shape:      {logits.shape} (Expected: [8, 1])")
    print(f"Embedding Shape:   {emb.shape} (Expected: [8, 128])")
    print(f"Probs Range:       [{probs.min().item():.4f}, {probs.max().item():.4f}] (Expected: [0, 1])")
    print(f"Config:            {model.get_config()}")
    assert logits.shape == (8, 1)
    assert emb.shape == (8, 128)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    print("Gait1DCNN architecture verified successfully!")
