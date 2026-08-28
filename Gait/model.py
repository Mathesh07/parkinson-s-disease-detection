"""1D CNN + BiLSTM Neural Network Architecture for Gait-Based Parkinson's Disease Detection."""

import sys
from pathlib import Path
from typing import Tuple, Union

import torch
import torch.nn as nn

# Support running as a standalone script or as a module
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
for p in (str(current_dir), str(parent_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from Gait import config
except ImportError:
    import config


class GaitCNNBiLSTM(nn.Module):
    """
    1D CNN + Bidirectional LSTM model for continuous Vertical Ground Reaction Force (VGRF) time-series.
    
    Pipeline:
        Raw VGRF (B, 500, 16)
               ↓
        Transpose to (B, 16, 500)
               ↓
        CNN Block 1 (16 -> 64, kernel=5, MaxPool /2) -> (B, 64, 250)
               ↓
        CNN Block 2 (64 -> 128, kernel=5, MaxPool /2) -> (B, 128, 125)
               ↓
        Permute to (B, 125, 128)
               ↓
        2-Layer BiLSTM (hidden=128, bidir=True) -> (B, 125, 256)
               ↓
        Global Temporal Pooling (Mean across time) -> (B, 256)
               ↓
        Dense Embedding Layer (256 -> 128, BN, ReLU, Dropout) -> 128-dim gait_feature_vector
               ↓
        Classification Head (Dropout, Linear 128 -> 1) -> binary logit
    """

    def __init__(
        self,
        in_channels: int = config.NUM_CHANNELS,
        cnn_channels: list = config.CNN_CHANNELS,
        cnn_kernel_size: int = 5,
        lstm_hidden_size: int = config.LSTM_HIDDEN_SIZE,
        lstm_num_layers: int = config.LSTM_NUM_LAYERS,
        embedding_dim: int = config.EMBEDDING_DIM,
        dropout: float = config.DROPOUT
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        # 1. 1D CNN Feature Extractor (Local temporal dynamics across 16 sensors)
        c1, c2 = cnn_channels
        self.cnn_block1 = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=cnn_kernel_size, padding=cnn_kernel_size // 2),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)  # (500 -> 250)
        )

        self.cnn_block2 = nn.Sequential(
            nn.Conv1d(c1, c2, kernel_size=cnn_kernel_size, padding=cnn_kernel_size // 2),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)  # (250 -> 125)
        )

        # 2. Bidirectional LSTM (Long-range temporal sequence dependencies)
        self.lstm = nn.LSTM(
            input_size=c2,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0
        )

        lstm_out_dim = lstm_hidden_size * 2  # 256 for BiLSTM

        # 3. Dense Embedding Layer (Produces fixed 128-dim gait feature vector)
        self.embedding_layer = nn.Sequential(
            nn.Linear(lstm_out_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # 4. Classification Head (BCEWithLogitsLoss)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, 16)
            return_embedding: If True, returns (logits, gait_feature_vector)
            
        Returns:
            logits: (batch_size,) if return_embedding=False
            (logits, embedding): if return_embedding=True
        """
        # 1. Transpose for Conv1d: (B, T, C) -> (B, C, T)
        x_conv = x.transpose(1, 2)

        # 2. CNN blocks
        c_out = self.cnn_block1(x_conv)  # (B, 64, 250)
        c_out = self.cnn_block2(c_out)   # (B, 128, 125)

        # 3. Permute back for LSTM: (B, C, T) -> (B, T, C)
        lstm_in = c_out.transpose(1, 2)  # (B, 125, 128)

        # 4. BiLSTM
        lstm_out, _ = self.lstm(lstm_in)  # (B, 125, 256)

        # 5. Global Temporal Mean Pooling across time steps
        pooled = torch.mean(lstm_out, dim=1)  # (B, 256)

        # 6. 128-dimensional Gait Feature Vector
        gait_feature_vector = self.embedding_layer(pooled)  # (B, 128)

        # 7. Classification Head Logit
        logits = self.classifier(gait_feature_vector).squeeze(-1)  # (B,)

        if return_embedding:
            return logits, gait_feature_vector
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method to extract only the 128-dimensional embedding."""
        with torch.no_grad():
            _, emb = self.forward(x, return_embedding=True)
        return emb


if __name__ == "__main__":
    print("Testing GaitCNNBiLSTM architecture...")
    model = GaitCNNBiLSTM()
    dummy_input = torch.randn(8, config.WINDOW_SIZE, config.NUM_CHANNELS)
    logits, emb = model(dummy_input, return_embedding=True)
    print(f"Input Shape:      {dummy_input.shape} (Expected: [8, 500, 16])")
    print(f"Logits Shape:     {logits.shape} (Expected: [8])")
    print(f"Embedding Shape:  {emb.shape} (Expected: [8, 128])")
    print("GaitCNNBiLSTM Architecture verified successfully!")
