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


class EvidentialHead(nn.Module):
    """
    Evidential classification head for subjective logic / evidential neural networks.
    
    Transforms feature embeddings into non-negative evidence values for each class:
        features (B, feature_dim) -> Dropout -> Linear (B, num_classes) -> Softplus -> Evidence (B, num_classes)
        
    Outputs:
        evidence: Non-negative tensor of shape (B, 2) representing [e_HC, e_PD] >= 0.
    """

    def __init__(self, in_features: int, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.evidence_layer = nn.Linear(in_features, num_classes)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor, return_raw_logits: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass producing non-negative evidence.
        
        Args:
            x: Feature embedding of shape (batch_size, in_features)
            return_raw_logits: If True, returns (evidence, raw_logits)
            
        Returns:
            evidence: Non-negative evidence of shape (batch_size, 2) where evidence >= 0.
        """
        dropped = self.dropout(x)
        raw_logits = self.evidence_layer(dropped)
        evidence = self.softplus(raw_logits)
        
        if return_raw_logits:
            return evidence, raw_logits
        return evidence


class GaitCNNBiLSTM(nn.Module):
    """
    1D CNN + Bidirectional LSTM model with Evidential Output Head for continuous VGRF time-series.
    
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
        Evidential Head (Dropout, Linear 128 -> 2, Softplus) -> [e_HC, e_PD] >= 0
    """

    def __init__(
        self,
        in_channels: int = config.NUM_CHANNELS,
        cnn_channels: list = config.CNN_CHANNELS,
        cnn_kernel_size: int = 5,
        lstm_hidden_size: int = config.LSTM_HIDDEN_SIZE,
        lstm_num_layers: int = config.LSTM_NUM_LAYERS,
        embedding_dim: int = config.EMBEDDING_DIM,
        num_classes: int = 2,
        dropout: float = config.DROPOUT
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

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

        # 4. Modular Evidential Head (Produces non-negative evidence [e_HC, e_PD] >= 0)
        self.evidential_head = EvidentialHead(
            in_features=embedding_dim,
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass producing evidential outputs.
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, 16)
            return_embedding: If True, returns (evidence, gait_feature_vector)
            
        Returns:
            evidence: Non-negative tensor of shape (batch_size, 2) -> [e_HC, e_PD]
            (evidence, embedding): if return_embedding=True
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

        # 7. Evidential Output Head
        evidence = self.evidential_head(gait_feature_vector)  # (B, 2)

        if return_embedding:
            return evidence, gait_feature_vector
        return evidence

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method to extract only the 128-dimensional embedding."""
        with torch.no_grad():
            _, emb = self.forward(x, return_embedding=True)
        return emb


if __name__ == "__main__":
    print("Testing GaitCNNBiLSTM evidential architecture...")
    model = GaitCNNBiLSTM()
    dummy_input = torch.randn(8, config.WINDOW_SIZE, config.NUM_CHANNELS)
    evidence, emb = model(dummy_input, return_embedding=True)
    print(f"Input Shape:          {dummy_input.shape} (Expected: [8, 500, 16])")
    print(f"Evidence Shape:       {evidence.shape} (Expected: [8, 2])")
    print(f"Embedding Shape:      {emb.shape} (Expected: [8, 128])")
    print(f"Min Evidence Value:   {evidence.min().item():.6f} (Must be >= 0)")
    print(f"Max Evidence Value:   {evidence.max().item():.6f}")
    print(f"Sample Evidence (0):  [e_HC={evidence[0,0].item():.4f}, e_PD={evidence[0,1].item():.4f}]")
    assert evidence.shape == (8, 2), "Evidence shape mismatch!"
    assert (evidence >= 0).all(), "All evidence values must be non-negative!"
    print("Gait evidential architecture verified successfully!")
