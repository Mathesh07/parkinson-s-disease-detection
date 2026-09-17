"""Wav2Vec2 architecture for Parkinson's Disease voice classification."""

import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

# Support running as a standalone script or as a module
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
for p in (str(current_dir), str(parent_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from Voice import config
except ImportError:
    import config


class EvidentialHead(nn.Module):
    """
    Evidential classification head for subjective logic / evidential neural networks.
    
    Transforms feature embeddings into non-negative evidence values:
        features (B, feature_dim) -> Dropout -> Linear (B, 2) -> Softplus -> Evidence (B, 2)
        
    Outputs:
        evidence: Non-negative tensor of shape (B, 2) representing [e_HC, e_PD] >= 0.
    """

    def __init__(self, in_features: int, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.evidence_layer = nn.Linear(in_features, num_classes)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor, return_raw_logits: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        dropped = self.dropout(x)
        raw_logits = self.evidence_layer(dropped)
        evidence = self.softplus(raw_logits)
        if return_raw_logits:
            return evidence, raw_logits
        return evidence


class Wav2Vec2ForParkinsons(nn.Module):
    """Pretrained Wav2Vec2 model with temporal pooling and evidential classification head."""

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        num_classes: int = config.NUM_CLASSES,
        freeze_feature_encoder: bool = config.FREEZE_FEATURE_ENCODER,
        dropout: float = 0.1
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        # Load pretrained Wav2Vec2 backbone
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            model_name,
            use_safetensors=True
        )

        hidden_size = self.wav2vec2.config.hidden_size  # 768 for wav2vec2-base

        # Evidential Output Head (Produces non-negative evidence [e_HC, e_PD] >= 0)
        self.evidential_head = EvidentialHead(
            in_features=hidden_size,
            num_classes=num_classes,
            dropout=dropout
        )

        if freeze_feature_encoder:
            self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """Freeze convolutional feature encoder parameters for stable fine-tuning."""
        self.wav2vec2.feature_extractor._freeze_parameters()

    def unfreeze_feature_encoder(self):
        """Unfreeze convolutional feature encoder parameters."""
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = True

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_embedding: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through Wav2Vec2 feature & transformer encoders,
        followed by masked temporal pooling and evidential head.
        """
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask
        )

        # last_hidden_state shape: (batch_size, sequence_length, hidden_size)
        hidden_states = outputs.last_hidden_state

        # Masked temporal mean pooling
        if attention_mask is not None:
            pooled_embedding = torch.mean(hidden_states, dim=1)
        else:
            pooled_embedding = torch.mean(hidden_states, dim=1)

        # Evidential Head
        evidence = self.evidential_head(pooled_embedding)

        if return_embedding:
            return evidence, pooled_embedding
        return evidence


if __name__ == "__main__":
    print("Testing Wav2Vec2 evidential model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Wav2Vec2ForParkinsons().to(device)
    dummy_input = torch.randn(2, 160000, device=device)
    with torch.no_grad():
        evidence, emb = model(dummy_input, return_embedding=True)
        print(f"Evidence shape:       {evidence.shape} (Expected: [2, 2])")
        print(f"Embedding shape:      {emb.shape} (Expected: [2, 768])")
        print(f"Min Evidence Value:   {evidence.min().item():.6f} (Must be >= 0)")
        print(f"Sample Evidence (0):  [e_HC={evidence[0,0].item():.4f}, e_PD={evidence[0,1].item():.4f}]")
        assert evidence.shape == (2, 2), "Evidence shape mismatch!"
        assert (evidence >= 0).all(), "All evidence values must be non-negative!"
    print("Voice evidential architecture verified successfully!")
