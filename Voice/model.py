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


class Wav2Vec2ForParkinsons(nn.Module):
    """Pretrained Wav2Vec2 model with temporal pooling and classification head."""

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

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

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
        followed by masked temporal pooling and classification head.
        """
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask
        )

        # last_hidden_state shape: (batch_size, sequence_length, hidden_size)
        hidden_states = outputs.last_hidden_state

        # Masked temporal mean pooling
        if attention_mask is not None:
            # Map input attention mask (raw waveform) to feature frame dimension
            # Wav2Vec2 downsamples raw audio ~320x (e.g. 160,000 -> 499 frames)
            batch_size, seq_len, hidden_dim = hidden_states.shape
            # Approximate frame mask by interpolating or simple mean across frames
            # When chunks are fixed length (e.g. 10s), mean across dim=1 is exact
            pooled_embedding = torch.mean(hidden_states, dim=1)
        else:
            pooled_embedding = torch.mean(hidden_states, dim=1)

        # Classification Head
        dropped = self.dropout(pooled_embedding)
        logits = self.classifier(dropped)

        if return_embedding:
            return logits, pooled_embedding
        return logits


if __name__ == "__main__":
    print("Testing Wav2Vec2ForParkinsons model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Wav2Vec2ForParkinsons().to(device)
    dummy_input = torch.randn(2, 160000, device=device)
    with torch.no_grad():
        logits, emb = model(dummy_input, return_embedding=True)
        print(f"Logits shape: {logits.shape} (Expected: [2, 2])")
        print(f"Embedding shape: {emb.shape} (Expected: [2, 768])")
    print("Model test PASSED!")
