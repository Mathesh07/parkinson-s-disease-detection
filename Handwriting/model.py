"""Model architectures for Parkinson's handwriting detection.

This module provides:
1. ViTBinaryClassifier: Pretrained Hugging Face Vision Transformer (ViT) with a custom binary classification head.
2. CNNBaseline: A 4-stage Convolutional Neural Network baseline model for baseline comparison (Phase 6).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers import ViTModel
from config import VIT_MODEL_NAME


class EvidentialHead(nn.Module):
    """
    Evidential classification head for subjective logic / evidential neural networks.
    
    Transforms feature embeddings into non-negative evidence values:
        features (B, feature_dim) -> Dropout -> Linear (B, 2) -> Softplus -> Evidence (B, 2)
        
    Outputs:
        evidence: Non-negative tensor of shape (B, 2) representing [e_Healthy, e_Parkinson] >= 0.
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


class ViTBinaryClassifier(nn.Module):
    """Vision Transformer (ViT) with Evidential Output Head for Parkinson's handwriting detection.
    
    Uses a pretrained ViT backbone from Hugging Face. The evidential head
    maps the pooled representation ([CLS] token output) to 2 non-negative evidence values (0=Healthy, 1=Parkinson).
    """

    def __init__(
        self,
        model_name: str = VIT_MODEL_NAME,
        num_labels: int = 2,
        freeze_backbone: bool = True,
        output_attentions: bool = False
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.output_attentions = output_attentions

        # Load pretrained ViT backbone from Hugging Face
        self.backbone = ViTModel.from_pretrained(
            model_name,
            output_attentions=output_attentions
        )
        hidden_size = self.backbone.config.hidden_size

        # Evidential Head (Produces non-negative evidence [e_Healthy, e_Parkinson] >= 0)
        self.evidential_head = EvidentialHead(
            in_features=hidden_size,
            num_classes=num_labels,
            dropout=0.1
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freeze all parameters in the ViT encoder backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, unfreeze_last_n_layers: int = 2) -> None:
        """Unfreeze the last N encoder layers for fine-tuning."""
        if hasattr(self.backbone, "layernorm") and self.backbone.layernorm is not None:
            for param in self.backbone.layernorm.parameters():
                param.requires_grad = True

        if hasattr(self.backbone, "encoder"):
            layers = self.backbone.encoder.layer
        elif hasattr(self.backbone, "layers"):
            layers = self.backbone.layers
        else:
            raise AttributeError("Could not find layers to unfreeze in the ViT backbone.")

        total_layers = len(layers)
        for i in range(total_layers - unfreeze_last_n_layers, total_layers):
            for param in layers[i].parameters():
                param.requires_grad = True

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values, output_attentions=self.output_attentions)
        pooled_output = outputs.pooler_output
        evidence = self.evidential_head(pooled_output)
        if self.output_attentions:
            return evidence, outputs.attentions
        return evidence


class CNNBaseline(nn.Module):
    """4-stage Convolutional Neural Network baseline classifier with Evidential Output Head."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Stage 1: 224x224 -> 112x112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Stage 2: 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Stage 3: 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Stage 4: 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True)
        )
        self.evidential_head = EvidentialHead(
            in_features=128,
            num_classes=num_classes,
            dropout=0.3
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        feat = self.dense(x)
        evidence = self.evidential_head(feat)
        return evidence


if __name__ == "__main__":
    print("Testing ViT and CNN evidential architectures...")
    vit_model = ViTBinaryClassifier()
    dummy_img = torch.randn(4, 3, 224, 224)
    vit_evidence = vit_model(dummy_img)
    print(f"ViT Evidence Shape:     {vit_evidence.shape} (Expected: [4, 2])")
    print(f"ViT Min Evidence:       {vit_evidence.min().item():.6f} (Must be >= 0)")
    print(f"Sample ViT Evidence(0): [e_HC={vit_evidence[0,0].item():.4f}, e_PD={vit_evidence[0,1].item():.4f}]")
    assert vit_evidence.shape == (4, 2), "ViT evidence shape mismatch!"
    assert (vit_evidence >= 0).all(), "ViT evidence values must be non-negative!"

    cnn_model = CNNBaseline()
    cnn_evidence = cnn_model(dummy_img)
    print(f"CNN Evidence Shape:     {cnn_evidence.shape} (Expected: [4, 2])")
    print(f"CNN Min Evidence:       {cnn_evidence.min().item():.6f} (Must be >= 0)")
    assert cnn_evidence.shape == (4, 2), "CNN evidence shape mismatch!"
    assert (cnn_evidence >= 0).all(), "CNN evidence values must be non-negative!"
    print("Handwriting evidential architectures verified successfully!")
