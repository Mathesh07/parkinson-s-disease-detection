"""Model architectures for Parkinson's handwriting detection.

This module provides:
1. ViTBinaryClassifier: Pretrained Hugging Face Vision Transformer (ViT) with a custom binary classification head.
2. CNNBaseline: A 4-stage Convolutional Neural Network baseline model for baseline comparison (Phase 6).
"""

import torch
import torch.nn as nn
from transformers import ViTModel
from config import VIT_MODEL_NAME


class ViTBinaryClassifier(nn.Module):
    """Vision Transformer (ViT) for Parkinson's handwriting binary classification.
    
    Uses a pretrained ViT backbone from Hugging Face. The classification head
    maps the pooled representation ([CLS] token output) to 2 classes (0=Healthy, 1=Parkinson).
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

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
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
        logits = self.classifier(pooled_output)
        if self.output_attentions:
            return logits, outputs.attentions
        return logits


class CNNBaseline(nn.Module):
    """Simple 4-stage Convolutional Neural Network baseline classifier.
    
    Used for Phase 6 to compare ViT performance against a standard CNN architecture.
    """

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
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits
