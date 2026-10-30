"""Unit tests for Gait Base-Paper 2D CNN Model Architecture."""

import pytest
import torch

from Gait.baselines.base_paper_cnn.model import GaitBasePaper2DCNN


def test_model_forward_shape():
    """Verify 2D CNN forward pass shape on (B, 16, 33, 32)."""
    model = GaitBasePaper2DCNN()
    x = torch.randn(4, 16, 33, 32)
    logits = model(x)

    assert logits.shape == (4, 1)
    probs = model.predict_proba(x)
    assert probs.shape == (4, 1)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_feature_extraction():
    """Verify feature extractor extracts 128-dim embeddings."""
    model = GaitBasePaper2DCNN()
    x = torch.randn(2, 16, 33, 32)
    features = model.extract_features(x)

    assert features.shape == (2, 128)
