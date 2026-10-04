"""Tests for Wav2Vec2 architecture, embedding shapes, and head outputs."""

import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn

# Ensure workspace root and Voice directory are in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
voice_dir = root_dir / "Voice"
for p in (str(root_dir), str(voice_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from Voice import config
from Voice.model import EvidentialHead, Wav2Vec2ForParkinsons


def test_evidential_head_non_negative_evidence():
    """Verify EvidentialHead produces non-negative evidence >= 0 via Softplus."""
    head = EvidentialHead(in_features=768, num_classes=2, dropout=0.1)
    dummy_feat = torch.randn(4, 768)

    evidence = head(dummy_feat)
    assert evidence.shape == (4, 2)
    assert (evidence >= 0.0).all(), "Evidence values must be non-negative!"
    assert not torch.isnan(evidence).any(), "Evidence contains NaN!"
    assert not torch.isinf(evidence).any(), "Evidence contains Inf!"

    # Test return_raw_logits flag
    evidence_val, raw_logits = head(dummy_feat, return_raw_logits=True)
    assert raw_logits.shape == (4, 2)
    assert torch.allclose(evidence_val, torch.nn.functional.softplus(raw_logits))


def test_wav2vec2_model_instantiation_and_freezing():
    """Test Wav2Vec2 model instantiation and feature encoder parameter freezing."""
    model = Wav2Vec2ForParkinsons(
        model_name=config.MODEL_NAME,
        num_classes=2,
        freeze_feature_encoder=True
    )

    assert isinstance(model, nn.Module)
    assert hasattr(model, "wav2vec2")
    assert hasattr(model, "evidential_head")

    # Check feature encoder parameters are frozen
    for param in model.wav2vec2.feature_extractor.parameters():
        assert not param.requires_grad, "Feature encoder parameters should be frozen!"


def test_model_forward_pass_dimensions_and_embedding():
    """Test Wav2Vec2 forward pass shapes, embedding dimension (768), and attention mask handling."""
    model = Wav2Vec2ForParkinsons(
        model_name=config.MODEL_NAME,
        num_classes=2,
        freeze_feature_encoder=True
    )
    model.eval()

    batch_size = 2
    seq_len = 160000  # 10s at 16kHz
    dummy_audio = torch.randn(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    with torch.no_grad():
        evidence = model(dummy_audio, attention_mask=attention_mask)
        assert evidence.shape == (batch_size, 2)
        assert (evidence >= 0.0).all()
        assert not torch.isnan(evidence).any()

        # Test with return_embedding=True
        evidence_emb, embedding = model(dummy_audio, attention_mask=attention_mask, return_embedding=True)
        assert evidence_emb.shape == (batch_size, 2)
        assert embedding.shape == (batch_size, 768), "Pooled embedding must be 768-dimensional!"
        assert not torch.isnan(embedding).any()


def test_baseline_logits_crossentropy_integration():
    """
    AUDIT VERIFICATION B:
    Verify that the baseline Wav2Vec2 model uses EvidentialHead which outputs non-negative
    evidence [e_HC, e_PD] >= 0. Standard CrossEntropyLoss treats these evidence outputs
    as logits for backpropagation.
    """
    model = Wav2Vec2ForParkinsons(model_name=config.MODEL_NAME, num_classes=2)
    model.eval()

    dummy_input = torch.randn(2, 768)
    targets = torch.tensor([0, 1], dtype=torch.long)

    evidence = model(dummy_input)  # Shape (2, 2), non-negative

    # Baseline training passes evidence directly into CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(evidence, targets)

    assert not torch.isnan(loss), "CrossEntropyLoss on evidence output returned NaN!"
    assert loss.item() > 0.0

    # Probabilities in baseline evaluation are computed via softmax(evidence)
    probs = torch.softmax(evidence, dim=-1)
    assert probs.shape == (2, 2)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2))
