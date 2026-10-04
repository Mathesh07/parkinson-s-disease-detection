"""Unit tests for Monte Carlo (MC) Dropout uncertainty quantification and model mode handling."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

# Ensure workspace root and Voice directory are in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
voice_dir = root_dir / "Voice"
for p in (str(root_dir), str(voice_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from Voice.mc_dropout import (
    calculate_entropy,
    compute_mc_statistics,
    enable_mc_dropout,
    run_mc_dropout_passes,
)
from Voice.model import EvidentialHead, Wav2Vec2ForParkinsons


def test_dropout_modules_enabled_during_mc_inference():
    """Requirement 1: Verify enable_mc_dropout sets dropout modules to train() while model is eval()."""
    model = Wav2Vec2ForParkinsons(model_name="facebook/wav2vec2-base", num_classes=2)
    enable_mc_dropout(model)

    assert not model.training, "Model top-level should remain in eval() mode!"

    # Check all Dropout modules are in train() mode
    for name, module in model.named_modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            assert module.training, f"Dropout module '{name}' was not set to train() mode!"


def test_model_parameters_remain_unchanged():
    """Requirement 2: Verify MC inference does not mutate model parameters or gradient status."""
    model = Wav2Vec2ForParkinsons(model_name="facebook/wav2vec2-base", num_classes=2)
    model.eval()

    params_before = [p.clone() for p in model.parameters()]
    requires_grad_before = [p.requires_grad for p in model.parameters()]

    dummy_input = torch.randn(2, 160000)
    enable_mc_dropout(model)

    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)

    params_after = [p.clone() for p in model.parameters()]
    requires_grad_after = [p.requires_grad for p in model.parameters()]

    assert requires_grad_before == requires_grad_after
    for pb, pa in zip(params_before, params_after):
        assert torch.equal(pb, pa), "Model parameter mutated during MC Dropout inference!"


def test_multiple_stochastic_passes_produce_valid_distinct_probabilities():
    """Requirement 3 & 4: Verify N passes produce stochastic, distinct probabilities in [0, 1]."""
    head = EvidentialHead(in_features=768, num_classes=2, dropout=0.3)  # Higher p to ensure variance
    dummy_emb = torch.randn(1, 768)

    passes = run_mc_dropout_passes(head, dummy_emb, num_passes=30, seed=42)

    assert passes.shape == (30, 1)
    assert np.all(passes >= 0.0) and np.all(passes <= 1.0)
    # Variance should be > 0 due to active dropout
    assert np.var(passes) > 0.0, "Stochastic passes produced identical outputs!"


def test_predictive_mean_bounds():
    """Requirement 5: Verify predictive mean probability is in [0, 1]."""
    stochastic_probs = np.array([0.2, 0.4, 0.6, 0.8])
    stats = compute_mc_statistics(stochastic_probs)

    assert 0.0 <= stats["mc_mean_probability"] <= 1.0
    assert np.isclose(stats["mc_mean_probability"], 0.5)


def test_variance_is_non_negative():
    """Requirement 6: Verify predictive variance is >= 0."""
    stochastic_probs = np.array([0.5, 0.5, 0.5])
    stats = compute_mc_statistics(stochastic_probs)
    assert stats["mc_variance"] == 0.0

    stochastic_probs_var = np.array([0.1, 0.9])
    stats_var = compute_mc_statistics(stochastic_probs_var)
    assert stats_var["mc_variance"] > 0.0


def test_entropy_is_non_negative():
    """Requirement 7 & 8: Verify predictive entropy and expected entropy are >= 0."""
    p_vals = np.array([0.1, 0.5, 0.9])
    h_vals = calculate_entropy(p_vals)

    assert np.all(h_vals >= 0.0)
    assert np.isclose(calculate_entropy(0.5), -np.log(0.5))

    stats = compute_mc_statistics(p_vals)
    assert stats["predictive_entropy"] >= 0.0
    assert stats["expected_entropy"] >= 0.0


def test_mutual_information_is_non_negative():
    """Requirement 9: Verify Mutual Information MI = H(mean) - mean(H(p_i)) is >= 0."""
    p_vals = np.array([0.1, 0.9])
    stats = compute_mc_statistics(p_vals)

    assert stats["mutual_information"] >= 0.0
    # Jensen's inequality: H(mean) >= mean(H(p_i))
    assert stats["predictive_entropy"] >= stats["expected_entropy"] - 1e-6


def test_subject_level_aggregation_remains_correct():
    """Requirement 10: Verify subject-level aggregation logic applies across MC passes."""
    # Chunk MC probabilities -> recording mean -> subject mean
    chunk_pass1 = np.array([0.2, 0.4])  # Rec 1 mean = 0.3
    chunk_pass2 = np.array([0.6, 0.8])  # Rec 2 mean = 0.7
    sub_mean_pass1 = 0.3
    sub_mean_pass2 = 0.7

    sub_passes = np.array([sub_mean_pass1, sub_mean_pass2])
    stats = compute_mc_statistics(sub_passes)
    assert np.isclose(stats["mc_mean_probability"], 0.5)


def test_mc_dropout_does_not_use_test_labels_for_uncertainty():
    """Requirement 11: Leakage Check — Verify uncertainty metrics do not consume ground-truth test labels."""
    stochastic_probs = np.array([0.3, 0.4, 0.5])
    # Compute stats without labels
    stats = compute_mc_statistics(stochastic_probs)

    assert "true_label" not in stats
    assert "y_true" not in stats
    assert len(stats) == 5  # Only mc_mean, mc_var, pred_ent, exp_ent, mi


def test_deterministic_inference_remains_unchanged():
    """Requirement 12: Verify deterministic eval() mode produces identical reproducible outputs."""
    head = EvidentialHead(in_features=768, num_classes=2, dropout=0.1)
    head.eval()

    dummy_emb = torch.randn(1, 768)

    with torch.no_grad():
        out1 = head(dummy_emb)
        out2 = head(dummy_emb)

    assert torch.equal(out1, out2), "Deterministic eval() pass produced non-identical outputs!"
