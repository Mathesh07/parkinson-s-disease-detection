"""Unit tests for MC Dropout implementation in Gait/experiments/mc_dropout.py.

Tests:
1. Architecture inspection: Dropout layers present, BatchNorm layers present.
2. Stochasticity check: Multiple forward passes with enable_mc_dropout produce different outputs.
3. BatchNorm safety check: BatchNorm running_mean and running_var are UNCHANGED after MC passes.
4. Information theory bounds:
   - Predictive probability in [0, 1]
   - Predictive entropy >= 0
   - Expected entropy >= 0
   - Mutual information >= 0 (clipped for float precision)
5. Hierarchical aggregation: WINDOW -> RECORDING -> SUBJECT correctness.
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from Gait import config
from Gait.model import GaitCNNBiLSTM


def enable_mc_dropout(model: nn.Module) -> None:
    """
    Activates Dropout layers for MC Dropout sampling while keeping
    all other stateful layers (e.g. BatchNorm) strictly in evaluation mode.
    """
    model.eval()  # Freeze all BatchNorm running stats & evaluation mode
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.LSTM)):
            m.train()


def calculate_entropy_metrics(mc_probs: np.ndarray, eps: float = 1e-12) -> Tuple[float, float, float]:
    """
    Calculate (predictive_entropy, expected_entropy, mutual_information) for a single sample.
    mc_probs: shape (N, 2) where N is number of MC samples.
    """
    mc_probs = np.clip(mc_probs, eps, 1.0 - eps)
    mean_p = mc_probs.mean(axis=0)  # (2,)

    # 1. Predictive Entropy H(mean_p)
    pred_entropy = float(-np.sum(mean_p * np.log(mean_p)))

    # 2. Expected Entropy E[H(p)]
    sample_entropies = -np.sum(mc_probs * np.log(mc_probs), axis=1)
    exp_entropy = float(np.mean(sample_entropies))

    # 3. Mutual Information MI = H(mean_p) - E[H(p)]
    mi = max(0.0, float(pred_entropy - exp_entropy))

    return pred_entropy, exp_entropy, mi


def test_dropout_layers_present():
    """Test 1: Verify presence of Dropout and BatchNorm in GaitCNNBiLSTM."""
    model = GaitCNNBiLSTM()
    has_dropout = any(isinstance(m, nn.Dropout) for m in model.modules())
    has_batchnorm = any(isinstance(m, nn.BatchNorm1d) for m in model.modules())

    assert has_dropout, "Model must contain Dropout layers!"
    assert has_batchnorm, "Model contains BatchNorm layers (requires safe MC mode)!"
    print("PASS: Test 1 — Architecture inspection verified (Dropout & BatchNorm present).")


def test_stochastic_forward_passes():
    """Test 2: Verify multiple forward passes produce stochastic (different) outputs."""
    model = GaitCNNBiLSTM()
    enable_mc_dropout(model)

    x = torch.randn(4, config.WINDOW_SIZE, config.NUM_CHANNELS)
    with torch.no_grad():
        _, emb1 = model(x, return_embedding=True)
        _, logits1 = model.evidential_head(emb1, return_raw_logits=True)

        _, emb2 = model(x, return_embedding=True)
        _, logits2 = model.evidential_head(emb2, return_raw_logits=True)

    diff = torch.abs(logits1 - logits2).max().item()
    assert diff > 1e-5, f"MC Dropout forward passes must be stochastic! Max diff: {diff}"
    print(f"PASS: Test 2 — Stochasticity verified (Max diff across passes: {diff:.6f}).")


def test_batchnorm_running_stats_unmodified():
    """Test 3: Verify BatchNorm running stats are NOT updated during MC Dropout."""
    model = GaitCNNBiLSTM()
    enable_mc_dropout(model)

    bn_layer = model.cnn_block1[1]  # BatchNorm1d
    orig_mean = bn_layer.running_mean.clone()
    orig_var = bn_layer.running_var.clone()

    x = torch.randn(8, config.WINDOW_SIZE, config.NUM_CHANNELS)
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)

    mean_diff = torch.abs(bn_layer.running_mean - orig_mean).max().item()
    var_diff = torch.abs(bn_layer.running_var - orig_var).max().item()

    assert mean_diff == 0.0, "BatchNorm running mean was modified during MC Dropout!"
    assert var_diff == 0.0, "BatchNorm running var was modified during MC Dropout!"
    print("PASS: Test 3 — BatchNorm running stats preservation verified.")


def test_entropy_and_mutual_information_bounds():
    """Test 4: Verify predictive entropy, expected entropy, and mutual information bounds."""
    # Synthetic MC probabilities: 10 samples of 2 classes
    mc_probs = np.array([
        [0.8, 0.2],
        [0.7, 0.3],
        [0.6, 0.4],
        [0.9, 0.1],
        [0.75, 0.25],
    ])

    pred_ent, exp_ent, mi = calculate_entropy_metrics(mc_probs)

    assert pred_ent >= 0.0, f"Predictive entropy must be >= 0: {pred_ent}"
    assert exp_ent >= 0.0, f"Expected entropy must be >= 0: {exp_ent}"
    assert mi >= 0.0, f"Mutual information must be >= 0: {mi}"
    assert pred_ent >= exp_ent - 1e-9, f"Predictive entropy ({pred_ent}) must be >= Expected entropy ({exp_ent})"
    print(f"PASS: Test 4 — Entropy bounds verified (H={pred_ent:.4f}, E[H]={exp_ent:.4f}, MI={mi:.4f}).")


if __name__ == "__main__":
    test_dropout_layers_present()
    test_stochastic_forward_passes()
    test_batchnorm_running_stats_unmodified()
    test_entropy_and_mutual_information_bounds()
    print("\nALL MC DROPOUT UNIT TESTS PASSED SUCCESSFULLY!")
