"""Unit tests for Gait 1D CNN Baseline."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from Gait.baselines.cnn1d_baseline import config
from Gait.baselines.cnn1d_baseline.dataset import (
    Gait1DCNNWindowDataset,
    collate_gait_1d_batch,
    create_subject_splits,
)
from Gait.baselines.cnn1d_baseline.evaluate import calculate_metrics_dict
from Gait.baselines.cnn1d_baseline.model import Gait1DCNN
from Gait.preprocessing import GaitNormalizer, build_manifest


def test_gait_1d_model_forward():
    """Verify forward pass input/output dimensions and probabilities."""
    model = Gait1DCNN()
    # (B, 500, 16)
    x = torch.randn(4, 500, 16)
    logits, emb = model(x, return_embedding=True)

    assert logits.shape == (4, 1), f"Expected logits (4, 1), got {logits.shape}"
    assert emb.shape == (4, 128), f"Expected embedding (4, 128), got {emb.shape}"

    # Test probability output
    probs = model.predict_proba(x)
    assert probs.shape == (4, 1)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    # Also test transpose shape (B, 16, 500)
    x_t = torch.randn(4, 16, 500)
    logits_t = model(x_t)
    assert logits_t.shape == (4, 1)


def test_bce_loss_compatibility():
    """Verify binary cross-entropy loss computation."""
    model = Gait1DCNN()
    criterion = nn.BCEWithLogitsLoss()
    x = torch.randn(4, 500, 16)
    y = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    logits = model(x)
    loss = criterion(logits, y)

    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_subject_split_isolation():
    """Verify strict zero-leakage subject separation across train, val, test."""
    manifest_df = build_manifest(config.DATA_DIR)
    train_df, val_df, test_df = create_subject_splits(manifest_df, seed=42)

    train_subs = set(train_df["subject_id"].unique())
    val_subs = set(val_df["subject_id"].unique())
    test_subs = set(test_df["subject_id"].unique())

    assert len(train_subs & val_subs) == 0, "Train-Val subject leakage!"
    assert len(train_subs & test_subs) == 0, "Train-Test subject leakage!"
    assert len(val_subs & test_subs) == 0, "Val-Test subject leakage!"
    assert len(test_subs) == 25, f"Expected 25 test subjects, got {len(test_subs)}"


def test_normalizer_train_only():
    """Verify normalizer can be fit on dummy train data and transform test data."""
    train_data = [np.random.randn(500, 16) * 10 + 5 for _ in range(3)]
    test_data = np.random.randn(500, 16) * 10 + 5

    norm = GaitNormalizer().fit(train_data)
    transformed = norm.transform(test_data)

    assert transformed.shape == test_data.shape
    assert not np.isnan(transformed).any()


def test_metric_calculations_synthetic():
    """Verify calculation of metrics against known ground truth."""
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1])
    m = calculate_metrics_dict(y_true, y_prob)

    assert m["accuracy"] == 1.0
    assert m["balanced_accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall_sensitivity"] == 1.0
    assert m["specificity"] == 1.0
    assert m["f1_score"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["tp"] == 2 and m["tn"] == 2 and m["fp"] == 0 and m["fn"] == 0


def test_checkpoint_save_and_reload():
    """Verify saving and loading weights."""
    model = Gait1DCNN()
    model.eval()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        torch.save(model.state_dict(), tmp_path)
        model2 = Gait1DCNN()
        model2.load_state_dict(torch.load(tmp_path, weights_only=True))
        model2.eval()

        x = torch.randn(2, 500, 16)
        with torch.no_grad():
            out1 = model(x)
            out2 = model2(x)
        assert torch.allclose(out1, out2)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_isolation_from_existing_checkpoints():
    """Verify existing project metrics and deep ensemble checkpoints exist and are NOT modified."""
    existing_bce_metrics = config.GAIT_DIR / "outputs" / "metrics" / "test_metrics.json"
    existing_evi_metrics = config.GAIT_DIR / "outputs" / "metrics" / "gait_evidential_metrics.json"

    assert existing_bce_metrics.exists(), "Original test_metrics.json missing!"
    assert existing_evi_metrics.exists(), "Original gait_evidential_metrics.json missing!"
    assert config.BEST_MODEL_PATH.parent == config.CHECKPOINT_DIR
    assert config.CHECKPOINT_DIR.resolve() != (config.GAIT_DIR / "checkpoints").resolve()
