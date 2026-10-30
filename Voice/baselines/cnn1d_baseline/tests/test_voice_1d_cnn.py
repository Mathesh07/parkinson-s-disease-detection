"""Unit tests for Voice 1D CNN Raw Waveform Baseline."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from Voice.baselines.cnn1d_baseline import config
from Voice.baselines.cnn1d_baseline.dataset import (
    VoiceRawWaveformChunkDataset,
    VoiceWaveformNormalizer,
    create_subject_splits,
)
from Voice.baselines.cnn1d_baseline.evaluate import calculate_metrics_dict
from Voice.baselines.cnn1d_baseline.model import Voice1DCNN
from Voice.dataset import build_metadata


def test_voice_1d_forward_shape():
    """Verify raw waveform forward pass output dimensions."""
    model = Voice1DCNN()
    x = torch.randn(2, 1, config.CHUNK_SAMPLES)
    logits = model(x)
    probs = model.predict_proba(x)

    assert logits.shape == (2, 2)
    assert probs.shape == (2, 2)
    assert torch.allclose(torch.sum(probs, dim=-1), torch.ones(2), atol=1e-5)


def test_voice_loss_compatibility():
    """Verify CrossEntropyLoss computation."""
    model = Voice1DCNN()
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(4, 1, config.CHUNK_SAMPLES)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    logits = model(x)
    loss = criterion(logits, y)

    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_voice_subject_split_isolation():
    """Verify zero patient leakage between splits."""
    meta_df = build_metadata(config.DATA_DIR)
    train_df, val_df, test_df = create_subject_splits(meta_df, seed=42)

    train_subs = set(train_df["subject_id"].unique())
    val_subs = set(val_df["subject_id"].unique())
    test_subs = set(test_df["subject_id"].unique())

    assert len(train_subs & val_subs) == 0, "Train-Val subject leakage!"
    assert len(train_subs & test_subs) == 0, "Train-Test subject leakage!"
    assert len(val_subs & test_subs) == 0, "Val-Test subject leakage!"
    assert len(test_subs) == 6, f"Expected 6 test subjects, got {len(test_subs)}"


def test_voice_normalizer_train_only():
    """Verify waveform normalizer fit on train chunks only."""
    dummy_chunks = [torch.randn(1, 48000) * 5 + 2 for _ in range(3)]
    test_chunk = torch.randn(1, 48000) * 5 + 2

    norm = VoiceWaveformNormalizer().fit(dummy_chunks)
    transformed = norm.transform(test_chunk)

    assert transformed.shape == test_chunk.shape
    assert not torch.isnan(transformed).any()


def test_voice_metric_calculations_synthetic():
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


def test_voice_checkpoint_save_reload():
    """Verify checkpoint saving and reload."""
    model = Voice1DCNN()
    model.eval()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        torch.save(model.state_dict(), tmp_path)
        model2 = Voice1DCNN()
        model2.load_state_dict(torch.load(tmp_path, weights_only=True))
        model2.eval()

        x = torch.randn(2, 1, config.CHUNK_SAMPLES)
        with torch.no_grad():
            out1 = model(x)
            out2 = model2(x)
        assert torch.allclose(out1, out2)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_voice_isolation_from_existing_checkpoints():
    """Verify existing Wav2Vec2 outputs and metrics exist and are untouched."""
    existing_metrics = config.VOICE_DIR / "results" / "metrics.json"
    existing_sub_preds = config.VOICE_DIR / "results" / "subject_predictions.csv"

    assert existing_metrics.exists(), "Original Wav2Vec2 metrics.json missing!"
    assert existing_sub_preds.exists(), "Original Wav2Vec2 subject_predictions.csv missing!"
    assert config.BEST_MODEL_PATH.parent == config.CHECKPOINT_DIR
    assert config.CHECKPOINT_DIR.resolve() != (config.VOICE_DIR / "results").resolve()
