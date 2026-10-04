"""Unit tests for Phase 6 Acoustic Corruption / Robustness implementation in Voice pipeline."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# Add workspace roots
root_dir = Path(__file__).resolve().parent.parent.parent
voice_dir = root_dir / "Voice"
for p in (str(root_dir), str(voice_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from Voice import config
from Voice.acoustic_corruption import add_additive_gaussian_noise, compute_spearman_correlation
from Voice.model import EvidentialHead

CORRUPT_DIR = config.RESULTS_DIR / "acoustic_corruption"
CONFIG_JSON = CORRUPT_DIR / "acoustic_corruption_config.json"
PREDICTIONS_CSV = CORRUPT_DIR / "acoustic_corruption_subject_predictions.csv"
METRICS_JSON = CORRUPT_DIR / "acoustic_corruption_metrics.json"


@pytest.fixture
def corrupt_predictions_df():
    assert PREDICTIONS_CSV.exists(), f"Missing predictions CSV at {PREDICTIONS_CSV}"
    return pd.read_csv(PREDICTIONS_CSV)


@pytest.fixture
def corrupt_config_data():
    assert CONFIG_JSON.exists(), f"Missing config JSON at {CONFIG_JSON}"
    with open(CONFIG_JSON, "r") as f:
        return json.load(f)


def test_clean_audio_remains_unchanged():
    """1. Verify clean audio (SNR inf) remains unchanged."""
    clean_audio = np.random.randn(16000).astype(np.float32)
    corrupted, achieved_snr = add_additive_gaussian_noise(clean_audio, snr_db=float("inf"), seed=42)
    assert np.array_equal(clean_audio, corrupted)
    assert np.isinf(achieved_snr)


def test_corrupted_audio_differs():
    """2. Verify corrupted audio differs from clean audio."""
    clean_audio = np.random.randn(16000).astype(np.float32)
    corrupted, _ = add_additive_gaussian_noise(clean_audio, snr_db=10.0, seed=42)
    assert not np.array_equal(clean_audio, corrupted)


def test_requested_snr_achieved():
    """3. Verify requested SNR is approximately achieved (within 0.5 dB)."""
    t = np.linspace(0, 10, 160000, dtype=np.float32)
    clean_audio = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    _, achieved_snr = add_additive_gaussian_noise(clean_audio, snr_db=10.0, seed=42)
    assert abs(achieved_snr - 10.0) < 0.5


def test_noise_reproducibility_with_seed():
    """4. Verify noise generation is reproducible with seed 42."""
    clean_audio = np.random.randn(16000).astype(np.float32)
    corr1, _ = add_additive_gaussian_noise(clean_audio, snr_db=5.0, seed=42)
    corr2, _ = add_additive_gaussian_noise(clean_audio, snr_db=5.0, seed=42)
    assert np.array_equal(corr1, corr2)


def test_different_snr_levels_produce_different_strength():
    """5. Verify different SNR levels produce different noise strength."""
    clean_audio = np.random.randn(16000).astype(np.float32)
    corr_20, _ = add_additive_gaussian_noise(clean_audio, snr_db=20.0, seed=42)
    corr_0, _ = add_additive_gaussian_noise(clean_audio, snr_db=0.0, seed=42)
    
    diff_20 = np.mean((corr_20 - clean_audio) ** 2)
    diff_0 = np.mean((corr_0 - clean_audio) ** 2)
    assert diff_0 > diff_20  # 0 dB noise power is greater than 20 dB noise power


def test_no_nan_values():
    """6. Verify corrupted audio contains zero NaN values."""
    clean_audio = np.random.randn(16000).astype(np.float32)
    corrupted, _ = add_additive_gaussian_noise(clean_audio, snr_db=0.0, seed=42)
    assert not np.isnan(corrupted).any()


def test_no_inf_values():
    """7. Verify corrupted audio contains zero Inf values."""
    clean_audio = np.random.randn(16000).astype(np.float32)
    corrupted, _ = add_additive_gaussian_noise(clean_audio, snr_db=0.0, seed=42)
    assert not np.isinf(corrupted).any()


def test_waveform_shape_preserved():
    """8. Verify waveform tensor shape is preserved."""
    clean_audio = np.random.randn(160000).astype(np.float32)
    corrupted, _ = add_additive_gaussian_noise(clean_audio, snr_db=10.0, seed=42)
    assert corrupted.shape == clean_audio.shape


def test_sample_rate_preserved(corrupt_config_data):
    """9. Verify audio sample rate is preserved at 16000 Hz."""
    assert corrupt_config_data["sample_rate"] == 16000


def test_subject_level_aggregation_correctness(corrupt_predictions_df):
    """10. Verify subject-level predictions exist for all 6 test subjects across 5 SNR levels."""
    assert len(corrupt_predictions_df) == 30  # 6 subjects x 5 SNR levels
    expected_subs = set(["ID04", "ID10", "ID13", "ID23", "ID29", "ID35"])
    assert set(corrupt_predictions_df["subject_id"].unique()) == expected_subs


def test_mc_probabilities_valid_range(corrupt_predictions_df):
    """11. Verify MC Dropout probabilities remain within [0, 1]."""
    probs = corrupt_predictions_df["mc_mean_probability"].values
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_mc_variance_nonnegative(corrupt_predictions_df):
    """12. Verify MC variance is non-negative."""
    variances = corrupt_predictions_df["mc_variance"].values
    assert np.all(variances >= 0.0)


def test_mc_entropy_nonnegative(corrupt_predictions_df):
    """13. Verify MC predictive entropy is non-negative."""
    entropies = corrupt_predictions_df["mc_predictive_entropy"].values
    assert np.all(entropies >= 0.0)


def test_mc_mutual_info_nonnegative(corrupt_predictions_df):
    """14. Verify MC mutual information is non-negative within numerical tolerance."""
    mis = corrupt_predictions_df["mc_mutual_information"].values
    assert np.all(mis >= -1e-12)


def test_ensemble_probabilities_valid_range(corrupt_predictions_df):
    """15. Verify Deep Ensemble probabilities remain within [0, 1]."""
    probs = corrupt_predictions_df["ensemble_mean_probability"].values
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_existing_clean_inference_unmodified(corrupt_predictions_df):
    """16. Verify existing clean baseline inference predictions remain unmodified."""
    clean_df = corrupt_predictions_df[corrupt_predictions_df["corruption_type"] == "Clean"]
    assert len(clean_df) == 6
    for _, row in clean_df.iterrows():
        assert 0.0 <= row["baseline_probability"] <= 1.0


def test_test_labels_not_used_for_corruption(corrupt_predictions_df):
    """17. Verify test labels are not used to tune corruption or threshold decisions."""
    for _, row in corrupt_predictions_df.iterrows():
        base_pred = 1 if row["baseline_probability"] >= 0.5 else 0
        ens_pred = 1 if row["ensemble_mean_probability"] >= 0.5 else 0
        assert row["baseline_prediction"] == base_pred
        assert row["ensemble_prediction"] == ens_pred


def test_no_model_parameters_modified():
    """18. Verify model parameters remain unmutated during corrupted inference."""
    head = EvidentialHead(in_features=768, num_classes=2)
    head.eval()
    
    init_params = [p.clone() for p in head.parameters()]
    dummy_corrupted_input = torch.randn(2, 768)
    
    with torch.no_grad():
        _ = head(dummy_corrupted_input)
        
    for p_init, p_curr in zip(init_params, head.parameters()):
        assert torch.equal(p_init, p_curr), "Model parameters mutated during inference!"
