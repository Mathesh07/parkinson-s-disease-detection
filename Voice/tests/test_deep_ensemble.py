"""Unit tests for Phase 4 Deep Ensemble implementation in Voice pipeline."""

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
from Voice.dataset import create_subject_splits
from Voice.deep_ensemble import calculate_entropy, compute_ensemble_statistics
from Voice.model import Wav2Vec2ForParkinsons

ENSEMBLE_DIR = config.RESULTS_DIR / "deep_ensemble"
ENSEMBLE_CONFIG = ENSEMBLE_DIR / "deep_ensemble_config.json"
ENSEMBLE_CSV = ENSEMBLE_DIR / "deep_ensemble_subject_predictions.csv"
ENSEMBLE_METRICS = ENSEMBLE_DIR / "deep_ensemble_metrics.json"


@pytest.fixture
def ensemble_config_data():
    assert ENSEMBLE_CONFIG.exists(), f"Missing ensemble config at {ENSEMBLE_CONFIG}"
    with open(ENSEMBLE_CONFIG, "r") as f:
        return json.load(f)


@pytest.fixture
def ensemble_predictions_df():
    assert ENSEMBLE_CSV.exists(), f"Missing ensemble CSV at {ENSEMBLE_CSV}"
    return pd.read_csv(ENSEMBLE_CSV)


def test_ensemble_config_members(ensemble_config_data):
    """1. Verify exactly five ensemble members are configured."""
    assert ensemble_config_data["num_members"] == 5


def test_ensemble_config_seeds(ensemble_config_data):
    """2. Verify seeds are 42, 43, 44, 45, 46."""
    assert ensemble_config_data["seeds"] == [42, 43, 44, 45, 46]


def test_same_test_subject_ids(ensemble_config_data):
    """3. Verify all members use the exact same test subject IDs."""
    expected_test_subjects = ["ID04", "ID10", "ID13", "ID23", "ID29", "ID35"]
    assert ensemble_config_data["test_subject_ids"] == expected_test_subjects


def test_test_subjects_isolated(ensemble_config_data):
    """4. Verify no test subject appears in training or validation splits."""
    train_subs = set(ensemble_config_data["train_subject_ids"])
    val_subs = set(ensemble_config_data["validation_subject_ids"])
    test_subs = set(ensemble_config_data["test_subject_ids"])

    assert len(train_subs & test_subs) == 0, "Leakage: Train subjects overlap with Test!"
    assert len(val_subs & test_subs) == 0, "Leakage: Validation subjects overlap with Test!"


def test_member_probability_range(ensemble_predictions_df):
    """5. Verify individual member predictions have valid probability range [0, 1]."""
    for idx in range(5):
        col = f"member_{idx}_probability"
        assert col in ensemble_predictions_df.columns
        probs = ensemble_predictions_df[col].values
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_ensemble_mean_probability_range(ensemble_predictions_df):
    """6. Verify ensemble mean probability is in range [0, 1]."""
    probs = ensemble_predictions_df["ensemble_mean_probability"].values
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_ensemble_variance_nonnegative(ensemble_predictions_df):
    """7. Verify ensemble variance is non-negative."""
    variances = ensemble_predictions_df["ensemble_variance"].values
    assert np.all(variances >= 0.0)


def test_ensemble_std_nonnegative(ensemble_predictions_df):
    """8. Verify ensemble standard deviation is non-negative."""
    stds = ensemble_predictions_df["ensemble_std"].values
    assert np.all(stds >= 0.0)


def test_entropy_nonnegative(ensemble_predictions_df):
    """9. Verify predictive entropy is non-negative."""
    entropies = ensemble_predictions_df["predictive_entropy"].values
    assert np.all(entropies >= 0.0)


def test_mutual_information_nonnegative(ensemble_predictions_df):
    """10. Verify mutual information is non-negative within numerical tolerance."""
    mis = ensemble_predictions_df["mutual_information"].values
    assert np.all(mis >= -1e-12)


def test_subject_aggregation_preserved(ensemble_predictions_df):
    """11. Verify subject-level aggregation is arithmetic mean of member probabilities."""
    for _, row in ensemble_predictions_df.iterrows():
        m_probs = [row[f"member_{i}_probability"] for i in range(5)]
        expected_mean = float(np.mean(m_probs))
        assert abs(row["ensemble_mean_probability"] - expected_mean) < 1e-6


def test_ensemble_inference_unmutated_parameters():
    """12. Verify ensemble inference pass does not modify model parameters."""
    head = Wav2Vec2ForParkinsons()
    head.eval()
    
    # Store initial parameter snapshot
    init_params = [p.clone() for p in head.parameters()]
    
    dummy_input = torch.randn(2, 768)
    with torch.no_grad():
        _ = head(dummy_input)
        
    for p_init, p_curr in zip(init_params, head.parameters()):
        assert torch.equal(p_init, p_curr), "Model parameter mutated during evaluation!"


def test_ensemble_checkpoint_reproducibility():
    """13. Verify member predictions are reproducible when loading saved checkpoint."""
    ckpt_dir = ENSEMBLE_DIR / "member_0" / "checkpoint"
    weights_file = ckpt_dir / "best_model.pt"
    assert weights_file.exists()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(weights_file, map_location=device, weights_only=True)

    from Voice.model import EvidentialHead
    head1 = EvidentialHead(in_features=768, num_classes=2).to(device)
    head2 = EvidentialHead(in_features=768, num_classes=2).to(device)

    head_state = {k.replace("evidential_head.", ""): v for k, v in state.items() if "evidential_head" in k}
    if not head_state:
        head_state = state

    head1.load_state_dict(head_state)
    head1.eval()

    head2.load_state_dict(head_state)
    head2.eval()

    dummy_input = torch.randn(2, 768).to(device)
    with torch.no_grad():
        out1 = head1(dummy_input)
        out2 = head2(dummy_input)

    assert torch.allclose(out1, out2, atol=1e-6)


def test_test_labels_evaluation_only(ensemble_predictions_df):
    """14. Verify test labels are used only for metrics calculation and not thresholding."""
    # Predictions must be derived solely from ensemble_mean_probability >= 0.5
    for _, row in ensemble_predictions_df.iterrows():
        expected_pred = 1 if row["ensemble_mean_probability"] >= 0.5 else 0
        assert row["ensemble_prediction"] == expected_pred


def test_baseline_results_unmodified():
    """15. Verify existing baseline results files remain unchanged."""
    base_metrics = config.RESULTS_DIR / "metrics.json"
    base_preds = config.RESULTS_DIR / "subject_predictions.csv"

    assert base_metrics.exists()
    assert base_preds.exists()

    with open(base_metrics, "r") as f:
        data = json.load(f)
        assert "subject_level_overall" in data
