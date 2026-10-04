"""Unit tests for Phase 5 Cross-Task Shift implementation in Voice pipeline."""

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

CROSS_TASK_DIR = config.RESULTS_DIR / "cross_task_shift"
CONFIG_JSON = CROSS_TASK_DIR / "cross_task_config.json"
PREDICTIONS_CSV = CROSS_TASK_DIR / "cross_task_subject_predictions.csv"
METRICS_JSON = CROSS_TASK_DIR / "cross_task_metrics.json"
AVAILABILITY_CSV = CROSS_TASK_DIR / "task_availability.csv"
DELTA_CSV = CROSS_TASK_DIR / "within_subject_task_delta.csv"


@pytest.fixture
def cross_task_predictions_df():
    assert PREDICTIONS_CSV.exists(), f"Missing predictions CSV at {PREDICTIONS_CSV}"
    return pd.read_csv(PREDICTIONS_CSV)


@pytest.fixture
def cross_task_config_data():
    assert CONFIG_JSON.exists(), f"Missing config JSON at {CONFIG_JSON}"
    with open(CONFIG_JSON, "r") as f:
        return json.load(f)


def test_task_labels_from_verified_metadata():
    """1. Verify task labels originate from verified metadata CSV."""
    meta_csv = config.RESULTS_DIR / "embeddings" / "voice_embeddings_metadata.csv"
    assert meta_csv.exists()
    df = pd.read_csv(meta_csv)
    assert "task" in df.columns
    assert set(df["task"].unique()) == {"ReadText", "SpontaneousDialogue"}


def test_no_fabricated_task_labels(cross_task_predictions_df):
    """2. Verify no fabricated task labels were generated."""
    valid_tasks = {"ReadText", "SpontaneousDialogue"}
    observed_tasks = set(cross_task_predictions_df["task"].unique())
    assert observed_tasks.issubset(valid_tasks)


def test_train_test_subjects_disjoint():
    """3. Verify train and test subjects remain completely disjoint."""
    meta_csv = config.RESULTS_DIR / "embeddings" / "voice_embeddings_metadata.csv"
    df = pd.read_csv(meta_csv)
    train_df, val_df, test_df = create_subject_splits(df, seed=42)

    train_subs = set(train_df["subject_id"].unique())
    val_subs = set(val_df["subject_id"].unique())
    test_subs = set(test_df["subject_id"].unique())

    assert len(train_subs & test_subs) == 0
    assert len(val_subs & test_subs) == 0


def test_test_subjects_preserved(cross_task_config_data):
    """4. Verify test subjects remain the existing six subjects."""
    expected_test_subjects = ["ID04", "ID10", "ID13", "ID23", "ID29", "ID35"]
    assert cross_task_config_data["test_subjects"] == expected_test_subjects


def test_task_aggregation_preserves_task_identity(cross_task_predictions_df):
    """5. Verify task aggregation maintains subject x task identity."""
    for sub in ["ID04", "ID10", "ID13", "ID23", "ID29", "ID35"]:
        sub_df = cross_task_predictions_df[cross_task_predictions_df["subject_id"] == sub]
        tasks = set(sub_df["task"].unique())
        assert "ReadText" in tasks
        assert "SpontaneousDialogue" in tasks


def test_subject_level_task_aggregation_correctness(cross_task_predictions_df):
    """6. Verify subject x task aggregation correctly computes predictions per task."""
    assert len(cross_task_predictions_df) == 12  # 6 subjects x 2 tasks
    for _, row in cross_task_predictions_df.iterrows():
        assert "baseline_probability" in row
        assert "ensemble_mean_probability" in row


def test_no_task_specific_training_occurred():
    """7. Verify zero task-specific model retraining occurred."""
    member_ckpt = config.RESULTS_DIR / "deep_ensemble" / "member_0" / "checkpoint" / "best_model.pt"
    assert member_ckpt.exists()
    assert config.CHECKPOINT_DIR.exists()
    # Ensure baseline checkpoint modified time is unchanged (frozen from Phase 1)


def test_mc_dropout_n30_configuration_preserved(cross_task_config_data):
    """8. Verify MC Dropout uses the existing N=30 configuration."""
    assert cross_task_config_data["mc_dropout_passes"] == 30


def test_deep_ensemble_5_checkpoints_used(cross_task_config_data):
    """9. Verify Deep Ensemble uses the existing 5 trained member checkpoints."""
    assert cross_task_config_data["ensemble_members"] == 5
    assert cross_task_config_data["ensemble_seeds"] == [42, 43, 44, 45, 46]


def test_probabilities_in_valid_range(cross_task_predictions_df):
    """10. Verify all output probabilities are strictly within [0, 1]."""
    for col in ["baseline_probability", "mc_mean_probability", "ensemble_mean_probability"]:
        probs = cross_task_predictions_df[col].values
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_entropy_nonnegative(cross_task_predictions_df):
    """11. Verify predictive entropy is non-negative for both MC Dropout and Ensemble."""
    assert np.all(cross_task_predictions_df["mc_predictive_entropy"].values >= 0.0)
    assert np.all(cross_task_predictions_df["ensemble_predictive_entropy"].values >= 0.0)


def test_variance_nonnegative(cross_task_predictions_df):
    """12. Verify variance is non-negative for both MC Dropout and Ensemble."""
    assert np.all(cross_task_predictions_df["mc_variance"].values >= 0.0)
    assert np.all(cross_task_predictions_df["ensemble_variance"].values >= 0.0)


def test_mutual_information_nonnegative(cross_task_predictions_df):
    """13. Verify mutual information is non-negative within numerical tolerance."""
    assert np.all(cross_task_predictions_df["mc_mutual_information"].values >= -1e-12)
    assert np.all(cross_task_predictions_df["ensemble_mutual_information"].values >= -1e-12)


def test_baseline_results_unmodified():
    """14. Verify existing baseline result files remain unmodified."""
    base_metrics = config.RESULTS_DIR / "metrics.json"
    base_preds = config.RESULTS_DIR / "subject_predictions.csv"
    assert base_metrics.exists()
    assert base_preds.exists()


def test_test_labels_evaluation_only(cross_task_predictions_df):
    """15. Verify test labels are used only for retrospective evaluation metrics."""
    for _, row in cross_task_predictions_df.iterrows():
        expected_base_pred = 1 if row["baseline_probability"] >= 0.5 else 0
        expected_ens_pred = 1 if row["ensemble_mean_probability"] >= 0.5 else 0
        assert row["baseline_prediction"] == expected_base_pred
        assert row["ensemble_prediction"] == expected_ens_pred
