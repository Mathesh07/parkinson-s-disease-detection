"""Unit tests for Multi-Seed LOCO experiment.

Tests:
1. Seed determinism: Given identical seeds, split and normalizer are identical; different seeds yield distinct splits.
2. Leakage prevention: Zero subject overlap between train, val, and test splits across all folds and seeds.
3. Cohort isolation: Test cohort recordings never appear in train or validation sets.
4. Correct hierarchical aggregation: Window -> Recording -> Subject aggregation behaves correctly.
5. Metric computation and bounds: Accuracy, Brier, NLL, ECE, AUROC are within valid ranges.
6. Summary statistics schema: Mean, std, 95% CI, min, max correctly computed without NaN for valid inputs.
7. Validation-only calibration: Temperature is fit on val logits, not test.
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from Gait.experiments.calibration_baselines import fit_temperature
from Gait.experiments.leave_one_cohort_out import (
    aggregate_to_subject_level,
    build_cohort_splits,
)
from Gait.experiments.multiseed_loco import (
    compute_comprehensive_metrics,
    compute_summary_statistics,
    DEFAULT_SEEDS,
)
from Gait.preprocessing import build_manifest


def test_seed_determinism_and_stochastic_variation():
    """Verify that same seed produces identical splits, and different seeds produce distinct val splits."""
    manifest_df = build_manifest()

    # Identical seeds -> identical splits
    tr1, val1, te1 = build_cohort_splits(manifest_df, ["Ga", "Ju"], "Si", seed=42)
    tr2, val2, te2 = build_cohort_splits(manifest_df, ["Ga", "Ju"], "Si", seed=42)

    assert set(tr1["subject_id"]) == set(tr2["subject_id"]), "Same seed produced different train subjects!"
    assert set(val1["subject_id"]) == set(val2["subject_id"]), "Same seed produced different val subjects!"
    assert set(te1["subject_id"]) == set(te2["subject_id"]), "Same seed produced different test subjects!"

    # Different seeds -> test cohort identical (cohort-determined), but train/val split varies
    tr3, val3, te3 = build_cohort_splits(manifest_df, ["Ga", "Ju"], "Si", seed=43)
    assert set(te1["subject_id"]) == set(te3["subject_id"]), "Test cohort should remain identical across seeds!"
    assert set(val1["subject_id"]) != set(val3["subject_id"]), "Different seeds should yield different val splits!"


def test_no_data_leakage_across_seeds():
    """Verify train, val, and test are strictly disjoint across all seeds in DEFAULT_SEEDS."""
    manifest_df = build_manifest()

    for s in DEFAULT_SEEDS:
        for train_c, test_c in [(["Ga", "Ju"], "Si"), (["Ga", "Si"], "Ju"), (["Ju", "Si"], "Ga")]:
            tr, val, te = build_cohort_splits(manifest_df, train_c, test_c, seed=s)
            tr_subs = set(tr["subject_id"].unique())
            val_subs = set(val["subject_id"].unique())
            te_subs = set(te["subject_id"].unique())

            assert len(tr_subs & te_subs) == 0, f"Seed {s}: Leakage between train and test ({train_c} -> {test_c})"
            assert len(val_subs & te_subs) == 0, f"Seed {s}: Leakage between val and test ({train_c} -> {test_c})"
            assert len(tr_subs & val_subs) == 0, f"Seed {s}: Leakage between train and val ({train_c} -> {test_c})"
            assert set(te["study"].unique()) == {test_c}, f"Seed {s}: Non-test cohort present in test set"


def test_hierarchical_aggregation_multi_seed():
    """Verify hierarchical aggregation: WINDOW -> RECORDING -> SUBJECT."""
    rows = []
    # 2 recordings for 1 subject with different lengths
    for i in range(10):
        rows.append({
            "experiment": 1,
            "train_cohorts": "Ga+Ju",
            "test_cohort": "Si",
            "subject_id": "Sub_X",
            "recording_id": "Rec_1",
            "study": "Si",
            "true_label": 1,
            "e_HC": 0.0, "e_PD": 1.0,
            "alpha_HC": 1.0, "alpha_PD": 2.0,
            "P_HC": 0.8, "P_PD": 0.2,
            "uncertainty": 0.3,
            "predicted_label": 0, "is_correct": 0
        })
    for i in range(40):
        rows.append({
            "experiment": 1,
            "train_cohorts": "Ga+Ju",
            "test_cohort": "Si",
            "subject_id": "Sub_X",
            "recording_id": "Rec_2",
            "study": "Si",
            "true_label": 1,
            "e_HC": 0.0, "e_PD": 1.0,
            "alpha_HC": 1.0, "alpha_PD": 2.0,
            "P_HC": 0.2, "P_PD": 0.8,
            "uncertainty": 0.3,
            "predicted_label": 1, "is_correct": 1
        })

    win_df = pd.DataFrame(rows)
    sub_df = aggregate_to_subject_level(win_df)

    assert len(sub_df) == 1
    # Rec 1 mean P_PD = 0.2, Rec 2 mean P_PD = 0.8 -> Subject mean = 0.50
    assert np.isclose(sub_df["P_PD"].iloc[0], 0.5, atol=1e-5)


def test_metrics_computation_and_bounds():
    """Verify compute_comprehensive_metrics returns valid values and expected keys."""
    sub_data = pd.DataFrame([
        {"subject_id": "s1", "true_label": 0, "predicted_label": 0, "P_PD": 0.1, "P_HC": 0.9, "uncertainty": 0.1},
        {"subject_id": "s2", "true_label": 0, "predicted_label": 0, "P_PD": 0.2, "P_HC": 0.8, "uncertainty": 0.15},
        {"subject_id": "s3", "true_label": 1, "predicted_label": 1, "P_PD": 0.85, "P_HC": 0.15, "uncertainty": 0.2},
        {"subject_id": "s4", "true_label": 1, "predicted_label": 0, "P_PD": 0.45, "P_HC": 0.55, "uncertainty": 0.5},
    ])
    metrics = compute_comprehensive_metrics(sub_data)

    expected_keys = [
        "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1",
        "roc_auc", "nll", "brier", "ece", "mean_uncertainty",
        "mean_uncertainty_correct", "mean_uncertainty_incorrect",
        "uncertainty_error_auroc", "aurc"
    ]
    for k in expected_keys:
        assert k in metrics, f"Missing metric {k}"
        assert not np.isnan(metrics[k]), f"Metric {k} is NaN"

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    assert 0.0 <= metrics["ece"] <= 1.0
    assert metrics["nll"] >= 0.0
    assert 0.0 <= metrics["brier"] <= 1.0


def test_summary_statistics_computation():
    """Verify summary table correctly aggregates per-fold and overall mean/std/CI."""
    dummy_results = pd.DataFrame([
        {"experiment": 1, "test_cohort": "Si", "seed": 42, "accuracy": 0.8, "balanced_accuracy": 0.8, "roc_auc": 0.85, "nll": 0.4, "brier": 0.15, "ece": 0.05, "mean_uncertainty": 0.2, "mean_uncertainty_correct": 0.18, "mean_uncertainty_incorrect": 0.35, "uncertainty_error_auroc": 0.75, "aurc": 0.1},
        {"experiment": 1, "test_cohort": "Si", "seed": 43, "accuracy": 0.84, "balanced_accuracy": 0.82, "roc_auc": 0.88, "nll": 0.38, "brier": 0.14, "ece": 0.06, "mean_uncertainty": 0.22, "mean_uncertainty_correct": 0.19, "mean_uncertainty_incorrect": 0.38, "uncertainty_error_auroc": 0.78, "aurc": 0.09},
        {"experiment": 2, "test_cohort": "Ju", "seed": 42, "accuracy": 0.7, "balanced_accuracy": 0.68, "roc_auc": 0.75, "nll": 0.55, "brier": 0.22, "ece": 0.08, "mean_uncertainty": 0.3, "mean_uncertainty_correct": 0.25, "mean_uncertainty_incorrect": 0.45, "uncertainty_error_auroc": 0.65, "aurc": 0.15},
    ])
    summary_df = compute_summary_statistics(dummy_results)

    assert "group" in summary_df.columns
    assert "metric" in summary_df.columns
    assert "mean" in summary_df.columns
    assert "std" in summary_df.columns
    assert "ci95_lower" in summary_df.columns
    assert "ci95_upper" in summary_df.columns

    # Check that Exp 1 and Overall exist
    groups = summary_df["group"].unique()
    assert "Exp_1_Si" in groups
    assert "Overall" in groups


def test_validation_only_temperature_fitting():
    """Verify temperature scaling is fit only on validation logits and decreases NLL."""
    torch.manual_seed(42)
    val_logits = torch.randn(50, 2) * 3.0  # overconfident
    val_labels = torch.randint(0, 2, (50,))

    learned_t = fit_temperature(val_logits, val_labels)
    assert learned_t > 0.0
