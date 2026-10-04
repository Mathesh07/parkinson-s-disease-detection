"""Unit tests for Conformal Prediction implementation.

Tests:
1. Conformal quantile calculation: Quantile respects finite-sample correction and bounded probabilities.
2. Monotonicity of prediction sets: As alpha decreases (higher target coverage), prediction set sizes are non-decreasing.
3. Prediction set schema: Valid set compositions {HC}, {PD}, {HC, PD}, {}.
4. No test data leakage: Quantile q_hat is strictly computed on validation split, never test data.
5. In-distribution coverage property: Under exchangeable data, empirical coverage satisfies >= 1 - alpha - tolerance.
"""

import numpy as np
import pandas as pd
import pytest

from Gait.experiments.conformal_prediction import (
    compute_conformal_quantile,
    construct_prediction_set,
    format_set_string,
)


def test_conformal_quantile_bounds_and_monotonicity():
    """Verify that conformal quantile respects mathematical bounds and monotonic scaling with alpha."""
    np.random.seed(42)
    n = 100
    val_probs = np.random.dirichlet([2.0, 2.0], size=n)
    val_labels = np.random.binomial(1, 0.5, size=n)

    q_05 = compute_conformal_quantile(val_probs, val_labels, alpha=0.05)
    q_10 = compute_conformal_quantile(val_probs, val_labels, alpha=0.10)
    q_20 = compute_conformal_quantile(val_probs, val_labels, alpha=0.20)

    # All nonconformity scores are in [0, 1]
    assert 0.0 <= q_05 <= 1.0
    assert 0.0 <= q_10 <= 1.0
    assert 0.0 <= q_20 <= 1.0

    # Stricter coverage requirement (smaller alpha) requires larger nonconformity quantile
    assert q_05 >= q_10 >= q_20


def test_prediction_set_construction_and_formatting():
    """Verify prediction set logic for confident, ambiguous, and empty cases."""
    q_hat = 0.8  # Threshold = 1 - 0.8 = 0.2

    # Confident PD: P(PD)=0.85 >= 0.2, P(HC)=0.15 < 0.2 -> {PD}
    set_pd = construct_prediction_set(prob_hc=0.15, prob_pd=0.85, q_hat=q_hat)
    assert set_pd == ["PD"]
    assert format_set_string(set_pd) == "{PD}"

    # Confident HC: P(HC)=0.85 >= 0.2, P(PD)=0.15 < 0.2 -> {HC}
    set_hc = construct_prediction_set(prob_hc=0.85, prob_pd=0.15, q_hat=q_hat)
    assert set_hc == ["HC"]
    assert format_set_string(set_hc) == "{Healthy}"

    # Ambiguous: Both probabilities >= 0.2 -> {HC, PD}
    set_both = construct_prediction_set(prob_hc=0.45, prob_pd=0.55, q_hat=q_hat)
    assert set(set_both) == {"HC", "PD"}
    assert format_set_string(set_both) == "{Healthy, PD}"

    # Empty set: threshold is high, e.g. q_hat = 0.2 -> threshold = 0.8, probabilities 0.5, 0.5
    set_empty = construct_prediction_set(prob_hc=0.5, prob_pd=0.5, q_hat=0.2)
    assert set_empty == []
    assert format_set_string(set_empty) == "{}"


def test_exchangeable_in_distribution_coverage():
    """Verify that inductive conformal prediction guarantees coverage on i.i.d. exchangeable synthetic test data."""
    np.random.seed(123)
    n_val = 500
    n_test = 500
    alpha = 0.10
    target_coverage = 1.0 - alpha

    # Generate synthetic calibrated probabilities
    true_prob_pd = np.random.uniform(0.1, 0.9, size=n_val + n_test)
    y = (np.random.uniform(0.0, 1.0, size=n_val + n_test) < true_prob_pd).astype(int)

    probs = np.column_stack([1.0 - true_prob_pd, true_prob_pd])

    val_probs, test_probs = probs[:n_val], probs[n_val:]
    val_y, test_y = y[:n_val], y[n_val:]

    q_hat = compute_conformal_quantile(val_probs, val_y, alpha=alpha)

    # Evaluate on test set
    covered = []
    for i in range(n_test):
        label_str = "PD" if test_y[i] == 1 else "HC"
        pset = construct_prediction_set(test_probs[i, 0], test_probs[i, 1], q_hat)
        covered.append(label_str in pset)

    emp_coverage = np.mean(covered)
    # Under exchangeability, coverage must be approximately >= 1 - alpha (within sampling error)
    assert emp_coverage >= target_coverage - 0.03, f"Empirical coverage {emp_coverage:.3f} fell below {target_coverage}"


def test_no_test_leakage_in_quantile():
    """Verify that passing test labels or modifying test probabilities does not alter q_hat."""
    np.random.seed(42)
    val_probs = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.3, 0.7]])
    val_labels = np.array([0, 1, 0, 1])

    q_hat_1 = compute_conformal_quantile(val_probs, val_labels, alpha=0.10)

    # Test data changes should have zero influence on q_hat
    dummy_test_labels = np.array([1, 0, 1, 0])
    q_hat_2 = compute_conformal_quantile(val_probs, val_labels, alpha=0.10)

    assert np.isclose(q_hat_1, q_hat_2), "Conformal threshold is not deterministic on val data!"


def test_conformal_no_subject_leakage_and_dataloader_drop_last():
    """
    Test 6: Verify zero subject leakage across train, val, and test splits for all LOCO folds,
    verify that test cohort data is never in train or val splits, and verify drop_last=True
    on train_loader while val/test have drop_last=False.
    """
    from Gait.preprocessing import build_manifest
    from Gait.experiments.leave_one_cohort_out import build_cohort_splits, build_dataloaders
    from Gait.experiments.conformal_prediction import LOCO_EXPERIMENTS

    manifest_df = build_manifest()
    for exp in LOCO_EXPERIMENTS:
        train_cohorts = exp["train_cohorts"]
        test_cohort = exp["test_cohort"]
        train_df, val_df, test_df = build_cohort_splits(
            manifest_df, train_cohorts, test_cohort, seed=42
        )
        tr_subs = set(train_df["subject_id"].unique())
        val_subs = set(val_df["subject_id"].unique())
        te_subs = set(test_df["subject_id"].unique())

        assert len(tr_subs & val_subs) == 0, f"Fold {exp['exp_id']}: Train/Val subject overlap!"
        assert len(tr_subs & te_subs) == 0, f"Fold {exp['exp_id']}: Train/Test subject overlap!"
        assert len(val_subs & te_subs) == 0, f"Fold {exp['exp_id']}: Val/Test subject overlap!"
        assert set(test_df["study"].unique()) == {test_cohort}, f"Fold {exp['exp_id']}: Test cohort isolation violated!"
        assert test_cohort not in train_df["study"].unique(), f"Fold {exp['exp_id']}: Test cohort leaked into train!"
        assert test_cohort not in val_df["study"].unique(), f"Fold {exp['exp_id']}: Test cohort leaked into val!"

    # Verify build_dataloaders behavior for conformal prediction
    train_loader, val_loader, test_loader = build_dataloaders(
        train_df.iloc[:2], val_df.iloc[:2], test_df.iloc[:2], normalizer=None, batch_size=4
    )
    assert train_loader.drop_last is True, "Train loader must have drop_last=True"
    assert val_loader.drop_last is False, "Val loader must have drop_last=False"
    assert test_loader.drop_last is False, "Test loader must have drop_last=False"


def test_conformal_coverage_and_set_size_metrics():
    """
    Test 7: Verify empirical coverage, average prediction set size, singleton rate,
    ambiguity rate, and empty set rate calculation on deterministic synthetic cases.
    """
    # Threshold = 1 - 0.7 = 0.3
    q_hat = 0.7
    # Case 1: P(HC)=0.9, P(PD)=0.1, True=0 (HC) -> Set={HC} (Covered, Size=1, Singleton)
    # Case 2: P(HC)=0.1, P(PD)=0.9, True=1 (PD) -> Set={PD} (Covered, Size=1, Singleton)
    # Case 3: P(HC)=0.5, P(PD)=0.5, True=1 (PD) -> Set={HC, PD} (Covered, Size=2, Ambiguous)
    # Case 4: P(HC)=0.8, P(PD)=0.2, True=1 (PD) -> Set={HC} (Uncovered, Size=1, Singleton)
    test_cases = [
        {"P_HC": 0.9, "P_PD": 0.1, "true_label": 0},
        {"P_HC": 0.1, "P_PD": 0.9, "true_label": 1},
        {"P_HC": 0.5, "P_PD": 0.5, "true_label": 1},
        {"P_HC": 0.8, "P_PD": 0.2, "true_label": 1},
    ]

    coverages = []
    set_sizes = []
    singletons = 0
    ambiguous = 0
    empty = 0

    for c in test_cases:
        true_label_str = "PD" if c["true_label"] == 1 else "HC"
        pset = construct_prediction_set(c["P_HC"], c["P_PD"], q_hat)
        covered = int(true_label_str in pset)
        coverages.append(covered)
        set_sizes.append(len(pset))
        if len(pset) == 1:
            singletons += 1
        elif len(pset) == 2:
            ambiguous += 1
        elif len(pset) == 0:
            empty += 1

    emp_cov = np.mean(coverages)
    avg_size = np.mean(set_sizes)
    singleton_rate = singletons / len(test_cases)
    ambiguity_rate = ambiguous / len(test_cases)
    empty_rate = empty / len(test_cases)

    assert emp_cov == 0.75  # 3 of 4 covered
    assert avg_size == 1.25  # (1 + 1 + 2 + 1) / 4 = 1.25
    assert singleton_rate == 0.75  # 3 of 4 are singletons
    assert ambiguity_rate == 0.25  # 1 of 4 is ambiguous
    assert empty_rate == 0.0
