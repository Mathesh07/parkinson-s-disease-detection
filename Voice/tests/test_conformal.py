"""Unit test suite for Voice Conformal Prediction (Phase 7).

Verifies nonconformity score computation, quantile threshold calculation,
prediction-set generation, singleton/ambiguous/empty set handling, calibration isolation,
and subject-level metrics.
"""

import math
import numpy as np
import pandas as pd
import pytest

from Voice.conformal_prediction import (
    calculate_conformal_quantile,
    compute_nonconformity_score,
    compute_nonconformity_scores,
    construct_prediction_set,
    evaluate_conformal_predictions,
    format_prediction_set_str,
)


def test_nonconformity_score_single():
    """Test nonconformity score for a single probability value."""
    # s = 1 - p_true
    assert math.isclose(compute_nonconformity_score(0.8), 0.2, abs_tol=1e-6)
    assert math.isclose(compute_nonconformity_score(1.0), 0.0, abs_tol=1e-6)
    assert math.isclose(compute_nonconformity_score(0.0), 1.0, abs_tol=1e-6)
    assert math.isclose(compute_nonconformity_score(0.5), 0.5, abs_tol=1e-6)


def test_nonconformity_scores_array_1d():
    """Test nonconformity scores for 1D array of P(PD) and binary labels."""
    probs = np.array([0.9, 0.2, 0.7, 0.4])  # P(PD)
    labels = np.array([1, 0, 0, 1])        # 1=PD, 0=HC

    # True probs: [0.9, 0.8, 0.3, 0.4] -> s = 1 - p_true: [0.1, 0.2, 0.7, 0.6]
    scores = compute_nonconformity_scores(probs, labels)
    expected = np.array([0.1, 0.2, 0.7, 0.6])
    np.testing.assert_allclose(scores, expected, atol=1e-6)


def test_nonconformity_scores_array_2d():
    """Test nonconformity scores for 2D probability matrix (N, 2)."""
    # col 0 = P(HC), col 1 = P(PD)
    probs = np.array([
        [0.1, 0.9],
        [0.8, 0.2],
        [0.3, 0.7],
        [0.6, 0.4]
    ])
    labels = np.array([1, 0, 0, 1])
    scores = compute_nonconformity_scores(probs, labels)
    expected = np.array([0.1, 0.2, 0.7, 0.6])
    np.testing.assert_allclose(scores, expected, atol=1e-6)


def test_calculate_conformal_quantile_n5():
    """Test finite-sample conformal quantile for n=5, alpha=0.10."""
    scores = np.array([0.1, 0.25, 0.3, 0.4, 0.5])
    # k = ceil(6 * 0.90) = 6 -> q_level = min(1.0, 6/5) = 1.0 -> max(scores) = 0.5
    q_hat = calculate_conformal_quantile(scores, alpha=0.10)
    assert math.isclose(q_hat, 0.5, abs_tol=1e-5)


def test_calculate_conformal_quantile_empty_error():
    """Test error handling when empty array is passed to quantile calculation."""
    with pytest.raises(ValueError, match="Cannot calculate quantile"):
        calculate_conformal_quantile(np.array([]))


def test_prediction_set_singleton_pd():
    """Test prediction set construction resulting in singleton PD set."""
    p_hc = 0.2
    p_pd = 0.8
    q_hat = 0.4  # Threshold = 1 - 0.4 = 0.6. P(PD) >= 0.6, P(HC) < 0.6
    pred_set = construct_prediction_set(p_hc, p_pd, q_hat)
    assert pred_set == {"Parkinson's Disease"}


def test_prediction_set_singleton_hc():
    """Test prediction set construction resulting in singleton HC set."""
    p_hc = 0.85
    p_pd = 0.15
    q_hat = 0.4  # Threshold = 1 - 0.4 = 0.6. P(HC) >= 0.6
    pred_set = construct_prediction_set(p_hc, p_pd, q_hat)
    assert pred_set == {"Healthy Control"}


def test_prediction_set_ambiguous():
    """Test prediction set construction resulting in ambiguous (both) set."""
    p_hc = 0.55
    p_pd = 0.45
    q_hat = 0.6  # Threshold = 1 - 0.6 = 0.4. Both P(HC) and P(PD) >= 0.4
    pred_set = construct_prediction_set(p_hc, p_pd, q_hat)
    assert pred_set == {"Healthy Control", "Parkinson's Disease"}


def test_prediction_set_empty():
    """Test prediction set construction resulting in empty set."""
    p_hc = 0.51
    p_pd = 0.49
    q_hat = 0.4  # Threshold = 1 - 0.4 = 0.6. Neither >= 0.6
    pred_set = construct_prediction_set(p_hc, p_pd, q_hat)
    assert pred_set == set()


def test_format_prediction_set_str():
    """Test formatting prediction set strings."""
    assert format_prediction_set_str(set()) == "Empty {}"
    assert format_prediction_set_str({"Healthy Control"}) == "{Healthy Control}"
    assert format_prediction_set_str({"Parkinson's Disease"}) == "{Parkinson's Disease}"
    assert format_prediction_set_str({"Healthy Control", "Parkinson's Disease"}) == "{Healthy Control, Parkinson's Disease}"


def test_evaluate_conformal_predictions():
    """Test calculation of empirical coverage and set size metrics."""
    df = pd.DataFrame([
        {"subject_id": "ID01", "true_label": 1, "p_hc": 0.2, "p_pd": 0.8, "set_size": 1, "covered_true_label": 1},
        {"subject_id": "ID02", "true_label": 0, "p_hc": 0.55, "p_pd": 0.45, "set_size": 2, "covered_true_label": 1},
        {"subject_id": "ID03", "true_label": 1, "p_hc": 0.51, "p_pd": 0.49, "set_size": 0, "covered_true_label": 0},
        {"subject_id": "ID04", "true_label": 0, "p_hc": 0.7, "p_pd": 0.3, "set_size": 1, "covered_true_label": 1},
    ])
    metrics = evaluate_conformal_predictions(df, q_hat=0.4, alpha=0.10)
    
    assert math.isclose(metrics["empirical_coverage"], 0.75, abs_tol=1e-5)  # 3/4
    assert math.isclose(metrics["average_set_size"], 1.0, abs_tol=1e-5)      # (1+2+0+1)/4 = 1.0
    assert math.isclose(metrics["singleton_rate"], 0.50, abs_tol=1e-5)       # 2/4
    assert math.isclose(metrics["ambiguous_rate"], 0.25, abs_tol=1e-5)       # 1/4
    assert math.isclose(metrics["empty_rate"], 0.25, abs_tol=1e-5)           # 1/4


def test_calibration_test_subject_isolation():
    """Verify strictly disjoint set membership between validation and test subjects."""
    from Voice.dataset import build_metadata, create_subject_splits
    meta_df = build_metadata()
    _, val_df, test_df = create_subject_splits(meta_df, seed=42)

    val_subs = set(val_df["subject_id"].unique())
    test_subs = set(test_df["subject_id"].unique())

    assert len(val_subs & test_subs) == 0, "Leakage detected: Validation and test subject IDs overlap!"
    assert val_subs == {"ID02", "ID11", "ID15", "ID21", "ID33"}
    assert test_subs == {"ID04", "ID10", "ID13", "ID23", "ID29", "ID35"}
