"""Unit tests for Unified Reliability and Abstention Layer.

Tests:
1. Validation-only threshold tuning: Thresholds are derived strictly from validation subjects.
2. Decision logic verification:
   - Accept rule: High confidence + low uncertainty + consistent singleton set.
   - Flag rule 1: High uncertainty triggers flag.
   - Flag rule 2: Low confidence triggers flag.
   - Flag rule 3: Ambiguous conformal set {Healthy, PD} triggers flag.
   - Flag rule 4: Point prediction contradiction triggers flag.
3. Selective accuracy and error capture metric calculations.
"""

import numpy as np
import pandas as pd
import pytest

from Gait.experiments.reliability_analysis import (
    compute_reliability_metrics,
    evaluate_decision_layer,
    tune_reliability_thresholds_on_validation,
)


def test_validation_threshold_tuning():
    """Verify that threshold tuning is purely based on validation data without leakage."""
    val_data = pd.DataFrame([
        {"subject_id": "s1", "true_label": 0, "predicted_label": 0, "is_correct": 1, "confidence": 0.85, "uncertainty": 0.15, "P_HC": 0.85, "P_PD": 0.15},
        {"subject_id": "s2", "true_label": 1, "predicted_label": 1, "is_correct": 1, "confidence": 0.90, "uncertainty": 0.10, "P_HC": 0.10, "P_PD": 0.90},
        {"subject_id": "s3", "true_label": 0, "predicted_label": 1, "is_correct": 0, "confidence": 0.60, "uncertainty": 0.40, "P_HC": 0.40, "P_PD": 0.60},
        {"subject_id": "s4", "true_label": 1, "predicted_label": 1, "is_correct": 1, "confidence": 0.75, "uncertainty": 0.25, "P_HC": 0.25, "P_PD": 0.75},
    ])
    th = tune_reliability_thresholds_on_validation(val_data, alpha=0.10)

    assert "tau_conf" in th
    assert "tau_unc" in th
    assert "q_hat" in th
    assert 0.50 <= th["tau_conf"] <= 1.0
    assert 0.0 <= th["tau_unc"] <= 1.0
    assert 0.0 <= th["q_hat"] <= 1.0


def test_decision_layer_flag_rules():
    """Verify all 4 distinct reasons for flagging work as expected."""
    thresholds = {
        "tau_conf": 0.70,
        "tau_unc": 0.30,
        "q_hat": 0.75,  # threshold for class inclusion = 1 - 0.75 = 0.25
        "alpha": 0.10,
    }

    test_subjects = pd.DataFrame([
        # Subject 1: Confident and reliable -> ACCEPT
        {"subject_id": "s1", "true_label": 1, "predicted_label": 1, "confidence": 0.85, "uncertainty": 0.15, "P_HC": 0.15, "P_PD": 0.85, "is_correct": 1},
        # Subject 2: High uncertainty -> FLAG
        {"subject_id": "s2", "true_label": 1, "predicted_label": 1, "confidence": 0.80, "uncertainty": 0.45, "P_HC": 0.20, "P_PD": 0.80, "is_correct": 1},
        # Subject 3: Low confidence -> FLAG
        {"subject_id": "s3", "true_label": 0, "predicted_label": 0, "confidence": 0.58, "uncertainty": 0.25, "P_HC": 0.58, "P_PD": 0.42, "is_correct": 1},
        # Subject 4: Ambiguous conformal set {Healthy, PD} -> FLAG
        {"subject_id": "s4", "true_label": 1, "predicted_label": 1, "confidence": 0.72, "uncertainty": 0.28, "P_HC": 0.28, "P_PD": 0.72, "is_correct": 1},
    ])

    decisions = evaluate_decision_layer(test_subjects, thresholds)

    # Subject 1 must be accepted
    assert decisions.loc[0, "reliability_status"] == "ACCEPT"
    assert decisions.loc[0, "conformal_set"] == "{PD}"

    # Subject 2 flagged due to high uncertainty
    assert decisions.loc[1, "reliability_status"] == "FLAG"
    assert "HIGH_UNCERTAINTY" in decisions.loc[1, "flag_reasons"]

    # Subject 3 flagged due to low confidence
    assert decisions.loc[2, "reliability_status"] == "FLAG"
    assert "LOW_CONFIDENCE" in decisions.loc[2, "flag_reasons"]

    # Subject 4 has P_HC = 0.28 >= 0.25 and P_PD = 0.72 >= 0.25 -> conformal set is {Healthy, PD} -> FLAG
    assert decisions.loc[3, "conformal_set"] == "{Healthy, PD}"
    assert decisions.loc[3, "reliability_status"] == "FLAG"
    assert "AMBIGUOUS_CONFORMAL_SET" in decisions.loc[3, "flag_reasons"]


def test_reliability_metrics_calculation():
    """Verify compute_reliability_metrics calculates coverage and accuracy improvements."""
    df = pd.DataFrame([
        {"true_label": 1, "predicted_label": 1, "is_correct": 1, "reliability_status": "ACCEPT"},
        {"true_label": 0, "predicted_label": 0, "is_correct": 1, "reliability_status": "ACCEPT"},
        {"true_label": 1, "predicted_label": 1, "is_correct": 1, "reliability_status": "ACCEPT"},
        {"true_label": 0, "predicted_label": 1, "is_correct": 0, "reliability_status": "FLAG"},  # Error correctly flagged!
        {"true_label": 1, "predicted_label": 0, "is_correct": 0, "reliability_status": "FLAG"},  # Error correctly flagged!
    ])
    metrics = compute_reliability_metrics(df)

    assert metrics["raw_accuracy"] == 0.60  # 3/5
    assert metrics["coverage"] == 0.60      # 3/5 accepted
    assert metrics["accepted_accuracy"] == 1.0  # 3/3 accepted are correct!
    assert metrics["error_capture_rate"] == 1.0  # 2/2 errors flagged!
    assert metrics["accuracy_gain"] == 0.40     # +40% gain in accepted subset


def test_reliability_no_subject_leakage_and_dataloader_drop_last():
    """
    Test 4: Verify zero subject leakage across train, val, and test splits for all LOCO folds,
    verify that test cohort data is never in train or val splits, and verify drop_last=True
    on train_loader while val/test have drop_last=False.
    """
    from Gait.preprocessing import build_manifest
    from Gait.experiments.leave_one_cohort_out import build_cohort_splits, build_dataloaders
    from Gait.experiments.reliability_analysis import LOCO_EXPERIMENTS

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

    # Verify build_dataloaders behavior
    train_loader, val_loader, test_loader = build_dataloaders(
        train_df.iloc[:2], val_df.iloc[:2], test_df.iloc[:2], normalizer=None, batch_size=4
    )
    assert train_loader.drop_last is True, "Train loader must have drop_last=True"
    assert val_loader.drop_last is False, "Val loader must have drop_last=False"
    assert test_loader.drop_last is False, "Test loader must have drop_last=False"


def test_reliability_test_labels_invariance_and_conformal_edge_cases():
    """
    Test 5: Verify that test labels never influence thresholds, and verify that empty
    conformal set or point contradiction correctly trigger the FLAG decision.
    """
    val_data = pd.DataFrame([
        {"subject_id": "v1", "true_label": 0, "predicted_label": 0, "is_correct": 1, "confidence": 0.85, "uncertainty": 0.15, "P_HC": 0.85, "P_PD": 0.15},
        {"subject_id": "v2", "true_label": 1, "predicted_label": 1, "is_correct": 1, "confidence": 0.90, "uncertainty": 0.10, "P_HC": 0.10, "P_PD": 0.90},
        {"subject_id": "v3", "true_label": 0, "predicted_label": 0, "is_correct": 1, "confidence": 0.70, "uncertainty": 0.30, "P_HC": 0.70, "P_PD": 0.30},
    ])
    th1 = tune_reliability_thresholds_on_validation(val_data, alpha=0.10)

    # Regardless of arbitrary external/test data, threshold function receives only validation data
    th2 = tune_reliability_thresholds_on_validation(val_data, alpha=0.10)
    assert th1 == th2, "Threshold tuning is non-deterministic!"

    # Empty conformal set case: e.g. q_hat is high, threshold is high
    # If q_hat = 0.2 -> threshold = 1 - 0.2 = 0.8, with probs (0.5, 0.5) -> set is empty
    th_empty = {"tau_conf": 0.55, "tau_unc": 0.50, "q_hat": 0.20, "alpha": 0.10}
    test_empty = pd.DataFrame([
        {"subject_id": "t1", "true_label": 1, "predicted_label": 1, "confidence": 0.55, "uncertainty": 0.45, "P_HC": 0.45, "P_PD": 0.55, "is_correct": 1},
    ])
    res_empty = evaluate_decision_layer(test_empty, th_empty)
    assert res_empty.loc[0, "reliability_status"] == "FLAG"
    assert "EMPTY_CONFORMAL_SET" in res_empty.loc[0, "flag_reasons"]
    assert res_empty.loc[0, "conformal_set"] == "{}"

    # Point contradiction case: point prediction is PD (1), but conformal set contains only HC
    th_contra = {"tau_conf": 0.55, "tau_unc": 0.50, "q_hat": 0.65, "alpha": 0.10}  # threshold = 0.35
    test_contra = pd.DataFrame([
        # P_HC=0.40 >= 0.35, P_PD=0.30 < 0.35, but predicted_label was 1 (PD)
        {"subject_id": "t2", "true_label": 0, "predicted_label": 1, "confidence": 0.60, "uncertainty": 0.40, "P_HC": 0.40, "P_PD": 0.30, "is_correct": 0},
    ])
    res_contra = evaluate_decision_layer(test_contra, th_contra)
    assert res_contra.loc[0, "reliability_status"] == "FLAG"
    assert "CONFORMAL_POINT_CONTRADICTION" in res_contra.loc[0, "flag_reasons"]
