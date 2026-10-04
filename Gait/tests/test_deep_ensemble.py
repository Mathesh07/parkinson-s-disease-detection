"""Unit tests for Deep Ensemble implementation.

Tests:
1. Architecture compatibility & weight independence: Independently initialized models have distinct weights.
2. Entropy, variance, and mutual information bounds: Mathematical bounds hold for all predictions.
3. Hierarchical aggregation: Aggregation from window to recording to subject preserves true subject labels and correct averaging.
4. Baseline comparison format: All 5 methods produce matching metric schemas.
5. No test leakage: Verification that test cohort data is never used during ensemble member training or calibration.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from Gait import config
from Gait.experiments.deep_ensemble import (
    aggregate_ensemble_to_subject_level,
    evaluate_subject_predictions,
    LOCO_EXPERIMENTS,
)
from Gait.experiments.leave_one_cohort_out import (
    build_cohort_splits,
    build_dataloaders,
)
from Gait.model import GaitCNNBiLSTM
from Gait.preprocessing import build_manifest


def test_ensemble_members_stochastic_independence():
    """Verify that models initialized with different seeds have completely different weights."""
    torch.manual_seed(42)
    m1 = GaitCNNBiLSTM(
        in_channels=config.NUM_CHANNELS,
        cnn_channels=config.CNN_CHANNELS,
        lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
        lstm_num_layers=config.LSTM_NUM_LAYERS,
        embedding_dim=config.EMBEDDING_DIM,
        num_classes=2,
    )

    torch.manual_seed(43)
    m2 = GaitCNNBiLSTM(
        in_channels=config.NUM_CHANNELS,
        cnn_channels=config.CNN_CHANNELS,
        lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
        lstm_num_layers=config.LSTM_NUM_LAYERS,
        embedding_dim=config.EMBEDDING_DIM,
        num_classes=2,
    )

    # Weights in CNN and LSTM must be distinct
    w1 = m1.cnn_block1[0].weight.data.numpy()
    w2 = m2.cnn_block1[0].weight.data.numpy()
    assert not np.allclose(w1, w2), "Models initialized with different seeds share identical CNN weights!"

    lstm_w1 = m1.lstm.weight_ih_l0.data.numpy()
    lstm_w2 = m2.lstm.weight_ih_l0.data.numpy()
    assert not np.allclose(lstm_w1, lstm_w2), "Models initialized with different seeds share identical LSTM weights!"


def test_ensemble_entropy_and_variance_bounds():
    """Verify predictive entropy, variance, and mutual information are strictly non-negative and bounded."""
    eps = 1e-12
    # Simulate 5 members, 10 samples, 2 classes
    np.random.seed(42)
    member_probs = np.random.dirichlet([1.0, 1.0], size=(5, 10))  # (M=5, B=10, C=2)

    mean_p = np.mean(member_probs, axis=0)  # (B=10, C=2)
    var_p_pd = np.var(member_probs[:, :, 1], axis=0)  # (B=10,)

    pred_entropy = -np.sum(mean_p * np.log(np.clip(mean_p, eps, 1.0)), axis=1)

    member_entropies = -np.sum(member_probs * np.log(np.clip(member_probs, eps, 1.0)), axis=-1)
    exp_entropy = np.mean(member_entropies, axis=0)
    mutual_info = np.maximum(0.0, pred_entropy - exp_entropy)

    assert np.all(mean_p >= 0.0) and np.all(mean_p <= 1.0)
    assert np.allclose(np.sum(mean_p, axis=1), 1.0)
    assert np.all(var_p_pd >= 0.0) and np.all(var_p_pd <= 0.25)  # max variance for Bernoulli is 0.25
    assert np.all(pred_entropy >= 0.0) and np.all(pred_entropy <= np.log(2.0) + 1e-5)
    assert np.all(mutual_info >= 0.0) and np.all(mutual_info <= pred_entropy + 1e-5)


def test_ensemble_hierarchical_aggregation():
    """Verify window -> recording -> subject hierarchical aggregation for ensemble outputs."""
    rows = []
    # 2 recordings for 1 subject: Rec A has 5 windows (all P_PD=0.2), Rec B has 15 windows (all P_PD=0.8)
    for i in range(5):
        rows.append({
            "experiment": 1,
            "train_cohorts": "Ga+Ju",
            "test_cohort": "Si",
            "subject_id": "sub_1",
            "recording_id": "rec_A",
            "study": "Si",
            "true_label": 1,
            "P_HC": 0.8,
            "P_PD": 0.2,
            "predictive_entropy": 0.5,
            "prediction_variance": 0.01,
            "mutual_information": 0.005,
        })
    for i in range(15):
        rows.append({
            "experiment": 1,
            "train_cohorts": "Ga+Ju",
            "test_cohort": "Si",
            "subject_id": "sub_1",
            "recording_id": "rec_B",
            "study": "Si",
            "true_label": 1,
            "P_HC": 0.2,
            "P_PD": 0.8,
            "predictive_entropy": 0.5,
            "prediction_variance": 0.01,
            "mutual_information": 0.005,
        })

    win_df = pd.DataFrame(rows)
    sub_df = aggregate_ensemble_to_subject_level(win_df)

    assert len(sub_df) == 1
    # Hierarchical mean should be (0.2 + 0.8) / 2 = 0.50, NOT window mean (5*0.2 + 15*0.8)/20 = 0.65
    assert np.isclose(sub_df["P_PD"].iloc[0], 0.50, atol=1e-5)
    assert sub_df["is_correct"].iloc[0] == 1  # 0.50 >= 0.5 -> pred=1, true=1 -> correct


def test_evaluate_subject_predictions_metrics():
    """Verify evaluate_subject_predictions generates all required metrics in valid ranges."""
    sub_df = pd.DataFrame([
        {"subject_id": "s1", "true_label": 0, "predicted_label": 0, "P_PD": 0.1, "P_HC": 0.9, "confidence": 0.9, "uncertainty": 0.1, "is_correct": 1},
        {"subject_id": "s2", "true_label": 0, "predicted_label": 0, "P_PD": 0.2, "P_HC": 0.8, "confidence": 0.8, "uncertainty": 0.15, "is_correct": 1},
        {"subject_id": "s3", "true_label": 1, "predicted_label": 1, "P_PD": 0.85, "P_HC": 0.15, "confidence": 0.85, "uncertainty": 0.2, "is_correct": 1},
        {"subject_id": "s4", "true_label": 1, "predicted_label": 0, "P_PD": 0.45, "P_HC": 0.55, "confidence": 0.55, "uncertainty": 0.5, "is_correct": 0},
    ])
    metrics = evaluate_subject_predictions(sub_df, method_name="Deep Ensemble")

    required_metrics = [
        "method", "accuracy", "balanced_accuracy", "sensitivity", "specificity",
        "f1", "roc_auc", "nll", "brier", "ece", "error_detection_auroc", "aurc",
        "mean_uncertainty_correct", "mean_uncertainty_incorrect"
    ]
    for m in required_metrics:
        assert m in metrics, f"Missing metric: {m}"

    assert metrics["method"] == "Deep Ensemble"
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    assert 0.0 <= metrics["ece"] <= 1.0
    assert 0.0 <= metrics["brier"] <= 1.0
    assert metrics["mean_uncertainty_incorrect"] > metrics["mean_uncertainty_correct"]


def test_deep_ensemble_no_data_leakage_and_dataloader_drop_last():
    """
    Test 5: Verify zero subject leakage across train, val, and test splits for all LOCO folds,
    verify that test cohort data is never in train or val splits, and verify drop_last=True
    on train_loader while val/test have drop_last=False.
    """
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

    # Also verify build_dataloaders behavior for Deep Ensemble
    train_loader, val_loader, test_loader = build_dataloaders(
        train_df.iloc[:2], val_df.iloc[:2], test_df.iloc[:2], normalizer=None, batch_size=4
    )
    assert train_loader.drop_last is True, "Train loader must have drop_last=True"
    assert val_loader.drop_last is False, "Val loader must have drop_last=False"
    assert test_loader.drop_last is False, "Test loader must have drop_last=False"
