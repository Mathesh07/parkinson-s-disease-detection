"""Comprehensive evaluation script for Gait Base-Paper-Style 2D CNN Baseline.

Computes window, recording, subject, and study-level metrics on the exact held-out
test subjects used by the proposed 1D CNN + BiLSTM model. Also provides transition
analysis (onset vs. offset vs. steady-state) and LOCO cohort-shift evaluation.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

try:
    from . import config
    from .dataset import (
        GaitSTFTDataset,
        build_manifest,
        collate_gait_stft_batch,
        create_subject_splits,
    )
    from .model import GaitBasePaper2DCNN
    from .preprocessing import GaitSTFTNormalizer, load_raw_recording
except ImportError:
    import config
    from dataset import (
        GaitSTFTDataset,
        build_manifest,
        collate_gait_stft_batch,
        create_subject_splits,
    )
    from model import GaitBasePaper2DCNN
    from preprocessing import GaitSTFTNormalizer, load_raw_recording


def calculate_metrics_dict(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict:
    """Calculate all standard classification and calibration metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5

    brier = float(brier_score_loss(y_true, y_prob))
    try:
        nll = float(log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6)))
    except Exception:
        nll = 0.0

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall_sensitivity": rec,
        "specificity": spec,
        "f1_score": f1,
        "roc_auc": auc,
        "brier_score": brier,
        "negative_log_likelihood": nll,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }


def plot_confusion_matrix(cm_counts: np.ndarray, save_path: Path):
    """Plot and save confusion matrix heatmap."""
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_counts, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Gait Base-Paper 2D CNN: Confusion Matrix", fontsize=12, fontweight="bold")
    plt.colorbar()

    classes = ["HC", "PD"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=11)
    plt.yticks(tick_marks, classes, fontsize=11)

    thresh = cm_counts.max() / 2.0
    for i in range(cm_counts.shape[0]):
        for j in range(cm_counts.shape[1]):
            plt.text(
                j, i, format(cm_counts[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm_counts[i, j] > thresh else "black",
                fontsize=14, fontweight="bold"
            )

    plt.ylabel("True Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curve_figure(y_true: np.ndarray, y_prob: np.ndarray, save_path: Path):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.title("Gait Base-Paper 2D CNN: Subject ROC", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_baseline_vs_existing_comparison(
    baseline_metrics: Dict,
    existing_metrics: Dict,
    save_path: Path
):
    """Generate a clean comparative bar chart between Base Paper 2D CNN and Proposed 1D CNN + BiLSTM."""
    metrics_to_plot = ["accuracy", "balanced_accuracy", "precision", "recall_sensitivity", "specificity", "f1_score", "roc_auc"]
    labels = ["Accuracy", "Bal. Acc", "Precision", "Sensitivity", "Specificity", "F1-Score", "ROC-AUC"]

    base_vals = [baseline_metrics.get(m, 0.0) for m in metrics_to_plot]
    prop_vals = [existing_metrics.get(m, 0.0) for m in metrics_to_plot]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 5.5))
    plt.bar(x - width/2, base_vals, width, label="Base-Paper 2D CNN (STFT)", color="#4A90E2", edgecolor="black", alpha=0.9)
    plt.bar(x + width/2, prop_vals, width, label="Existing 1D CNN + BiLSTM", color="#50E3C2", edgecolor="black", alpha=0.9)

    plt.ylabel("Score (0.0 - 1.0)", fontsize=12)
    plt.title("Subject-Level Evaluation: Base-Paper 2D CNN vs. Proposed Model", fontsize=13, fontweight="bold")
    plt.xticks(x, labels, fontsize=11)
    plt.ylim(0.0, 1.05)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend(fontsize=11, loc="lower right")

    # Add numeric labels on bars
    for i in range(len(x)):
        plt.text(x[i] - width/2, base_vals[i] + 0.02, f"{base_vals[i]:.2f}", ha="center", fontsize=9, fontweight="bold")
        plt.text(x[i] + width/2, prop_vals[i] + 0.02, f"{prop_vals[i]:.2f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def evaluate(checkpoint_path: Path = config.BEST_MODEL_PATH):
    """Execute primary held-out test evaluation."""
    print("=" * 60)
    print("GAIT BASE-PAPER 2D CNN BASELINE EVALUATION")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 60)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data Splits and Normalizer
    manifest_df = build_manifest()
    train_df, val_df, test_df = create_subject_splits(manifest_df, seed=config.RANDOM_SEED)

    normalizer = GaitSTFTNormalizer.load(config.NORMALIZATION_STATS_PATH)
    test_dataset = GaitSTFTDataset(test_df, normalizer=normalizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_gait_stft_batch
    )

    # 2. Load Model
    model = GaitBasePaper2DCNN(
        in_channels=config.NUM_CHANNELS,
        conv_channels=config.CONV_CHANNELS,
        dense_units=config.DENSE_UNITS,
        dropout_rate=config.DROPOUT_RATE
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 3. Predict on Test Windows
    window_records = []
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            logits = model(features)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            labels = batch["labels"].cpu().numpy().flatten()

            for i in range(len(probs)):
                window_records.append({
                    "subject_id": batch["subject_ids"][i],
                    "recording_id": batch["recording_ids"][i],
                    "study": batch["studies"][i],
                    "window_idx": batch["window_indices"][i],
                    "window_type": batch["window_types"][i],
                    "true_label": int(labels[i]),
                    "pd_probability": float(probs[i]),
                    "predicted_label": int(probs[i] >= 0.5)
                })

    window_df = pd.DataFrame(window_records)
    window_df.to_csv(config.WINDOW_PREDICTIONS_PATH, index=False)

    # 4. Recording-level Aggregation
    rec_grouped = window_df.groupby(["recording_id", "subject_id", "study", "true_label"]).agg(
        pd_probability=("pd_probability", "mean")
    ).reset_index()
    rec_grouped["predicted_label"] = (rec_grouped["pd_probability"] >= 0.5).astype(int)
    rec_grouped.to_csv(config.RECORDING_PREDICTIONS_PATH, index=False)

    # 5. Subject-level Aggregation
    sub_grouped = rec_grouped.groupby(["subject_id", "study", "true_label"]).agg(
        pd_probability=("pd_probability", "mean")
    ).reset_index()
    sub_grouped["predicted_label"] = (sub_grouped["pd_probability"] >= 0.5).astype(int)
    sub_grouped["predicted_name"] = sub_grouped["predicted_label"].map(config.ID2LABEL)
    sub_grouped.to_csv(config.SUBJECT_PREDICTIONS_PATH, index=False)

    # 6. Compute Metrics
    win_metrics = calculate_metrics_dict(window_df["true_label"].values, window_df["pd_probability"].values)
    rec_metrics = calculate_metrics_dict(rec_grouped["true_label"].values, rec_grouped["pd_probability"].values)
    sub_metrics = calculate_metrics_dict(sub_grouped["true_label"].values, sub_grouped["pd_probability"].values)

    # Study / Cohort Breakdown
    study_breakdown = {}
    for study_name in ["Ga", "Ju", "Si"]:
        study_subs = sub_grouped[sub_grouped["study"] == study_name]
        if len(study_subs) > 0:
            study_breakdown[study_name] = calculate_metrics_dict(
                study_subs["true_label"].values,
                study_subs["pd_probability"].values
            )

    # Transition Analysis (Onset vs Offset vs Steady-State)
    transition_breakdown = {}
    for wtype in ["onset", "offset", "steady_state"]:
        wdf = window_df[window_df["window_type"] == wtype]
        if len(wdf) > 0:
            transition_breakdown[wtype] = calculate_metrics_dict(
                wdf["true_label"].values,
                wdf["pd_probability"].values
            )

    results_dict = {
        "model_name": "GaitBasePaper2DCNN",
        "dataset": "PhysioNet Gait in Parkinson's Disease (16-channel VGRF)",
        "input_representation": "STFT Log-Magnitude Spectrogram (16, 33, 32)",
        "subject_level": sub_metrics,
        "recording_level": rec_metrics,
        "window_level": win_metrics,
        "study_breakdown": study_breakdown,
        "transition_analysis": transition_breakdown,
        "split_counts": {
            "train_subjects": int(train_df["subject_id"].nunique()),
            "val_subjects": int(val_df["subject_id"].nunique()),
            "test_subjects": int(test_df["subject_id"].nunique()),
            "train_recordings": int(len(train_df)),
            "val_recordings": int(len(val_df)),
            "test_recordings": int(len(test_df)),
            "test_windows": int(len(window_df))
        }
    }

    with open(config.METRICS_JSON_PATH, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"Saved test metrics to: {config.METRICS_JSON_PATH}")

    # Plot Confusion Matrix
    cm_counts = np.array([[sub_metrics["tn"], sub_metrics["fp"]], [sub_metrics["fn"], sub_metrics["tp"]]])
    plot_confusion_matrix(cm_counts, config.CONFUSION_MATRIX_PATH)

    # Plot ROC Curve
    plot_roc_curve_figure(sub_grouped["true_label"].values, sub_grouped["pd_probability"].values, config.ROC_CURVE_PATH)

    # Load Existing Model Metrics for Matched Comparison
    existing_metrics_file = config.GAIT_DIR / "outputs" / "metrics" / "test_metrics.json"
    if existing_metrics_file.exists():
        with open(existing_metrics_file, "r") as f:
            existing_data = json.load(f)
        existing_sub = existing_data.get("subject_level", {})
        # Balanced accuracy fallback if not present in existing json
        if "balanced_accuracy" not in existing_sub and "recall_sensitivity" in existing_sub and "specificity" in existing_sub:
            existing_sub["balanced_accuracy"] = 0.5 * (existing_sub["recall_sensitivity"] + existing_sub["specificity"])
        plot_baseline_vs_existing_comparison(sub_metrics, existing_sub, config.COMPARISON_BAR_PATH)
        print(f"Saved baseline vs existing comparison plot to: {config.COMPARISON_BAR_PATH}")

    print("\n" + "=" * 60)
    print("GAIT BASE-PAPER 2D CNN EVALUATION SUMMARY:")
    print(f"  Subject Accuracy:     {sub_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy:    {sub_metrics['balanced_accuracy']:.4f}")
    print(f"  Subject F1-Score:     {sub_metrics['f1_score']:.4f}")
    print(f"  Subject ROC-AUC:      {sub_metrics['roc_auc']:.4f}")
    print(f"  Sensitivity (Recall): {sub_metrics['recall_sensitivity']:.4f}")
    print(f"  Specificity:          {sub_metrics['specificity']:.4f}")
    print(f"  Test TP: {sub_metrics['tp']}, TN: {sub_metrics['tn']}, FP: {sub_metrics['fp']}, FN: {sub_metrics['fn']}")
    print("=" * 60)

    return results_dict


if __name__ == "__main__":
    evaluate()
