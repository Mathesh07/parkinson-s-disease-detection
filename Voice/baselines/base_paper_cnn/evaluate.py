"""Comprehensive evaluation script for Voice Base-Paper-Style 2D CNN Baseline.

Evaluates continuous STFT 2D CNN, Onset, and Offset models on the exact held-out
test subjects used by the proposed Wav2Vec2 model.
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
        VoiceSTFTChunkDataset,
        build_voice_metadata,
        collate_voice_stft_batch,
        create_subject_splits,
    )
    from .model import VoiceBasePaper2DCNN
    from .preprocessing import (
        VoiceSTFTNormalizer,
        detect_speech_transitions,
        load_and_resample_audio,
    )
except ImportError:
    import config
    from dataset import (
        VoiceSTFTChunkDataset,
        build_voice_metadata,
        collate_voice_stft_batch,
        create_subject_splits,
    )
    from model import VoiceBasePaper2DCNN
    from preprocessing import (
        VoiceSTFTNormalizer,
        detect_speech_transitions,
        load_and_resample_audio,
    )


def calculate_metrics_dict(y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
    """Calculate classification and calibration metrics for binary outcomes."""
    y_pred = (y_prob >= 0.5).astype(int)
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
    plt.title("Voice Base-Paper 2D CNN: Confusion Matrix", fontsize=12, fontweight="bold")
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
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("Voice Base-Paper 2D CNN: Subject ROC", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_comparison_chart(base_sub: Dict, exist_sub: Dict, save_path: Path):
    """Plot side-by-side bar chart comparison against existing Wav2Vec2 model."""
    metrics = ["accuracy", "balanced_accuracy", "precision", "recall_sensitivity", "specificity", "f1_score", "roc_auc"]
    labels = ["Accuracy", "Bal. Acc", "Precision", "Sensitivity", "Specificity", "F1-Score", "ROC-AUC"]

    b_vals = [base_sub.get(m, 0.0) for m in metrics]
    e_vals = [exist_sub.get(m, 0.0) for m in metrics]

    x = np.arange(len(labels))
    w = 0.35

    plt.figure(figsize=(10, 5.5))
    plt.bar(x - w/2, b_vals, w, label="Base-Paper 2D CNN (STFT)", color="#FF6F61", edgecolor="black", alpha=0.9)
    plt.bar(x + w/2, e_vals, w, label="Proposed Wav2Vec2 Model", color="#6B5B95", edgecolor="black", alpha=0.9)

    plt.ylabel("Score (0.0 - 1.0)", fontsize=12)
    plt.title("Voice Subject-Level Evaluation: Base-Paper 2D CNN vs. Proposed Model", fontsize=13, fontweight="bold")
    plt.xticks(x, labels, fontsize=11)
    plt.ylim(0.0, 1.05)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend(fontsize=11, loc="lower right")

    for i in range(len(x)):
        plt.text(x[i] - w/2, b_vals[i] + 0.02, f"{b_vals[i]:.2f}", ha="center", fontsize=9, fontweight="bold")
        plt.text(x[i] + w/2, e_vals[i] + 0.02, f"{e_vals[i]:.2f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def evaluate_voice():
    """Run thorough evaluation on held-out test split."""
    print("=" * 60)
    print("VOICE BASE-PAPER 2D CNN EVALUATION")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data Splits
    metadata_df = build_voice_metadata()
    train_df, val_df, test_df = create_subject_splits(metadata_df, seed=config.RANDOM_SEED)

    normalizer = VoiceSTFTNormalizer.load(config.NORMALIZATION_STATS_PATH)
    test_ds = VoiceSTFTChunkDataset(test_df, normalizer=normalizer)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_voice_stft_batch)

    # 2. Load Continuous STFT Model
    model = VoiceBasePaper2DCNN().to(device)
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    # 3. Evaluate Chunks
    chunk_records = []
    with torch.no_grad():
        for batch in test_loader:
            feats = batch["features"].to(device)
            logits = model(feats)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()

            for i in range(len(labels)):
                chunk_records.append({
                    "subject_id": batch["subject_ids"][i],
                    "recording_id": batch["recording_ids"][i],
                    "task": batch["tasks"][i],
                    "chunk_idx": batch["chunk_indices"][i],
                    "true_label": int(labels[i]),
                    "pd_probability": float(probs[i, 1]),
                    "hc_probability": float(probs[i, 0]),
                    "predicted_label": int(probs[i, 1] >= 0.5)
                })

    chunk_df = pd.DataFrame(chunk_records)
    chunk_df.to_csv(config.CHUNK_PREDICTIONS_PATH, index=False)

    # 4. Recording Aggregation
    rec_grouped = chunk_df.groupby(["recording_id", "subject_id", "task", "true_label"]).agg(
        pd_probability=("pd_probability", "mean")
    ).reset_index()
    rec_grouped["predicted_label"] = (rec_grouped["pd_probability"] >= 0.5).astype(int)
    rec_grouped["hc_probability"] = 1.0 - rec_grouped["pd_probability"]
    rec_grouped.to_csv(config.RECORDING_PREDICTIONS_PATH, index=False)

    # 5. Subject Aggregation
    sub_grouped = rec_grouped.groupby(["subject_id", "true_label"]).agg(
        pd_probability=("pd_probability", "mean")
    ).reset_index()
    sub_grouped["predicted_label"] = (sub_grouped["pd_probability"] >= 0.5).astype(int)
    sub_grouped["hc_probability"] = 1.0 - sub_grouped["pd_probability"]
    sub_grouped["predicted_name"] = sub_grouped["predicted_label"].map(config.ID2LABEL)
    sub_grouped.to_csv(config.SUBJECT_PREDICTIONS_PATH, index=False)

    # 6. Compute Metrics
    chunk_m = calculate_metrics_dict(chunk_df["true_label"].values, chunk_df["pd_probability"].values)
    rec_m = calculate_metrics_dict(rec_grouped["true_label"].values, rec_grouped["pd_probability"].values)
    sub_m = calculate_metrics_dict(sub_grouped["true_label"].values, sub_grouped["pd_probability"].values)

    # Task breakdown
    task_breakdown = {}
    for task_name in ["ReadText", "SpontaneousDialogue"]:
        tdf = rec_grouped[rec_grouped["task"] == task_name]
        if len(tdf) > 0:
            task_breakdown[task_name] = calculate_metrics_dict(tdf["true_label"].values, tdf["pd_probability"].values)

    # 7. Transition Analysis (Onset & Offset Models)
    transition_m = {}
    if config.ONSET_MODEL_PATH.exists():
        onset_model = VoiceBasePaper2DCNN().to(device)
        onset_model.load_state_dict(torch.load(config.ONSET_MODEL_PATH, map_location=device))
        onset_model.eval()

        onset_test_records = []
        with torch.no_grad():
            for _, r in test_df.iterrows():
                fp = r.get("filepath", "")
                if fp and Path(fp).exists():
                    audio = load_and_resample_audio(fp)
                    ons, _ = detect_speech_transitions(audio)
                    if ons:
                        stacked = torch.stack(ons).to(device)
                        probs = torch.softmax(onset_model(stacked), dim=-1)[:, 1].cpu().numpy()
                        mean_p = float(np.mean(probs))
                        onset_test_records.append({"true_label": r["label"], "prob": mean_p})

        if onset_test_records:
            odf = pd.DataFrame(onset_test_records)
            transition_m["onset_model"] = calculate_metrics_dict(odf["true_label"].values, odf["prob"].values)

    if config.OFFSET_MODEL_PATH.exists():
        offset_model = VoiceBasePaper2DCNN().to(device)
        offset_model.load_state_dict(torch.load(config.OFFSET_MODEL_PATH, map_location=device))
        offset_model.eval()

        offset_test_records = []
        with torch.no_grad():
            for _, r in test_df.iterrows():
                fp = r.get("filepath", "")
                if fp and Path(fp).exists():
                    audio = load_and_resample_audio(fp)
                    _, offs = detect_speech_transitions(audio)
                    if offs:
                        stacked = torch.stack(offs).to(device)
                        probs = torch.softmax(offset_model(stacked), dim=-1)[:, 1].cpu().numpy()
                        mean_p = float(np.mean(probs))
                        offset_test_records.append({"true_label": r["label"], "prob": mean_p})

        if offset_test_records:
            odf = pd.DataFrame(offset_test_records)
            transition_m["offset_model"] = calculate_metrics_dict(odf["true_label"].values, odf["prob"].values)

    results_dict = {
        "model_name": "VoiceBasePaper2DCNN",
        "dataset": "MDVR-KCL Audio (ReadText & SpontaneousDialogue)",
        "input_representation": "STFT Log-Magnitude Spectrogram (1, 257, 301)",
        "subject_level": sub_m,
        "recording_level": rec_m,
        "chunk_level": chunk_m,
        "task_breakdown": task_breakdown,
        "transition_models": transition_m,
        "test_subjects": sorted(sub_grouped["subject_id"].tolist())
    }

    with open(config.METRICS_JSON_PATH, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"Saved Voice metrics to: {config.METRICS_JSON_PATH}")

    # Plots
    cm_counts = np.array([[sub_m["tn"], sub_m["fp"]], [sub_m["fn"], sub_m["tp"]]])
    plot_confusion_matrix(cm_counts, config.CONFUSION_MATRIX_PATH)
    plot_roc_curve_figure(sub_grouped["true_label"].values, sub_grouped["pd_probability"].values, config.ROC_CURVE_PATH)

    # Comparison Plot against Proposed Wav2Vec2 Model
    exist_metrics_path = config.VOICE_DIR / "results" / "metrics.json"
    if exist_metrics_path.exists():
        with open(exist_metrics_path, "r") as f:
            exist_data = json.load(f)
        exist_sub = exist_data.get("subject_level_overall", {})
        if "balanced_accuracy" not in exist_sub and "sensitivity" in exist_sub and "specificity" in exist_sub:
            exist_sub["balanced_accuracy"] = 0.5 * (exist_sub["sensitivity"] + exist_sub["specificity"])
        plot_comparison_chart(sub_m, exist_sub, config.COMPARISON_BAR_PATH)
        print(f"Saved baseline vs existing comparison plot to: {config.COMPARISON_BAR_PATH}")

    print("\n" + "=" * 60)
    print("VOICE BASE-PAPER 2D CNN EVALUATION SUMMARY:")
    print(f"  Subject Accuracy:     {sub_m['accuracy']:.4f}")
    print(f"  Balanced Accuracy:    {sub_m['balanced_accuracy']:.4f}")
    print(f"  Subject F1-Score:     {sub_m['f1_score']:.4f}")
    print(f"  Subject ROC-AUC:      {sub_m['roc_auc']:.4f}")
    print(f"  Sensitivity (Recall): {sub_m['recall_sensitivity']:.4f}")
    print(f"  Specificity:          {sub_m['specificity']:.4f}")
    print(f"  Test TP: {sub_m['tp']}, TN: {sub_m['tn']}, FP: {sub_m['fp']}, FN: {sub_m['fn']}")
    print("=" * 60)

    return results_dict


if __name__ == "__main__":
    evaluate_voice()
