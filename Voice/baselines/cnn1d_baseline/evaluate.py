"""Evaluation script for Voice 1D CNN Baseline on Raw Waveforms."""

import json
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

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Voice.baselines.cnn1d_baseline import config
from Voice.baselines.cnn1d_baseline.dataset import (
    VoiceRawWaveformChunkDataset,
    VoiceWaveformNormalizer,
    collate_voice_1d_batch,
)
from Voice.baselines.cnn1d_baseline.model import Voice1DCNN
from Voice.dataset import build_metadata, create_subject_splits


def calculate_metrics_dict(y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
    """Calculate classification, ranking, and calibration metrics."""
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


def compute_bootstrap_subject_ci(
    subject_df: pd.DataFrame,
    n_bootstraps: int = 1000,
    seed: int = 42
) -> Dict[str, Tuple[float, float]]:
    """Compute 95% Confidence Intervals by resampling subjects."""
    rng = np.random.RandomState(seed)
    n_subjects = len(subject_df)
    boot_acc, boot_f1 = [], []

    for _ in range(n_bootstraps):
        sample_indices = rng.choice(n_subjects, size=n_subjects, replace=True)
        boot_sub = subject_df.iloc[sample_indices]
        if len(np.unique(boot_sub["true_label"])) < 2:
            continue
        y_t = boot_sub["true_label"].values
        y_p = boot_sub["pd_probability"].values
        y_pred = (y_p >= 0.5).astype(int)
        boot_acc.append(accuracy_score(y_t, y_pred))
        boot_f1.append(f1_score(y_t, y_pred, zero_division=0))

    ci_dict = {}
    if boot_acc:
        ci_dict["accuracy"] = (float(np.percentile(boot_acc, 2.5)), float(np.percentile(boot_acc, 97.5)))
    else:
        ci_dict["accuracy"] = (0.0, 0.0)
    if boot_f1:
        ci_dict["f1_score"] = (float(np.percentile(boot_f1, 2.5)), float(np.percentile(boot_f1, 97.5)))
    else:
        ci_dict["f1_score"] = (0.0, 0.0)
    return ci_dict


def plot_confusion_matrix(cm_counts: np.ndarray, save_path: Path):
    """Plot and save confusion matrix heatmap."""
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_counts, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Voice 1D CNN: Subject Confusion Matrix", fontsize=12, fontweight="bold")
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


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: Path):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#d62728", lw=2.5, label=f"1D CNN (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("Voice 1D CNN: Subject ROC Curve", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_comparison_chart(base_1d_m: Dict, exist_w2v_m: Dict, save_path: Path):
    """Plot comparison between Voice 1D CNN baseline and proposed Wav2Vec2 model."""
    metrics = ["accuracy", "balanced_accuracy", "precision", "recall_sensitivity", "specificity", "f1_score", "roc_auc"]
    labels = ["Accuracy", "Bal. Acc", "Precision", "Sensitivity", "Specificity", "F1-Score", "ROC-AUC"]

    v_1d = [base_1d_m.get(m, 0.0) for m in metrics]
    v_w2v = [exist_w2v_m.get(m, 0.0) for m in metrics]

    x = np.arange(len(labels))
    w = 0.35

    plt.figure(figsize=(10, 5.5))
    plt.bar(x - w/2, v_1d, w, label="Voice 1D CNN Baseline (Raw Waveform)", color="#E67E22", edgecolor="black", alpha=0.9)
    plt.bar(x + w/2, v_w2v, w, label="Proposed Wav2Vec2 Model", color="#3498DB", edgecolor="black", alpha=0.9)

    plt.ylabel("Score (0.0 - 1.0)", fontsize=12)
    plt.title("Voice Subject-Level: 1D CNN Baseline vs. Proposed Wav2Vec2", fontsize=13, fontweight="bold")
    plt.xticks(x, labels, fontsize=11)
    plt.ylim(0.0, 1.05)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend(fontsize=11, loc="lower right")

    for i in range(len(x)):
        plt.text(x[i] - w/2, v_1d[i] + 0.02, f"{v_1d[i]:.2f}", ha="center", fontsize=9, fontweight="bold")
        plt.text(x[i] + w/2, v_w2v[i] + 0.02, f"{v_w2v[i]:.2f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def evaluate_voice_1d_cnn() -> Dict:
    """Run thorough evaluation of Voice 1D CNN baseline."""
    print("=" * 60)
    print("VOICE 1D CNN BASELINE EVALUATION (TEST SET)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta_df = build_metadata(config.DATA_DIR)
    train_df, val_df, test_df = create_subject_splits(meta_df, seed=config.RANDOM_SEED)

    normalizer = VoiceWaveformNormalizer.load(config.NORMALIZATION_STATS_PATH)
    test_ds = VoiceRawWaveformChunkDataset(test_df, normalizer=normalizer)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_voice_1d_batch)

    model = Voice1DCNN().to(device)
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    chunk_records = []
    with torch.no_grad():
        for batch in test_loader:
            feats = batch["features"].to(device)
            logits = model(feats)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()

            for i in range(len(labels)):
                p_pd = float(probs[i, 1])
                p_hc = float(probs[i, 0])
                pred = int(p_pd >= 0.5)
                y = int(labels[i])

                p_clamped = np.clip(p_pd, 1e-6, 1.0 - 1e-6)
                entropy = float(-(p_clamped * np.log2(p_clamped) + (1.0 - p_clamped) * np.log2(1.0 - p_clamped)))

                chunk_records.append({
                    "subject_id": batch["subject_ids"][i],
                    "recording_id": batch["recording_ids"][i],
                    "task": batch["tasks"][i],
                    "chunk_idx": batch["chunk_indices"][i],
                    "true_label": y,
                    "pd_probability": p_pd,
                    "hc_probability": p_hc,
                    "predicted_label": pred,
                    "is_correct": int(pred == y),
                    "predictive_entropy": entropy
                })

    chunk_df = pd.DataFrame(chunk_records)
    chunk_df.to_csv(config.CHUNK_PREDICTIONS_PATH, index=False)

    rec_grouped = chunk_df.groupby(["recording_id", "subject_id", "task", "true_label"]).agg(
        pd_probability=("pd_probability", "mean"),
        predictive_entropy=("predictive_entropy", "mean")
    ).reset_index()
    rec_grouped["predicted_label"] = (rec_grouped["pd_probability"] >= 0.5).astype(int)
    rec_grouped["hc_probability"] = 1.0 - rec_grouped["pd_probability"]
    rec_grouped["is_correct"] = (rec_grouped["predicted_label"] == rec_grouped["true_label"]).astype(int)
    rec_grouped.to_csv(config.RECORDING_PREDICTIONS_PATH, index=False)

    sub_grouped = rec_grouped.groupby(["subject_id", "true_label"]).agg(
        pd_probability=("pd_probability", "mean"),
        predictive_entropy=("predictive_entropy", "mean")
    ).reset_index()
    sub_grouped["predicted_label"] = (sub_grouped["pd_probability"] >= 0.5).astype(int)
    sub_grouped["hc_probability"] = 1.0 - sub_grouped["pd_probability"]
    sub_grouped["is_correct"] = (sub_grouped["predicted_label"] == sub_grouped["true_label"]).astype(int)
    sub_grouped["predicted_name"] = sub_grouped["predicted_label"].map(config.ID2LABEL)
    sub_grouped.to_csv(config.SUBJECT_PREDICTIONS_PATH, index=False)

    chunk_m = calculate_metrics_dict(chunk_df["true_label"].values, chunk_df["pd_probability"].values)
    rec_m = calculate_metrics_dict(rec_grouped["true_label"].values, rec_grouped["pd_probability"].values)
    sub_m = calculate_metrics_dict(sub_grouped["true_label"].values, sub_grouped["pd_probability"].values)

    sub_ci = compute_bootstrap_subject_ci(sub_grouped)
    sub_m["confidence_intervals_95"] = sub_ci

    correct_ent = float(sub_grouped[sub_grouped["is_correct"] == 1]["predictive_entropy"].mean()) if (sub_grouped["is_correct"] == 1).any() else 0.0
    incorrect_ent = float(sub_grouped[sub_grouped["is_correct"] == 0]["predictive_entropy"].mean()) if (sub_grouped["is_correct"] == 0).any() else 0.0

    entropy_stats = {
        "mean_predictive_entropy_all": float(sub_grouped["predictive_entropy"].mean()),
        "mean_predictive_entropy_correct": correct_ent,
        "mean_predictive_entropy_incorrect": incorrect_ent,
        "uncertainty_type": "binary_predictive_entropy_proxy"
    }

    results_dict = {
        "model_name": "Voice1DCNN_RawWaveform_Baseline",
        "description": "Conventional 1D CNN on 16 kHz raw waveforms without pretrained transformers or STFT",
        "subject_level": sub_m,
        "recording_level": rec_m,
        "chunk_level": chunk_m,
        "entropy_uncertainty_stats": entropy_stats,
        "test_subjects": sorted(sub_grouped["subject_id"].tolist()),
        "test_sample_size": {
            "total_subjects": len(sub_grouped),
            "hc_subjects": int((sub_grouped["true_label"] == 0).sum()),
            "pd_subjects": int((sub_grouped["true_label"] == 1).sum()),
            "recordings": len(rec_grouped),
            "chunks": len(chunk_df)
        }
    }

    with open(config.METRICS_JSON_PATH, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"Saved Voice 1D CNN test metrics to: {config.METRICS_JSON_PATH}")

    cm_counts = np.array([[sub_m["tn"], sub_m["fp"]], [sub_m["fn"], sub_m["tp"]]])
    plot_confusion_matrix(cm_counts, config.CONFUSION_MATRIX_PATH)
    plot_roc_curve(sub_grouped["true_label"].values, sub_grouped["pd_probability"].values, config.ROC_CURVE_PATH)

    # Comparison plot against proposed Wav2Vec2 model
    exist_w2v_path = config.VOICE_DIR / "results" / "metrics.json"
    if exist_w2v_path.exists():
        with open(exist_w2v_path, "r") as f:
            exist_data = json.load(f)
        exist_sub = exist_data.get("subject_level_overall", {})
        if "balanced_accuracy" not in exist_sub and "sensitivity" in exist_sub and "specificity" in exist_sub:
            exist_sub["balanced_accuracy"] = 0.5 * (exist_sub["sensitivity"] + exist_sub["specificity"])
        plot_comparison_chart(sub_m, exist_sub, config.COMPARISON_BAR_PATH)

    print("\n" + "=" * 60)
    print("VOICE 1D CNN TEST RESULTS SUMMARY (SUBJECT-LEVEL):")
    print(f"  Subject Accuracy:     {sub_m['accuracy']:.4f}")
    print(f"  Balanced Accuracy:    {sub_m['balanced_accuracy']:.4f}")
    print(f"  Sensitivity (Recall): {sub_m['recall_sensitivity']:.4f}")
    print(f"  Specificity:          {sub_m['specificity']:.4f}")
    print(f"  Precision:            {sub_m['precision']:.4f}")
    print(f"  F1-Score:             {sub_m['f1_score']:.4f}")
    print(f"  ROC-AUC:              {sub_m['roc_auc']:.4f}")
    print(f"  Brier Score:          {sub_m['brier_score']:.4f}")
    print(f"  Negative Log-Likelihood: {sub_m['negative_log_likelihood']:.4f}")
    print(f"  TP: {sub_m['tp']}, TN: {sub_m['tn']}, FP: {sub_m['fp']}, FN: {sub_m['fn']}")
    print("-" * 60)
    print(f"  NOTE ON TEST COHORT UNCERTAINTY: N = {len(sub_grouped)} subjects (3 HC, 3 PD).")
    print(f"  Small sample size causes broad bootstrap intervals.")
    print("=" * 60)

    return results_dict


if __name__ == "__main__":
    evaluate_voice_1d_cnn()
