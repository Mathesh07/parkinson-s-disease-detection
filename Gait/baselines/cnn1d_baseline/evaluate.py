"""Comprehensive evaluation and ablation study script for Gait 1D CNN Baseline.

Computes window, recording, and subject-level metrics, bootstrap confidence intervals,
predictive entropy uncertainty proxies, and comparison against the proposed 1D CNN + BiLSTM models.
"""

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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Gait.baselines.cnn1d_baseline import config
from Gait.baselines.cnn1d_baseline.dataset import Gait1DCNNWindowDataset, collate_gait_1d_batch
from Gait.baselines.cnn1d_baseline.model import Gait1DCNN
from Gait.dataset import create_subject_splits
from Gait.preprocessing import GaitNormalizer, build_manifest


def calculate_metrics_dict(y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
    """Calculate all classification, ranking, and calibration metrics."""
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
    """
    Compute 95% Confidence Intervals by resampling SUBJECTS with replacement.
    
    Treats patients as the fundamental sampling unit rather than correlated windows.
    """
    rng = np.random.RandomState(seed)
    n_subjects = len(subject_df)
    boot_metrics = {
        "accuracy": [],
        "balanced_accuracy": [],
        "f1_score": [],
        "roc_auc": []
    }

    for _ in range(n_bootstraps):
        sample_indices = rng.choice(n_subjects, size=n_subjects, replace=True)
        boot_sub = subject_df.iloc[sample_indices]

        # Check that both classes exist in resample
        if len(np.unique(boot_sub["true_label"])) < 2:
            continue

        y_t = boot_sub["true_label"].values
        y_p = boot_sub["pd_probability"].values
        y_pred = (y_p >= 0.5).astype(int)

        boot_metrics["accuracy"].append(accuracy_score(y_t, y_pred))
        boot_metrics["balanced_accuracy"].append(balanced_accuracy_score(y_t, y_pred))
        boot_metrics["f1_score"].append(f1_score(y_t, y_pred, zero_division=0))
        try:
            boot_metrics["roc_auc"].append(roc_auc_score(y_t, y_p))
        except Exception:
            pass

    ci_results = {}
    for k, v in boot_metrics.items():
        if len(v) > 0:
            ci_results[k] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        else:
            ci_results[k] = (0.0, 0.0)

    return ci_results


def plot_confusion_matrix(cm_counts: np.ndarray, save_path: Path):
    """Plot and save confusion matrix heatmap."""
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_counts, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Gait 1D CNN: Subject Confusion Matrix", fontsize=12, fontweight="bold")
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
    plt.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"1D CNN (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("Gait 1D CNN: Subject ROC Curve", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: Path):
    """Plot and save Precision-Recall curve."""
    prec, rec, _ = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, color="#2ca02c", lw=2.5, label="1D CNN Precision-Recall")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Gait 1D CNN: Subject Precision-Recall Curve", fontsize=12, fontweight="bold")
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_ablation_comparison(
    cnn1d_m: Dict,
    bilstm_m: Dict,
    evidential_m: Dict,
    save_path: Path
):
    """Plot controlled 3-stage ablation chart."""
    metrics = ["accuracy", "balanced_accuracy", "recall_sensitivity", "specificity", "f1_score", "roc_auc"]
    labels = ["Accuracy", "Bal. Acc", "Sensitivity", "Specificity", "F1-Score", "ROC-AUC"]

    v_cnn = [cnn1d_m.get(m, 0.0) for m in metrics]
    v_bilstm = [bilstm_m.get(m, 0.0) for m in metrics]
    v_evi = [evidential_m.get(m, 0.0) for m in metrics]

    x = np.arange(len(labels))
    w = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - w, v_cnn, w, label="1. 1D CNN (Baseline)", color="#4A90E2", edgecolor="black", alpha=0.9)
    plt.bar(x, v_bilstm, w, label="2. 1D CNN + BiLSTM (Proposed)", color="#50E3C2", edgecolor="black", alpha=0.9)
    plt.bar(x + w, v_evi, w, label="3. 1D CNN + BiLSTM + Evidential (Proposed)", color="#B8E986", edgecolor="black", alpha=0.9)

    plt.ylabel("Score (0.0 - 1.0)", fontsize=12)
    plt.title("Gait Ablation Study: Impact of BiLSTM Temporal Modelling & Evidential Head", fontsize=13, fontweight="bold")
    plt.xticks(x, labels, fontsize=11)
    plt.ylim(0.0, 1.05)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend(fontsize=10, loc="lower right")

    for i in range(len(x)):
        plt.text(x[i] - w, v_cnn[i] + 0.02, f"{v_cnn[i]:.2f}", ha="center", fontsize=8, fontweight="bold")
        plt.text(x[i], v_bilstm[i] + 0.02, f"{v_bilstm[i]:.2f}", ha="center", fontsize=8, fontweight="bold")
        plt.text(x[i] + w, v_evi[i] + 0.02, f"{v_evi[i]:.2f}", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def evaluate_gait_1d_cnn() -> Dict:
    """Run thorough evaluation of Gait 1D CNN baseline on held-out test split."""
    print("=" * 60)
    print("GAIT 1D CNN BASELINE EVALUATION (TEST SET)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest_df = build_manifest(config.DATA_DIR)
    train_df, val_df, test_df = create_subject_splits(manifest_df, seed=config.RANDOM_SEED)

    normalizer = GaitNormalizer.load(config.NORMALIZATION_STATS_PATH)
    test_ds = Gait1DCNNWindowDataset(test_df, normalizer=normalizer)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_gait_1d_batch)

    model = Gait1DCNN().to(device)
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    window_records = []
    with torch.no_grad():
        for batch in test_loader:
            feats = batch["features"].to(device)
            logits = model(feats)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()

            for i in range(len(labels)):
                p = float(probs[i])
                y = int(labels[i])
                pred = int(p >= 0.5)

                # Binary predictive entropy proxy
                p_clamped = np.clip(p, 1e-6, 1.0 - 1e-6)
                entropy = float(-(p_clamped * np.log2(p_clamped) + (1.0 - p_clamped) * np.log2(1.0 - p_clamped)))

                window_records.append({
                    "subject_id": batch["subject_ids"][i],
                    "recording_id": batch["recording_ids"][i],
                    "study": batch["studies"][i],
                    "window_idx": batch["window_indices"][i],
                    "true_label": y,
                    "pd_probability": p,
                    "hc_probability": 1.0 - p,
                    "predicted_label": pred,
                    "is_correct": int(pred == y),
                    "predictive_entropy": entropy
                })

    window_df = pd.DataFrame(window_records)
    window_df.to_csv(config.WINDOW_PREDICTIONS_PATH, index=False)

    # Recording-level aggregation
    rec_grouped = window_df.groupby(["recording_id", "subject_id", "study", "true_label"]).agg(
        pd_probability=("pd_probability", "mean"),
        predictive_entropy=("predictive_entropy", "mean")
    ).reset_index()
    rec_grouped["predicted_label"] = (rec_grouped["pd_probability"] >= 0.5).astype(int)
    rec_grouped["hc_probability"] = 1.0 - rec_grouped["pd_probability"]
    rec_grouped["is_correct"] = (rec_grouped["predicted_label"] == rec_grouped["true_label"]).astype(int)
    rec_grouped.to_csv(config.RECORDING_PREDICTIONS_PATH, index=False)

    # Subject-level aggregation
    sub_grouped = rec_grouped.groupby(["subject_id", "study", "true_label"]).agg(
        pd_probability=("pd_probability", "mean"),
        predictive_entropy=("predictive_entropy", "mean")
    ).reset_index()
    sub_grouped["predicted_label"] = (sub_grouped["pd_probability"] >= 0.5).astype(int)
    sub_grouped["hc_probability"] = 1.0 - sub_grouped["pd_probability"]
    sub_grouped["is_correct"] = (sub_grouped["predicted_label"] == sub_grouped["true_label"]).astype(int)
    sub_grouped["predicted_name"] = sub_grouped["predicted_label"].map(config.ID2LABEL)
    sub_grouped.to_csv(config.SUBJECT_PREDICTIONS_PATH, index=False)

    # Metrics computation
    win_m = calculate_metrics_dict(window_df["true_label"].values, window_df["pd_probability"].values)
    rec_m = calculate_metrics_dict(rec_grouped["true_label"].values, rec_grouped["pd_probability"].values)
    sub_m = calculate_metrics_dict(sub_grouped["true_label"].values, sub_grouped["pd_probability"].values)

    # Bootstrap 95% Confidence Intervals
    sub_ci = compute_bootstrap_subject_ci(sub_grouped)
    sub_m["confidence_intervals_95"] = sub_ci

    # Predictive Entropy Uncertainty breakdown
    correct_ent = float(sub_grouped[sub_grouped["is_correct"] == 1]["predictive_entropy"].mean())
    incorrect_ent = float(sub_grouped[sub_grouped["is_correct"] == 0]["predictive_entropy"].mean()) if (sub_grouped["is_correct"] == 0).any() else 0.0
    hc_ent = float(sub_grouped[sub_grouped["true_label"] == 0]["predictive_entropy"].mean())
    pd_ent = float(sub_grouped[sub_grouped["true_label"] == 1]["predictive_entropy"].mean())

    entropy_stats = {
        "mean_predictive_entropy_all": float(sub_grouped["predictive_entropy"].mean()),
        "mean_predictive_entropy_correct": correct_ent,
        "mean_predictive_entropy_incorrect": incorrect_ent,
        "mean_predictive_entropy_HC": hc_ent,
        "mean_predictive_entropy_PD": pd_ent,
        "uncertainty_type": "binary_predictive_entropy_proxy"
    }

    # Study breakdown
    study_breakdown = {}
    for s_name in ["Ga", "Ju", "Si"]:
        sdf = sub_grouped[sub_grouped["study"] == s_name]
        if len(sdf) > 0:
            study_breakdown[s_name] = calculate_metrics_dict(sdf["true_label"].values, sdf["pd_probability"].values)

    results_dict = {
        "model_name": "Gait1DCNN_Baseline",
        "description": "Conventional 1D CNN without LSTM, BiLSTM, or Evidential Head",
        "input_representation": "Raw VGRF 16-channel time series (500 time steps, 16 sensors)",
        "subject_level": sub_m,
        "recording_level": rec_m,
        "window_level": win_m,
        "entropy_uncertainty_stats": entropy_stats,
        "study_breakdown": study_breakdown,
        "test_subjects": sorted(sub_grouped["subject_id"].tolist()),
        "class_distribution": {
            "total_test_subjects": len(sub_grouped),
            "hc_test_subjects": int((sub_grouped["true_label"] == 0).sum()),
            "pd_test_subjects": int((sub_grouped["true_label"] == 1).sum())
        }
    }

    with open(config.TEST_METRICS_PATH, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"Saved Gait 1D CNN test metrics to: {config.TEST_METRICS_PATH}")

    # Visualizations
    cm_counts = np.array([[sub_m["tn"], sub_m["fp"]], [sub_m["fn"], sub_m["tp"]]])
    plot_confusion_matrix(cm_counts, config.CONFUSION_MATRIX_PATH)
    plot_roc_curve(sub_grouped["true_label"].values, sub_grouped["pd_probability"].values, config.ROC_CURVE_PATH)
    plot_pr_curve(sub_grouped["true_label"].values, sub_grouped["pd_probability"].values, config.PR_CURVE_PATH)

    # 3-Stage Ablation Comparison against existing proposed models
    exist_bce_path = config.GAIT_DIR / "outputs" / "metrics" / "test_metrics.json"
    exist_evi_path = config.GAIT_DIR / "outputs" / "metrics" / "gait_evidential_metrics.json"

    bilstm_m = {}
    evidential_m = {}

    if exist_bce_path.exists():
        with open(exist_bce_path, "r") as f:
            bce_data = json.load(f)
        bilstm_m = bce_data.get("subject_level", {})
        if "balanced_accuracy" not in bilstm_m and "recall_sensitivity" in bilstm_m and "specificity" in bilstm_m:
            bilstm_m["balanced_accuracy"] = 0.5 * (bilstm_m["recall_sensitivity"] + bilstm_m["specificity"])

    if exist_evi_path.exists():
        with open(exist_evi_path, "r") as f:
            evi_data = json.load(f)
        evidential_m = evi_data.get("subject_metrics", {})
        if "balanced_accuracy" not in evidential_m and "recall_sensitivity" in evidential_m and "specificity" in evidential_m:
            evidential_m["balanced_accuracy"] = 0.5 * (evidential_m["recall_sensitivity"] + evidential_m["specificity"])

    if bilstm_m and evidential_m:
        plot_ablation_comparison(sub_m, bilstm_m, evidential_m, config.ABLATION_BAR_PATH)

        ablation_summary = {
            "ablation_levels": {
                "Level_1_1D_CNN_Baseline": sub_m,
                "Level_2_1D_CNN_BiLSTM_BCE": bilstm_m,
                "Level_3_1D_CNN_BiLSTM_Evidential": evidential_m
            }
        }
        with open(config.ABLATION_METRICS_PATH, "w") as f:
            json.dump(ablation_summary, f, indent=4)
        print(f"Saved ablation study summary to: {config.ABLATION_METRICS_PATH}")

    print("\n" + "=" * 60)
    print("GAIT 1D CNN TEST RESULTS SUMMARY (SUBJECT-LEVEL):")
    print(f"  Subject Accuracy:     {sub_m['accuracy']:.4f} (95% CI: [{sub_ci['accuracy'][0]:.4f}, {sub_ci['accuracy'][1]:.4f}])")
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
    print("PREDICTIVE ENTROPY (UNCERTAINTY PROXY):")
    print(f"  Mean Entropy (Correct):   {entropy_stats['mean_predictive_entropy_correct']:.4f}")
    print(f"  Mean Entropy (Incorrect): {entropy_stats['mean_predictive_entropy_incorrect']:.4f}")
    print("=" * 60)

    return results_dict


if __name__ == "__main__":
    evaluate_gait_1d_cnn()
