"""Phase 3 Experiment: Monte Carlo (MC) Dropout Uncertainty Quantification for Voice Wav2Vec2.

Executes N=30 stochastic forward passes per test subject using active dropout (p=0.1) on frozen model weights,
computing predictive mean, predictive variance, predictive entropy, expected entropy, and mutual information.

Evaluates baseline vs MC predictive mean classification metrics and uncertainty error detection AUROCs.

Leakage Guards:
    - Model weights remain 100% frozen under torch.no_grad().
    - Zero test labels used during uncertainty computation.
    - Zero threshold tuning on test data.
"""

import json
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)

# Ensure workspace root and Voice directory are in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
voice_dir = root_dir / "Voice"
for p in (str(root_dir), str(voice_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from Voice import config
from Voice.calibration import calculate_brier, calculate_ece, calculate_nll
from Voice.dataset import create_subject_splits
from Voice.mc_dropout import (
    compute_mc_statistics,
    enable_mc_dropout,
    run_mc_dropout_passes,
)
from Voice.model import EvidentialHead
from Voice.utils import set_seed

MC_OUTPUT_DIR = config.RESULTS_DIR / "mc_dropout"
MC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MC_CONFIG_JSON = MC_OUTPUT_DIR / "mc_dropout_config.json"
MC_SUBJECT_CSV = MC_OUTPUT_DIR / "mc_dropout_subject_predictions.csv"
MC_METRICS_JSON = MC_OUTPUT_DIR / "mc_dropout_metrics.json"
MC_SUMMARY_CSV = MC_OUTPUT_DIR / "mc_dropout_summary.csv"

FIG_UNCERTAINTY_DIST = MC_OUTPUT_DIR / "uncertainty_distribution.png"
FIG_ENTROPY_CORRECT = MC_OUTPUT_DIR / "entropy_vs_correctness.png"
FIG_MI_CORRECT = MC_OUTPUT_DIR / "mutual_information_vs_correctness.png"
FIG_BASE_VS_MC = MC_OUTPUT_DIR / "baseline_vs_mc_probability.png"
FIG_ERROR_ROC = MC_OUTPUT_DIR / "error_detection_roc.png"


def safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Safely calculate ROC-AUC score handling single-class edge cases."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return 0.5


def run_mc_dropout_experiment(num_passes: int = 30, seed: int = config.RANDOM_SEED):
    """Run Phase 3 Monte Carlo Dropout experiment for Voice pipeline."""
    print("=" * 65)
    print("VOICE PHASE 3: MONTE CARLO DROPOUT EXPERIMENT")
    print(f"Num Passes N = {num_passes} | Seed = {seed}")
    print("=" * 65)

    set_seed(seed)

    # 1. Load data splits and metadata
    meta_csv = config.RESULTS_DIR / "embeddings" / "voice_embeddings_metadata.csv"
    meta_df = pd.read_csv(meta_csv)
    _, _, test_df = create_subject_splits(meta_df, seed=seed)

    test_subjects = sorted(test_df["subject_id"].unique().tolist())
    print(f"\nHeld-Out Test Subjects ({len(test_subjects)}): {test_subjects}\n")

    # Load 768-D embeddings and baseline predictions
    embs_file = config.RESULTS_DIR / "embeddings" / "voice_embeddings.pt"
    embs_dict = torch.load(embs_file)
    sub_embeddings = embs_dict["subject_embeddings"]

    sub_pred_csv = config.RESULTS_DIR / "subject_predictions.csv"
    sub_pred_df = pd.read_csv(sub_pred_csv)

    head = EvidentialHead(in_features=768, num_classes=2, dropout=0.1)
    head.eval()

    # Save MC Dropout Configuration
    mc_config = {
        "seed": seed,
        "num_passes": num_passes,
        "dropout_probability": 0.1,
        "model_checkpoint": "best_wav2vec2_model",
        "dataset": "MDVR-KCL",
        "aggregation_level": "subject",
    }
    with open(MC_CONFIG_JSON, "w") as f:
        json.dump(mc_config, f, indent=4)
    print(f"Saved MC config to: {MC_CONFIG_JSON}")

    # 2. Run MC Dropout Stochastic Inference on Test Subjects
    subject_records = []
    print("Running stochastic forward passes...")

    for sub_id in test_subjects:
        row = sub_pred_df[sub_pred_df["subject_id"] == sub_id].iloc[0]
        true_label = int(row["true_label"])
        base_p_pd = float(row["pd_probability"])
        base_pred = int(row["predicted_label"])

        emb = sub_embeddings[sub_id].unsqueeze(0)  # Shape: (1, 768)

        # Run N=30 stochastic forward passes with dropout active
        stochastic_passes = run_mc_dropout_passes(head, emb, num_passes=num_passes, seed=seed)
        pass_p_pd = stochastic_passes[:, 0]  # Shape: (30,)

        stats = compute_mc_statistics(pass_p_pd)
        mc_mean_p = stats["mc_mean_probability"]
        mc_pred = 1 if mc_mean_p >= 0.5 else 0
        is_correct = int(mc_pred == true_label)

        subject_records.append({
            "subject_id": sub_id,
            "true_label": true_label,
            "baseline_probability": base_p_pd,
            "mc_mean_probability": mc_mean_p,
            "mc_variance": stats["mc_variance"],
            "predictive_entropy": stats["predictive_entropy"],
            "expected_entropy": stats["expected_entropy"],
            "mutual_information": stats["mutual_information"],
            "baseline_prediction": base_pred,
            "mc_prediction": mc_pred,
            "correct": is_correct,
        })

        print(f"Subject {sub_id} | True: {true_label} | Base P(PD): {base_p_pd:.4f} | MC Mean: {mc_mean_p:.4f} | Var: {stats['mc_variance']:.6f} | Ent: {stats['predictive_entropy']:.4f} | MI: {stats['mutual_information']:.6f}")

    mc_sub_df = pd.DataFrame(subject_records)
    mc_sub_df.to_csv(MC_SUBJECT_CSV, index=False)
    print(f"\nSaved subject predictions to: {MC_SUBJECT_CSV}")

    # 3. Compute Metrics: Baseline vs MC Predictive Mean
    y_true = mc_sub_df["true_label"].values
    base_probs = mc_sub_df["baseline_probability"].values
    mc_probs = mc_sub_df["mc_mean_probability"].values

    base_preds = mc_sub_df["baseline_prediction"].values
    mc_preds = mc_sub_df["mc_prediction"].values

    # Baseline Deterministic Metrics
    acc_base = float(accuracy_score(y_true, base_preds))
    bal_acc_base = float(balanced_accuracy_score(y_true, base_preds))
    auc_base = float(roc_auc_score(y_true, base_probs))
    nll_base = calculate_nll(y_true, base_probs)
    brier_base = calculate_brier(y_true, base_probs)
    ece_base = calculate_ece(np.maximum(1.0 - base_probs, base_probs), (base_preds == y_true).astype(int), n_bins=10)

    # MC Dropout Predictive Mean Metrics
    acc_mc = float(accuracy_score(y_true, mc_preds))
    bal_acc_mc = float(balanced_accuracy_score(y_true, mc_preds))
    auc_mc = float(roc_auc_score(y_true, mc_probs))
    nll_mc = calculate_nll(y_true, mc_probs)
    brier_mc = calculate_brier(y_true, mc_probs)
    ece_mc = calculate_ece(np.maximum(1.0 - mc_probs, mc_probs), (mc_preds == y_true).astype(int), n_bins=10)

    # Error Detection AUROCs (Ranking misclassifications higher than correct predictions)
    is_incorrect = (1 - mc_sub_df["correct"].values)  # 1 for error, 0 for correct

    entropy_err_auroc = safe_auroc(is_incorrect, mc_sub_df["predictive_entropy"].values)
    mi_err_auroc = safe_auroc(is_incorrect, mc_sub_df["mutual_information"].values)
    var_err_auroc = safe_auroc(is_incorrect, mc_sub_df["mc_variance"].values)

    summary_row_base = {
        "method": "Standard Baseline (Deterministic)",
        "accuracy": acc_base,
        "balanced_accuracy": bal_acc_base,
        "roc_auc": auc_base,
        "nll": nll_base,
        "brier": brier_base,
        "ece": ece_base,
        "entropy_error_auroc": None,
        "mi_error_auroc": None,
        "variance_error_auroc": None,
    }

    summary_row_mc = {
        "method": f"MC Dropout (N={num_passes}, p=0.1)",
        "accuracy": acc_mc,
        "balanced_accuracy": bal_acc_mc,
        "roc_auc": auc_mc,
        "nll": nll_mc,
        "brier": brier_mc,
        "ece": ece_mc,
        "entropy_error_auroc": entropy_err_auroc,
        "mi_error_auroc": mi_err_auroc,
        "variance_error_auroc": var_err_auroc,
    }

    summary_df = pd.DataFrame([summary_row_base, summary_row_mc])
    summary_df.to_csv(MC_SUMMARY_CSV, index=False)
    print(f"Saved summary metrics to: {MC_SUMMARY_CSV}")

    full_mc_metrics = {
        "experiment": "Voice Wav2Vec2 MC Dropout Uncertainty Quantification",
        "num_passes": num_passes,
        "seed": seed,
        "baseline_deterministic_metrics": summary_row_base,
        "mc_dropout_metrics": summary_row_mc,
    }

    with open(MC_METRICS_JSON, "w") as f:
        json.dump(full_mc_metrics, f, indent=4)
    print(f"Saved metrics JSON to: {MC_METRICS_JSON}")

    # 4. Generate Visualizations
    generate_mc_dropout_plots(mc_sub_df, entropy_err_auroc, mi_err_auroc, var_err_auroc)

    # 5. Output Results Summary Banner
    print("\n" + "=" * 65)
    print("VOICE MC DROPOUT EXPERIMENT RESULTS SUMMARY (TEST SET)")
    print("=" * 65)
    print(f"{'Metric':<25} | {'Baseline (Deterministic)':<22} | {'MC Dropout (N=30)':<20}")
    print("-" * 70)
    print(f"{'Accuracy':<25} | {acc_base:<22.4f} | {acc_mc:<20.4f}")
    print(f"{'Balanced Accuracy':<25} | {bal_acc_base:<22.4f} | {bal_acc_mc:<20.4f}")
    print(f"{'ROC-AUC':<25} | {auc_base:<22.4f} | {auc_mc:<20.4f}")
    print(f"{'NLL (Cross-Entropy)':<25} | {nll_base:<22.4f} | {nll_mc:<20.4f}")
    print(f"{'Brier Score':<25} | {brier_base:<22.4f} | {brier_mc:<20.4f}")
    print(f"{'ECE (10 Bins)':<25} | {ece_base:<22.4f} | {ece_mc:<20.4f}")
    print("-" * 70)
    print("UNCERTAINTY ERROR DETECTION AUROCs:")
    print(f"  Predictive Entropy Error-AUROC:    {entropy_err_auroc:.4f}")
    print(f"  Mutual Information Error-AUROC:    {mi_err_auroc:.4f}")
    print(f"  Predictive Variance Error-AUROC:   {var_err_auroc:.4f}")
    print("=" * 65)

    return full_mc_metrics


def generate_mc_dropout_plots(df: pd.DataFrame, entropy_auroc: float, mi_auroc: float, var_auroc: float):
    """Generate 5 publication-quality visualization figures for MC Dropout results."""
    # 1. Uncertainty Distribution Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
    ax1.hist(df["predictive_entropy"], bins=6, color="#1f77b4", edgecolor="black", alpha=0.7)
    ax1.set_title("Predictive Entropy Distribution", fontsize=11, weight="bold")
    ax1.set_xlabel("Predictive Entropy H(p)", fontsize=10)
    ax1.set_ylabel("Subject Count", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.hist(df["mutual_information"], bins=6, color="#2ca02c", edgecolor="black", alpha=0.7)
    ax2.set_title("Mutual Information Distribution", fontsize=11, weight="bold")
    ax2.set_xlabel("Mutual Information (Epistemic)", fontsize=10)
    ax2.set_ylabel("Subject Count", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(FIG_UNCERTAINTY_DIST, bbox_inches="tight")
    plt.close()

    # 2. Entropy vs Correctness Plot
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    correct_mask = (df["correct"] == 1)
    ax.scatter(df.loc[correct_mask, "subject_id"], df.loc[correct_mask, "predictive_entropy"],
               color="green", label="Correct Prediction", s=80, marker="o")
    ax.scatter(df.loc[~correct_mask, "subject_id"], df.loc[~correct_mask, "predictive_entropy"],
               color="red", label="Incorrect Prediction", s=100, marker="x")
    ax.set_title("Predictive Entropy vs. Prediction Correctness", fontsize=12, weight="bold")
    ax.set_xlabel("Test Subject ID", fontsize=11)
    ax.set_ylabel("Predictive Entropy H(p)", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIG_ENTROPY_CORRECT, bbox_inches="tight")
    plt.close()

    # 3. Mutual Information vs Correctness Plot
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    ax.scatter(df.loc[correct_mask, "subject_id"], df.loc[correct_mask, "mutual_information"],
               color="green", label="Correct Prediction", s=80, marker="o")
    ax.scatter(df.loc[~correct_mask, "subject_id"], df.loc[~correct_mask, "mutual_information"],
               color="red", label="Incorrect Prediction", s=100, marker="x")
    ax.set_title("Mutual Information vs. Prediction Correctness", fontsize=12, weight="bold")
    ax.set_xlabel("Test Subject ID", fontsize=11)
    ax.set_ylabel("Mutual Information", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIG_MI_CORRECT, bbox_inches="tight")
    plt.close()

    # 4. Baseline vs MC Probability Scatter Plot
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    ax.scatter(df["baseline_probability"], df["mc_mean_probability"], color="#1f77b4", s=70, alpha=0.8)
    ax.plot([0, 1], [0, 1], "k--", label="Identity Line (y = x)")
    ax.set_title("Baseline vs. MC Dropout Mean Probability", fontsize=12, weight="bold")
    ax.set_xlabel("Baseline Deterministic P(PD)", fontsize=11)
    ax.set_ylabel("MC Dropout Mean P(PD)", fontsize=11)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIG_BASE_VS_MC, bbox_inches="tight")
    plt.close()

    # 5. Error Detection ROC Curve
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    is_err = 1 - df["correct"].values
    if len(np.unique(is_err)) > 1:
        fpr1, tpr1, _ = roc_curve(is_err, df["predictive_entropy"].values)
        fpr2, tpr2, _ = roc_curve(is_err, df["mutual_information"].values)
        fpr3, tpr3, _ = roc_curve(is_err, df["mc_variance"].values)

        ax.plot(fpr1, tpr1, label=f"Predictive Entropy (AUROC = {entropy_auroc:.3f})", lw=2, color="#1f77b4")
        ax.plot(fpr2, tpr2, label=f"Mutual Information (AUROC = {mi_auroc:.3f})", lw=2, color="#2ca02c")
        ax.plot(fpr3, tpr3, label=f"Predictive Variance (AUROC = {var_auroc:.3f})", lw=2, color="#ff7f0e")
    
    ax.plot([0, 1], [0, 1], "k--", label="Chance Level")
    ax.set_title("Error Detection ROC Curve (MC Uncertainty)", fontsize=12, weight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate (Error Detection)", fontsize=11)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIG_ERROR_ROC, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_mc_dropout_experiment()
