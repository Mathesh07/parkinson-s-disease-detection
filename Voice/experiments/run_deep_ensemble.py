"""Phase 4 Experiment: Deep Ensemble Uncertainty Quantification for Voice Wav2Vec2.

Evaluates M=5 independently trained ensemble members on 6 held-out test subjects,
calculating predictive mean, variance, standard deviation, predictive entropy, expected entropy,
mutual information, member agreement (m/5), and uncertainty error detection AUROCs.

Compares Baseline vs MC Dropout vs Deep Ensemble.

Leakage Guards:
    - Model weights remain 100% frozen under torch.no_grad().
    - Zero test labels used during inference or uncertainty estimation.
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
from Voice.deep_ensemble import compute_ensemble_statistics
from Voice.model import Wav2Vec2ForParkinsons
from Voice.utils import set_seed

ENSEMBLE_DIR = config.RESULTS_DIR / "deep_ensemble"
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

ENSEMBLE_CONFIG_JSON = ENSEMBLE_DIR / "deep_ensemble_config.json"
ENSEMBLE_SUBJECT_CSV = ENSEMBLE_DIR / "deep_ensemble_subject_predictions.csv"
ENSEMBLE_METRICS_JSON = ENSEMBLE_DIR / "deep_ensemble_metrics.json"
ENSEMBLE_SUMMARY_CSV = ENSEMBLE_DIR / "deep_ensemble_summary.csv"

FIG_PROB_DIST = ENSEMBLE_DIR / "ensemble_probability_distribution.png"
FIG_MEMBER_AGREE = ENSEMBLE_DIR / "member_probability_agreement.png"
FIG_UNCERTAINTY_DIST = ENSEMBLE_DIR / "ensemble_uncertainty_distribution.png"
FIG_ENTROPY_CORRECT = ENSEMBLE_DIR / "entropy_vs_correctness.png"
FIG_MI_CORRECT = ENSEMBLE_DIR / "mutual_information_vs_correctness.png"
FIG_BASE_VS_ENSEMBLE = ENSEMBLE_DIR / "baseline_vs_ensemble_probability.png"
FIG_ERROR_ROC = ENSEMBLE_DIR / "uncertainty_error_detection_roc.png"
FIG_MEMBER_PERF = ENSEMBLE_DIR / "member_performance_comparison.png"


def safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Safely calculate ROC-AUC score handling single-class edge cases."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return 0.5


def run_deep_ensemble_evaluation(seeds: List[int] = [42, 43, 44, 45, 46], split_seed: int = 42):
    """Run Phase 4 Deep Ensemble evaluation on held-out test subjects."""
    print("=" * 65)
    print("VOICE PHASE 4: DEEP ENSEMBLE EVALUATION EXPERIMENT")
    print(f"Num Members M = {len(seeds)} | Member Seeds = {seeds}")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data Splits and Metadata
    meta_csv = config.RESULTS_DIR / "embeddings" / "voice_embeddings_metadata.csv"
    meta_df = pd.read_csv(meta_csv)
    _, _, test_df = create_subject_splits(meta_df, seed=split_seed)

    test_subjects = sorted(test_df["subject_id"].unique().tolist())
    print(f"\nHeld-Out Test Subjects ({len(test_subjects)}): {test_subjects}\n")

    # Load 768-D embeddings and baseline predictions
    embs_file = config.RESULTS_DIR / "embeddings" / "voice_embeddings.pt"
    embs_dict = torch.load(embs_file)
    sub_embeddings = embs_dict["subject_embeddings"]

    sub_pred_csv = config.RESULTS_DIR / "subject_predictions.csv"
    sub_pred_df = pd.read_csv(sub_pred_csv)

    # Load each ensemble member model
    member_models = []
    member_val_metrics = []

    for idx, seed in enumerate(seeds):
        ckpt_dir = ENSEMBLE_DIR / f"member_{idx}" / "checkpoint"
        weights_path = ckpt_dir / "best_model.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing ensemble member checkpoint at: {weights_path}")

        model = Wav2Vec2ForParkinsons(
            model_name=config.MODEL_NAME,
            num_classes=config.NUM_CLASSES,
            freeze_feature_encoder=config.FREEZE_FEATURE_ENCODER
        )
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        member_models.append(model)

        # Load member training metrics
        meta_json = ENSEMBLE_DIR / f"member_{idx}" / "metrics.json"
        if meta_json.exists():
            with open(meta_json, "r") as f:
                member_val_metrics.append(json.load(f))
        else:
            member_val_metrics.append({"seed": seed})

    # 2. Perform Inference for each member on test subjects
    member_subject_probs = {idx: {} for idx in range(len(seeds))}
    
    with torch.no_grad():
        for sub_id in test_subjects:
            emb = sub_embeddings[sub_id].unsqueeze(0).to(device)  # Shape: (1, 768)
            for idx, model in enumerate(member_models):
                evidence = model(emb)
                probs = torch.softmax(evidence, dim=-1)
                p_pd = float(probs[0, 1].cpu().item())
                member_subject_probs[idx][sub_id] = p_pd

    # 3. Aggregate Ensemble Statistics per Subject
    subject_records = []

    for sub_id in test_subjects:
        row = sub_pred_df[sub_pred_df["subject_id"] == sub_id].iloc[0]
        true_label = int(row["true_label"])
        base_p_pd = float(row["pd_probability"])

        m_probs = [member_subject_probs[idx][sub_id] for idx in range(len(seeds))]
        stats = compute_ensemble_statistics(m_probs)

        mean_p = stats["ensemble_mean_probability"]
        ens_pred = 1 if mean_p >= 0.5 else 0
        is_correct = int(ens_pred == true_label)

        rec = {
            "subject_id": sub_id,
            "true_label": true_label,
            "member_0_probability": m_probs[0],
            "member_1_probability": m_probs[1],
            "member_2_probability": m_probs[2],
            "member_3_probability": m_probs[3],
            "member_4_probability": m_probs[4],
            "ensemble_mean_probability": mean_p,
            "ensemble_variance": stats["ensemble_variance"],
            "ensemble_std": stats["ensemble_std"],
            "predictive_entropy": stats["predictive_entropy"],
            "expected_entropy": stats["expected_entropy"],
            "mutual_information": stats["mutual_information"],
            "baseline_probability": base_p_pd,
            "ensemble_prediction": ens_pred,
            "member_agreement": stats["member_agreement"],
            "prediction_agreement": stats["prediction_agreement"],
            "correct": is_correct
        }
        subject_records.append(rec)

        print(
            f"Subject {sub_id} | True: {true_label} | M0..M4: [{m_probs[0]:.3f}, {m_probs[1]:.3f}, {m_probs[2]:.3f}, {m_probs[3]:.3f}, {m_probs[4]:.3f}] | "
            f"Mean P: {mean_p:.4f} | Var: {stats['ensemble_variance']:.6f} | Ent: {stats['predictive_entropy']:.4f} | MI: {stats['mutual_information']:.6f} | "
            f"Agree: {stats['member_agreement']}"
        )

    ens_sub_df = pd.DataFrame(subject_records)
    ens_sub_df.to_csv(ENSEMBLE_SUBJECT_CSV, index=False)
    print(f"\nSaved subject predictions CSV to: {ENSEMBLE_SUBJECT_CSV}")

    # 4. Compute Method Comparison Metrics: Baseline vs MC Dropout vs Deep Ensemble
    y_true = ens_sub_df["true_label"].values

    # Baseline Deterministic Metrics
    base_probs = ens_sub_df["baseline_probability"].values
    base_preds = (base_probs >= 0.5).astype(int)

    acc_base = float(accuracy_score(y_true, base_preds))
    bal_acc_base = float(balanced_accuracy_score(y_true, base_preds))
    auc_base = float(roc_auc_score(y_true, base_probs))
    nll_base = calculate_nll(y_true, base_probs)
    brier_base = calculate_brier(y_true, base_probs)
    ece_base = calculate_ece(np.maximum(1.0 - base_probs, base_probs), (base_preds == y_true).astype(int), n_bins=10)

    # MC Dropout Metrics (Load from MC Dropout Phase 3 results)
    mc_metrics_json = config.RESULTS_DIR / "mc_dropout" / "mc_dropout_metrics.json"
    if mc_metrics_json.exists():
        with open(mc_metrics_json, "r") as f:
            mc_data = json.load(f)
            mc_row_dict = mc_data.get("mc_dropout_metrics", {})
            acc_mc = mc_row_dict.get("accuracy", acc_base)
            bal_acc_mc = mc_row_dict.get("balanced_accuracy", bal_acc_base)
            auc_mc = mc_row_dict.get("roc_auc", auc_base)
            nll_mc = mc_row_dict.get("nll", nll_base)
            brier_mc = mc_row_dict.get("brier", brier_base)
            ece_mc = mc_row_dict.get("ece", ece_base)
            mc_entropy_err_auroc = mc_row_dict.get("entropy_error_auroc", 0.5)
            mc_mi_err_auroc = mc_row_dict.get("mi_error_auroc", 0.5)
            mc_var_err_auroc = mc_row_dict.get("variance_error_auroc", 0.5)
    else:
        acc_mc = bal_acc_mc = auc_mc = nll_mc = brier_mc = ece_mc = 0.0
        mc_entropy_err_auroc = mc_mi_err_auroc = mc_var_err_auroc = 0.5

    # Deep Ensemble Metrics
    ens_probs = ens_sub_df["ensemble_mean_probability"].values
    ens_preds = ens_sub_df["ensemble_prediction"].values

    acc_ens = float(accuracy_score(y_true, ens_preds))
    bal_acc_ens = float(balanced_accuracy_score(y_true, ens_preds))
    auc_ens = float(roc_auc_score(y_true, ens_probs))
    nll_ens = calculate_nll(y_true, ens_probs)
    brier_ens = calculate_brier(y_true, ens_probs)
    ece_ens = calculate_ece(np.maximum(1.0 - ens_probs, ens_probs), (ens_preds == y_true).astype(int), n_bins=10)

    # Uncertainty Error Detection AUROCs for Deep Ensemble
    is_incorrect = (1 - ens_sub_df["correct"].values)  # 1 for error, 0 for correct
    entropy_err_auroc = safe_auroc(is_incorrect, ens_sub_df["predictive_entropy"].values)
    mi_err_auroc = safe_auroc(is_incorrect, ens_sub_df["mutual_information"].values)
    var_err_auroc = safe_auroc(is_incorrect, ens_sub_df["ensemble_variance"].values)

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
        "variance_error_auroc": None
    }

    summary_row_mc = {
        "method": "MC Dropout (N=30, p=0.1)",
        "accuracy": acc_mc,
        "balanced_accuracy": bal_acc_mc,
        "roc_auc": auc_mc,
        "nll": nll_mc,
        "brier": brier_mc,
        "ece": ece_mc,
        "entropy_error_auroc": mc_entropy_err_auroc,
        "mi_error_auroc": mc_mi_err_auroc,
        "variance_error_auroc": mc_var_err_auroc
    }

    summary_row_ens = {
        "method": f"Deep Ensemble (M={len(seeds)})",
        "accuracy": acc_ens,
        "balanced_accuracy": bal_acc_ens,
        "roc_auc": auc_ens,
        "nll": nll_ens,
        "brier": brier_ens,
        "ece": ece_ens,
        "entropy_error_auroc": entropy_err_auroc,
        "mi_error_auroc": mi_err_auroc,
        "variance_error_auroc": var_err_auroc
    }

    summary_df = pd.DataFrame([summary_row_base, summary_row_mc, summary_row_ens])
    summary_df.to_csv(ENSEMBLE_SUMMARY_CSV, index=False)
    print(f"Saved summary metrics CSV to: {ENSEMBLE_SUMMARY_CSV}")

    full_metrics = {
        "experiment": "Voice Wav2Vec2 Deep Ensemble Uncertainty Quantification",
        "num_members": len(seeds),
        "seeds": seeds,
        "split_seed": split_seed,
        "baseline_deterministic_metrics": summary_row_base,
        "mc_dropout_metrics": summary_row_mc,
        "deep_ensemble_metrics": summary_row_ens,
        "individual_member_test_performance": [
            {
                "member": idx,
                "seed": seed,
                "accuracy": float(accuracy_score(y_true, (ens_sub_df[f"member_{idx}_probability"].values >= 0.5).astype(int))),
                "roc_auc": safe_auroc(y_true, ens_sub_df[f"member_{idx}_probability"].values),
                "nll": calculate_nll(y_true, ens_sub_df[f"member_{idx}_probability"].values),
                "brier": calculate_brier(y_true, ens_sub_df[f"member_{idx}_probability"].values)
            }
            for idx, seed in enumerate(seeds)
        ]
    }

    with open(ENSEMBLE_METRICS_JSON, "w") as f:
        json.dump(full_metrics, f, indent=4)
    print(f"Saved detailed metrics JSON to: {ENSEMBLE_METRICS_JSON}")

    # 5. Generate 8 Required Visualizations
    generate_ensemble_plots(ens_sub_df, seeds, entropy_err_auroc, mi_err_auroc, var_err_auroc, full_metrics)

    # 6. Display Comparison Summary Banner
    print("\n" + "=" * 80)
    print("VOICE DEEP ENSEMBLE COMPARISON SUMMARY (TEST SET = 6 SUBJECTS)")
    print("=" * 80)
    print(f"{'Method':<32} | {'Acc':<7} | {'BA':<7} | {'AUC':<7} | {'NLL':<8} | {'Brier':<7} | {'ECE':<7}")
    print("-" * 80)
    print(f"{'Standard Baseline (Deterministic)':<32} | {acc_base:<7.4f} | {bal_acc_base:<7.4f} | {auc_base:<7.4f} | {nll_base:<8.4f} | {brier_base:<7.4f} | {ece_base:<7.4f}")
    print(f"{'MC Dropout (N=30, p=0.1)':<32} | {acc_mc:<7.4f} | {bal_acc_mc:<7.4f} | {auc_mc:<7.4f} | {nll_mc:<8.4f} | {brier_mc:<7.4f} | {ece_mc:<7.4f}")
    print(f"{'Deep Ensemble (M=5)':<32} | {acc_ens:<7.4f} | {bal_acc_ens:<7.4f} | {auc_ens:<7.4f} | {nll_ens:<8.4f} | {brier_ens:<7.4f} | {ece_ens:<7.4f}")
    print("-" * 80)
    print("DEEP ENSEMBLE UNCERTAINTY ERROR DETECTION AUROCs:")
    print(f"  Predictive Entropy Error-AUROC:    {entropy_err_auroc:.4f}")
    print(f"  Mutual Information Error-AUROC:    {mi_err_auroc:.4f}")
    print(f"  Predictive Variance Error-AUROC:   {var_err_auroc:.4f}")
    print("=" * 80)

    return full_metrics


def generate_ensemble_plots(df: pd.DataFrame, seeds: List[int], entropy_auroc: float, mi_auroc: float, var_auroc: float, metrics: dict):
    """Generate 8 publication-quality visualization figures for Deep Ensemble results."""
    
    # 1. ensemble_probability_distribution.png
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.hist(df["ensemble_mean_probability"], bins=6, color="#1f77b4", edgecolor="black", alpha=0.7)
    ax.set_title("Ensemble Mean Probability Distribution", fontsize=12, weight="bold")
    ax.set_xlabel("Ensemble Mean P(PD)", fontsize=11)
    ax.set_ylabel("Subject Count", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIG_PROB_DIST, bbox_inches="tight")
    plt.close()

    # 2. member_probability_agreement.png
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    x = np.arange(len(df))
    width = 0.15
    for idx in range(len(seeds)):
        ax.bar(x + idx * width, df[f"member_{idx}_probability"], width, label=f"Member {idx} (Seed {seeds[idx]})")
    ax.plot(x + width * 2, df["ensemble_mean_probability"], "k-o", lw=2, label="Ensemble Mean")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(df["subject_id"])
    ax.set_title("Member Probability Agreement per Subject", fontsize=12, weight="bold")
    ax.set_xlabel("Test Subject ID", fontsize=11)
    ax.set_ylabel("P(PD)", fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIG_MEMBER_AGREE, bbox_inches="tight")
    plt.close()

    # 3. ensemble_uncertainty_distribution.png
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

    # 4. entropy_vs_correctness.png
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    correct_mask = (df["correct"] == 1)
    ax.scatter(df.loc[correct_mask, "subject_id"], df.loc[correct_mask, "predictive_entropy"],
               color="green", label="Correct Prediction", s=80, marker="o")
    ax.scatter(df.loc[~correct_mask, "subject_id"], df.loc[~correct_mask, "predictive_entropy"],
               color="red", label="Incorrect Prediction", s=100, marker="x")
    ax.set_title("Ensemble Predictive Entropy vs. Correctness", fontsize=12, weight="bold")
    ax.set_xlabel("Test Subject ID", fontsize=11)
    ax.set_ylabel("Predictive Entropy H(p)", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIG_ENTROPY_CORRECT, bbox_inches="tight")
    plt.close()

    # 5. mutual_information_vs_correctness.png
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    ax.scatter(df.loc[correct_mask, "subject_id"], df.loc[correct_mask, "mutual_information"],
               color="green", label="Correct Prediction", s=80, marker="o")
    ax.scatter(df.loc[~correct_mask, "subject_id"], df.loc[~correct_mask, "mutual_information"],
               color="red", label="Incorrect Prediction", s=100, marker="x")
    ax.set_title("Mutual Information vs. Correctness", fontsize=12, weight="bold")
    ax.set_xlabel("Test Subject ID", fontsize=11)
    ax.set_ylabel("Mutual Information", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIG_MI_CORRECT, bbox_inches="tight")
    plt.close()

    # 6. baseline_vs_ensemble_probability.png
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    ax.scatter(df["baseline_probability"], df["ensemble_mean_probability"], color="#1f77b4", s=70, alpha=0.8)
    ax.plot([0, 1], [0, 1], "k--", label="Identity Line (y = x)")
    ax.set_title("Baseline vs. Deep Ensemble Mean Probability", fontsize=12, weight="bold")
    ax.set_xlabel("Baseline Deterministic P(PD)", fontsize=11)
    ax.set_ylabel("Deep Ensemble Mean P(PD)", fontsize=11)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIG_BASE_VS_ENSEMBLE, bbox_inches="tight")
    plt.close()

    # 7. uncertainty_error_detection_roc.png
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    is_err = 1 - df["correct"].values
    if len(np.unique(is_err)) > 1:
        fpr1, tpr1, _ = roc_curve(is_err, df["predictive_entropy"].values)
        fpr2, tpr2, _ = roc_curve(is_err, df["mutual_information"].values)
        fpr3, tpr3, _ = roc_curve(is_err, df["ensemble_variance"].values)

        ax.plot(fpr1, tpr1, label=f"Predictive Entropy (AUROC = {entropy_auroc:.3f})", lw=2, color="#1f77b4")
        ax.plot(fpr2, tpr2, label=f"Mutual Information (AUROC = {mi_auroc:.3f})", lw=2, color="#2ca02c")
        ax.plot(fpr3, tpr3, label=f"Ensemble Variance (AUROC = {var_auroc:.3f})", lw=2, color="#ff7f0e")
    
    ax.plot([0, 1], [0, 1], "k--", label="Chance Level")
    ax.set_title("Error Detection ROC Curve (Deep Ensemble)", fontsize=12, weight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate (Error Detection)", fontsize=11)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIG_ERROR_ROC, bbox_inches="tight")
    plt.close()

    # 8. member_performance_comparison.png
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    member_perf = metrics.get("individual_member_test_performance", [])
    m_indices = [f"M{p['member']} (s{p['seed']})" for p in member_perf]
    m_accs = [p["accuracy"] for p in member_perf]
    m_aucs = [p["roc_auc"] for p in member_perf]

    x = np.arange(len(m_indices))
    width = 0.35
    ax.bar(x - width/2, m_accs, width, label="Test Accuracy", color="#1f77b4", alpha=0.8)
    ax.bar(x + width/2, m_aucs, width, label="Test ROC-AUC", color="#2ca02c", alpha=0.8)
    ax.axhline(metrics["deep_ensemble_metrics"]["accuracy"], color="#1f77b4", linestyle="--", label="Ensemble Acc")
    ax.axhline(metrics["deep_ensemble_metrics"]["roc_auc"], color="#2ca02c", linestyle=":", label="Ensemble AUC")

    ax.set_xticks(x)
    ax.set_xticklabels(m_indices)
    ax.set_title("Individual Member Test Performance vs. Ensemble", fontsize=12, weight="bold")
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIG_MEMBER_PERF, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_deep_ensemble_evaluation()
