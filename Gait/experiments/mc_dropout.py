"""MC Dropout Uncertainty Analysis for Gait LOCO Cross-Validation.

Research Question:
    Does Monte Carlo (MC) Dropout capture epistemic uncertainty under domain/cohort shift,
    and how does its uncertainty quality compare against Softmax, Temperature Scaling, and EDL?

MC Dropout Protocol:
    - Performs N stochastic forward passes with Dropout layers enabled and BatchNorm frozen in eval mode.
    - Computes Predictive Entropy H(mean_p) (total uncertainty) and Mutual Information MI = H(mean_p) - E[H(p)] (epistemic uncertainty).
    - 3 LOCO folds: (Ga+Ju -> Si), (Ga+Si -> Ju), (Ju+Si -> Ga)
    - Subject-level evaluation via hierarchical aggregation: WINDOW -> RECORDING -> SUBJECT
    - Output directory: Gait/outputs/mc_dropout/

Leakage Guards & Safety:
    - BatchNorm running statistics remain frozen during stochastic sampling.
    - Test cohort subjects are completely held out.
    - Model selection and normalization statistics use train/val cohorts only.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from Gait import config
from Gait.experiments.calibration_baselines import calculate_aurc, calculate_ece
from Gait.experiments.leave_one_cohort_out import (
    _safe_auroc,
    _safe_brier,
    _safe_nll,
    build_cohort_splits,
    build_dataloaders,
    build_manifest,
    fit_normalizer_on_train,
    json_default,
    run_safety_checks,
    set_seed,
    train_loco_model,
)
from Gait.model import GaitCNNBiLSTM

# ---------------------------------------------------------------------------
# Paths and Configurations
# ---------------------------------------------------------------------------
SEED = 42
DEFAULT_MC_SAMPLES = 30

_THIS_FILE = Path(__file__).resolve()
_EXPERIMENTS_DIR = _THIS_FILE.parent          # Gait/experiments/
_GAIT_DIR = _EXPERIMENTS_DIR.parent           # Gait/
_ROOT_DIR = _GAIT_DIR.parent                  # repo root

OUTPUT_DIR = _GAIT_DIR / "outputs" / "mc_dropout"
MC_SUBJECTS_CSV = OUTPUT_DIR / "mc_dropout_subject_predictions.csv"
MC_RESULTS_CSV = OUTPUT_DIR / "mc_dropout_results.csv"
MC_SUMMARY_JSON = OUTPUT_DIR / "mc_dropout_results.json"

PLOT_RELIABILITY_MC = OUTPUT_DIR / "reliability_mc_dropout.png"
PLOT_RISK_COVERAGE_MC = OUTPUT_DIR / "risk_coverage_mc_dropout.png"
PLOT_ERROR_DETECTION_MC = OUTPUT_DIR / "uncertainty_error_detection_mc_dropout.png"


# ===========================================================================
# STOCHASTIC MC DROPOUT UTILITIES
# ===========================================================================

def enable_mc_dropout(model: nn.Module) -> None:
    """
    Activates Dropout layers for MC Dropout sampling while keeping
    all other stateful layers (e.g. BatchNorm) strictly in evaluation mode.
    """
    model.eval()  # Freeze all BatchNorm running stats & evaluation mode
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.LSTM)):
            m.train()


def compute_window_entropy_metrics(mc_probs: np.ndarray, eps: float = 1e-12) -> Tuple[float, float, float]:
    """
    Calculate (predictive_entropy, expected_entropy, mutual_information) for a single window.
    mc_probs: shape (N, 2) where N is number of MC samples.
    """
    mc_probs = np.clip(mc_probs, eps, 1.0 - eps)
    mean_p = mc_probs.mean(axis=0)  # (2,)

    # 1. Predictive Entropy H(mean_p)
    pred_entropy = float(-np.sum(mean_p * np.log(mean_p)))

    # 2. Expected Entropy E[H(p)]
    sample_entropies = -np.sum(mc_probs * np.log(mc_probs), axis=1)
    exp_entropy = float(np.mean(sample_entropies))

    # 3. Mutual Information MI = H(mean_p) - E[H(p)]
    mi = max(0.0, float(pred_entropy - exp_entropy))

    return pred_entropy, exp_entropy, mi


# ===========================================================================
# INFERENCE & AGGREGATION
# ===========================================================================

@torch.no_grad()
def run_mc_dropout_inference(
    model: GaitCNNBiLSTM,
    test_loader: DataLoader,
    device: torch.device,
    mc_samples: int,
    exp_name: str,
    train_cohorts: List[str],
    test_cohort: str,
) -> pd.DataFrame:
    """
    Run N stochastic MC Dropout forward passes for each test window.
    Returns window-level predictions DataFrame.
    """
    enable_mc_dropout(model)
    window_records = []

    for batch in test_loader:
        x = batch["features"].to(device)
        y = batch["labels"].to(device)
        batch_size = len(y)

        # N stochastic forward passes
        batch_mc_probs = []
        for _ in range(mc_samples):
            _, emb = model(x, return_embedding=True)
            _, logits = model.evidential_head(emb, return_raw_logits=True)
            probs = torch.softmax(logits, dim=1).cpu().numpy()  # (B, 2)
            batch_mc_probs.append(probs)

        # shape: (N, B, 2)
        batch_mc_probs = np.stack(batch_mc_probs, axis=0)
        y_np = y.cpu().numpy()

        for i in range(batch_size):
            window_mc = batch_mc_probs[:, i, :]  # (N, 2)
            mean_p = window_mc.mean(axis=0)       # (2,)
            pred_ent, exp_ent, mi = compute_window_entropy_metrics(window_mc)

            window_records.append({
                "experiment": exp_name,
                "train_cohorts": "_".join(train_cohorts),
                "test_cohort": test_cohort,
                "subject_id": batch["subject_ids"][i],
                "recording_id": batch["recording_ids"][i],
                "study": batch["studies"][i],
                "true_label": int(y_np[i]),
                "mc_P_HC": float(mean_p[0]),
                "mc_P_PD": float(mean_p[1]),
                "predictive_entropy": float(pred_ent),
                "expected_entropy": float(exp_ent),
                "mutual_information": float(mi),
            })

    return pd.DataFrame(window_records)


def aggregate_mc_dropout_subject_level(window_df: pd.DataFrame) -> pd.DataFrame:
    """
    Hierarchical aggregation: WINDOW -> RECORDING -> SUBJECT.

    Computes subject-level predicted probabilities, predictive entropy,
    expected entropy, and mutual information.
    """
    rec_group_keys = [
        "experiment", "train_cohorts", "test_cohort",
        "subject_id", "recording_id", "study", "true_label"
    ]
    sub_group_keys = [
        "experiment", "train_cohorts", "test_cohort",
        "subject_id", "study", "true_label"
    ]
    num_cols = ["mc_P_HC", "mc_P_PD", "predictive_entropy", "expected_entropy", "mutual_information"]

    # Step 1: WINDOW -> RECORDING
    rec_df = window_df.groupby(rec_group_keys)[num_cols].mean().reset_index()

    # Step 2: RECORDING -> SUBJECT
    sub_df = rec_df.groupby(sub_group_keys)[num_cols].mean().reset_index()

    # Step 3: Compute Subject-level Metrics & Predictions
    sub_df["predicted_probability"] = sub_df["mc_P_PD"]
    sub_df["confidence"] = np.maximum(sub_df["mc_P_HC"], sub_df["mc_P_PD"])
    sub_df["prediction"] = (sub_df["predicted_probability"] >= 0.5).astype(int)
    sub_df["is_correct"] = (sub_df["prediction"] == sub_df["true_label"]).astype(int)

    return sub_df


def calculate_mc_dropout_metrics(sub_df: pd.DataFrame, mc_samples: int) -> Dict:
    """
    Calculate complete classification, calibration, error detection, and selective prediction metrics.
    """
    y_true = sub_df["true_label"].values
    pred = sub_df["prediction"].values
    prob = sub_df["predicted_probability"].values
    conf = sub_df["confidence"].values
    is_correct = sub_df["is_correct"].values
    is_incorrect = 1 - is_correct

    pred_ent = sub_df["predictive_entropy"].values
    exp_ent = sub_df["expected_entropy"].values
    mi = sub_df["mutual_information"].values

    acc = float(accuracy_score(y_true, pred))
    bal_acc = float(balanced_accuracy_score(y_true, pred))
    sens = float(recall_score(y_true, pred, pos_label=1, zero_division=0))
    spec = float(recall_score(y_true, pred, pos_label=0, zero_division=0))
    f1 = float(f1_score(y_true, pred, zero_division=0))

    roc_auc = _safe_auroc(y_true, prob)
    nll = _safe_nll(y_true, prob)
    brier = _safe_brier(y_true, prob)
    ece = calculate_ece(conf, is_correct, n_bins=10)

    # Error Detection AUROCs
    err_auroc_ent = _safe_auroc(is_incorrect, pred_ent)
    err_auroc_mi = _safe_auroc(is_incorrect, mi)

    # Risk-Coverage AURCs
    aurc_ent = calculate_aurc(pred_ent, is_correct)
    aurc_mi = calculate_aurc(mi, is_correct)

    # Means
    m_ent_corr = float(pred_ent[is_correct == 1].mean()) if (is_correct == 1).sum() > 0 else np.nan
    m_ent_inc = float(pred_ent[is_correct == 0].mean()) if (is_correct == 0).sum() > 0 else np.nan
    m_mi_corr = float(mi[is_correct == 1].mean()) if (is_correct == 1).sum() > 0 else np.nan
    m_mi_inc = float(mi[is_correct == 0].mean()) if (is_correct == 0).sum() > 0 else np.nan

    return {
        "mc_samples": mc_samples,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "sensitivity": sens,
        "specificity": spec,
        "f1": f1,
        "roc_auc": roc_auc,
        "nll": nll,
        "brier": brier,
        "ece": ece,
        "error_auroc_entropy": err_auroc_ent,
        "error_auroc_mutual_information": err_auroc_mi,
        "aurc_entropy": aurc_ent,
        "aurc_mutual_information": aurc_mi,
        "mean_entropy_correct": m_ent_corr,
        "mean_entropy_incorrect": m_ent_inc,
        "mean_mi_correct": m_mi_corr,
        "mean_mi_incorrect": m_mi_inc,
    }


# ===========================================================================
# PLOTTING FUNCTIONS
# ===========================================================================

def plot_mc_reliability_diagram(sub_df: pd.DataFrame, save_path: Path, n_bins: int = 10) -> None:
    """Plot calibration curve (Reliability Diagram) for MC Dropout."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    confidences = sub_df["confidence"].values
    is_correct = sub_df["is_correct"].values

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_means, bin_accs = [], []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        if mask.sum() == 0:
            continue
        bin_means.append(float(confidences[mask].mean()))
        bin_accs.append(float(is_correct[mask].mean()))

    if not bin_means:
        plt.close()
        return

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    ax.plot(bin_means, bin_accs, "o-", color="#9b59b6", lw=2, markersize=7, label="MC Dropout (LOCO pooled)")

    ax.set_title("Reliability Diagram — MC Dropout Baseline\n(Pooled across LOCO folds)", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Mean Confidence", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot -> {save_path}")


def plot_mc_risk_coverage_curve(sub_df: pd.DataFrame, save_path: Path) -> None:
    """Plot Risk-Coverage curve comparing Predictive Entropy vs Mutual Information."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    n = len(sub_df)
    is_correct = sub_df["is_correct"].values

    for unc_col, color, label_prefix in [
        ("mutual_information", "#9b59b6", "Mutual Information (Epistemic)"),
        ("predictive_entropy", "#e67e22", "Predictive Entropy (Total)"),
    ]:
        unc = sub_df[unc_col].values
        order = np.argsort(unc)
        sorted_error = 1.0 - is_correct[order]
        cum_errors = np.cumsum(sorted_error)
        coverages = np.arange(1, n + 1) / n
        risks = cum_errors / np.arange(1, n + 1)
        aurc = float(np.mean(risks))

        ax.plot(coverages, risks, color=color, lw=2.0, label=f"{label_prefix} (AURC={aurc:.4f})")

    ax.set_title("Risk-Coverage Curve — MC Dropout Baseline\n(Pooled across LOCO folds)", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Coverage (Fraction of least uncertain samples)", fontsize=10)
    ax.set_ylabel("Selective Risk (Error Rate)", fontsize=10)
    ax.set_xlim([0, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot -> {save_path}")


def plot_mc_error_detection_roc(sub_df: pd.DataFrame, save_path: Path) -> None:
    """Plot ROC curve for uncertainty-based error detection."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    is_incorrect = 1 - sub_df["is_correct"].values

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance (AUROC = 0.500)")

    for unc_col, color, label_name in [
        ("mutual_information", "#9b59b6", "Mutual Information"),
        ("predictive_entropy", "#e67e22", "Predictive Entropy"),
    ]:
        scores = sub_df[unc_col].values
        auroc = _safe_auroc(is_incorrect, scores)
        if auroc is not None:
            fpr, tpr, _ = roc_curve(is_incorrect, scores)
            ax.plot(fpr, tpr, color=color, lw=2.0, label=f"{label_name} (AUROC={auroc:.4f})")

    ax.set_title("Uncertainty-Based Error Detection ROC Curve\n(MC Dropout pooled across LOCO folds)", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot -> {save_path}")


# ===========================================================================
# EXPERIMENT EXECUTION PIPELINE
# ===========================================================================

def run_mc_dropout_experiment(epochs: int = config.EPOCHS, mc_samples: int = DEFAULT_MC_SAMPLES, seed: int = SEED) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n=======================================================")
    print(f"  Gait MC Dropout Uncertainty Analysis")
    print(f"  Device: {device} | Epochs per fold: {epochs} | MC Samples: {mc_samples} | Seed: {seed}")
    print(f"=======================================================\n")

    if not config.DATA_DIR.exists():
        print(f"\nERROR: Dataset directory not found.\n  Expected: {config.DATA_DIR}")
        return

    manifest_df = build_manifest(config.DATA_DIR)

    LOCO_SCENARIOS = [
        {"train": ["Ga", "Ju"], "test": "Si", "name": "Exp1_GaJu_to_Si"},
        {"train": ["Ga", "Si"], "test": "Ju", "name": "Exp2_GaSi_to_Ju"},
        {"train": ["Ju", "Si"], "test": "Ga", "name": "Exp3_JuSi_to_Ga"},
    ]

    all_subject_dfs = []
    summary_results = []

    for idx, sc in enumerate(LOCO_SCENARIOS, 1):
        train_cohorts = sc["train"]
        test_cohort = sc["test"]
        exp_name = sc["name"]

        print(f"\n{'='*65}")
        print(f"LOCO MC Dropout Fold {idx}/3: Train={'+'.join(train_cohorts)} -> Test={test_cohort}")
        print(f"{'='*65}")

        # 1. Subject-level split
        train_df, val_df, test_df = build_cohort_splits(
            manifest_df, train_cohorts, test_cohort, val_fraction=config.VAL_SPLIT_RATIO, seed=seed
        )

        # 2. Normalization strictly on train
        normalizer = fit_normalizer_on_train(train_df)

        # 3. Dataloaders
        train_loader, val_loader, test_loader = build_dataloaders(
            train_df, val_df, test_df, normalizer, batch_size=config.BATCH_SIZE
        )

        # 4. Train model
        model = train_loco_model(
            train_loader, val_loader, device, epochs=epochs, seed=seed, exp_label=exp_name
        )

        # 5. Safety checks
        run_safety_checks(train_df, val_df, test_df, normalizer, train_cohorts, test_cohort)

        # 6. Run stochastic MC Dropout inference
        window_df = run_mc_dropout_inference(
            model, test_loader, device, mc_samples, exp_name, train_cohorts, test_cohort
        )

        # 7. Aggregate to subject level
        sub_df = aggregate_mc_dropout_subject_level(window_df)
        all_subject_dfs.append(sub_df)

        # 8. Compute metrics
        met = calculate_mc_dropout_metrics(sub_df, mc_samples)

        row = {
            "experiment": exp_name,
            "test_cohort": test_cohort,
            "method": "mc_dropout",
            **met
        }
        summary_results.append(row)

        print(f"\n  Results for Fold {exp_name} (Test: {test_cohort}):")
        print(f"    MC Dropout (N={mc_samples}): BalAcc={met['balanced_accuracy']:.4f}, ROC-AUC={met['roc_auc']:.4f}, NLL={met['nll']:.4f}, ECE={met['ece']:.4f}, ErrAUC(MI)={met['error_auroc_mutual_information']:.4f}, ErrAUC(Ent)={met['error_auroc_entropy']:.4f}")

    # Combine all subject predictions
    all_sub_df = pd.concat(all_subject_dfs, ignore_index=True)
    summary_df = pd.DataFrame(summary_results)

    # Compute Macro Averages
    macro_row = {
        "experiment": "MACRO_AVERAGE",
        "test_cohort": "ALL",
        "method": "mc_dropout",
        "mc_samples": mc_samples,
        "accuracy": float(summary_df["accuracy"].mean()),
        "balanced_accuracy": float(summary_df["balanced_accuracy"].mean()),
        "sensitivity": float(summary_df["sensitivity"].mean()),
        "specificity": float(summary_df["specificity"].mean()),
        "f1": float(summary_df["f1"].mean()),
        "roc_auc": float(summary_df["roc_auc"].mean()) if not summary_df["roc_auc"].isna().any() else None,
        "nll": float(summary_df["nll"].mean()),
        "brier": float(summary_df["brier"].mean()),
        "ece": float(summary_df["ece"].mean()),
        "error_auroc_entropy": float(summary_df["error_auroc_entropy"].mean()),
        "error_auroc_mutual_information": float(summary_df["error_auroc_mutual_information"].mean()),
        "aurc_entropy": float(summary_df["aurc_entropy"].mean()),
        "aurc_mutual_information": float(summary_df["aurc_mutual_information"].mean()),
        "mean_entropy_correct": float(summary_df["mean_entropy_correct"].mean()),
        "mean_entropy_incorrect": float(summary_df["mean_entropy_incorrect"].mean()),
        "mean_mi_correct": float(summary_df["mean_mi_correct"].mean()),
        "mean_mi_incorrect": float(summary_df["mean_mi_incorrect"].mean()),
    }

    final_summary_df = pd.concat([summary_df, pd.DataFrame([macro_row])], ignore_index=True)

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Prepare clean subject predictions columns
    pred_export_cols = [
        "experiment", "test_cohort", "subject_id", "true_label",
        "predicted_probability", "prediction", "confidence",
        "predictive_entropy", "expected_entropy", "mutual_information",
        "is_correct"
    ]
    all_sub_df[pred_export_cols].to_csv(MC_SUBJECTS_CSV, index=False)
    final_summary_df.to_csv(MC_RESULTS_CSV, index=False)
    print(f"\nSaved subject predictions -> {MC_SUBJECTS_CSV}")
    print(f"Saved MC Dropout results   -> {MC_RESULTS_CSV}")

    # Save JSON summary
    summary_dict = {
        "experiment": "Gait MC Dropout Uncertainty Analysis",
        "description": "Monte Carlo Dropout stochastic uncertainty analysis across LOCO folds.",
        "seed": seed,
        "epochs": epochs,
        "mc_samples": mc_samples,
        "results": summary_results,
        "macro_average": macro_row,
    }
    with open(MC_SUMMARY_JSON, "w") as fh:
        json.dump(summary_dict, fh, indent=4, default=json_default)
    print(f"Saved summary JSON          -> {MC_SUMMARY_JSON}")

    # Generate Plots
    print("\nGenerating MC Dropout calibration and risk-coverage plots...")
    plot_mc_reliability_diagram(all_sub_df, PLOT_RELIABILITY_MC)
    plot_mc_risk_coverage_curve(all_sub_df, PLOT_RISK_COVERAGE_MC)
    plot_mc_error_detection_roc(all_sub_df, PLOT_ERROR_DETECTION_MC)

    print("\n=======================================================")
    print("MC DROPOUT EXPERIMENT COMPLETE!")
    print("=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gait MC Dropout Analysis")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs per fold")
    parser.add_argument("--mc-samples", type=int, default=DEFAULT_MC_SAMPLES, help="Number of MC Dropout forward passes")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility")
    args = parser.parse_args()

    run_mc_dropout_experiment(epochs=args.epochs, mc_samples=args.mc_samples, seed=args.seed)
