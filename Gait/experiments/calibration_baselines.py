"""Calibration & Uncertainty Baselines (Softmax Confidence & Temperature Scaling) for Gait LOCO Cross-Validation.

Research Question:
    How well calibrated are standard neural network probabilities (Softmax and Temperature Scaling)
    under cohort shift in gait-based Parkinson's detection, and can uncertainty (1 - confidence)
    accurately identify predictive errors?

Baselines Evaluated:
    1. Method A: Softmax Confidence (T = 1.0)
    2. Method B: Post-Hoc Temperature Scaling (T > 0 learned on validation set ONLY)

Evaluation Protocol:
    - 3 LOCO folds: (Ga+Ju -> Si), (Ga+Si -> Ju), (Ju+Si -> Ga)
    - Subject-level evaluation via hierarchical aggregation: WINDOW -> RECORDING -> SUBJECT
    - Metrics: Accuracy, Balanced Acc, Sensitivity, Specificity, F1, ROC-AUC, NLL, Brier, ECE, Error AUROC, AURC
    - Output directory: Gait/outputs/calibration/

Leakage Guards:
    - Temperature parameter T is fitted ONLY on validation subjects.
    - Test subjects are completely excluded during temperature optimization.
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
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from Gait import config
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
# Paths and Configuration
# ---------------------------------------------------------------------------
SEED = 42

_THIS_FILE = Path(__file__).resolve()
_EXPERIMENTS_DIR = _THIS_FILE.parent          # Gait/experiments/
_GAIT_DIR = _EXPERIMENTS_DIR.parent           # Gait/
_ROOT_DIR = _GAIT_DIR.parent                  # repo root

OUTPUT_DIR = _GAIT_DIR / "outputs" / "calibration"
CALIB_SUBJECTS_CSV = OUTPUT_DIR / "calibration_subject_predictions.csv"
CALIB_RESULTS_CSV = OUTPUT_DIR / "calibration_results.csv"
CALIB_SUMMARY_JSON = OUTPUT_DIR / "calibration_results.json"

PLOT_RELIABILITY_SOFTMAX = OUTPUT_DIR / "reliability_softmax.png"
PLOT_RELIABILITY_TEMP = OUTPUT_DIR / "reliability_temperature_scaled.png"
PLOT_RISK_COVERAGE_SOFTMAX = OUTPUT_DIR / "risk_coverage_softmax.png"
PLOT_RISK_COVERAGE_TEMP = OUTPUT_DIR / "risk_coverage_temperature_scaled.png"


# ===========================================================================
# CALIBRATION UTILITIES
# ===========================================================================

def fit_temperature(val_logits: torch.Tensor, val_labels: torch.Tensor) -> float:
    """
    Fit temperature parameter T > 0 strictly on validation logits and labels.

    Objective: Minimize Negative Log Likelihood (NLL) on validation logits.
    Returns:
        best_t: Learned temperature parameter T > 0.
    """
    val_logits = val_logits.float()
    val_labels = val_labels.long()

    def nll_func(t_val: float) -> float:
        t_val = max(1e-4, float(t_val))
        scaled_logits = val_logits / t_val
        log_probs = torch.log_softmax(scaled_logits, dim=1)
        loss = -log_probs[torch.arange(len(val_labels)), val_labels].mean().item()
        return loss

    res = minimize_scalar(nll_func, bounds=(0.01, 50.0), method="bounded")
    best_t = float(res.x)

    uncalibrated_nll = nll_func(1.0)
    calibrated_nll = nll_func(best_t)

    assert best_t > 0.0, f"ERROR: Invalid non-positive temperature {best_t}"
    assert calibrated_nll <= uncalibrated_nll + 1e-6, (
        f"ERROR: Temperature scaling increased validation NLL ({uncalibrated_nll:.6f} -> {calibrated_nll:.6f})"
    )
    return best_t


def calculate_ece(confidences: np.ndarray, is_correct: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    Handles confidence == 1.0 by including it in the final bin [0.9, 1.0].
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(confidences)
    if n_samples == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_size = in_bin.sum()
        if bin_size > 0:
            bin_acc = is_correct[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def calculate_aurc(uncertainties: np.ndarray, is_correct: np.ndarray) -> float:
    """
    Calculate Area Under the Risk-Coverage curve (AURC).
    Uncertainties are sorted ascending (lowest uncertainty / highest confidence first).
    Risk(theta) = mean error rate on top theta fraction of samples.
    """
    n = len(uncertainties)
    if n == 0:
        return 0.0

    order = np.argsort(uncertainties)
    sorted_correct = is_correct[order]
    sorted_error = 1.0 - sorted_correct

    cum_errors = np.cumsum(sorted_error)
    coverages = np.arange(1, n + 1)
    risks = cum_errors / coverages

    return float(np.mean(risks))


# ===========================================================================
# INFERENCE & AGGREGATION
# ===========================================================================

@torch.no_grad()
def extract_validation_logits(
    model: GaitCNNBiLSTM,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract validation logits and labels for temperature scaling.
    """
    model.eval()
    val_logits_list = []
    val_labels_list = []

    for batch in val_loader:
        x = batch["features"].to(device)
        y = batch["labels"].to(device)
        _, emb = model(x, return_embedding=True)
        _, logits = model.evidential_head(emb, return_raw_logits=True)
        val_logits_list.append(logits.cpu())
        val_labels_list.append(y.cpu())

    return torch.cat(val_logits_list, dim=0), torch.cat(val_labels_list, dim=0)


@torch.no_grad()
def run_calibration_inference(
    model: GaitCNNBiLSTM,
    test_loader: DataLoader,
    device: torch.device,
    learned_T: float,
    exp_name: str,
    train_cohorts: List[str],
    test_cohort: str,
) -> pd.DataFrame:
    """
    Run inference on test set, computing both Softmax (T=1.0) and Temperature Scaled probabilities.
    Outputs window-level predictions DataFrame.
    """
    model.eval()
    window_records = []

    for batch in test_loader:
        x = batch["features"].to(device)
        y = batch["labels"].to(device)
        _, emb = model(x, return_embedding=True)
        _, logits = model.evidential_head(emb, return_raw_logits=True)

        probs_softmax = torch.softmax(logits, dim=1).cpu().numpy()
        probs_temp = torch.softmax(logits / learned_T, dim=1).cpu().numpy()

        y_np = y.cpu().numpy()
        logits_np = logits.cpu().numpy()

        for i in range(len(y_np)):
            window_records.append({
                "experiment": exp_name,
                "train_cohorts": "_".join(train_cohorts),
                "test_cohort": test_cohort,
                "subject_id": batch["subject_ids"][i],
                "recording_id": batch["recording_ids"][i],
                "study": batch["studies"][i],
                "true_label": int(y_np[i]),
                "logit_HC": float(logits_np[i, 0]),
                "logit_PD": float(logits_np[i, 1]),
                "softmax_P_HC": float(probs_softmax[i, 0]),
                "softmax_P_PD": float(probs_softmax[i, 1]),
                "temp_P_HC": float(probs_temp[i, 0]),
                "temp_P_PD": float(probs_temp[i, 1]),
            })

    return pd.DataFrame(window_records)


def aggregate_calibration_subject_level(window_df: pd.DataFrame, learned_T: float) -> pd.DataFrame:
    """
    Hierarchical aggregation: WINDOW -> RECORDING -> SUBJECT.
    
    Computes subject-level confidence, uncertainty, and predictions for both
    Softmax and Temperature Scaling.
    """
    rec_group_keys = [
        "experiment", "train_cohorts", "test_cohort",
        "subject_id", "recording_id", "study", "true_label"
    ]
    sub_group_keys = [
        "experiment", "train_cohorts", "test_cohort",
        "subject_id", "study", "true_label"
    ]
    prob_cols = ["softmax_P_HC", "softmax_P_PD", "temp_P_HC", "temp_P_PD"]

    # Step 1: WINDOW -> RECORDING
    rec_df = window_df.groupby(rec_group_keys)[prob_cols].mean().reset_index()

    # Step 2: RECORDING -> SUBJECT
    sub_df = rec_df.groupby(sub_group_keys)[prob_cols].mean().reset_index()

    # Step 3: Compute Softmax & Temperature Scaled subject metrics
    # Softmax (T = 1.0)
    sub_df["softmax_probability"] = sub_df["softmax_P_PD"]
    sub_df["softmax_confidence"] = np.maximum(sub_df["softmax_P_HC"], sub_df["softmax_P_PD"])
    sub_df["softmax_uncertainty"] = 1.0 - sub_df["softmax_confidence"]
    sub_df["softmax_prediction"] = (sub_df["softmax_probability"] >= 0.5).astype(int)
    sub_df["softmax_is_correct"] = (sub_df["softmax_prediction"] == sub_df["true_label"]).astype(int)

    # Temperature Scaling (T = learned_T)
    sub_df["temperature"] = learned_T
    sub_df["temperature_probability"] = sub_df["temp_P_PD"]
    sub_df["temperature_confidence"] = np.maximum(sub_df["temp_P_HC"], sub_df["temp_P_PD"])
    sub_df["temperature_uncertainty"] = 1.0 - sub_df["temperature_confidence"]
    sub_df["temperature_prediction"] = (sub_df["temperature_probability"] >= 0.5).astype(int)
    sub_df["temperature_is_correct"] = (sub_df["temperature_prediction"] == sub_df["true_label"]).astype(int)

    return sub_df


def calculate_method_metrics(sub_df: pd.DataFrame, method: str) -> Dict:
    """
    Calculate classification, calibration, and uncertainty error detection metrics
    for a specific method ('softmax' or 'temperature').
    """
    y_true = sub_df["true_label"].values
    pred = sub_df[f"{method}_prediction"].values
    prob = sub_df[f"{method}_probability"].values
    conf = sub_df[f"{method}_confidence"].values
    unc = sub_df[f"{method}_uncertainty"].values
    is_correct = sub_df[f"{method}_is_correct"].values
    is_incorrect = 1 - is_correct

    acc = float(accuracy_score(y_true, pred))
    bal_acc = float(balanced_accuracy_score(y_true, pred))
    sens = float(recall_score(y_true, pred, pos_label=1, zero_division=0))
    spec = float(recall_score(y_true, pred, pos_label=0, zero_division=0))
    f1 = float(f1_score(y_true, pred, zero_division=0))

    roc_auc = _safe_auroc(y_true, prob)
    nll = _safe_nll(y_true, prob)
    brier = _safe_brier(y_true, prob)
    ece = calculate_ece(conf, is_correct, n_bins=10)

    u_correct = float(unc[is_correct == 1].mean()) if (is_correct == 1).sum() > 0 else np.nan
    u_incorrect = float(unc[is_correct == 0].mean()) if (is_correct == 0).sum() > 0 else np.nan
    err_auroc = _safe_auroc(is_incorrect, unc)
    aurc = calculate_aurc(unc, is_correct)

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "sensitivity": sens,
        "specificity": spec,
        "f1": f1,
        "roc_auc": roc_auc,
        "nll": nll,
        "brier": brier,
        "ece": ece,
        "mean_uncertainty_correct": u_correct,
        "mean_uncertainty_incorrect": u_incorrect,
        "uncertainty_error_auroc": err_auroc,
        "aurc": aurc,
    }


# ===========================================================================
# PLOTTING FUNCTIONS
# ===========================================================================

def plot_reliability_diagram(sub_df: pd.DataFrame, conf_col: str, correct_col: str, title: str, save_path: Path, n_bins: int = 10) -> None:
    """
    Plot calibration curve (Reliability Diagram) pooled across LOCO folds.
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    confidences = sub_df[conf_col].values
    is_correct = sub_df[correct_col].values

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
    ax.plot(bin_means, bin_accs, "o-", color="#1f77b4", lw=2, markersize=7, label="Model (LOCO pooled)")

    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
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


def plot_risk_coverage_curve(sub_df: pd.DataFrame, unc_col: str, correct_col: str, title: str, save_path: Path) -> None:
    """
    Plot Risk-Coverage curve pooled across LOCO folds.
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    unc = sub_df[unc_col].values
    is_correct = sub_df[correct_col].values
    n = len(unc)

    order = np.argsort(unc)
    sorted_correct = is_correct[order]
    sorted_error = 1.0 - sorted_correct

    cum_errors = np.cumsum(sorted_error)
    coverages = np.arange(1, n + 1) / n
    risks = cum_errors / np.arange(1, n + 1)

    aurc = float(np.mean(risks))

    ax.plot(coverages, risks, color="#e74c3c", lw=2.5, label=f"Risk-Coverage (AURC = {aurc:.4f})")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
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


# ===========================================================================
# EXPERIMENT EXECUTION PIPELINE
# ===========================================================================

def run_calibration_experiment(epochs: int = config.EPOCHS, seed: int = SEED) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n=======================================================")
    print(f"  Gait Calibration Baselines (Softmax & Temp Scaling)")
    print(f"  Device: {device} | Epochs per fold: {epochs} | Seed: {seed}")
    print(f"=======================================================\n")

    if not config.DATA_DIR.exists():
        print(f"\nERROR: Dataset directory not found.\n  Expected: {config.DATA_DIR}")
        sys.exit(1)
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
        print(f"LOCO Calibration Fold {idx}/3: Train={'+'.join(train_cohorts)} -> Test={test_cohort}")
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

        # 6. Fit Temperature Scaling strictly on validation logits
        val_logits, val_labels = extract_validation_logits(model, val_loader, device)
        learned_T = fit_temperature(val_logits, val_labels)
        print(f"  [{exp_name}] Learned Validation Temperature T = {learned_T:.4f}")

        # 7. Run test inference
        window_df = run_calibration_inference(
            model, test_loader, device, learned_T, exp_name, train_cohorts, test_cohort
        )

        # 8. Aggregate to subject level
        sub_df = aggregate_calibration_subject_level(window_df, learned_T)
        all_subject_dfs.append(sub_df)

        # 9. Compute metrics per method
        sm_met = calculate_method_metrics(sub_df, "softmax")
        ts_met = calculate_method_metrics(sub_df, "temperature")

        # Record summary rows
        row_sm = {
            "experiment": exp_name,
            "test_cohort": test_cohort,
            "method": "softmax",
            "temperature": 1.0,
            **sm_met
        }
        row_ts = {
            "experiment": exp_name,
            "test_cohort": test_cohort,
            "method": "temperature_scaling",
            "temperature": learned_T,
            **ts_met
        }

        summary_results.extend([row_sm, row_ts])

        print(f"\n  Results for Fold {exp_name} (Test: {test_cohort}):")
        print(f"    Softmax      (T=1.0000): BalAcc={sm_met['balanced_accuracy']:.4f}, ROC-AUC={sm_met['roc_auc']:.4f}, NLL={sm_met['nll']:.4f}, Brier={sm_met['brier']:.4f}, ECE={sm_met['ece']:.4f}, ErrAUC={sm_met['uncertainty_error_auroc']:.4f}")
        print(f"    Temp Scaling (T={learned_T:.4f}): BalAcc={ts_met['balanced_accuracy']:.4f}, ROC-AUC={ts_met['roc_auc']:.4f}, NLL={ts_met['nll']:.4f}, Brier={ts_met['brier']:.4f}, ECE={ts_met['ece']:.4f}, ErrAUC={ts_met['uncertainty_error_auroc']:.4f}")

    # Combine all subject predictions
    all_sub_df = pd.concat(all_subject_dfs, ignore_index=True)
    summary_df = pd.DataFrame(summary_results)

    # Compute Macro Averages
    macro_rows = []
    for method_name in ["softmax", "temperature_scaling"]:
        m_df = summary_df[summary_df["method"] == method_name]
        m_row = {
            "experiment": "MACRO_AVERAGE",
            "test_cohort": "ALL",
            "method": method_name,
            "temperature": float(m_df["temperature"].mean()),
            "accuracy": float(m_df["accuracy"].mean()),
            "balanced_accuracy": float(m_df["balanced_accuracy"].mean()),
            "sensitivity": float(m_df["sensitivity"].mean()),
            "specificity": float(m_df["specificity"].mean()),
            "f1": float(m_df["f1"].mean()),
            "roc_auc": float(m_df["roc_auc"].mean()) if not m_df["roc_auc"].isna().any() else None,
            "nll": float(m_df["nll"].mean()),
            "brier": float(m_df["brier"].mean()),
            "ece": float(m_df["ece"].mean()),
            "mean_uncertainty_correct": float(m_df["mean_uncertainty_correct"].mean()),
            "mean_uncertainty_incorrect": float(m_df["mean_uncertainty_incorrect"].mean()),
            "uncertainty_error_auroc": float(m_df["uncertainty_error_auroc"].mean()) if not m_df["uncertainty_error_auroc"].isna().any() else None,
            "aurc": float(m_df["aurc"].mean()),
        }
        macro_rows.append(m_row)

    final_summary_df = pd.concat([summary_df, pd.DataFrame(macro_rows)], ignore_index=True)

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_sub_df.to_csv(CALIB_SUBJECTS_CSV, index=False)
    final_summary_df.to_csv(CALIB_RESULTS_CSV, index=False)
    print(f"\nSaved subject predictions -> {CALIB_SUBJECTS_CSV}")
    print(f"Saved calibration results   -> {CALIB_RESULTS_CSV}")

    # Save JSON summary
    summary_dict = {
        "experiment": "Gait Calibration & Uncertainty Baselines",
        "description": "Softmax Confidence (T=1.0) vs Temperature Scaling (learned T > 0 on Val)",
        "seed": seed,
        "epochs": epochs,
        "results": summary_results,
        "macro_averages": macro_rows,
    }
    with open(CALIB_SUMMARY_JSON, "w") as fh:
        json.dump(summary_dict, fh, indent=4, default=json_default)
    print(f"Saved summary JSON          -> {CALIB_SUMMARY_JSON}")

    # Generate Plots
    print("\nGenerating calibration and risk-coverage plots...")
    plot_reliability_diagram(
        all_sub_df, "softmax_confidence", "softmax_is_correct",
        "Reliability Diagram — Softmax Baseline (T=1.0)\n(Pooled across LOCO folds)",
        PLOT_RELIABILITY_SOFTMAX
    )
    plot_reliability_diagram(
        all_sub_df, "temperature_confidence", "temperature_is_correct",
        "Reliability Diagram — Temperature Scaling\n(Pooled across LOCO folds)",
        PLOT_RELIABILITY_TEMP
    )
    plot_risk_coverage_curve(
        all_sub_df, "softmax_uncertainty", "softmax_is_correct",
        "Risk-Coverage Curve — Softmax Baseline (T=1.0)\n(Pooled across LOCO folds)",
        PLOT_RISK_COVERAGE_SOFTMAX
    )
    plot_risk_coverage_curve(
        all_sub_df, "temperature_uncertainty", "temperature_is_correct",
        "Risk-Coverage Curve — Temperature Scaling\n(Pooled across LOCO folds)",
        PLOT_RISK_COVERAGE_TEMP
    )

    print("\n=======================================================")
    print("CALIBRATION BASELINES EXPERIMENT COMPLETE!")
    print("=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gait Calibration Baselines")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs per fold")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility")
    args = parser.parse_args()

    run_calibration_experiment(epochs=args.epochs, seed=args.seed)
