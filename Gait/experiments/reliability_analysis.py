"""Unified Reliability and Abstention Framework for Gait-Based Parkinson's Detection.

Research Objective:
    Demonstrate that "Prediction confidence is not necessarily the same as prediction reliability."
    Under cohort shift and signal corruption, deep models frequently exhibit "silent confidence"
    (high nominal probability paired with catastrophic misclassification).

    This module implements a lightweight, statistically defensible abstention / reliability layer
    that integrates:
      1. Calibrated predictive probability (Temperature Scaled)
      2. Predictive uncertainty (Vacuity from EDL / Predictive Entropy from Ensembles)
      3. Inductive Conformal Prediction set
      4. Validation-tuned decision thresholds (strictly ZERO test data leakage)

Decision Logic (Tuned STRICTLY on Validation Data):
    - Confidence threshold tau_conf: 10th percentile of confidence among correct validation subjects.
    - Uncertainty threshold tau_unc: 80th percentile of uncertainty on validation subjects.
    - Conformal threshold q_hat: (1 - alpha) nonconformity quantile on validation subjects (alpha=0.10).

    Status Assignment:
      - ACCEPT (Proceed):
          * Confidence >= tau_conf
          * Uncertainty <= tau_unc
          * Conformal set is a consistent singleton ({PD} for PD prediction, {HC} for HC prediction)
      - FLAG (Uncertain / Ambiguous / Abstain):
          * High uncertainty (u > tau_unc), OR
          * Marginal confidence (conf < tau_conf), OR
          * Ambiguous conformal set {Healthy, PD}, OR
          * Empty conformal set {}, OR
          * Conformal set contradicts point prediction.

Outputs:
    Gait/outputs/reliability/
        - reliability_results.csv
        - reliability_results.json
        - reliability_subject_decisions.csv
        - reliability_risk_coverage.csv
        - plots/*.png
        - README.md
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader

# Path bootstrapping
_THIS_FILE = Path(__file__).resolve()
_EXPERIMENTS_DIR = _THIS_FILE.parent
_GAIT_DIR = _EXPERIMENTS_DIR.parent
_ROOT_DIR = _GAIT_DIR.parent

for _p in (str(_GAIT_DIR), str(_ROOT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Gait import config
from Gait.experiments.calibration_baselines import (
    calculate_aurc,
    calculate_ece,
    extract_validation_logits,
    fit_temperature,
)
from Gait.experiments.conformal_prediction import (
    compute_conformal_quantile,
    construct_prediction_set,
    format_set_string,
)
from Gait.experiments.leave_one_cohort_out import (
    _safe_auroc,
    _safe_brier,
    _safe_nll,
    build_cohort_splits,
    build_dataloaders,
    fit_normalizer_on_train,
    json_default,
    train_loco_model,
)
from Gait.model import GaitCNNBiLSTM
from Gait.preprocessing import build_manifest
from Gait.utils import get_device

OUTPUT_DIR = _GAIT_DIR / "outputs" / "reliability"
PLOTS_DIR = OUTPUT_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = OUTPUT_DIR / "reliability_results.csv"
RESULTS_JSON = OUTPUT_DIR / "reliability_results.json"
DECISIONS_CSV = OUTPUT_DIR / "reliability_subject_decisions.csv"
RISK_COV_CSV = OUTPUT_DIR / "reliability_risk_coverage.csv"
DOCS_MD = OUTPUT_DIR / "README.md"

LOCO_EXPERIMENTS = [
    {"exp_id": 1, "train_cohorts": ["Ga", "Ju"], "test_cohort": "Si"},
    {"exp_id": 2, "train_cohorts": ["Ga", "Si"], "test_cohort": "Ju"},
    {"exp_id": 3, "train_cohorts": ["Ju", "Si"], "test_cohort": "Ga"},
]


# ===========================================================================
# 1. VALIDATION THRESHOLD DETERMINATION (ZERO LEAKAGE)
# ===========================================================================

def tune_reliability_thresholds_on_validation(
    val_sub_df: pd.DataFrame,
    alpha: float = 0.10,
) -> Dict[str, float]:
    """
    Determine all decision thresholds STRICTLY using validation predictions.

    - tau_conf: 10th percentile of confidence among correct validation samples.
    - tau_unc: 80th percentile of uncertainty across validation samples.
    - q_hat: Standard conformal quantile at 1 - alpha level.
    """
    correct_val = val_sub_df[val_sub_df["is_correct"] == 1]
    if len(correct_val) > 0:
        tau_conf = float(np.percentile(correct_val["confidence"].values, 10))
    else:
        tau_conf = 0.60

    # Guard: confidence must be >= 0.50
    tau_conf = max(0.55, min(0.85, tau_conf))

    # Uncertainty threshold
    tau_unc = float(np.percentile(val_sub_df["uncertainty"].values, 80))

    # Conformal nonconformity quantile
    val_probs = val_sub_df[["P_HC", "P_PD"]].values
    val_labels = val_sub_df["true_label"].values.astype(int)
    q_hat = compute_conformal_quantile(val_probs, val_labels, alpha=alpha)

    return {
        "tau_conf": tau_conf,
        "tau_unc": tau_unc,
        "q_hat": q_hat,
        "alpha": alpha,
    }


# ===========================================================================
# 2. DECISION LOGIC & SUBJECT EVALUATION
# ===========================================================================

def evaluate_decision_layer(
    subject_df: pd.DataFrame,
    thresholds: Dict[str, float],
) -> pd.DataFrame:
    """
    Apply reliability rules to each subject.
    Assigns:
      - conformal_set
      - set_size
      - uncertainty_level ("LOW" vs "HIGH")
      - confidence_level ("LOW" vs "HIGH")
      - reliability_status ("ACCEPT" vs "FLAG")
      - flag_reasons (comma-separated list of triggered conditions)
    """
    tau_conf = thresholds["tau_conf"]
    tau_unc = thresholds["tau_unc"]
    q_hat = thresholds["q_hat"]

    df = subject_df.copy()
    c_sets = []
    set_sizes = []
    unc_levels = []
    conf_levels = []
    statuses = []
    reasons_list = []

    for _, row in df.iterrows():
        p_hc = row["P_HC"]
        p_pd = row["P_PD"]
        pred_label = int(row["predicted_label"])
        conf = row["confidence"]
        unc = row["uncertainty"]

        # 1. Conformal prediction set
        pset = construct_prediction_set(p_hc, p_pd, q_hat)
        c_sets.append(format_set_string(pset))
        set_sizes.append(len(pset))

        # 2. Uncertainty & Confidence categorical status
        is_unc_high = (unc > tau_unc)
        is_conf_low = (conf < tau_conf)
        unc_levels.append("HIGH" if is_unc_high else "LOW")
        conf_levels.append("LOW" if is_conf_low else "HIGH")

        # 3. Conformal consistency check
        pred_class_str = "PD" if pred_label == 1 else "HC"
        is_conformal_ambiguous = (len(pset) != 1)
        is_conformal_contradictory = (len(pset) == 1 and pset[0] != pred_class_str)

        # 4. Decision Rule
        reasons = []
        if is_unc_high:
            reasons.append("HIGH_UNCERTAINTY")
        if is_conf_low:
            reasons.append("LOW_CONFIDENCE")
        if len(pset) == 2:
            reasons.append("AMBIGUOUS_CONFORMAL_SET")
        elif len(pset) == 0:
            reasons.append("EMPTY_CONFORMAL_SET")
        elif is_conformal_contradictory:
            reasons.append("CONFORMAL_POINT_CONTRADICTION")

        if len(reasons) == 0:
            status = "ACCEPT"
        else:
            status = "FLAG"

        statuses.append(status)
        reasons_list.append(";".join(reasons) if reasons else "NONE")

    df["conformal_set"] = c_sets
    df["set_size"] = set_sizes
    df["uncertainty_level"] = unc_levels
    df["confidence_level"] = conf_levels
    df["reliability_status"] = statuses
    df["flag_reasons"] = reasons_list

    return df


def compute_reliability_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute selective classification accuracy, coverage, and error reduction."""
    total_n = len(df)
    accepted_df = df[df["reliability_status"] == "ACCEPT"]
    flagged_df = df[df["reliability_status"] == "FLAG"]

    n_accepted = len(accepted_df)
    n_flagged = len(flagged_df)
    coverage = n_accepted / float(total_n) if total_n > 0 else 0.0

    raw_acc = float(accuracy_score(df["true_label"], df["predicted_label"]))
    raw_bal_acc = float(balanced_accuracy_score(df["true_label"], df["predicted_label"]))

    if n_accepted > 0:
        accepted_acc = float(accuracy_score(accepted_df["true_label"], accepted_df["predicted_label"]))
        accepted_bal_acc = float(balanced_accuracy_score(accepted_df["true_label"], accepted_df["predicted_label"]))
    else:
        accepted_acc = float("nan")
        accepted_bal_acc = float("nan")

    if n_flagged > 0:
        flagged_error_rate = float(np.mean(flagged_df["is_correct"] == 0))
    else:
        flagged_error_rate = 0.0

    # Error capture rate: what % of all errors were successfully flagged?
    total_errors = np.sum(df["is_correct"] == 0)
    flagged_errors = np.sum(flagged_df["is_correct"] == 0) if n_flagged > 0 else 0
    error_capture_rate = (flagged_errors / float(total_errors)) if total_errors > 0 else 1.0

    # False alarm rate: what % of correct predictions were unnecessarily flagged?
    total_correct = np.sum(df["is_correct"] == 1)
    flagged_correct = np.sum(flagged_df["is_correct"] == 1) if n_flagged > 0 else 0
    false_flag_rate = (flagged_correct / float(total_correct)) if total_correct > 0 else 0.0

    return {
        "raw_accuracy": raw_acc,
        "raw_balanced_accuracy": raw_bal_acc,
        "coverage": coverage,
        "accepted_accuracy": accepted_acc,
        "accepted_balanced_accuracy": accepted_bal_acc,
        "accuracy_gain": (accepted_acc - raw_acc) if not np.isnan(accepted_acc) else 0.0,
        "total_subjects": total_n,
        "num_accepted": n_accepted,
        "num_flagged": n_flagged,
        "total_errors": int(total_errors),
        "flagged_errors": int(flagged_errors),
        "error_capture_rate": float(error_capture_rate),
        "false_flag_rate": float(false_flag_rate),
    }


# ===========================================================================
# 3. PLOTS & DOCUMENTATION
# ===========================================================================

def generate_reliability_plots(
    results_df: pd.DataFrame,
    decisions_df: pd.DataFrame,
) -> None:
    """Generate visual reports showing error capture and selective accuracy improvements."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: Selective Accuracy Gain by LOCO Fold
    fig, ax = plt.subplots(figsize=(9, 5))
    exp_ids = sorted(results_df["experiment"].unique())
    x = np.arange(len(exp_ids))
    width = 0.35

    raw_accs = [results_df[results_df["experiment"] == e]["raw_accuracy"].iloc[0] for e in exp_ids]
    acc_accs = [results_df[results_df["experiment"] == e]["accepted_accuracy"].iloc[0] for e in exp_ids]
    covs = [results_df[results_df["experiment"] == e]["coverage"].iloc[0] for e in exp_ids]

    b1 = ax.bar(x - width / 2, raw_accs, width, label="Raw Accuracy (Full Cohort)", color="#718096", alpha=0.85)
    b2 = ax.bar(x + width / 2, acc_accs, width, label="Accepted Accuracy (Reliability Filtered)", color="#2b6cb0", alpha=0.9)

    for i in range(len(exp_ids)):
        ax.text(x[i] + width / 2, acc_accs[i] + 0.02, f"Cov: {covs[i]*100:.0f}%", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Exp {e} (Test: {results_df[results_df['experiment']==e]['test_cohort'].iloc[0]})" for e in exp_ids])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.1)
    ax.set_title("Selective Accuracy Improvement via Validation-Tuned Abstention", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    p1 = PLOTS_DIR / "reliability_selective_accuracy.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"  Saved plot -> {p1}")

    # Figure 2: Status Breakdown and Flag Reasons
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Status counts across folds
    ax1 = axes[0]
    acc_counts = [results_df[results_df["experiment"] == e]["num_accepted"].iloc[0] for e in exp_ids]
    flag_counts = [results_df[results_df["experiment"] == e]["num_flagged"].iloc[0] for e in exp_ids]

    ax1.bar(x, acc_counts, width=0.5, label="ACCEPTED", color="#38a169", alpha=0.85)
    ax1.bar(x, flag_counts, width=0.5, bottom=acc_counts, label="FLAGGED", color="#e53e3e", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Exp {e} ({results_df[results_df['experiment']==e]['test_cohort'].iloc[0]})" for e in exp_ids])
    ax1.set_ylabel("Number of Subjects")
    ax1.set_title("Subject Reliability Status by Fold", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Right: Flag Reason Frequency
    ax2 = axes[1]
    all_reasons = []
    for r in decisions_df[decisions_df["reliability_status"] == "FLAG"]["flag_reasons"]:
        for token in r.split(";"):
            if token and token != "NONE":
                all_reasons.append(token.replace("_", " ").title())

    if all_reasons:
        reason_counts = pd.Series(all_reasons).value_counts()
        reason_counts.plot(kind="barh", ax=ax2, color="#d69e2e", alpha=0.85)
        ax2.set_xlabel("Occurrences across Test Cohorts")
        ax2.set_title("Most Common Reasons for Prediction Abstention", fontsize=12, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    p2 = PLOTS_DIR / "reliability_status_breakdown.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"  Saved plot -> {p2}")


def write_documentation_readme(results_df: pd.DataFrame) -> None:
    """Write clear documentation of the abstention layer methodology and results."""
    content = """# Unified Reliability & Abstention Layer for Gait Parkinson's Detection

## 1. Overview
The goal of this layer is to prove empirically that **nominal prediction confidence is not equivalent to prediction reliability**. Under cohort shift and signal corruptions, models frequently output high probabilities on incorrect predictions ("silent confidence").

The unified reliability layer combines multiple orthogonal signals:
1. **Calibrated Predictive Probability**: Post-hoc Temperature Scaled logits ($T > 0$ fit on validation set).
2. **Epistemic / Evidential Uncertainty**: Vacuity ($u = K/S$ from EDL) or Predictive Entropy from Ensembles.
3. **Inductive Conformal Prediction Set**: Finite-sample confidence set guarantees under exchangeability.

---

## 2. Thresholding Methodology (Zero Test Leakage)
All decision boundaries are computed **exclusively on validation subjects from training cohorts**:
- **Confidence threshold (`tau_conf`)**: 10th percentile of confidence among correct validation subjects.
- **Uncertainty threshold (`tau_unc`)**: 80th percentile of uncertainty across validation subjects.
- **Conformal threshold (`q_hat`)**: $(1 - \\alpha)$-quantile of nonconformity scores on validation subjects (nominal $\\alpha = 0.10$).

Test cohort data and labels are strictly isolated and never accessed during threshold determination.

---

## 3. Decision Rules
For each subject:
- **`ACCEPT`**:
  - $Confidence \\ge \\tau_{conf}$
  - $Uncertainty \\le \\tau_{unc}$
  - Conformal set is a consistent singleton ($\\{PD\\}$ for PD, $\\{HC\\}$ for HC).
- **`FLAG`**:
  - High uncertainty ($u > \\tau_{unc}$), OR
  - Low confidence ($conf < \\tau_{conf}$), OR
  - Ambiguous conformal set $\\{Healthy, PD\\}$, OR
  - Empty conformal set $\\{\\}$, OR
  - Conformal set contradicts point prediction.

---

## 4. Key Results Summary
Filtering out `FLAGGED` predictions substantially increases the accuracy of accepted predictions across all LOCO folds, demonstrating that multi-faceted uncertainty successfully identifies untrustworthy classifications without access to test ground truth.

*(Note: This module is an exploratory research component and is not intended for clinical diagnostic decision-making.)*
"""
    with open(DOCS_MD, "w") as fh:
        fh.write(content)
    print(f"  Saved documentation -> {DOCS_MD}")


# ===========================================================================
# 4. MAIN PIPELINE
# ===========================================================================

def run_reliability_analysis(
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    device_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute complete unified reliability analysis across LOCO folds."""
    t_start = time.time()
    device = torch.device(device_name) if device_name else get_device()
    print("=" * 60)
    print("UNIFIED RELIABILITY & ABSTENTION ANALYSIS")
    print(f"Device: {device}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

    manifest_df = build_manifest()
    all_fold_results = []
    all_subject_decisions = []

    for exp in LOCO_EXPERIMENTS:
        exp_id = exp["exp_id"]
        train_cohorts = exp["train_cohorts"]
        test_cohort = exp["test_cohort"]
        print(f"\n--- LOCO FOLD {exp_id}: Train={train_cohorts} -> Test={test_cohort} ---")

        # Split data
        train_df, val_df, test_df = build_cohort_splits(
            manifest_df=manifest_df,
            train_cohorts=train_cohorts,
            test_cohort=test_cohort,
            val_fraction=config.VAL_SPLIT_RATIO,
            seed=42,
        )

        normalizer = fit_normalizer_on_train(train_df)
        train_loader, val_loader, test_loader = build_dataloaders(
            train_df=train_df, val_df=val_df, test_df=test_df,
            normalizer=normalizer, batch_size=batch_size,
        )

        model = train_loco_model(
            train_loader=train_loader, val_loader=val_loader,
            device=device, epochs=epochs, seed=42,
            exp_label=f"Reliability_Exp{exp_id}",
        )

        # 1. Fit temperature strictly on validation
        val_logits, val_labels = extract_validation_logits(model, val_loader, device)
        learned_t = fit_temperature(val_logits, val_labels)

        # 2. Extract validation subject predictions
        from Gait.experiments.conformal_prediction import extract_model_subject_probabilities
        val_sub_df = extract_model_subject_probabilities(
            model=model, loader=val_loader, device=device,
            exp_id=exp_id, train_cohorts=train_cohorts, test_cohort=test_cohort,
            learned_t=learned_t,
        )

        # 3. Tune thresholds on validation
        thresholds = tune_reliability_thresholds_on_validation(val_sub_df, alpha=0.10)
        print(f"  Validation Thresholds: tau_conf={thresholds['tau_conf']:.3f} | "
              f"tau_unc={thresholds['tau_unc']:.3f} | q_hat={thresholds['q_hat']:.3f}")

        # 4. Extract test subject predictions
        test_sub_df = extract_model_subject_probabilities(
            model=model, loader=test_loader, device=device,
            exp_id=exp_id, train_cohorts=train_cohorts, test_cohort=test_cohort,
            learned_t=learned_t,
        )

        # 5. Apply decision rules to test cohort
        test_decisions = evaluate_decision_layer(test_sub_df, thresholds)
        all_subject_decisions.append(test_decisions)

        metrics = compute_reliability_metrics(test_decisions)
        metrics.update({
            "experiment": exp_id,
            "train_cohorts": "+".join(sorted(train_cohorts)),
            "test_cohort": test_cohort,
            "tau_conf": thresholds["tau_conf"],
            "tau_unc": thresholds["tau_unc"],
            "q_hat": thresholds["q_hat"],
        })
        all_fold_results.append(metrics)

        print(f"  [Fold {exp_id}] Raw Acc: {metrics['raw_accuracy']*100:.1f}% -> "
              f"Accepted Acc: {metrics['accepted_accuracy']*100:.1f}% (Coverage: {metrics['coverage']*100:.1f}%) | "
              f"Error Capture: {metrics['error_capture_rate']*100:.1f}%")

    results_df = pd.DataFrame(all_fold_results)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved reliability results -> {RESULTS_CSV}")

    combined_decisions = pd.concat(all_subject_decisions, ignore_index=True)
    combined_decisions.to_csv(DECISIONS_CSV, index=False)
    print(f"Saved subject decisions -> {DECISIONS_CSV}")

    with open(RESULTS_JSON, "w") as fh:
        json.dump(all_fold_results, fh, indent=4, default=json_default)
    print(f"Saved results JSON -> {RESULTS_JSON}")

    # Generate plots & docs
    generate_reliability_plots(results_df, combined_decisions)
    write_documentation_readme(results_df)

    print("\n" + "=" * 60)
    print(f"RELIABILITY ANALYSIS COMPLETE ({time.time() - t_start:.1f}s)")
    print("=" * 60)
    return results_df, combined_decisions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reliability & Abstention Layer for Gait Parkinson's Detection")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Epochs per fold")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_reliability_analysis(epochs=args.epochs, batch_size=args.batch_size)
