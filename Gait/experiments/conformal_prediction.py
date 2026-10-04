"""Conformal Prediction Layer for Gait-Based Parkinson's Disease Detection.

Research Objective:
    Determine whether standard inductive conformal prediction provides valid empirical coverage
    under cohort shift (LOCO) and controlled signal corruption in gait-based Parkinson's detection,
    and demonstrate how distribution shift violates the foundational exchangeability assumption.

Theoretical Foundation:
    - Standard Inductive Conformal Prediction (ICP) (Vovk et al.; Angelopoulos & Bates 2021).
    - Calibration data: Split validation subjects ONLY (no test cohort labels seen).
    - Nonconformity score for true label y:
          s_i = 1 - P(Y = y_i | x_i)
    - Conformal quantile at significance level alpha (e.g., alpha = 0.10 for 90% target coverage):
          q_hat = Quantile_{ (1 - alpha)(1 + 1/n) } ( {s_i}_{i=1}^n )
    - Prediction set for test instance x:
          C(x) = { y in {HC, PD} : P(Y = y | x) >= 1 - q_hat }
    - Output sets:
          {PD}           -> Confident Parkinson's Disease
          {Healthy}      -> Confident Healthy Control
          {Healthy, PD}  -> Ambiguous / High Epistemic Uncertainty
          {} (Empty)     -> Extreme Out-of-Distribution / Model Vacuity

Methodological Rigor:
    - q_hat is constructed STRICTLY on validation subjects from training cohorts.
    - Test cohort labels are NEVER used for calibration or quantile selection.
    - Evaluated across:
        1. All 3 LOCO folds (held-out cohorts: Si, Ju, Ga).
        2. Signal corruption sweeps (e.g., Gaussian noise and amplitude scaling severities 0-4).
    - Explicitly documents that finite-sample marginal coverage guarantees P(Y in C(X)) >= 1 - alpha
      require exchangeability, which cohort shift directly violates.
    - No clinical validity claims.

Outputs:
    Gait/outputs/conformal/
        - conformal_results.csv
        - conformal_results.json
        - conformal_subject_predictions.csv
        - conformal_corruption_coverage.csv
        - plots/*.png
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score
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
from Gait.dataset import GaitWindowDataset, collate_gait_batch
from Gait.experiments.calibration_baselines import (
    calculate_aurc,
    calculate_ece,
    extract_validation_logits,
    fit_temperature,
)
from Gait.experiments.corruption_shift import apply_corruption
from Gait.experiments.leave_one_cohort_out import (
    _safe_auroc,
    _safe_brier,
    _safe_nll,
    aggregate_to_subject_level,
    build_cohort_splits,
    build_dataloaders,
    fit_normalizer_on_train,
    json_default,
    train_loco_model,
)
from Gait.model import GaitCNNBiLSTM
from Gait.preprocessing import build_manifest
from Gait.utils import calculate_metrics, get_device, set_seed

OUTPUT_DIR = _GAIT_DIR / "outputs" / "conformal"
PLOTS_DIR = OUTPUT_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = OUTPUT_DIR / "conformal_results.csv"
RESULTS_JSON = OUTPUT_DIR / "conformal_results.json"
SUBJECTS_CSV = OUTPUT_DIR / "conformal_subject_predictions.csv"
CORRUPTION_CSV = OUTPUT_DIR / "conformal_corruption_coverage.csv"

LOCO_EXPERIMENTS = [
    {"exp_id": 1, "train_cohorts": ["Ga", "Ju"], "test_cohort": "Si"},
    {"exp_id": 2, "train_cohorts": ["Ga", "Si"], "test_cohort": "Ju"},
    {"exp_id": 3, "train_cohorts": ["Ju", "Si"], "test_cohort": "Ga"},
]

DEFAULT_ALPHAS = [0.05, 0.10, 0.15, 0.20]  # Target coverages: 95%, 90%, 85%, 80%


# ===========================================================================
# 1. CONFORMAL CALIBRATION & THRESHOLD COMPUTATION
# ===========================================================================

def compute_conformal_quantile(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    alpha: float = 0.10,
) -> float:
    """
    Compute split-conformal nonconformity quantile strictly on validation data.

    Nonconformity score: s_i = 1 - P(Y = y_i | x_i).
    Finite-sample quantile level: q_level = ceil((n + 1) * (1 - alpha)) / n.
    """
    n = len(val_labels)
    if n == 0:
        raise ValueError("Validation set is empty.")

    true_class_probs = val_probs[np.arange(n), val_labels]
    scores = 1.0 - true_class_probs

    # Conformal quantile level with finite-sample correction
    q_level = np.ceil((n + 1) * (1.0 - alpha)) / float(n)
    q_level = min(1.0, max(0.0, q_level))

    q_hat = float(np.quantile(scores, q_level, method="higher"))
    return q_hat


def construct_prediction_set(
    prob_hc: float,
    prob_pd: float,
    q_hat: float,
) -> List[str]:
    """
    Construct binary prediction set for given class probabilities and conformal threshold.
    Class c is included if P(Y = c | x) >= 1 - q_hat.
    """
    threshold = max(0.0, 1.0 - q_hat)
    pred_set = []
    if prob_hc >= threshold:
        pred_set.append("HC")
    if prob_pd >= threshold:
        pred_set.append("PD")
    return pred_set


def format_set_string(pred_set: List[str]) -> str:
    """Format prediction set for CSV reporting."""
    if not pred_set:
        return "{}"
    if len(pred_set) == 2:
        return "{Healthy, PD}"
    if "PD" in pred_set:
        return "{PD}"
    return "{Healthy}"


# ===========================================================================
# 2. VALIDATION & TEST INFERENCE
# ===========================================================================

@torch.no_grad()
def extract_model_subject_probabilities(
    model: GaitCNNBiLSTM,
    loader: DataLoader,
    device: torch.device,
    exp_id: int,
    train_cohorts: List[str],
    test_cohort: str,
    learned_t: float = 1.0,
) -> pd.DataFrame:
    """
    Extract hierarchical subject-level probabilities (WINDOW -> RECORDING -> SUBJECT)
    using temperature-scaled logits for optimal probabilistic calibration.
    """
    model.eval()
    records = []

    for batch in loader:
        x = batch["features"].to(device)
        y = batch["labels"].to(device)
        batch_size = len(y)
        y_np = y.cpu().numpy()

        _, emb = model(x, return_embedding=True)
        _, raw_logits = model.evidential_head(emb, return_raw_logits=True)
        scaled_logits = raw_logits / learned_t
        probs = torch.softmax(scaled_logits, dim=1).cpu().numpy()

        for i in range(batch_size):
            records.append({
                "experiment": exp_id,
                "train_cohorts": "+".join(sorted(train_cohorts)),
                "test_cohort": test_cohort,
                "subject_id": batch["subject_ids"][i],
                "recording_id": batch["recording_ids"][i],
                "study": batch["studies"][i],
                "true_label": int(y_np[i]),
                "P_HC": float(probs[i, 0]),
                "P_PD": float(probs[i, 1]),
            })

    win_df = pd.DataFrame(records)
    rec_keys = ["experiment", "train_cohorts", "test_cohort", "subject_id", "recording_id", "study", "true_label"]
    sub_keys = ["experiment", "train_cohorts", "test_cohort", "subject_id", "study", "true_label"]

    rec_df = win_df.groupby(rec_keys)[["P_HC", "P_PD"]].mean().reset_index()
    sub_df = rec_df.groupby(sub_keys)[["P_HC", "P_PD"]].mean().reset_index()

    # Re-normalize to sum to 1.0
    p_sum = sub_df["P_HC"] + sub_df["P_PD"]
    sub_df["P_HC"] = sub_df["P_HC"] / p_sum
    sub_df["P_PD"] = sub_df["P_PD"] / p_sum
    sub_df["predicted_label"] = (sub_df["P_PD"] >= 0.5).astype(int)
    sub_df["is_correct"] = (sub_df["predicted_label"] == sub_df["true_label"]).astype(int)
    sub_df["confidence"] = np.maximum(sub_df["P_HC"], sub_df["P_PD"])
    sub_df["uncertainty"] = 1.0 - sub_df["confidence"]
    return sub_df


# ===========================================================================
# 3. CORRUPTED EVALUATION FOR CONFORMAL PREDICTION
# ===========================================================================

@torch.no_grad()
def evaluate_conformal_under_corruption(
    model: GaitCNNBiLSTM,
    test_loader: DataLoader,
    device: torch.device,
    learned_t: float,
    q_hat: float,
    alpha: float,
    exp_id: int,
    test_cohort: str,
    corruption_types: List[str] = ["gaussian_noise", "amplitude_scaling"],
    severities: List[int] = [0, 1, 2, 3, 4],
) -> List[Dict]:
    """
    Evaluate empirical coverage and set size under increasing signal corruption.
    Demonstrates violation of exchangeability under distribution shift.
    """
    model.eval()
    corruption_results = []

    for c_type in corruption_types:
        for sev in severities:
            records = []
            for batch_idx, batch in enumerate(test_loader):
                clean_x = batch["features"].to(device)
                y = batch["labels"].to(device)
                batch_size = len(y)
                y_np = y.cpu().numpy()
                batch_seed = 42 + batch_idx * 100 + sev * 10

                corr_x = apply_corruption(clean_x, c_type, sev, seed=batch_seed).to(device)

                _, emb = model(corr_x, return_embedding=True)
                _, raw_logits = model.evidential_head(emb, return_raw_logits=True)
                scaled_logits = raw_logits / learned_t
                probs = torch.softmax(scaled_logits, dim=1).cpu().numpy()

                for i in range(batch_size):
                    records.append({
                        "subject_id": batch["subject_ids"][i],
                        "recording_id": batch["recording_ids"][i],
                        "study": batch["studies"][i],
                        "true_label": int(y_np[i]),
                        "P_HC": float(probs[i, 0]),
                        "P_PD": float(probs[i, 1]),
                    })

            win_df = pd.DataFrame(records)
            rec_df = win_df.groupby(["subject_id", "recording_id", "study", "true_label"])[["P_HC", "P_PD"]].mean().reset_index()
            sub_df = rec_df.groupby(["subject_id", "study", "true_label"])[["P_HC", "P_PD"]].mean().reset_index()

            p_sum = sub_df["P_HC"] + sub_df["P_PD"]
            sub_df["P_HC"] = sub_df["P_HC"] / p_sum
            sub_df["P_PD"] = sub_df["P_PD"] / p_sum

            # Conformal evaluation on corrupted test subjects
            coverages = []
            set_sizes = []
            singletons = 0
            both_classes = 0
            empty_sets = 0

            for _, row in sub_df.iterrows():
                true_label_str = "PD" if int(row["true_label"]) == 1 else "HC"
                pset = construct_prediction_set(row["P_HC"], row["P_PD"], q_hat)
                covered = int(true_label_str in pset)
                coverages.append(covered)
                set_sizes.append(len(pset))
                if len(pset) == 1:
                    singletons += 1
                elif len(pset) == 2:
                    both_classes += 1
                elif len(pset) == 0:
                    empty_sets += 1

            n_sub = len(sub_df)
            corruption_results.append({
                "experiment": exp_id,
                "test_cohort": test_cohort,
                "corruption_type": c_type,
                "severity": sev,
                "target_coverage": 1.0 - alpha,
                "alpha": alpha,
                "q_hat": q_hat,
                "empirical_coverage": float(np.mean(coverages)),
                "avg_set_size": float(np.mean(set_sizes)),
                "singleton_rate": singletons / float(n_sub),
                "ambiguity_rate": both_classes / float(n_sub),
                "empty_rate": empty_sets / float(n_sub),
                "num_subjects": n_sub,
            })

    return corruption_results


# ===========================================================================
# 4. PLOTTING ROUTINES
# ===========================================================================

def generate_conformal_plots(
    results_df: pd.DataFrame,
    corruption_df: pd.DataFrame,
    subject_df: pd.DataFrame,
) -> None:
    """Generate publication figures illustrating conformal coverage and shift breakdown."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: Target Coverage vs Empirical Coverage across Folds
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    exp_ids = sorted(results_df["experiment"].unique())
    alphas = sorted(results_df["alpha"].unique())

    for exp_id in exp_ids:
        sub = results_df[results_df["experiment"] == exp_id]
        cohort = sub["test_cohort"].iloc[0]
        ax1.plot(sub["target_coverage"], sub["empirical_coverage"], marker="o", linewidth=2,
                 label=f"Exp {exp_id} (Test: {cohort})")

    ax1.plot([0.7, 1.0], [0.7, 1.0], "k--", alpha=0.7, label="Ideal Calibration Line")
    ax1.set_xlabel("Nominal Target Coverage (1 - alpha)", fontsize=11)
    ax1.set_ylabel("Empirical Coverage on Test Cohort", fontsize=11)
    ax1.set_title("Conformal Coverage under Cohort Shift", fontsize=12, fontweight="bold")
    ax1.set_xlim(0.75, 0.98)
    ax1.set_ylim(0.40, 1.02)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Right: Average Set Size vs Target Coverage
    ax2 = axes[1]
    for exp_id in exp_ids:
        sub = results_df[results_df["experiment"] == exp_id]
        cohort = sub["test_cohort"].iloc[0]
        ax2.plot(sub["target_coverage"], sub["avg_set_size"], marker="s", linewidth=2,
                 label=f"Exp {exp_id} (Test: {cohort})")

    ax2.set_xlabel("Nominal Target Coverage (1 - alpha)", fontsize=11)
    ax2.set_ylabel("Average Prediction Set Size", fontsize=11)
    ax2.set_title("Prediction Set Size vs Coverage Requirement", fontsize=12, fontweight="bold")
    ax2.set_ylim(0.5, 2.1)
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Conformal Prediction Evaluation Across LOCO Folds", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p1 = PLOTS_DIR / "conformal_coverage_and_set_size.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"  Saved plot -> {p1}")

    # Figure 2: Conformal Coverage Breakdown Under Controlled Signal Corruption
    if not corruption_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        corr_types = corruption_df["corruption_type"].unique()

        for idx, c_type in enumerate(corr_types):
            ax = axes[idx]
            c_sub = corruption_df[corruption_df["corruption_type"] == c_type]
            for exp_id in sorted(c_sub["experiment"].unique()):
                exp_sub = c_sub[c_sub["experiment"] == exp_id]
                cohort = exp_sub["test_cohort"].iloc[0]
                ax.plot(exp_sub["severity"], exp_sub["empirical_coverage"], marker="o", linewidth=2,
                        label=f"Exp {exp_id} ({cohort})")

            nominal = c_sub["target_coverage"].iloc[0]
            ax.axhline(nominal, color="red", linestyle="--", label=f"Target ({nominal*100:.0f}%)")
            ax.set_title(f"Coverage: {c_type.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Corruption Severity Level (0 = Clean)", fontsize=11)
            ax.set_xticks([0, 1, 2, 3, 4])
            ax.set_ylim(0.0, 1.05)
            ax.grid(True, linestyle="--", alpha=0.5)
            if idx == 0:
                ax.set_ylabel("Empirical Conformal Coverage", fontsize=11)
                ax.legend(loc="lower left")

        plt.suptitle("Conformal Coverage Breakdown Under Distribution Shift (Exchangeability Violation)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        p2 = PLOTS_DIR / "conformal_coverage_vs_corruption.png"
        plt.savefig(p2, dpi=300)
        plt.close()
        print(f"  Saved plot -> {p2}")


# ===========================================================================
# 5. MAIN EXECUTION PIPELINE
# ===========================================================================

def run_conformal_prediction(
    alphas: List[float] = DEFAULT_ALPHAS,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    device_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute complete Conformal Prediction experiment across all 3 LOCO folds."""
    t_start = time.time()
    device = torch.device(device_name) if device_name else get_device()
    print("=" * 60)
    print("CONFORMAL PREDICTION LOCO EVALUATION")
    print(f"Alpha levels: {alphas} (Target coverages: {[1 - a for a in alphas]})")
    print(f"Device: {device}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

    manifest_df = build_manifest()
    all_conformal_results = []
    all_subject_rows = []
    all_corruption_results = []

    for exp in LOCO_EXPERIMENTS:
        exp_id = exp["exp_id"]
        train_cohorts = exp["train_cohorts"]
        test_cohort = exp["test_cohort"]
        print(f"\n--- LOCO FOLD {exp_id}: Train={train_cohorts} -> Test={test_cohort} ---")

        # Split data using master seed 42
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

        # Train model
        model = train_loco_model(
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            seed=42,
            exp_label=f"Conformal_Exp{exp_id}",
        )

        # 1. Fit temperature scaling on validation set
        val_logits, val_labels = extract_validation_logits(model, val_loader, device)
        learned_t = fit_temperature(val_logits, val_labels)

        # 2. Extract validation probabilities for nonconformity score calibration
        val_sub_df = extract_model_subject_probabilities(
            model=model, loader=val_loader, device=device,
            exp_id=exp_id, train_cohorts=train_cohorts, test_cohort=test_cohort,
            learned_t=learned_t,
        )
        val_probs_arr = val_sub_df[["P_HC", "P_PD"]].values
        val_labels_arr = val_sub_df["true_label"].values.astype(int)

        # 3. Extract test probabilities
        test_sub_df = extract_model_subject_probabilities(
            model=model, loader=test_loader, device=device,
            exp_id=exp_id, train_cohorts=train_cohorts, test_cohort=test_cohort,
            learned_t=learned_t,
        )

        # 4. Evaluate each alpha level
        for alpha in alphas:
            target_cov = 1.0 - alpha
            q_hat = compute_conformal_quantile(val_probs_arr, val_labels_arr, alpha=alpha)

            coverages = []
            set_sizes = []
            singletons = 0
            both_classes = 0
            empty_sets = 0

            for _, row in test_sub_df.iterrows():
                true_label_str = "PD" if int(row["true_label"]) == 1 else "HC"
                pset = construct_prediction_set(row["P_HC"], row["P_PD"], q_hat)
                covered = int(true_label_str in pset)
                coverages.append(covered)
                set_sizes.append(len(pset))

                if len(pset) == 1:
                    singletons += 1
                elif len(pset) == 2:
                    both_classes += 1
                elif len(pset) == 0:
                    empty_sets += 1

                # Record per-subject prediction details for primary alpha=0.10
                if np.isclose(alpha, 0.10):
                    all_subject_rows.append({
                        "experiment": exp_id,
                        "test_cohort": test_cohort,
                        "subject_id": row["subject_id"],
                        "true_label": int(row["true_label"]),
                        "P_HC": float(row["P_HC"]),
                        "P_PD": float(row["P_PD"]),
                        "confidence": float(row["confidence"]),
                        "uncertainty": float(row["uncertainty"]),
                        "conformal_set": format_set_string(pset),
                        "set_size": len(pset),
                        "is_covered": covered,
                    })

            n_test = len(test_sub_df)
            emp_cov = float(np.mean(coverages))
            avg_size = float(np.mean(set_sizes))

            all_conformal_results.append({
                "experiment": exp_id,
                "train_cohorts": "+".join(sorted(train_cohorts)),
                "test_cohort": test_cohort,
                "alpha": alpha,
                "target_coverage": target_cov,
                "q_hat": q_hat,
                "empirical_coverage": emp_cov,
                "coverage_gap": emp_cov - target_cov,
                "avg_set_size": avg_size,
                "singleton_rate": singletons / float(n_test),
                "ambiguity_rate": both_classes / float(n_test),
                "empty_rate": empty_sets / float(n_test),
                "num_test_subjects": n_test,
                "num_val_subjects": len(val_sub_df),
                "learned_temperature": learned_t,
            })

            print(f"  [Alpha={alpha:.2f} | Target={target_cov*100:.0f}%] -> "
                  f"Empirical Coverage: {emp_cov*100:.1f}% | Avg Set Size: {avg_size:.2f} | "
                  f"q_hat: {q_hat:.4f} | Singletons: {singletons}/{n_test}")

        # 5. Evaluate conformal under controlled signal corruption for alpha=0.10
        q_hat_10 = compute_conformal_quantile(val_probs_arr, val_labels_arr, alpha=0.10)
        corr_results = evaluate_conformal_under_corruption(
            model=model, test_loader=test_loader, device=device,
            learned_t=learned_t, q_hat=q_hat_10, alpha=0.10,
            exp_id=exp_id, test_cohort=test_cohort,
        )
        all_corruption_results.extend(corr_results)

    # Save CSVs
    results_df = pd.DataFrame(all_conformal_results)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved conformal results -> {RESULTS_CSV}")

    subject_df = pd.DataFrame(all_subject_rows)
    subject_df.to_csv(SUBJECTS_CSV, index=False)
    print(f"Saved subject conformal predictions -> {SUBJECTS_CSV}")

    corruption_df = pd.DataFrame(all_corruption_results)
    corruption_df.to_csv(CORRUPTION_CSV, index=False)
    print(f"Saved corruption coverage analysis -> {CORRUPTION_CSV}")

    with open(RESULTS_JSON, "w") as fh:
        json.dump({
            "alphas": alphas,
            "results": all_conformal_results,
            "corruption_evaluation": all_corruption_results,
        }, fh, indent=4, default=json_default)
    print(f"Saved results JSON -> {RESULTS_JSON}")

    # Generate plots
    generate_conformal_plots(results_df, corruption_df, subject_df)

    print("\n" + "=" * 60)
    print(f"CONFORMAL PREDICTION EVALUATION COMPLETE ({time.time() - t_start:.1f}s)")
    print("=" * 60)
    return results_df, corruption_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conformal Prediction for Gait Parkinson's Detection")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Epochs per fold")
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS, help="Significance levels")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_conformal_prediction(alphas=args.alphas, epochs=args.epochs, batch_size=args.batch_size)
