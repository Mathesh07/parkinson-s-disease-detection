"""Multi-Seed Leave-One-Cohort-Out (LOCO) Validation for Gait-Based Parkinson's Detection.

Research Objective:
    Determine whether model performance and uncertainty behavior (error-detection AUROC,
    calibration, and silent confidence) are consistent across multiple independent random seeds,
    or whether individual cohort-shift results are artifacts of initialization/splitting.

Seeds:
    42, 43, 44, 45, 46

Preserved Principles:
    - Strict subject-level splitting (no subject appears in both train/val and test).
    - Train-only normalization (GaitNormalizer fitted exclusively on training cohort recordings).
    - Validation-only calibration (temperature scaling parameter T fitted strictly on val split).
    - Hierarchical aggregation: WINDOW -> RECORDING -> SUBJECT.
    - Zero data leakage into test evaluations.

Outputs:
    Gait/outputs/multiseed/
        - multiseed_loco_results.csv
        - multiseed_loco_results.json
        - multiseed_loco_subject_predictions.csv
        - multiseed_summary.csv
        - plots/*.png
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
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
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
from Gait.experiments.leave_one_cohort_out import (
    _safe_auroc,
    _safe_brier,
    _safe_nll,
    aggregate_to_subject_level,
    build_cohort_splits,
    build_dataloaders,
    fit_normalizer_on_train,
    json_default,
    run_inference_on_test,
    train_loco_model,
)
from Gait.model import GaitCNNBiLSTM
from Gait.preprocessing import build_manifest
from Gait.utils import calculate_metrics, get_device, set_seed
from evidential import dirichlet_from_evidence

DEFAULT_SEEDS = [42, 43, 44, 45, 46]

MULTISEED_DIR = _GAIT_DIR / "outputs" / "multiseed"
MULTISEED_PLOTS_DIR = MULTISEED_DIR / "plots"
MULTISEED_DIR.mkdir(parents=True, exist_ok=True)
MULTISEED_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = MULTISEED_DIR / "multiseed_loco_results.csv"
RESULTS_JSON = MULTISEED_DIR / "multiseed_loco_results.json"
SUBJECTS_CSV = MULTISEED_DIR / "multiseed_loco_subject_predictions.csv"
SUMMARY_CSV = MULTISEED_DIR / "multiseed_summary.csv"

LOCO_EXPERIMENTS = [
    {"exp_id": 1, "train_cohorts": ["Ga", "Ju"], "test_cohort": "Si"},
    {"exp_id": 2, "train_cohorts": ["Ga", "Si"], "test_cohort": "Ju"},
    {"exp_id": 3, "train_cohorts": ["Ju", "Si"], "test_cohort": "Ga"},
]


def compute_comprehensive_metrics(sub_df: pd.DataFrame) -> Dict[str, float]:
    """Compute complete classification, probabilistic, calibration, and uncertainty metrics."""
    y_true = sub_df["true_label"].values.astype(int)
    y_pred = sub_df["predicted_label"].values.astype(int)
    p_pd = sub_df["P_PD"].values
    u = sub_df["uncertainty"].values
    p_hc = sub_df["P_HC"].values if "P_HC" in sub_df.columns else (1.0 - p_pd)
    conf = np.maximum(p_hc, p_pd)

    correct_mask = (y_pred == y_true)
    incorrect_mask = ~correct_mask

    base_metrics = calculate_metrics(y_true.tolist(), y_pred.tolist(), p_pd.tolist())
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    nll = _safe_nll(y_true, p_pd)
    brier = _safe_brier(y_true, p_pd)
    ece = calculate_ece(conf, correct_mask.astype(int), n_bins=10)

    mean_u_corr = float(u[correct_mask].mean()) if correct_mask.any() else float("nan")
    mean_u_inc = float(u[incorrect_mask].mean()) if incorrect_mask.any() else float("nan")
    med_u_corr = float(np.median(u[correct_mask])) if correct_mask.any() else float("nan")
    med_u_inc = float(np.median(u[incorrect_mask])) if incorrect_mask.any() else float("nan")

    error_target = (~correct_mask).astype(int)
    unc_auroc = _safe_auroc(error_target, u)
    aurc = calculate_aurc(u, correct_mask.astype(int))

    return {
        "accuracy": float(base_metrics["accuracy"]),
        "balanced_accuracy": bal_acc,
        "sensitivity": float(base_metrics["recall_sensitivity"]),
        "specificity": float(base_metrics["specificity"]),
        "f1": float(base_metrics["f1_score"]),
        "roc_auc": float(base_metrics["roc_auc"]) if base_metrics["roc_auc"] is not None else float("nan"),
        "nll": nll,
        "brier": brier,
        "ece": ece,
        "mean_uncertainty": float(np.mean(u)),
        "mean_uncertainty_correct": mean_u_corr,
        "mean_uncertainty_incorrect": mean_u_inc,
        "median_uncertainty_correct": med_u_corr,
        "median_uncertainty_incorrect": med_u_inc,
        "uncertainty_error_auroc": float(unc_auroc) if unc_auroc is not None else float("nan"),
        "aurc": aurc,
    }


def compute_summary_statistics(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean, std, 95% CI, min, max across seeds for each fold and overall."""
    metric_cols = [
        "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1",
        "roc_auc", "nll", "brier", "ece", "mean_uncertainty",
        "mean_uncertainty_correct", "mean_uncertainty_incorrect",
        "uncertainty_error_auroc", "aurc",
    ]

    summary_rows = []

    # Per-fold summary across seeds
    for exp_id in sorted(results_df["experiment"].unique()):
        exp_df = results_df[results_df["experiment"] == exp_id]
        test_cohort = exp_df["test_cohort"].iloc[0]
        n_seeds = len(exp_df)

        for col in metric_cols:
            if col not in exp_df.columns:
                continue
            vals = exp_df[col].dropna().values
            n_vals = len(vals)
            if n_vals == 0:
                continue
            m = float(np.mean(vals))
            s = float(np.std(vals, ddof=1)) if n_vals > 1 else 0.0
            se = s / np.sqrt(n_vals) if n_vals > 1 else 0.0
            ci_factor = stats.t.ppf(0.975, df=n_vals - 1) if n_vals > 1 else 1.96
            ci95_lower = m - ci_factor * se
            ci95_upper = m + ci_factor * se

            summary_rows.append({
                "group": f"Exp_{exp_id}_{test_cohort}",
                "experiment": exp_id,
                "test_cohort": test_cohort,
                "metric": col,
                "n_seeds": n_vals,
                "mean": m,
                "std": s,
                "ci95_lower": ci95_lower,
                "ci95_upper": ci95_upper,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            })

    # Overall summary across all folds and seeds
    for col in metric_cols:
        if col not in results_df.columns:
            continue
        vals = results_df[col].dropna().values
        n_vals = len(vals)
        if n_vals == 0:
            continue
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1)) if n_vals > 1 else 0.0
        se = s / np.sqrt(n_vals) if n_vals > 1 else 0.0
        ci_factor = stats.t.ppf(0.975, df=n_vals - 1) if n_vals > 1 else 1.96
        ci95_lower = m - ci_factor * se
        ci95_upper = m + ci_factor * se

        summary_rows.append({
            "group": "Overall",
            "experiment": "All",
            "test_cohort": "All",
            "metric": col,
            "n_seeds": n_vals,
            "mean": m,
            "std": s,
            "ci95_lower": ci95_lower,
            "ci95_upper": ci95_upper,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        })

    return pd.DataFrame(summary_rows)


def generate_multiseed_plots(results_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """Generate publication-ready visualizations of multi-seed LOCO performance."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot 1: Performance metrics across seeds per fold
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    metrics_to_plot = ["balanced_accuracy", "roc_auc", "f1"]
    titles = ["Balanced Accuracy", "ROC-AUC", "F1-Score"]

    for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        ax = axes[idx]
        exp_labels = []
        box_data = []
        for exp_id in sorted(results_df["experiment"].unique()):
            exp_sub = results_df[results_df["experiment"] == exp_id]
            cohort = exp_sub["test_cohort"].iloc[0]
            exp_labels.append(f"Exp {exp_id}\n(Test: {cohort})")
            box_data.append(exp_sub[metric].dropna().values)

        try:
            bp = ax.boxplot(box_data, tick_labels=exp_labels, patch_artist=True, widths=0.5)
        except TypeError:
            bp = ax.boxplot(box_data, labels=exp_labels, patch_artist=True, widths=0.5)
        for box in bp["boxes"]:
            box.set(facecolor="#2b5c8f", alpha=0.7, edgecolor="#1a365d")
        for median in bp["medians"]:
            median.set(color="#c53030", linewidth=2)

        # Overlay individual seed points
        for i, data in enumerate(box_data):
            x = np.random.normal(i + 1, 0.04, size=len(data))
            ax.scatter(x, data, color="#1a365d", alpha=0.9, s=40, zorder=5)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Multi-Seed LOCO Performance Stability Across Folds (5 Seeds)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p1 = MULTISEED_PLOTS_DIR / "multiseed_performance_boxplots.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"  Saved plot -> {p1}")

    # Plot 2: Uncertainty metrics & Error Detection AUROC
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Error Detection AUROC across folds and seeds
    ax1 = axes[0]
    fold_names = []
    auroc_data = []
    for exp_id in sorted(results_df["experiment"].unique()):
        exp_sub = results_df[results_df["experiment"] == exp_id]
        cohort = exp_sub["test_cohort"].iloc[0]
        fold_names.append(f"Exp {exp_id} ({cohort})")
        auroc_data.append(exp_sub["uncertainty_error_auroc"].dropna().values)

    ax1.axhline(0.5, color="gray", linestyle="--", label="Random Chance (0.50)")
    try:
        bp1 = ax1.boxplot(auroc_data, tick_labels=fold_names, patch_artist=True, widths=0.45)
    except TypeError:
        bp1 = ax1.boxplot(auroc_data, labels=fold_names, patch_artist=True, widths=0.45)
    for box in bp1["boxes"]:
        box.set(facecolor="#319795", alpha=0.7, edgecolor="#234e52")
    for median in bp1["medians"]:
        median.set(color="#dd6b20", linewidth=2)
    for i, data in enumerate(auroc_data):
        x = np.random.normal(i + 1, 0.04, size=len(data))
        ax1.scatter(x, data, color="#234e52", alpha=0.9, s=40, zorder=5)

    ax1.set_title("Uncertainty Error-Detection AUROC", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Error Detection AUROC")
    ax1.set_ylim(0.2, 1.0)
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Right: Mean Uncertainty (Correct vs Incorrect) across folds
    ax2 = axes[1]
    exp_ids = sorted(results_df["experiment"].unique())
    x_pos = np.arange(len(exp_ids))
    width = 0.35

    mean_corr = [results_df[results_df["experiment"] == e]["mean_uncertainty_correct"].mean() for e in exp_ids]
    std_corr = [results_df[results_df["experiment"] == e]["mean_uncertainty_correct"].std() for e in exp_ids]
    mean_inc = [results_df[results_df["experiment"] == e]["mean_uncertainty_incorrect"].mean() for e in exp_ids]
    std_inc = [results_df[results_df["experiment"] == e]["mean_uncertainty_incorrect"].std() for e in exp_ids]

    ax2.bar(x_pos - width / 2, mean_corr, width, yerr=std_corr, capsize=4,
            label="Correct Predictions", color="#2b6cb0", alpha=0.85)
    ax2.bar(x_pos + width / 2, mean_inc, width, yerr=std_inc, capsize=4,
            label="Incorrect Predictions", color="#c53030", alpha=0.85)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"Exp {e} ({results_df[results_df['experiment']==e]['test_cohort'].iloc[0]})" for e in exp_ids])
    ax2.set_ylabel("Mean Uncertainty")
    ax2.set_title("Uncertainty Separation (Correct vs. Incorrect)", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Uncertainty Quality & Error Detection Across Seeds", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p2 = MULTISEED_PLOTS_DIR / "multiseed_uncertainty_analysis.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"  Saved plot -> {p2}")


def run_multiseed_loco(
    seeds: List[int] = DEFAULT_SEEDS,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    device_name: Optional[str] = None,
    resume: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute full multi-seed LOCO evaluation.

    Resumable: reads existing RESULTS_CSV if present, avoiding redundant runs.
    """
    t_start = time.time()
    device = torch.device(device_name) if device_name else get_device()
    print("=" * 60)
    print("MULTI-SEED LOCO EXPERIMENT")
    print(f"Seeds: {seeds}")
    print(f"Epochs per fold: {epochs}")
    print(f"Device: {device}")
    print(f"Output directory: {MULTISEED_DIR}")
    print("=" * 60)

    manifest_df = build_manifest()

    # Load existing results if resuming
    existing_results: List[Dict] = []
    completed_runs = set()
    if resume and RESULTS_CSV.exists():
        try:
            prev_df = pd.read_csv(RESULTS_CSV)
            existing_results = prev_df.to_dict("records")
            for r in existing_results:
                completed_runs.add((int(r["seed"]), int(r["experiment"])))
            print(f"Resuming: Found {len(completed_runs)} already completed (seed, exp) runs.")
        except Exception as e:
            print(f"Warning: Could not read existing results: {e}. Starting fresh.")
            existing_results = []
            completed_runs = set()

    all_results = list(existing_results)
    all_subject_preds: List[pd.DataFrame] = []

    if resume and SUBJECTS_CSV.exists():
        try:
            prev_sub_df = pd.read_csv(SUBJECTS_CSV)
            all_subject_preds.append(prev_sub_df)
        except Exception:
            pass

    for seed in seeds:
        print(f"\n{'#'*60}")
        print(f"STARTING SEED: {seed}")
        print(f"{'#'*60}")

        for exp in LOCO_EXPERIMENTS:
            exp_id = exp["exp_id"]
            train_cohorts = exp["train_cohorts"]
            test_cohort = exp["test_cohort"]
            exp_label = f"Seed{seed}_Exp{exp_id}_{test_cohort}"

            if (seed, exp_id) in completed_runs:
                print(f"\nSkipping {exp_label} (already completed).")
                continue

            print(f"\n--- {exp_label}: Train={train_cohorts} -> Test={test_cohort} ---")
            t_fold_start = time.time()

            # 1. Subject-level split with explicit seed
            train_df, val_df, test_df = build_cohort_splits(
                manifest_df=manifest_df,
                train_cohorts=train_cohorts,
                test_cohort=test_cohort,
                val_fraction=config.VAL_SPLIT_RATIO,
                seed=seed,
            )

            # Leakage assertions
            train_subs = set(train_df["subject_id"].unique())
            val_subs = set(val_df["subject_id"].unique())
            test_subs = set(test_df["subject_id"].unique())
            assert len(train_subs & test_subs) == 0, f"LEAKAGE: Train/Test overlap in {exp_label}"
            assert len(val_subs & test_subs) == 0, f"LEAKAGE: Val/Test overlap in {exp_label}"
            assert len(train_subs & val_subs) == 0, f"LEAKAGE: Train/Val overlap in {exp_label}"
            assert set(test_df["study"].unique()) == {test_cohort}, f"LEAKAGE: Unexpected cohort in test"

            # 2. Fit normalizer on training recordings ONLY
            normalizer = fit_normalizer_on_train(train_df)

            # 3. Build data loaders
            train_loader, val_loader, test_loader = build_dataloaders(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                normalizer=normalizer,
                batch_size=batch_size,
            )

            # 4. Train LOCO model
            model = train_loco_model(
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=epochs,
                seed=seed,
                exp_label=exp_label,
            )

            # 5. Extract validation logits to fit temperature scaling strictly on val
            val_logits, val_labels = extract_validation_logits(model, val_loader, device)
            learned_t = fit_temperature(val_logits, val_labels)

            # 6. Test inference
            window_df = run_inference_on_test(
                model=model,
                test_loader=test_loader,
                device=device,
                exp_id=exp_id,
                train_cohorts=train_cohorts,
                test_cohort=test_cohort,
            )

            # 7. Subject-level hierarchical aggregation
            sub_df = aggregate_to_subject_level(window_df)
            sub_df["seed"] = seed
            all_subject_preds.append(sub_df)

            # 8. Compute metrics
            metrics = compute_comprehensive_metrics(sub_df)
            metrics.update({
                "seed": seed,
                "experiment": exp_id,
                "train_cohorts": "+".join(sorted(train_cohorts)),
                "test_cohort": test_cohort,
                "num_train_subjects": len(train_subs),
                "num_val_subjects": len(val_subs),
                "num_test_subjects": len(test_subs),
                "learned_temperature": learned_t,
                "fold_runtime_seconds": time.time() - t_fold_start,
            })
            all_results.append(metrics)
            completed_runs.add((seed, exp_id))

            # Incremental save
            current_results_df = pd.DataFrame(all_results)
            current_results_df.to_csv(RESULTS_CSV, index=False)
            if all_subject_preds:
                combined_subs = pd.concat(all_subject_preds, ignore_index=True).drop_duplicates(
                    subset=["seed", "experiment", "subject_id"]
                )
                combined_subs.to_csv(SUBJECTS_CSV, index=False)

            print(f"  {exp_label} Finished in {time.time() - t_fold_start:.1f}s | "
                  f"BalAcc: {metrics['balanced_accuracy']:.4f} | "
                  f"ROC-AUC: {metrics['roc_auc']:.4f} | "
                  f"Err-AUROC: {metrics['uncertainty_error_auroc']:.4f} | "
                  f"Val-T: {learned_t:.3f}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved all run results -> {RESULTS_CSV}")

    # JSON output
    with open(RESULTS_JSON, "w") as fh:
        json.dump({
            "seeds": seeds,
            "epochs": epochs,
            "total_runs": len(results_df),
            "results": all_results,
        }, fh, indent=4, default=json_default)
    print(f"Saved results JSON -> {RESULTS_JSON}")

    # Summary statistics
    summary_df = compute_summary_statistics(results_df)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved summary statistics -> {SUMMARY_CSV}")

    # Visualizations
    generate_multiseed_plots(results_df, summary_df)

    print("\n" + "=" * 60)
    print(f"MULTI-SEED VALIDATION COMPLETE ({time.time() - t_start:.1f}s total)")
    print("=" * 60)
    return results_df, summary_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Seed LOCO Experiment for Gait")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Epochs per fold")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Random seeds to evaluate")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from existing results")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_multiseed_loco(
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        resume=not args.no_resume,
    )
