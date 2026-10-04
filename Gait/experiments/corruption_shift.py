"""Controlled Distribution Corruption Experiment for Gait-Based Parkinson's Detection.

Research Question:
    "Can Uncertainty Warn Us? Cohort-Shift Reliability of Parkinson's Disease Classifiers Across Voice, Handwriting and Gait"
    
    Specifically for the Gait modality:
    Does controlled corruption of held-out test signals cause:
    1. Predictive performance to degrade?
    2. Model uncertainty to increase?
    3. Uncertainty to become more useful for identifying unreliable / error-prone predictions?

Experimental Protocol & Leakage Guards:
    1. Subject-level LOCO splitting is preserved across 3 folds:
       - Exp1: Train Ga+Ju -> Test Si
       - Exp2: Train Ga+Si -> Test Ju
       - Exp3: Train Ju+Si -> Test Ga
    2. Model checkpoint / parameters are trained ONCE per fold and reused across all severities.
    3. Corruption is applied ONLY to held-out test signals AFTER train-fitted normalization (Option B).
    4. Normalization statistics and Temperature Scaling T are fit strictly on train / validation cohorts.
    5. Corrupted test data is NEVER used for calibration or fitting.
    6. Severity 0 represents the original clean test input.
    7. Evaluates 4 Uncertainty Methods: Softmax, Temperature Scaling, MC Dropout, Evidential Deep Learning (EDL).

Outputs:
    Gait/outputs/corruption_shift/
      ├── corruption_shift_results.csv
      ├── corruption_shift_results.json
      ├── corruption_shift_subject_predictions.csv
      └── plots/
          ├── uncertainty_vs_severity.png
          ├── performance_vs_severity.png
          ├── error_rate_vs_severity.png
          ├── ece_vs_severity.png
          └── risk_coverage_corruption.png

Usage:
    python -m Gait.experiments.corruption_shift [--dry-run] [--epochs 50] [--mc-samples 20] [--seed 42]
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
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

# ---------------------------------------------------------------------------
# Path bootstrapping — supports running as module or standalone script
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_EXPERIMENTS_DIR = _THIS_FILE.parent          # Gait/experiments/
_GAIT_DIR = _EXPERIMENTS_DIR.parent           # Gait/
_ROOT_DIR = _GAIT_DIR.parent                  # repo root

for _p in (str(_GAIT_DIR), str(_ROOT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Project Imports
# ---------------------------------------------------------------------------
from Gait import config
from Gait.dataset import GaitWindowDataset, collate_gait_batch
from Gait.model import GaitCNNBiLSTM
from Gait.preprocessing import GaitNormalizer, build_manifest, load_raw_recording
from Gait.utils import get_device, set_seed
from evidential import dirichlet_from_evidence

from Gait.experiments.leave_one_cohort_out import (
    _safe_auroc,
    _safe_brier,
    _safe_nll,
    build_cohort_splits,
    build_dataloaders,
    fit_normalizer_on_train,
    train_loco_model,
)
from Gait.experiments.calibration_baselines import (
    calculate_aurc,
    calculate_ece,
    extract_validation_logits,
    fit_temperature,
)
from Gait.experiments.mc_dropout import (
    compute_window_entropy_metrics,
    enable_mc_dropout,
)

# ---------------------------------------------------------------------------
# Output Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = _GAIT_DIR / "outputs" / "corruption_shift"
PLOTS_DIR = OUTPUT_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV_PATH = OUTPUT_DIR / "corruption_shift_results.csv"
RESULTS_JSON_PATH = OUTPUT_DIR / "corruption_shift_results.json"
SUBJECT_PREDS_CSV_PATH = OUTPUT_DIR / "corruption_shift_subject_predictions.csv"

PLOT_UNCERTAINTY_PATH = PLOTS_DIR / "uncertainty_vs_severity.png"
PLOT_PERFORMANCE_PATH = PLOTS_DIR / "performance_vs_severity.png"
PLOT_ERROR_RATE_PATH = PLOTS_DIR / "error_rate_vs_severity.png"
PLOT_ECE_PATH = PLOTS_DIR / "ece_vs_severity.png"
PLOT_RISK_COVERAGE_PATH = PLOTS_DIR / "risk_coverage_corruption.png"


# ===========================================================================
# NORMALIZATION & CORRUPTION ORDERING DECISION
# ===========================================================================
"""
NORMALIZATION & CORRUPTION ORDERING DECISION:

Decision: Apply corruption AFTER train-fitted z-score normalization (Option B).

Rationale:
1. Preserve Perturbation Intent & Scale:
   Gait VGRF signals are normalized using channel-wise Z-score statistics (mean mu_c, std sigma_c)
   fitted strictly on the training subjects. In normalized space, features have zero mean and unit variance.
   Applying corruption directly to normalized features ensures that corruption parameters (e.g., Gaussian noise
   std sigma=0.25, amplitude scaling factor=1.5, channel dropout=4 channels, temporal masking=100 steps)
   have exact, invariant physical meanings relative to the normalized input domain of the neural network.

2. Prevention of Normalization Masking:
   If corruption were applied before normalization, any test-set normalization or re-standardization would
   undo or attenuate the controlled perturbation (e.g., re-centering amplitude scaling or rescaling noise).
   Applying corruption after train-fitted normalization (directly on normalized inputs) guarantees that
   the perturbation reaches the model intact without being altered or removed by normalization.

3. Zero-Leakage Compliance:
   Normalizer parameters (mean, std) remain fit strictly on clean training cohort recordings.
   Corrupted test signals never influence normalizer statistics.
"""

# ===========================================================================
# CORRUPTION DEFINITIONS
# ===========================================================================

CORRUPTION_TYPES = [
    "gaussian_noise",
    "amplitude_scaling",
    "channel_dropout",
    "temporal_masking",
    "smoothing",
]

SEVERITY_LEVELS = [0, 1, 2, 3, 4]

# Parameter Mappings across Severities
GAUSSIAN_NOISE_STDS = {0: 0.0, 1: 0.10, 2: 0.25, 3: 0.50, 4: 0.75}
AMPLITUDE_SCALE_FACTORS = {0: 1.0, 1: 1.25, 2: 1.50, 3: 1.75, 4: 2.00}
CHANNEL_DROPOUT_COUNTS = {0: 0, 1: 2, 2: 4, 3: 6, 4: 8}
TEMPORAL_MASK_STEPS = {0: 0, 1: 50, 2: 100, 3: 150, 4: 200}
SMOOTHING_KERNEL_SIZES = {0: 1, 1: 5, 2: 11, 3: 21, 4: 31}


def apply_gaussian_noise(x: np.ndarray, severity: int, seed: int = 42) -> np.ndarray:
    """Add zero-mean Gaussian noise to normalized gait signals."""
    std = GAUSSIAN_NOISE_STDS[severity]
    if severity == 0 or std == 0.0:
        return x.copy()
    rng = np.random.RandomState(seed)
    noise = rng.normal(loc=0.0, scale=std, size=x.shape).astype(x.dtype)
    return x + noise


def apply_amplitude_scaling(x: np.ndarray, severity: int) -> np.ndarray:
    """Multiplicative amplitude scaling of normalized gait signals."""
    scale = AMPLITUDE_SCALE_FACTORS[severity]
    if severity == 0 or scale == 1.0:
        return x.copy()
    return x * scale


def apply_channel_dropout(x: np.ndarray, severity: int, seed: int = 42) -> np.ndarray:
    """Mask a proportion of the 16 sensor channels by setting them to 0.0 (normalized mean)."""
    n_drop = CHANNEL_DROPOUT_COUNTS[severity]
    if severity == 0 or n_drop == 0:
        return x.copy()
    out = x.copy()
    num_channels = x.shape[-1]
    rng = np.random.RandomState(seed)
    dropped_indices = rng.choice(num_channels, size=n_drop, replace=False)
    out[..., dropped_indices] = 0.0
    return out


def apply_temporal_masking(x: np.ndarray, severity: int, seed: int = 42) -> np.ndarray:
    """Mask a contiguous block of time steps by setting them to 0.0 (normalized mean)."""
    mask_steps = TEMPORAL_MASK_STEPS[severity]
    if severity == 0 or mask_steps == 0:
        return x.copy()
    out = x.copy()
    time_steps = x.shape[-2]
    rng = np.random.RandomState(seed)
    start_idx = rng.randint(0, time_steps - mask_steps + 1)
    out[..., start_idx : start_idx + mask_steps, :] = 0.0
    return out


def apply_smoothing(x: np.ndarray, severity: int) -> np.ndarray:
    """Apply 1D moving-average box filter smoothing along the temporal axis."""
    kernel_size = SMOOTHING_KERNEL_SIZES[severity]
    if severity == 0 or kernel_size <= 1:
        return x.copy()

    out = np.empty_like(x)
    kernel = np.ones(kernel_size, dtype=x.dtype) / float(kernel_size)
    pad_len = kernel_size // 2

    if x.ndim == 2:  # (500, 16)
        T, C = x.shape
        for c in range(C):
            padded = np.pad(x[:, c], pad_len, mode="edge")
            smoothed = np.convolve(padded, kernel, mode="valid")
            out[:, c] = smoothed[:T]
    elif x.ndim == 3:  # (B, 500, 16)
        B, T, C = x.shape
        for b in range(B):
            for c in range(C):
                padded = np.pad(x[b, :, c], pad_len, mode="edge")
                smoothed = np.convolve(padded, kernel, mode="valid")
                out[b, :, c] = smoothed[:T]
    else:
        raise ValueError(f"Unsupported input dimension for smoothing: {x.ndim}")

    return out


def apply_corruption(
    x: Union[np.ndarray, torch.Tensor],
    corruption_type: str,
    severity: int,
    seed: int = 42
) -> torch.Tensor:
    """
    Unified entry point for applying controlled signal corruption.
    Accepts PyTorch Tensor or numpy array, returns PyTorch Tensor of shape (B, 500, 16).
    Guarantees x is not modified in-place.
    """
    is_tensor = isinstance(x, torch.Tensor)
    device = x.device if is_tensor else torch.device("cpu")

    if is_tensor:
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.asarray(x)

    if severity == 0:
        corrupted_np = x_np.copy()
    elif corruption_type == "gaussian_noise":
        corrupted_np = apply_gaussian_noise(x_np, severity, seed=seed)
    elif corruption_type == "amplitude_scaling":
        corrupted_np = apply_amplitude_scaling(x_np, severity)
    elif corruption_type == "channel_dropout":
        corrupted_np = apply_channel_dropout(x_np, severity, seed=seed)
    elif corruption_type == "temporal_masking":
        corrupted_np = apply_temporal_masking(x_np, severity, seed=seed)
    elif corruption_type == "smoothing":
        corrupted_np = apply_smoothing(x_np, severity)
    else:
        raise ValueError(f"Unknown corruption type: '{corruption_type}'")

    return torch.tensor(corrupted_np, dtype=torch.float32, device=device)


# ===========================================================================
# INFERENCE & UNCERTAINTY COMPUTATION
# ===========================================================================

@torch.no_grad()
def run_corrupted_inference(
    model: GaitCNNBiLSTM,
    test_loader: DataLoader,
    device: torch.device,
    learned_T: float,
    corruption_type: str,
    severity: int,
    mc_samples: int,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Run inference on corrupted test loader for a given (corruption_type, severity).
    Evaluates 4 uncertainty methods using the EXACT SAME corrupted input batches:
      1. Softmax
      2. Temperature Scaling (using val-fitted learned_T)
      3. MC Dropout (N stochastic passes)
      4. EDL (Evidential Deep Learning)

    Returns dictionary mapping method name to window-level predictions DataFrame.
    """
    model.eval()

    softmax_windows = []
    temp_windows = []
    edl_windows = []
    mc_windows = []

    for batch_idx, batch in enumerate(test_loader):
        clean_x = batch["features"].to(device)
        labels = batch["labels"].to(device)
        batch_size = len(labels)
        batch_seed = seed + batch_idx * 1000 + severity * 10

        # Apply corruption to test batch AFTER normalization
        corr_x = apply_corruption(clean_x, corruption_type, severity, seed=batch_seed).to(device)

        # -------------------------------------------------------------------
        # 1. Standard Pass: Softmax, Temperature Scaling, EDL
        # -------------------------------------------------------------------
        model.eval()
        evidence, emb = model(corr_x, return_embedding=True)
        _, raw_logits = model.evidential_head(emb, return_raw_logits=True)

        alpha, S, edl_probs, u_edl = dirichlet_from_evidence(evidence, num_classes=2)

        probs_softmax = torch.softmax(raw_logits, dim=1).cpu().numpy()
        probs_temp = torch.softmax(raw_logits / learned_T, dim=1).cpu().numpy()
        probs_edl = edl_probs.cpu().numpy()
        u_edl_np = u_edl.squeeze(-1).cpu().numpy()
        y_np = labels.cpu().numpy()

        for i in range(batch_size):
            meta = {
                "subject_id": batch["subject_ids"][i],
                "recording_id": batch["recording_ids"][i],
                "study": batch["studies"][i],
                "true_label": int(y_np[i]),
            }

            # Softmax
            softmax_windows.append({
                **meta,
                "P_HC": float(probs_softmax[i, 0]),
                "P_PD": float(probs_softmax[i, 1]),
            })

            # Temperature Scaling
            temp_windows.append({
                **meta,
                "P_HC": float(probs_temp[i, 0]),
                "P_PD": float(probs_temp[i, 1]),
            })

            # EDL
            edl_windows.append({
                **meta,
                "P_HC": float(probs_edl[i, 0]),
                "P_PD": float(probs_edl[i, 1]),
                "u_edl": float(u_edl_np[i]),
            })

        # -------------------------------------------------------------------
        # 2. MC Dropout Pass: N stochastic forward passes
        # -------------------------------------------------------------------
        enable_mc_dropout(model)
        batch_mc_probs = []
        for _ in range(mc_samples):
            _, emb_mc = model(corr_x, return_embedding=True)
            _, logits_mc = model.evidential_head(emb_mc, return_raw_logits=True)
            p_mc = torch.softmax(logits_mc, dim=1).cpu().numpy()
            batch_mc_probs.append(p_mc)

        # shape (N, B, 2)
        batch_mc_probs = np.stack(batch_mc_probs, axis=0)

        for i in range(batch_size):
            win_mc = batch_mc_probs[:, i, :]  # (N, 2)
            mean_p = win_mc.mean(axis=0)
            pred_ent, exp_ent, mi = compute_window_entropy_metrics(win_mc)
            mc_windows.append({
                "subject_id": batch["subject_ids"][i],
                "recording_id": batch["recording_ids"][i],
                "study": batch["studies"][i],
                "true_label": int(y_np[i]),
                "P_HC": float(mean_p[0]),
                "P_PD": float(mean_p[1]),
                "predictive_entropy": float(pred_ent),
                "expected_entropy": float(exp_ent),
                "mutual_information": float(mi),
            })

    return {
        "Softmax": pd.DataFrame(softmax_windows),
        "Temperature Scaling": pd.DataFrame(temp_windows),
        "EDL": pd.DataFrame(edl_windows),
        "MC Dropout": pd.DataFrame(mc_windows),
    }


# ===========================================================================
# SUBJECT-LEVEL HIERARCHICAL AGGREGATION & METRICS
# ===========================================================================

def aggregate_method_predictions(
    method_name: str,
    window_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Hierarchical aggregation: WINDOW -> RECORDING -> SUBJECT.
    Computes subject-level predicted probability, confidence, uncertainty,
    prediction, and correctness.
    """
    rec_keys = ["subject_id", "recording_id", "study", "true_label"]
    sub_keys = ["subject_id", "study", "true_label"]

    if method_name in ["Softmax", "Temperature Scaling"]:
        prob_cols = ["P_HC", "P_PD"]
        rec_df = window_df.groupby(rec_keys)[prob_cols].mean().reset_index()
        sub_df = rec_df.groupby(sub_keys)[prob_cols].mean().reset_index()
        sub_df["probability"] = sub_df["P_PD"]
        sub_df["confidence"] = np.maximum(sub_df["P_HC"], sub_df["P_PD"])
        sub_df["uncertainty"] = 1.0 - sub_df["confidence"]

    elif method_name == "EDL":
        prob_cols = ["P_HC", "P_PD", "u_edl"]
        rec_df = window_df.groupby(rec_keys)[prob_cols].mean().reset_index()
        sub_df = rec_df.groupby(sub_keys)[prob_cols].mean().reset_index()
        sub_df["probability"] = sub_df["P_PD"]
        sub_df["confidence"] = np.maximum(sub_df["P_HC"], sub_df["P_PD"])
        sub_df["uncertainty"] = sub_df["u_edl"]

    elif method_name == "MC Dropout":
        prob_cols = ["P_HC", "P_PD", "predictive_entropy", "expected_entropy", "mutual_information"]
        rec_df = window_df.groupby(rec_keys)[prob_cols].mean().reset_index()
        sub_df = rec_df.groupby(sub_keys)[prob_cols].mean().reset_index()
        sub_df["probability"] = sub_df["P_PD"]
        sub_df["confidence"] = np.maximum(sub_df["P_HC"], sub_df["P_PD"])
        sub_df["uncertainty"] = sub_df["predictive_entropy"]

    else:
        raise ValueError(f"Unknown method name: {method_name}")

    sub_df["prediction"] = (sub_df["probability"] >= 0.5).astype(int)
    sub_df["is_correct"] = (sub_df["prediction"] == sub_df["true_label"]).astype(int)
    sub_df["method"] = method_name
    return sub_df


def compute_evaluation_metrics(sub_df: pd.DataFrame) -> Dict[str, float]:
    """Compute comprehensive performance and uncertainty metrics from subject-level DataFrame."""
    y_true = sub_df["true_label"].values
    y_pred = sub_df["prediction"].values
    y_prob = sub_df["probability"].values
    conf = sub_df["confidence"].values
    unc = sub_df["uncertainty"].values
    is_correct = sub_df["is_correct"].values

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    sens = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    spec = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = _safe_auroc(y_true, y_prob)

    nll = _safe_nll(y_true, y_prob)
    brier = _safe_brier(y_true, y_prob)
    ece = calculate_ece(conf, is_correct, n_bins=10)

    # Error detection AUROC: using uncertainty to predict incorrect prediction (is_error = 1 - is_correct)
    err_auroc = _safe_auroc(1 - is_correct, unc)
    aurc = calculate_aurc(unc, is_correct)

    mean_unc = float(np.mean(unc))
    median_unc = float(np.median(unc))

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "sensitivity": sens,
        "specificity": spec,
        "f1_score": f1,
        "roc_auc": float(roc_auc) if roc_auc is not None else 0.5,
        "nll": nll,
        "brier_score": brier,
        "ece": ece,
        "error_detection_auroc": float(err_auroc) if err_auroc is not None else 0.5,
        "aurc": aurc,
        "mean_uncertainty": mean_unc,
        "median_uncertainty": median_unc,
        "error_rate": 1.0 - acc,
    }


# ===========================================================================
# PLOTTING ROUTINES
# ===========================================================================

def generate_corruption_plots(results_df: pd.DataFrame, subject_df: pd.DataFrame) -> None:
    """
    Generate 5 primary publication-quality figures:
      1. uncertainty_vs_severity.png
      2. performance_vs_severity.png
      3. error_rate_vs_severity.png
      4. ece_vs_severity.png
      5. risk_coverage_corruption.png
    """
    print("\nGenerating corruption shift visualization figures...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    methods = ["Softmax", "Temperature Scaling", "MC Dropout", "EDL"]
    method_colors = {
        "Softmax": "#1f77b4",
        "Temperature Scaling": "#ff7f0e",
        "MC Dropout": "#2ca02c",
        "EDL": "#d62728",
    }
    corruption_titles = {
        "gaussian_noise": "Gaussian Noise",
        "amplitude_scaling": "Amplitude Scaling",
        "channel_dropout": "Channel Dropout",
        "temporal_masking": "Temporal Masking",
        "smoothing": "Smoothing",
    }

    # Aggregate across LOCO folds for overall plots
    avg_results = (
        results_df.groupby(["corruption_type", "severity", "method"])
        .mean(numeric_only=True)
        .reset_index()
    )

    # -----------------------------------------------------------------------
    # Figure 1: uncertainty_vs_severity.png
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=False)
    for idx, ctype in enumerate(CORRUPTION_TYPES):
        ax = axes[idx]
        sub_c = avg_results[avg_results["corruption_type"] == ctype]
        for m in methods:
            m_data = sub_c[sub_c["method"] == m].sort_values("severity")
            ax.plot(
                m_data["severity"],
                m_data["mean_uncertainty"],
                marker="o",
                linewidth=2,
                color=method_colors[m],
                label=m,
            )
        ax.set_title(corruption_titles[ctype], fontsize=12, fontweight="bold")
        ax.set_xlabel("Corruption Severity", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Mean Model Uncertainty", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xticks(SEVERITY_LEVELS)
    axes[0].legend(loc="upper left", fontsize=9)
    plt.suptitle("Model Uncertainty vs. Corruption Severity Across Modality Corruptions", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(PLOT_UNCERTAINTY_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {PLOT_UNCERTAINTY_PATH}")

    # -----------------------------------------------------------------------
    # Figure 2: performance_vs_severity.png
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=True)
    for idx, ctype in enumerate(CORRUPTION_TYPES):
        ax = axes[idx]
        sub_c = avg_results[avg_results["corruption_type"] == ctype]
        for m in methods:
            m_data = sub_c[sub_c["method"] == m].sort_values("severity")
            ax.plot(
                m_data["severity"],
                m_data["f1_score"],
                marker="s",
                linewidth=2,
                color=method_colors[m],
                label=m,
            )
        ax.set_title(corruption_titles[ctype], fontsize=12, fontweight="bold")
        ax.set_xlabel("Corruption Severity", fontsize=10)
        if idx == 0:
            ax.set_ylabel("F1 Score", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xticks(SEVERITY_LEVELS)
    axes[0].legend(loc="lower left", fontsize=9)
    plt.suptitle("Predictive Performance (F1 Score) vs. Corruption Severity", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(PLOT_PERFORMANCE_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {PLOT_PERFORMANCE_PATH}")

    # -----------------------------------------------------------------------
    # Figure 3: error_rate_vs_severity.png
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=True)
    for idx, ctype in enumerate(CORRUPTION_TYPES):
        ax = axes[idx]
        sub_c = avg_results[avg_results["corruption_type"] == ctype]
        for m in methods:
            m_data = sub_c[sub_c["method"] == m].sort_values("severity")
            ax.plot(
                m_data["severity"],
                m_data["error_rate"],
                marker="^",
                linewidth=2,
                color=method_colors[m],
                label=m,
            )
        ax.set_title(corruption_titles[ctype], fontsize=12, fontweight="bold")
        ax.set_xlabel("Corruption Severity", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Error Rate (1 - Accuracy)", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xticks(SEVERITY_LEVELS)
    axes[0].legend(loc="upper left", fontsize=9)
    plt.suptitle("Classification Error Rate vs. Corruption Severity", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(PLOT_ERROR_RATE_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {PLOT_ERROR_RATE_PATH}")

    # -----------------------------------------------------------------------
    # Figure 4: ece_vs_severity.png
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=False)
    for idx, ctype in enumerate(CORRUPTION_TYPES):
        ax = axes[idx]
        sub_c = avg_results[avg_results["corruption_type"] == ctype]
        for m in methods:
            m_data = sub_c[sub_c["method"] == m].sort_values("severity")
            ax.plot(
                m_data["severity"],
                m_data["ece"],
                marker="d",
                linewidth=2,
                color=method_colors[m],
                label=m,
            )
        ax.set_title(corruption_titles[ctype], fontsize=12, fontweight="bold")
        ax.set_xlabel("Corruption Severity", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Expected Calibration Error (ECE)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xticks(SEVERITY_LEVELS)
    axes[0].legend(loc="upper left", fontsize=9)
    plt.suptitle("Probability Calibration Error (ECE) vs. Corruption Severity", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(PLOT_ECE_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {PLOT_ECE_PATH}")

    # -----------------------------------------------------------------------
    # Figure 5: risk_coverage_corruption.png
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Clean (Severity 0) vs Severe (Severity 4) Risk-Coverage Curves
    for ax, sev, label_str in [(ax1, 0, "Clean (Severity 0)"), (ax2, 4, "Severe (Severity 4)")]:
        sev_sub = subject_df[subject_df["severity"] == sev]
        for m in methods:
            m_sub = sev_sub[sev_sub["method"] == m]
            if len(m_sub) > 0:
                uncs = m_sub["uncertainty"].values
                correct = m_sub["is_correct"].values
                order = np.argsort(uncs)
                sorted_correct = correct[order]
                sorted_error = 1.0 - sorted_correct
                n = len(sorted_error)
                coverages = np.linspace(1 / n, 1.0, n)
                risks = np.cumsum(sorted_error) / np.arange(1, n + 1)
                ax.plot(coverages, risks, label=f"{m}", color=method_colors[m], linewidth=2)

        ax.set_title(f"Selective Risk vs. Coverage — {label_str}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Coverage (Fraction of Retained Samples)", fontsize=11)
        ax.set_ylabel("Selective Risk (Error Rate on Retained)", fontsize=11)
        ax.set_xlim(0.0, 1.05)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle="--", alpha=0.6)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper left", fontsize=10)

    plt.suptitle("Selective Prediction Risk-Coverage Curves under Controlled Corruption", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_RISK_COVERAGE_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {PLOT_RISK_COVERAGE_PATH}")


# ===========================================================================
# EXPERIMENT EXECUTION PIPELINE
# ===========================================================================

def run_corruption_experiment(
    epochs: int = config.EPOCHS,
    mc_samples: int = 20,
    seed: int = config.RANDOM_SEED,
    dry_run: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Executes the controlled distribution corruption experiment across LOCO folds.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n=======================================================================")
    print("  Gait Controlled Distribution Corruption Experiment")
    print(f"  Device: {device} | Epochs: {epochs} | MC Samples: {mc_samples} | Dry Run: {dry_run}")
    print("=======================================================================\n")

    if not config.DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {config.DATA_DIR}")

    manifest_df = build_manifest(config.DATA_DIR)

    # Scenarios / LOCO folds
    if dry_run:
        loco_scenarios = [
            {"train": ["Ga", "Ju"], "test": "Si", "name": "Exp1_GaJu_to_Si"},
        ]
        corruption_types_to_run = ["gaussian_noise"]
        severities_to_run = [0, 1]
    else:
        loco_scenarios = [
            {"train": ["Ga", "Ju"], "test": "Si", "name": "Exp1_GaJu_to_Si"},
            {"train": ["Ga", "Si"], "test": "Ju", "name": "Exp2_GaSi_to_Ju"},
            {"train": ["Ju", "Si"], "test": "Ga", "name": "Exp3_JuSi_to_Ga"},
        ]
        corruption_types_to_run = CORRUPTION_TYPES
        severities_to_run = SEVERITY_LEVELS

    all_subject_records = []
    all_summary_rows = []

    for fold_idx, sc in enumerate(loco_scenarios, 1):
        train_cohorts = sc["train"]
        test_cohort = sc["test"]
        exp_name = sc["name"]

        print(f"\n{'='*70}")
        print(f"LOCO Fold {fold_idx}/{len(loco_scenarios)}: Train={'+'.join(train_cohorts)} -> Test={test_cohort}")
        print(f"{'='*70}")

        # 1. Subject-level split
        train_df, val_df, test_df = build_cohort_splits(
            manifest_df, train_cohorts, test_cohort, val_fraction=config.VAL_SPLIT_RATIO, seed=seed
        )

        # 2. Normalization fit strictly on train
        normalizer = fit_normalizer_on_train(train_df)

        # 3. Build DataLoaders
        train_loader, val_loader, test_loader = build_dataloaders(
            train_df, val_df, test_df, normalizer, batch_size=config.BATCH_SIZE
        )

        # 4. Train LOCO model ONCE per fold
        model = train_loco_model(
            train_loader, val_loader, device, epochs=epochs, seed=seed, exp_label=exp_name
        )

        # 5. Fit Temperature Scaling strictly on validation logits (val cohort only)
        val_logits, val_labels = extract_validation_logits(model, val_loader, device)
        learned_T = fit_temperature(val_logits, val_labels)
        print(f"  [{exp_name}] Learned Val Temperature T = {learned_T:.4f}")

        # 6. Evaluate all corruption types & severities using the SAME trained model
        for ctype in corruption_types_to_run:
            print(f"  Evaluating Corruption: {ctype}...")
            # Store clean baseline (sev 0) metrics per method for relative comparison
            sev0_metrics_by_method = {}

            for sev in severities_to_run:
                method_windows = run_corrupted_inference(
                    model, test_loader, device, learned_T,
                    corruption_type=ctype, severity=sev,
                    mc_samples=mc_samples, seed=seed
                )

                for method_name, win_df in method_windows.items():
                    sub_df = aggregate_method_predictions(method_name, win_df)
                    metrics = compute_evaluation_metrics(sub_df)

                    # Save subject predictions for auditing
                    sub_df["experiment"] = exp_name
                    sub_df["train_cohorts"] = "_".join(train_cohorts)
                    sub_df["test_cohort"] = test_cohort
                    sub_df["corruption_type"] = ctype
                    sub_df["severity"] = sev
                    all_subject_records.append(sub_df)

                    if sev == 0:
                        sev0_metrics_by_method[method_name] = metrics

                    sev0_m = sev0_metrics_by_method[method_name]
                    perf_deg = float(sev0_m["accuracy"] - metrics["accuracy"])
                    f1_deg = float(sev0_m["f1_score"] - metrics["f1_score"])
                    unc_inc = float(metrics["mean_uncertainty"] - sev0_m["mean_uncertainty"])

                    summary_row = {
                        "experiment": exp_name,
                        "train_cohorts": "_".join(train_cohorts),
                        "test_cohort": test_cohort,
                        "corruption_type": ctype,
                        "severity": sev,
                        "method": method_name,
                        "learned_T": learned_T,
                        **metrics,
                        "acc_degradation": perf_deg,
                        "f1_degradation": f1_deg,
                        "uncertainty_increase": unc_inc,
                    }
                    all_summary_rows.append(summary_row)

    master_subject_df = pd.concat(all_subject_records, ignore_index=True)
    summary_df = pd.DataFrame(all_summary_rows)

    # -----------------------------------------------------------------------
    # Compute Spearman Rank Correlations across Severities
    # -----------------------------------------------------------------------
    trend_analysis = []
    grouped = summary_df.groupby(["experiment", "corruption_type", "method"])

    for (exp_name, ctype, method_name), group in grouped:
        group_sorted = group.sort_values("severity")
        sevs = group_sorted["severity"].values
        uncs = group_sorted["mean_uncertainty"].values
        errs = group_sorted["error_rate"].values
        degs = group_sorted["acc_degradation"].values

        rho_unc, p_unc = stats.spearmanr(sevs, uncs) if len(set(uncs)) > 1 else (0.0, 1.0)
        rho_err, p_err = stats.spearmanr(sevs, errs) if len(set(errs)) > 1 else (0.0, 1.0)
        rho_deg, p_deg = stats.spearmanr(sevs, degs) if len(set(degs)) > 1 else (0.0, 1.0)

        trend_analysis.append({
            "experiment": exp_name,
            "corruption_type": ctype,
            "method": method_name,
            "spearman_rho_uncertainty": float(rho_unc) if not np.isnan(rho_unc) else 0.0,
            "p_value_uncertainty": float(p_unc) if not np.isnan(p_unc) else 1.0,
            "spearman_rho_error_rate": float(rho_err) if not np.isnan(rho_err) else 0.0,
            "p_value_error_rate": float(p_err) if not np.isnan(p_err) else 1.0,
            "spearman_rho_degradation": float(rho_deg) if not np.isnan(rho_deg) else 0.0,
            "p_value_degradation": float(p_deg) if not np.isnan(p_deg) else 1.0,
        })

    trend_df = pd.DataFrame(trend_analysis)

    # Save CSV & JSON outputs
    summary_df.to_csv(RESULTS_CSV_PATH, index=False)
    master_subject_df.to_csv(SUBJECT_PREDS_CSV_PATH, index=False)
    print(f"\nSaved results CSV: {RESULTS_CSV_PATH}")
    print(f"Saved subject predictions CSV: {SUBJECT_PREDS_CSV_PATH}")

    json_dict = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epochs": epochs,
            "mc_samples": mc_samples,
            "seed": seed,
            "dry_run": dry_run,
        },
        "results": summary_df.to_dict(orient="records"),
        "spearman_trends": trend_df.to_dict(orient="records"),
    }
    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(json_dict, f, indent=2)
    print(f"Saved summary JSON: {RESULTS_JSON_PATH}")

    # Generate figures
    generate_corruption_plots(summary_df, master_subject_df)

    print("\nCorruption shift experiment completed successfully!")
    return summary_df, master_subject_df, json_dict


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gait Controlled Distribution Corruption Experiment")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs per fold")
    parser.add_argument("--mc-samples", type=int, default=20, help="Number of MC Dropout forward passes")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="Random seed for reproducibility")
    parser.add_argument("--dry-run", action="store_true", help="Run a quick dry run (1 fold, 1 corruption, 2 severities)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.dry_run:
        run_corruption_experiment(epochs=1, mc_samples=5, seed=args.seed, dry_run=True)
    else:
        run_corruption_experiment(epochs=args.epochs, mc_samples=args.mc_samples, seed=args.seed, dry_run=False)
