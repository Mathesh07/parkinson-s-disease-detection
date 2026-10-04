"""Post-Hoc Temperature Scaling for Voice Wav2Vec2 Parkinson's Classifier.

Implements validation-only scalar Temperature Scaling (Guo et al., ICML 2017)
to calibrate predictive confidence without altering classification accuracy or model weights.

Leakage Guards:
    - Temperature parameter T > 0 is fitted ONLY on validation subjects.
    - Test set subjects and labels are completely isolated until final evaluation.
    - Pretrained Wav2Vec2 model weights remain strictly frozen.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
import torch
import torch.nn as nn

# Ensure workspace root and Voice directory are in sys.path
root_dir = Path(__file__).resolve().parent.parent
voice_dir = root_dir / "Voice"
for p in (str(root_dir), str(voice_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from Voice import config


class TemperatureScaler:
    """
    Post-hoc scalar Temperature Scaler for classification logits/evidence.
    
    Transforms logits z -> z / T before applying Softmax:
        P_k(T) = exp(z_k / T) / sum_j exp(z_j / T)
        
    Learns scalar parameter T > 0 strictly on Validation set predictions via NLL minimization.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    def calibrate_logits(self, logits: Union[np.ndarray, torch.Tensor], T: Optional[float] = None) -> Union[np.ndarray, torch.Tensor]:
        """Scale logits by temperature T."""
        temp = T if T is not None else self.temperature
        if isinstance(logits, torch.Tensor):
            return logits / temp
        return logits / temp

    def calibrate_probabilities(self, logits: Union[np.ndarray, torch.Tensor], T: Optional[float] = None) -> Union[np.ndarray, torch.Tensor]:
        """Compute temperature-scaled probabilities via Softmax."""
        temp = T if T is not None else self.temperature
        if isinstance(logits, torch.Tensor):
            return torch.softmax(logits / temp, dim=-1)
        
        # NumPy softmax implementation
        scaled = logits / temp
        exp_scaled = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        return exp_scaled / np.sum(exp_scaled, axis=-1, keepdims=True)

    def fit(self, val_logits: Union[np.ndarray, torch.Tensor], val_labels: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Fit scalar temperature T > 0 exclusively on validation logits and labels.
        
        Objective: Bounded scalar minimization of Validation Negative Log Likelihood (NLL).
        Guarantees zero test label usage.
        """
        if isinstance(val_logits, torch.Tensor):
            val_logits_np = val_logits.cpu().numpy()
        else:
            val_logits_np = np.array(val_logits)

        if isinstance(val_labels, torch.Tensor):
            val_labels_np = val_labels.cpu().numpy()
        else:
            val_labels_np = np.array(val_labels)

        val_labels_np = val_labels_np.astype(int)

        def nll_obj(t_val: float) -> float:
            t_val = max(1e-4, float(t_val))
            probs = self.calibrate_probabilities(val_logits_np, T=t_val)
            # Clip for numerical stability
            probs_clipped = np.clip(probs, 1e-12, 1.0 - 1e-12)
            # NLL loss
            sample_nll = -np.log(probs_clipped[np.arange(len(val_labels_np)), val_labels_np])
            return float(np.mean(sample_nll))

        res = opt.minimize_scalar(nll_obj, bounds=(0.01, 50.0), method="bounded")
        best_t = float(res.x)

        uncalibrated_nll = nll_obj(1.0)
        calibrated_nll = nll_obj(best_t)

        assert best_t > 0.0, f"Temperature T must be positive, got {best_t}"
        assert calibrated_nll <= uncalibrated_nll + 1e-5, (
            f"Temperature scaling increased validation NLL ({uncalibrated_nll:.6f} -> {calibrated_nll:.6f})"
        )

        self.temperature = best_t
        return best_t


def calculate_nll(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """Calculate Negative Log Likelihood (NLL) with eps clipping."""
    y_true = np.array(y_true, dtype=int)
    y_probs = np.array(y_probs, dtype=float)
    eps = 1e-12
    if y_probs.ndim == 1:
        y_probs = np.column_stack([1.0 - y_probs, y_probs])

    probs_clipped = np.clip(y_probs, eps, 1.0 - eps)
    nll = -np.log(probs_clipped[np.arange(len(y_true)), y_true])
    return float(np.mean(nll))


def calculate_brier(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """Calculate Brier Score (Mean Squared Error between one-hot label and target probability)."""
    y_true = np.array(y_true, dtype=int)
    y_probs = np.array(y_probs, dtype=float)

    if y_probs.ndim == 1:
        p_pd = y_probs
    else:
        p_pd = y_probs[:, 1]

    brier = np.mean((p_pd - y_true) ** 2)
    return float(brier)


def calculate_ece(confidences: np.ndarray, is_correct: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE) across n_bins equal-width bins in [0, 1].
    Includes confidence == 1.0 in the final bin.
    """
    confidences = np.array(confidences, dtype=float)
    is_correct = np.array(is_correct, dtype=float)
    n_samples = len(confidences)

    if n_samples == 0:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

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


def plot_reliability_diagram(
    sub_df: pd.DataFrame,
    save_path: Path,
    title: str = "Voice Reliability Diagram (Before vs. After Calibration)",
    n_bins: int = 10
) -> None:
    """Generate and save Reliability Diagram comparing Uncalibrated (T=1.0) vs. Calibrated Temperature Scaling."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    # 1. Uncalibrated (Softmax T=1.0)
    conf_uncal = sub_df["uncalibrated_confidence"].values
    correct_uncal = sub_df["uncalibrated_is_correct"].values
    uncal_means, uncal_accs = [], []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf_uncal >= lo) & (conf_uncal <= hi) if i == n_bins - 1 else (conf_uncal >= lo) & (conf_uncal < hi)
        if mask.sum() > 0:
            uncal_means.append(conf_uncal[mask].mean())
            uncal_accs.append(correct_uncal[mask].mean())

    # 2. Calibrated (Temperature Scaled T)
    conf_cal = sub_df["calibrated_confidence"].values
    correct_cal = sub_df["calibrated_is_correct"].values
    cal_means, cal_accs = [], []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf_cal >= lo) & (conf_cal <= hi) if i == n_bins - 1 else (conf_cal >= lo) & (conf_cal < hi)
        if mask.sum() > 0:
            cal_means.append(conf_cal[mask].mean())
            cal_accs.append(correct_cal[mask].mean())

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect Calibration")
    if uncal_means:
        ax.plot(uncal_means, uncal_accs, "o--", color="#d62728", lw=2, markersize=7, label="Uncalibrated Baseline (T=1.0)")
    if cal_means:
        ax.plot(cal_means, cal_accs, "s-", color="#1f77b4", lw=2, markersize=7, label=f"Calibrated (T={sub_df['temperature'].iloc[0]:.4f})")

    ax.set_title(title, fontsize=12, pad=12, weight="bold")
    ax.set_xlabel("Mean Confidence", fontsize=11, labelpad=8)
    ax.set_ylabel("Empirical Accuracy", fontsize=11, labelpad=8)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_calibration_comparison(
    uncal_metrics: Dict[str, float],
    cal_metrics: Dict[str, float],
    save_path: Path,
    title: str = "Voice Calibration Metrics Comparison"
) -> None:
    """Generate side-by-side bar chart comparing NLL, Brier Score, and ECE before vs. after calibration."""
    metrics = ["NLL", "Brier Score", "ECE"]
    uncal_vals = [uncal_metrics["nll"], uncal_metrics["brier"], uncal_metrics["ece"]]
    cal_vals = [cal_metrics["nll"], cal_metrics["brier"], cal_metrics["ece"]]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    rects1 = ax.bar(x - width/2, uncal_vals, width, label="Uncalibrated (T=1.0)", color="#d62728", alpha=0.85)
    rects2 = ax.bar(x + width/2, cal_vals, width, label="Calibrated (Temp Scaled)", color="#1f77b4", alpha=0.85)

    ax.set_ylabel("Metric Value (Lower is Better)", fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=13, pad=12, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11, weight="bold")
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    # Annotate bars
    for bar in rects1:
        height = bar.get_height()
        ax.annotate(f"{height:.4f}", xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for bar in rects2:
        height = bar.get_height()
        ax.annotate(f"{height:.4f}", xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
