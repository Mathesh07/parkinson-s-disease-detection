"""Utility functions for seeding, GPU detection, metrics calculation, aggregation, and plotting."""

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Support running as a standalone script or as a module
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
for p in (str(current_dir), str(parent_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from Voice import config
except ImportError:
    import config


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Detect available compute device and display a detailed banner."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print("=" * 50)
        print("COMPUTE DEVICE CONFIGURATION")
        print("=" * 50)
        print("Device: CUDA")
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {vram_gb:.2f} GB")
        print("CUDA available: True")
        print("Mixed Precision: Enabled (torch.amp.autocast)")
        print("=" * 50)
    else:
        device = torch.device("cpu")
        print("=" * 50)
        print("WARNING: CUDA is not available! Falling back to CPU.")
        print("Device: CPU")
        print("=" * 50)
    return device


def calculate_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_scores: Optional[List[float]] = None
) -> Dict[str, float]:
    """Calculate comprehensive classification metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix for Sensitivity (TPR) & Specificity (TNR)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    roc_auc = 0.0
    if y_scores is not None and len(np.unique(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, y_scores))
        except Exception:
            roc_auc = 0.0

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "roc_auc": float(roc_auc),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }


def aggregate_predictions(
    chunk_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate chunk-level probabilities to recording-level and subject-level.
    
    Expected chunk_df columns:
    [recording_id, subject_id, task, true_label, pd_probability]
    """
    # 1. Recording-level aggregation (mean of chunk probabilities)
    rec_grouped = chunk_df.groupby(["recording_id", "subject_id", "task", "true_label"])
    recording_df = rec_grouped["pd_probability"].mean().reset_index()
    recording_df["predicted_label"] = (recording_df["pd_probability"] >= 0.5).astype(int)
    recording_df["predicted_name"] = recording_df["predicted_label"].map(config.ID2LABEL)
    recording_df["hc_probability"] = 1.0 - recording_df["pd_probability"]

    # 2. Subject-level aggregation (mean of recording probabilities)
    sub_grouped = recording_df.groupby(["subject_id", "true_label"])
    subject_df = sub_grouped["pd_probability"].mean().reset_index()
    subject_df["predicted_label"] = (subject_df["pd_probability"] >= 0.5).astype(int)
    subject_df["predicted_name"] = subject_df["predicted_label"].map(config.ID2LABEL)
    subject_df["hc_probability"] = 1.0 - subject_df["pd_probability"]

    return recording_df, subject_df


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str] = ["HC", "PD"],
    save_path: Path = config.CONFUSION_MATRIX_PATH,
    title: str = "Voice Wav2Vec2 Confusion Matrix"
) -> None:
    """Generate and save a clear annotated confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", ax=ax, values_format="d", colorbar=True)
    ax.set_title(title, fontsize=14, pad=15, weight="bold")
    ax.set_xlabel("Predicted Class", fontsize=12, labelpad=10)
    ax.set_ylabel("True Class", fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_roc_curve(
    y_true: List[int],
    y_scores: List[float],
    save_path: Path = config.ROC_CURVE_PATH,
    title: str = "Voice Wav2Vec2 ROC Curve"
) -> None:
    """Generate and save a ROC curve plot."""
    plt.figure(figsize=(6, 5), dpi=300)
    
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        plt.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"Wav2Vec2 (AUC = {auc:.3f})")
    else:
        plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")

    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", alpha=0.6, label="Chance Level")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, labelpad=10)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12, labelpad=10)
    plt.title(title, fontsize=14, pad=15, weight="bold")
    plt.legend(loc="lower right", frameon=True, fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Path = config.TRAINING_CURVES_PATH
) -> None:
    """Plot training & validation loss and validation F1-score across epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # 1. Loss Plot
    ax1.plot(epochs, history["train_loss"], "o-", color="#d62728", lw=2, label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "s--", color="#1f77b4", lw=2, label="Val Loss")
    ax1.set_title("Training & Validation Loss", fontsize=13, weight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=11)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. Validation F1 & Accuracy Plot
    ax2.plot(epochs, history["val_f1"], "^-", color="#2ca02c", lw=2, label="Val Subject F1")
    ax2.plot(epochs, history["val_acc"], "d--", color="#ff7f0e", lw=2, label="Val Subject Accuracy")
    ax2.set_title("Validation Metrics (Subject-Level)", fontsize=13, weight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Score", fontsize=11)
    ax2.set_ylim([0.0, 1.05])
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Wav2Vec2 Fine-Tuning Performance", fontsize=15, weight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
