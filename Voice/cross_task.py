"""Cross-Task Shift Module for Voice Wav2Vec2 Parkinson's Classification.

Computes task-specific (ReadText vs SpontaneousDialogue) performance and uncertainty
metrics across Baseline, MC Dropout (N=30), and Deep Ensemble (M=5).
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

# Ensure workspace root and Voice directory are in sys.path
root_dir = Path(__file__).resolve().parent.parent
voice_dir = root_dir / "Voice"
for p in (str(root_dir), str(voice_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from Voice import config


def calculate_entropy(p: Union[float, np.ndarray], eps: float = 1e-12) -> Union[float, np.ndarray]:
    """Calculate binary predictive entropy H(p) = -p log(p) - (1-p) log(1-p)."""
    p_clamped = np.clip(p, eps, 1.0 - eps)
    h = -p_clamped * np.log(p_clamped) - (1.0 - p_clamped) * np.log(1.0 - p_clamped)
    return h


def compute_task_statistics(probs: np.ndarray) -> Dict[str, float]:
    """
    Compute descriptive statistics for a array of subject probabilities under a specific task.
    """
    probs = np.array(probs, dtype=float)
    if len(probs) == 0:
        return {
            "mean_probability": 0.0,
            "median_probability": 0.0,
            "mean_confidence": 0.0,
            "mean_entropy": 0.0,
            "median_entropy": 0.0,
            "mean_variance": 0.0,
            "median_variance": 0.0,
            "mean_mutual_info": 0.0,
            "median_mutual_info": 0.0
        }

    mean_p = float(np.mean(probs))
    median_p = float(np.median(probs))
    mean_conf = float(np.mean(np.abs(probs - 0.5)))
    entropies = calculate_entropy(probs)

    return {
        "mean_probability": mean_p,
        "median_probability": median_p,
        "mean_confidence": mean_conf,
        "mean_entropy": float(np.mean(entropies)),
        "median_entropy": float(np.median(entropies))
    }


def compute_within_subject_deltas(rt_row: pd.Series, sd_row: pd.Series) -> Dict[str, float]:
    """
    Compute within-subject paired differences between ReadText and SpontaneousDialogue tasks.
    
    Delta = ReadText_value - SpontaneousDialogue_value
    """
    return {
        "subject_id": rt_row["subject_id"],
        "true_label": rt_row["true_label"],
        "baseline_p_readtext": rt_row["baseline_probability"],
        "baseline_p_spontaneous": sd_row["baseline_probability"],
        "delta_baseline_p": float(rt_row["baseline_probability"] - sd_row["baseline_probability"]),
        
        "mc_mean_p_readtext": rt_row["mc_mean_probability"],
        "mc_mean_p_spontaneous": sd_row["mc_mean_probability"],
        "delta_mc_mean_p": float(rt_row["mc_mean_probability"] - sd_row["mc_mean_probability"]),
        
        "mc_entropy_readtext": rt_row["mc_predictive_entropy"],
        "mc_entropy_spontaneous": sd_row["mc_predictive_entropy"],
        "delta_mc_entropy": float(rt_row["mc_predictive_entropy"] - sd_row["mc_predictive_entropy"]),
        
        "mc_variance_readtext": rt_row["mc_variance"],
        "mc_variance_spontaneous": sd_row["mc_variance"],
        "delta_mc_variance": float(rt_row["mc_variance"] - sd_row["mc_variance"]),
        
        "mc_mi_readtext": rt_row["mc_mutual_information"],
        "mc_mi_spontaneous": sd_row["mc_mutual_information"],
        "delta_mc_mi": float(rt_row["mc_mutual_information"] - sd_row["mc_mutual_information"]),
        
        "ens_mean_p_readtext": rt_row["ensemble_mean_probability"],
        "ens_mean_p_spontaneous": sd_row["ensemble_mean_probability"],
        "delta_ens_mean_p": float(rt_row["ensemble_mean_probability"] - sd_row["ensemble_mean_probability"]),
        
        "ens_entropy_readtext": rt_row["ensemble_predictive_entropy"],
        "ens_entropy_spontaneous": sd_row["ensemble_predictive_entropy"],
        "delta_ens_entropy": float(rt_row["ensemble_predictive_entropy"] - sd_row["ensemble_predictive_entropy"]),
        
        "ens_variance_readtext": rt_row["ensemble_variance"],
        "ens_variance_spontaneous": sd_row["ensemble_variance"],
        "delta_ens_variance": float(rt_row["ensemble_variance"] - sd_row["ensemble_variance"]),
        
        "ens_mi_readtext": rt_row["ensemble_mutual_information"],
        "ens_mi_spontaneous": sd_row["ensemble_mutual_information"],
        "delta_ens_mi": float(rt_row["ensemble_mutual_information"] - sd_row["ensemble_mutual_information"])
    }
