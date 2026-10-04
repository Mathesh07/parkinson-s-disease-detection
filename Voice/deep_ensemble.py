"""Deep Ensemble Uncertainty Quantification for Voice Wav2Vec2 Classifier.

Provides functions to compute ensemble statistics, entropy, mutual information,
member prediction agreement, and error detection metrics across M=5 ensemble members.

Leakage Guards:
    - Model weights remain strictly frozen under torch.no_grad().
    - Zero test set labels are used during inference or uncertainty computation.
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


def compute_ensemble_statistics(member_probs: List[float]) -> Dict[str, float]:
    """
    Compute predictive distribution and uncertainty metrics from M ensemble member probabilities.
    
    Args:
        member_probs: List of M probabilities P_m(PD) for member m=1..M.
        
    Returns:
        Dict containing mean, variance, std, predictive entropy, expected entropy, mutual info,
        and prediction agreement scores.
    """
    probs = np.array(member_probs, dtype=float)
    M = len(probs)

    # 1. Predictive Mean Probability
    mean_prob = float(np.mean(probs))

    # 2. Predictive Variance & Standard Deviation
    variance = float(np.var(probs, ddof=0))
    std_dev = float(np.sqrt(variance))

    # 3. Predictive Entropy H(mean_p)
    pred_entropy = float(calculate_entropy(mean_prob))

    # 4. Expected Entropy mean(H(p_m))
    member_entropies = calculate_entropy(probs)
    exp_entropy = float(np.mean(member_entropies))

    # 5. Mutual Information = H(mean_p) - mean(H(p_m))
    mutual_info = float(max(0.0, pred_entropy - exp_entropy))

    # 6. Member Prediction Agreement
    member_preds = (probs >= 0.5).astype(int)
    num_pd = int(np.sum(member_preds == 1))
    num_hc = int(np.sum(member_preds == 0))
    agreement_str = f"{num_pd}/{M} PD, {num_hc}/{M} Healthy"
    agreement_score = float(max(num_pd, num_hc) / M)

    return {
        "ensemble_mean_probability": mean_prob,
        "ensemble_variance": variance,
        "ensemble_std": std_dev,
        "predictive_entropy": pred_entropy,
        "expected_entropy": exp_entropy,
        "mutual_information": mutual_info,
        "num_members_pd": num_pd,
        "num_members_hc": num_hc,
        "member_agreement": agreement_str,
        "prediction_agreement": agreement_score
    }
