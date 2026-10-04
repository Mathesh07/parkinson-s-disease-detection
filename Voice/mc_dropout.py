"""Monte Carlo (MC) Dropout Uncertainty Quantification for Voice Wav2Vec2 Classifier.

Performs N stochastic forward passes with active dropout during test time to capture
predictive mean, predictive variance, predictive entropy, expected entropy, and mutual information.

Leakage Guards:
    - Model weights remain strictly frozen under torch.no_grad().
    - Test set labels are never used during uncertainty estimation.
    - Deterministic eval() mode is preserved when not running MC inference.
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
import torch
import torch.nn as nn

# Ensure workspace root and Voice directory are in sys.path
root_dir = Path(__file__).resolve().parent.parent
voice_dir = root_dir / "Voice"
for p in (str(root_dir), str(voice_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from Voice import config


def enable_mc_dropout(model: torch.nn.Module) -> None:
    """
    Set entire model to eval() mode (freezing BatchNorm/LayerNorm),
    then selectively enable train() mode ONLY on Dropout modules.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def calculate_entropy(p: Union[float, np.ndarray], eps: float = 1e-12) -> Union[float, np.ndarray]:
    """
    Calculate binary predictive entropy H(p) = -p log(p) - (1-p) log(1-p).
    """
    p_clamped = np.clip(p, eps, 1.0 - eps)
    h = -p_clamped * np.log(p_clamped) - (1.0 - p_clamped) * np.log(1.0 - p_clamped)
    return h


def compute_mc_statistics(stochastic_probs: np.ndarray) -> Dict[str, float]:
    """
    Compute predictive distribution statistics from N stochastic pass probabilities.
    
    Args:
        stochastic_probs: 1D array of shape (N,) containing P_i(PD) for pass i=1..N.
        
    Returns:
        Dict with mc_mean, mc_variance, predictive_entropy, expected_entropy, mutual_information.
    """
    stochastic_probs = np.array(stochastic_probs, dtype=float)
    N = len(stochastic_probs)

    # 1. Predictive Mean Probability
    mc_mean = float(np.mean(stochastic_probs))

    # 2. Predictive Variance
    mc_var = float(np.var(stochastic_probs, ddof=0))

    # 3. Predictive Entropy H(mean_p)
    pred_entropy = float(calculate_entropy(mc_mean))

    # 4. Expected Entropy mean(H(p_i))
    pass_entropies = calculate_entropy(stochastic_probs)
    exp_entropy = float(np.mean(pass_entropies))

    # 5. Mutual Information = H(mean_p) - mean(H(p_i))
    mi = float(max(0.0, pred_entropy - exp_entropy))

    return {
        "mc_mean_probability": mc_mean,
        "mc_variance": mc_var,
        "predictive_entropy": pred_entropy,
        "expected_entropy": exp_entropy,
        "mutual_information": mi,
    }


def run_mc_dropout_passes(
    head: nn.Module,
    embedding: torch.Tensor,
    num_passes: int = 30,
    seed: int = config.RANDOM_SEED
) -> np.ndarray:
    """
    Execute N stochastic forward passes on a 768-D representation using EvidentialHead with dropout.
    
    Args:
        head: EvidentialHead model module.
        embedding: Tensor of shape (1, 768) or (B, 768).
        num_passes: Number of stochastic passes (default 30).
        seed: Random seed for PyTorch dropout.
        
    Returns:
        Array of shape (num_passes,) or (num_passes, B) containing P(PD) for each pass.
    """
    torch.manual_seed(seed)
    enable_mc_dropout(head)

    pass_probs = []
    with torch.no_grad():
        for _ in range(num_passes):
            evidence = head(embedding)  # Dropout is active
            probs = torch.softmax(evidence, dim=-1)
            pd_prob = probs[:, 1].cpu().numpy()
            pass_probs.append(pd_prob)

    head.eval()  # Reset back to standard eval mode
    return np.array(pass_probs)  # Shape: (N, B)
