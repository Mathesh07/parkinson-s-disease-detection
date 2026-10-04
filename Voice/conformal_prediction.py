"""Voice Conformal Prediction Module.

Implements Split Conformal Prediction for binary Parkinson's Disease classification
using subject-level calibration on held-out validation subjects.

Methodology:
    - Calibration Set: Validation subjects (ID02, ID11, ID15, ID21, ID33; n=5).
    - Nonconformity Score: s_i = 1 - P(Y_i | X_i).
    - Significance Level: alpha = 0.10.
    - Finite-Sample Quantile: q_hat = Quantile_{1-alpha}(s_1, ..., s_n) using finite-sample (n+1)(1-alpha)/n convention.
    - Prediction Set: C(X) = { c in {0, 1} | 1 - P(Y=c | X) <= q_hat }.
    - Leakage Guards: Calibration uses ONLY validation subjects; test set (n=6) is strictly held out.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def compute_nonconformity_score(p_true: float) -> float:
    """
    Compute nonconformity score for a single sample given the model probability for the true class.
    
    s = 1 - P(Y_true | X)
    """
    p_true = float(np.clip(p_true, 0.0, 1.0))
    return 1.0 - p_true


def compute_nonconformity_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute nonconformity scores for an array of probabilities and true labels.
    
    Args:
        probs: Array of shape (N, 2) where col 0 is P(HC) and col 1 is P(PD), OR 1D array of P(PD).
        labels: 1D array of binary true labels (0 = HC, 1 = PD).
        
    Returns:
        1D numpy array of nonconformity scores s_i = 1 - P(Y_i | X_i).
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)

    if probs.ndim == 1:
        p_pd = probs
        p_hc = 1.0 - p_pd
        p_true = np.where(labels == 1, p_pd, p_hc)
    elif probs.ndim == 2 and probs.shape[1] == 2:
        p_true = np.where(labels == 1, probs[:, 1], probs[:, 0])
    else:
        raise ValueError(f"Invalid probs shape {probs.shape}. Expected 1D array of P(PD) or (N, 2) array.")

    p_true = np.clip(p_true, 0.0, 1.0)
    scores = 1.0 - p_true
    return scores


def calculate_conformal_quantile(scores: np.ndarray, alpha: float = 0.10) -> float:
    """
    Calculate finite-sample split conformal quantile q_hat for target significance level alpha.
    
    Quantile formula:
        q_level = min(1.0, ceil((n + 1) * (1 - alpha)) / n)
        q_hat = Quantile(scores, q_level)
        
    For n=5, alpha=0.10:
        (n+1)(1-alpha) = 6 * 0.90 = 5.4 -> ceil(5.4) = 6 -> q_level = 6/5 = 1.2 -> clipped to 1.0.
        Therefore q_hat = max(scores).
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        raise ValueError("Cannot calculate quantile on empty calibration scores array.")

    k = math.ceil((n + 1) * (1.0 - alpha))
    q_level = min(1.0, k / float(n))

    # Calculate empirical quantile with higher-order interpolation to handle small n exact bounds
    q_hat = float(np.quantile(scores, q_level, method="higher" if hasattr(np, "quantile") else "linear"))
    q_hat = float(np.clip(q_hat, 0.0, 1.0))
    return q_hat


def construct_prediction_set(
    p_hc: float,
    p_pd: float,
    q_hat: float,
    class_names: List[str] = None
) -> Set[str]:
    """
    Construct conformal prediction set for a single subject given class probabilities and q_hat.
    
    A class c is included in C(X) if:
        1 - P(c) <= q_hat  <==>  P(c) >= 1 - q_hat
        
    Possible set outputs:
        - {"Healthy Control"} (Singleton HC)
        - {"Parkinson's Disease"} (Singleton PD)
        - {"Healthy Control", "Parkinson's Disease"} (Ambiguous / Both)
        - set() (Empty set)
    """
    if class_names is None:
        class_names = ["Healthy Control", "Parkinson's Disease"]

    threshold = 1.0 - q_hat
    prediction_set = set()

    if p_hc >= threshold - 1e-9:
        prediction_set.add(class_names[0])
    if p_pd >= threshold - 1e-9:
        prediction_set.add(class_names[1])

    return prediction_set


def format_prediction_set_str(pred_set: Set[str]) -> str:
    """Format prediction set into standardized clean string representation."""
    if not pred_set:
        return "Empty {}"
    elif len(pred_set) == 2:
        return "{Healthy Control, Parkinson's Disease}"
    else:
        return "{" + list(pred_set)[0] + "}"


def evaluate_conformal_predictions(
    predictions_df: pd.DataFrame,
    q_hat: float,
    alpha: float = 0.10
) -> Dict:
    """
    Evaluate conformal prediction sets on a test dataset DataFrame.
    
    Expected columns in predictions_df:
        - subject_id
        - true_label (0=HC, 1=PD)
        - p_hc
        - p_pd
        - prediction_set (Set or formatted string)
        - set_size (int)
        - covered_true_label (int 0 or 1)
    """
    n_total = len(predictions_df)
    if n_total == 0:
        return {}

    empirical_coverage = float(np.mean(predictions_df["covered_true_label"]))
    avg_set_size = float(np.mean(predictions_df["set_size"]))
    singleton_rate = float(np.mean(predictions_df["set_size"] == 1))
    ambiguous_rate = float(np.mean(predictions_df["set_size"] == 2))
    empty_rate = float(np.mean(predictions_df["set_size"] == 0))

    # Class-specific coverage
    hc_df = predictions_df[predictions_df["true_label"] == 0]
    pd_df = predictions_df[predictions_df["true_label"] == 1]

    coverage_hc = float(np.mean(hc_df["covered_true_label"])) if len(hc_df) > 0 else 0.0
    coverage_pd = float(np.mean(pd_df["covered_true_label"])) if len(pd_df) > 0 else 0.0

    # Point prediction metrics (derived from max probability or 0.5 threshold)
    point_preds = (predictions_df["p_pd"] >= 0.5).astype(int)
    true_labels = predictions_df["true_label"].astype(int)
    cm = confusion_matrix(true_labels, point_preds, labels=[0, 1]).tolist()

    return {
        "n_test_subjects": n_total,
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "q_hat": q_hat,
        "empirical_coverage": empirical_coverage,
        "average_set_size": avg_set_size,
        "singleton_rate": singleton_rate,
        "ambiguous_rate": ambiguous_rate,
        "empty_rate": empty_rate,
        "class_coverage": {
            "Healthy_Control": coverage_hc,
            "Parkinsons_Disease": coverage_pd
        },
        "point_prediction_confusion_matrix": cm
    }
