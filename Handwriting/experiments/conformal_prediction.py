"""Split Conformal Prediction calibration and evaluation for Handwriting Modality.

Calibrates finite-sample conformal prediction sets on held-out validation patients
to guarantee 1 - alpha coverage on test patients.
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Support running as a standalone script or module
current_dir = Path(__file__).resolve().parent
hw_dir = current_dir.parent
root_dir = hw_dir.parent
for p in (str(current_dir), str(hw_dir), str(root_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)


def compute_nonconformity_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute nonconformity scores s_i = 1 - P(Y_i | X_i)."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if probs.ndim == 1:
        p_pd = probs
        p_hc = 1.0 - p_pd
        p_true = np.where(labels == 1, p_pd, p_hc)
    else:
        p_true = np.where(labels == 1, probs[:, 1], probs[:, 0])
    return 1.0 - np.clip(p_true, 0.0, 1.0)


def calibrate_conformal_quantile(cal_scores: np.ndarray, alpha: float = 0.10) -> float:
    """
    Compute finite-sample calibrated quantile threshold:
    q_hat = Quantile_{(n+1)(1-alpha)/n}(s_1, ..., s_n)
    """
    n = len(cal_scores)
    if n == 0:
        return 0.5
    level = math.ceil((n + 1) * (1.0 - alpha)) / n
    level = min(1.0, max(0.0, level))
    q_hat = float(np.quantile(cal_scores, level, method="higher"))
    return q_hat


def construct_prediction_set(p_hc: float, p_pd: float, q_hat: float) -> List[str]:
    """Construct conformal prediction set C(x) = { c in {HC, PD} | 1 - P(c|x) <= q_hat }."""
    pred_set = []
    if (1.0 - p_hc) <= q_hat:
        pred_set.append("Healthy Control")
    if (1.0 - p_pd) <= q_hat:
        pred_set.append("Parkinson's Disease")
    return pred_set


def evaluate_conformal_coverage(
    probs: np.ndarray,
    labels: np.ndarray,
    q_hat: float
) -> Dict[str, float]:
    """Evaluate empirical coverage, average set size, and set size distributions."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    if n == 0:
        return {}

    covered = 0
    set_sizes = []
    empty_cnt = 0
    singleton_cnt = 0
    ambiguous_cnt = 0

    for i in range(n):
        p_pd = probs[i] if probs.ndim == 1 else probs[i, 1]
        p_hc = 1.0 - p_pd
        true_label_name = "Parkinson's Disease" if labels[i] == 1 else "Healthy Control"
        c_set = construct_prediction_set(p_hc, p_pd, q_hat)
        sz = len(c_set)
        set_sizes.append(sz)

        if true_label_name in c_set:
            covered += 1
        if sz == 0:
            empty_cnt += 1
        elif sz == 1:
            singleton_cnt += 1
        elif sz == 2:
            ambiguous_cnt += 1

    return {
        "n_samples": n,
        "q_hat": q_hat,
        "empirical_coverage": float(covered / n),
        "mean_set_size": float(np.mean(set_sizes)),
        "empty_rate": float(empty_cnt / n),
        "singleton_rate": float(singleton_cnt / n),
        "ambiguous_rate": float(ambiguous_cnt / n)
    }


def run_conformal_calibration(alpha: float = 0.10) -> Dict:
    """Run conformal calibration using existing validation and test manifests."""
    preds_csv = hw_dir / "outputs" / "handwriting_evidential_predictions.csv"
    if not preds_csv.exists():
        print(f"Predictions file not found at {preds_csv}. Using synthetic calibration demo.", file=sys.stderr)
        cal_scores = np.random.uniform(0.05, 0.40, size=30)
        q_hat = calibrate_conformal_quantile(cal_scores, alpha=alpha)
        metrics = {"q_hat": q_hat, "alpha": alpha, "target_coverage": 1.0 - alpha}
        return metrics

    df = pd.read_csv(preds_csv)
    # Split calibration vs evaluation (80/20 or by split column if present)
    if "split" in df.columns:
        cal_df = df[df["split"] == "val"]
        test_df = df[df["split"] == "test"]
    else:
        # 50% split for calibration and test
        n_half = len(df) // 2
        cal_df = df.iloc[:n_half]
        test_df = df.iloc[n_half:]

    cal_probs = cal_df["p_pd"].to_numpy() if "p_pd" in cal_df.columns else cal_df["probability"].to_numpy()
    cal_labels = cal_df["target"].to_numpy() if "target" in cal_df.columns else cal_df["label"].to_numpy()
    cal_scores = compute_nonconformity_scores(cal_probs, cal_labels)

    q_hat = calibrate_conformal_quantile(cal_scores, alpha=alpha)

    test_probs = test_df["p_pd"].to_numpy() if "p_pd" in test_df.columns else test_df["probability"].to_numpy()
    test_labels = test_df["target"].to_numpy() if "target" in test_df.columns else test_df["label"].to_numpy()
    eval_metrics = evaluate_conformal_coverage(test_probs, test_labels, q_hat)

    out_file = hw_dir / "outputs" / "conformal_prediction_results.json"
    with open(out_file, "w") as f:
        json.dump(eval_metrics, f, indent=2)

    print(f"Handwriting Conformal Calibration (alpha={alpha}): q_hat = {q_hat:.4f}")
    print(f"Empirical Coverage on Test Set: {eval_metrics.get('empirical_coverage', 0.0) * 100:.1f}%")
    print(f"Results saved to: {out_file}")
    return eval_metrics


if __name__ == "__main__":
    run_conformal_calibration()
