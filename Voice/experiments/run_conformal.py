"""Phase 7 Experiment: Voice Conformal Prediction Experiment Runner.

Performs Split Conformal Prediction for binary Parkinson's Disease classification
calibrated on 5 held-out validation subjects (ID02, ID11, ID15, ID21, ID33) and evaluated
on 6 held-out test subjects (ID04, ID10, ID13, ID23, ID29, ID35).

Evaluates BOTH:
    A. Deterministic Baseline Model
    B. Deep Ensemble Model (M=5)

Leakage Guards:
    - Pure inference mode under torch.no_grad().
    - Zero model retraining or fine-tuning.
    - Calibration uses ONLY validation subjects. Test labels are NEVER used to construct sets or compute q_hat.
    - Test set labels used strictly for retrospective coverage and accuracy metrics.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Ensure workspace root and Voice directory are in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
voice_dir = root_dir / "Voice"
for p in (str(root_dir), str(voice_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from Voice import config
from Voice.conformal_prediction import (
    calculate_conformal_quantile,
    compute_nonconformity_scores,
    construct_prediction_set,
    evaluate_conformal_predictions,
    format_prediction_set_str,
)
from Voice.dataset import build_metadata, create_subject_splits
from Voice.model import EvidentialHead, Wav2Vec2ForParkinsons
from Voice.utils import set_seed

CONFORMAL_DIR = config.RESULTS_DIR / "conformal"
CONFORMAL_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_JSON = CONFORMAL_DIR / "conformal_config.json"
RESULTS_JSON = CONFORMAL_DIR / "conformal_results.json"
PREDICTIONS_CSV = CONFORMAL_DIR / "conformal_subject_predictions.csv"
SUMMARY_CSV = CONFORMAL_DIR / "conformal_summary.csv"
FIG_SETS = CONFORMAL_DIR / "conformal_prediction_sets.png"


def run_conformal_experiment(alpha: float = 0.10, seed: int = 42):
    """Run Phase 7 Voice Conformal Prediction experiment."""
    print("=" * 70)
    print("VOICE PHASE 7: CONFORMAL PREDICTION EXPERIMENT")
    print(f"Target Coverage: {(1.0 - alpha) * 100:.1f}% (Significance Level alpha = {alpha:.2f})")
    print("=" * 70)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data Splits and Unique Subject IDs
    meta_df = build_metadata()
    train_df, val_df, test_df = create_subject_splits(meta_df, seed=seed)

    val_subjects = sorted(val_df["subject_id"].unique().tolist())
    test_subjects = sorted(test_df["subject_id"].unique().tolist())

    print(f"\nCalibration Set (Validation Subjects, n={len(val_subjects)}): {val_subjects}")
    print(f"Test Set (Held-Out Test Subjects, n={len(test_subjects)}):        {test_subjects}\n")

    # Subject True Labels Map
    sub_labels = {}
    for idx, row in meta_df.iterrows():
        sub_labels[row["subject_id"]] = int(row["label"])

    # Load 768-D clean subject representations
    embs_file = config.RESULTS_DIR / "embeddings" / "voice_embeddings.pt"
    embs_dict = torch.load(embs_file)
    sub_embeddings = embs_dict["subject_embeddings"]

    # Load Baseline Head
    baseline_head = EvidentialHead(in_features=768, num_classes=2, dropout=0.1)
    baseline_head.to(device)
    baseline_head.eval()

    # Load 5 Ensemble Member Models
    ensemble_models = []
    for idx in range(5):
        ckpt = config.RESULTS_DIR / "deep_ensemble" / f"member_{idx}" / "checkpoint" / "best_model.pt"
        m = Wav2Vec2ForParkinsons(model_name=config.MODEL_NAME, num_classes=2)
        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        m.to(device)
        m.eval()
        ensemble_models.append(m)

    # 2. Extract Probabilities for Calibration Set (Validation Subjects)
    val_records = []
    with torch.no_grad():
        for sub in val_subjects:
            y_true = sub_labels[sub]
            emb = sub_embeddings[sub].unsqueeze(0).to(device)

            # Baseline
            base_ev = baseline_head(emb)
            base_p_pd = float(torch.softmax(base_ev, dim=-1)[0, 1].cpu().item())
            base_p_hc = 1.0 - base_p_pd

            # Ensemble
            m_probs = []
            for m in ensemble_models:
                m_ev = m(emb)
                m_p = float(torch.softmax(m_ev, dim=-1)[0, 1].cpu().item())
                m_probs.append(m_p)
            ens_p_pd = float(np.mean(m_probs))
            ens_p_hc = 1.0 - ens_p_pd

            val_records.append({
                "subject_id": sub,
                "true_label": y_true,
                "base_p_hc": base_p_hc,
                "base_p_pd": base_p_pd,
                "ens_p_hc": ens_p_hc,
                "ens_p_pd": ens_p_pd
            })

    val_res_df = pd.DataFrame(val_records)

    # Compute Calibration Nonconformity Scores s_i = 1 - P(Y_i | X_i)
    base_val_scores = compute_nonconformity_scores(
        val_res_df[["base_p_hc", "base_p_pd"]].values,
        val_res_df["true_label"].values
    )
    ens_val_scores = compute_nonconformity_scores(
        val_res_df[["ens_p_hc", "ens_p_pd"]].values,
        val_res_df["true_label"].values
    )

    # 3. Calculate Finite-Sample Conformal Quantile q_hat
    q_hat_base = calculate_conformal_quantile(base_val_scores, alpha=alpha)
    q_hat_ens = calculate_conformal_quantile(ens_val_scores, alpha=alpha)

    print("-" * 70)
    print(f"CALIBRATION QUANTILE RESULTS (n={len(val_subjects)}, alpha={alpha:.2f}):")
    print(f"  Baseline q_hat:      {q_hat_base:.6f}  (Nonconformity scores: {np.round(base_val_scores, 4).tolist()})")
    print(f"  Deep Ensemble q_hat: {q_hat_ens:.6f}  (Nonconformity scores: {np.round(ens_val_scores, 4).tolist()})")
    print("-" * 70)

    # 4. Perform Conformal Inference on Held-Out Test Subjects
    test_records = []
    class_names = ["Healthy Control", "Parkinson's Disease"]

    with torch.no_grad():
        for sub in test_subjects:
            y_true = sub_labels[sub]
            true_name = class_names[y_true]
            emb = sub_embeddings[sub].unsqueeze(0).to(device)

            # A. Baseline Model
            base_ev = baseline_head(emb)
            base_p_pd = float(torch.softmax(base_ev, dim=-1)[0, 1].cpu().item())
            base_p_hc = 1.0 - base_p_pd
            base_point_pred = 1 if base_p_pd >= 0.5 else 0

            base_set = construct_prediction_set(base_p_hc, base_p_pd, q_hat_base, class_names)
            base_set_str = format_prediction_set_str(base_set)
            base_covered = int(true_name in base_set)
            base_s_true = 1.0 - (base_p_pd if y_true == 1 else base_p_hc)

            # B. Deep Ensemble Model
            m_probs = []
            for m in ensemble_models:
                m_ev = m(emb)
                m_p = float(torch.softmax(m_ev, dim=-1)[0, 1].cpu().item())
                m_probs.append(m_p)
            ens_p_pd = float(np.mean(m_probs))
            ens_p_hc = 1.0 - ens_p_pd
            ens_point_pred = 1 if ens_p_pd >= 0.5 else 0

            ens_set = construct_prediction_set(ens_p_hc, ens_p_pd, q_hat_ens, class_names)
            ens_set_str = format_prediction_set_str(ens_set)
            ens_covered = int(true_name in ens_set)
            ens_s_true = 1.0 - (ens_p_pd if y_true == 1 else ens_p_hc)

            test_records.append({
                "subject_id": sub,
                "true_label": y_true,
                "true_name": true_name,
                
                # Baseline Conformal Output
                "base_p_hc": base_p_hc,
                "base_p_pd": base_p_pd,
                "base_point_pred": base_point_pred,
                "base_nonconformity_score": base_s_true,
                "base_q_hat": q_hat_base,
                "base_prediction_set": base_set_str,
                "base_set_size": len(base_set),
                "base_covered_true_label": base_covered,

                # Deep Ensemble Conformal Output
                "ens_p_hc": ens_p_hc,
                "ens_p_pd": ens_p_pd,
                "ens_point_pred": ens_point_pred,
                "ens_nonconformity_score": ens_s_true,
                "ens_q_hat": q_hat_ens,
                "ens_prediction_set": ens_set_str,
                "ens_set_size": len(ens_set),
                "ens_covered_true_label": ens_covered
            })

    test_res_df = pd.DataFrame(test_records)
    test_res_df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"\nSaved conformal subject predictions to: {PREDICTIONS_CSV}\n")

    # 5. Compute Conformal Evaluation Metrics
    base_eval_df = pd.DataFrame({
        "subject_id": test_res_df["subject_id"],
        "true_label": test_res_df["true_label"],
        "p_hc": test_res_df["base_p_hc"],
        "p_pd": test_res_df["base_p_pd"],
        "prediction_set": test_res_df["base_prediction_set"],
        "set_size": test_res_df["base_set_size"],
        "covered_true_label": test_res_df["base_covered_true_label"]
    })
    base_metrics = evaluate_conformal_predictions(base_eval_df, q_hat_base, alpha=alpha)

    ens_eval_df = pd.DataFrame({
        "subject_id": test_res_df["subject_id"],
        "true_label": test_res_df["true_label"],
        "p_hc": test_res_df["ens_p_hc"],
        "p_pd": test_res_df["ens_p_pd"],
        "prediction_set": test_res_df["ens_prediction_set"],
        "set_size": test_res_df["ens_set_size"],
        "covered_true_label": test_res_df["ens_covered_true_label"]
    })
    ens_metrics = evaluate_conformal_predictions(ens_eval_df, q_hat_ens, alpha=alpha)

    # 6. Save Configuration and Results JSON
    config_data = {
        "experiment": "Voice Wav2Vec2 Split Conformal Prediction",
        "seed": seed,
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "calibration_subjects": val_subjects,
        "test_subjects": test_subjects,
        "n_calibration": len(val_subjects),
        "n_test": len(test_subjects),
        "small_sample_warning": "n=5 calibration subjects yields coarse finite-sample quantile (q_hat = max(s_i)). Results are exploratory and do not constitute clinical validation."
    }
    with open(CONFIG_JSON, "w") as f:
        json.dump(config_data, f, indent=4)

    results_data = {
        "config": config_data,
        "baseline": base_metrics,
        "deep_ensemble": ens_metrics
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(results_data, f, indent=4)
    print(f"Saved conformal results JSON to: {RESULTS_JSON}")

    # Summary CSV
    summary_rows = [
        {
            "model_type": "Baseline",
            "alpha": alpha,
            "q_hat": q_hat_base,
            "empirical_coverage": base_metrics["empirical_coverage"],
            "avg_set_size": base_metrics["average_set_size"],
            "singleton_rate": base_metrics["singleton_rate"],
            "ambiguous_rate": base_metrics["ambiguous_rate"],
            "empty_rate": base_metrics["empty_rate"],
            "hc_coverage": base_metrics["class_coverage"]["Healthy_Control"],
            "pd_coverage": base_metrics["class_coverage"]["Parkinsons_Disease"]
        },
        {
            "model_type": "Deep Ensemble (M=5)",
            "alpha": alpha,
            "q_hat": q_hat_ens,
            "empirical_coverage": ens_metrics["empirical_coverage"],
            "avg_set_size": ens_metrics["average_set_size"],
            "singleton_rate": ens_metrics["singleton_rate"],
            "ambiguous_rate": ens_metrics["ambiguous_rate"],
            "empty_rate": ens_metrics["empty_rate"],
            "hc_coverage": ens_metrics["class_coverage"]["Healthy_Control"],
            "pd_coverage": ens_metrics["class_coverage"]["Parkinsons_Disease"]
        }
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved conformal summary CSV to: {SUMMARY_CSV}\n")

    # 7. Generate Plot Visualization
    generate_conformal_plot(test_res_df, q_hat_base, q_hat_ens, alpha)

    # 8. Print Results Banners
    print("=" * 80)
    print("VOICE CONFORMAL PREDICTION RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} | {'q_hat':<8} | {'Coverage':<10} | {'Avg Size':<9} | {'Singleton':<10} | {'Ambiguous':<10} | {'Empty':<8}")
    print("-" * 80)
    for _, r in summary_df.iterrows():
        print(
            f"{r['model_type']:<20} | {r['q_hat']:<8.4f} | {r['empirical_coverage']*100:<9.1f}% | "
            f"{r['avg_set_size']:<9.2f} | {r['singleton_rate']*100:<9.1f}% | {r['ambiguous_rate']*100:<9.1f}% | {r['empty_rate']*100:<7.1f}%"
        )
    print("=" * 80)

    print("\nINDIVIDUAL TEST SUBJECT CONFORMAL PREDICTION SETS:")
    print("-" * 80)
    for _, r in test_res_df.iterrows():
        print(
            f"Subject {r['subject_id']} (True: {r['true_name']:<18}) | "
            f"Baseline: {r['base_prediction_set']:<42} (Covered: {r['base_covered_true_label']}) | "
            f"Ensemble: {r['ens_prediction_set']:<42} (Covered: {r['ens_covered_true_label']})"
        )
    print("=" * 80 + "\n")

    return results_data


def generate_conformal_plot(df: pd.DataFrame, q_hat_base: float, q_hat_ens: float, alpha: float):
    """Generate visualization plot comparing Baseline vs Ensemble conformal prediction sets."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    sub_ids = df["subject_id"].tolist()
    x = np.arange(len(sub_ids))

    # Baseline Plot
    ax1.scatter(x, df["base_p_pd"], color="#1f77b4", s=80, zorder=3, label="Baseline P(PD)")
    ax1.axhline(1.0 - q_hat_base, color="#d62728", linestyle="--", lw=1.5, label=f"Lower Threshold (1-q_hat={1-q_hat_base:.3f})")
    ax1.axhline(q_hat_base, color="#2ca02c", linestyle="--", lw=1.5, label=f"Upper Threshold (q_hat={q_hat_base:.3f})")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sub_ids)
    ax1.set_ylabel("Probability P(PD)")
    ax1.set_title(f"A. Baseline Conformal Sets (q_hat={q_hat_base:.3f})", weight="bold")
    ax1.set_ylim([-0.05, 1.05])
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8)

    # Ensemble Plot
    ax2.scatter(x, df["ens_p_pd"], color="#9467bd", s=80, zorder=3, label="Ensemble Mean P(PD)")
    ax2.axhline(1.0 - q_hat_ens, color="#d62728", linestyle="--", lw=1.5, label=f"Lower Threshold (1-q_hat={1-q_hat_ens:.3f})")
    ax2.axhline(q_hat_ens, color="#2ca02c", linestyle="--", lw=1.5, label=f"Upper Threshold (q_hat={q_hat_ens:.3f})")
    ax2.set_xticks(x)
    ax2.set_xticklabels(sub_ids)
    ax2.set_ylabel("Probability P(PD)")
    ax2.set_title(f"B. Deep Ensemble Conformal Sets (q_hat={q_hat_ens:.3f})", weight="bold")
    ax2.set_ylim([-0.05, 1.05])
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8)

    plt.suptitle(f"Voice Wav2Vec2 Split Conformal Prediction Sets (Target Coverage: {(1-alpha)*100:.0f}%)", weight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_SETS, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_conformal_experiment()
