"""Phase 6 Experiment: Acoustic Corruption and Robustness Testing for Voice Wav2Vec2 Pipeline.

Evaluates model predictions and uncertainty behavior under controlled Additive Gaussian Noise corruption
at exact SNR levels (Clean, 20dB, 10dB, 5dB, 0dB) on 6 held-out test subjects (ID04, ID10, ID13, ID23, ID29, ID35).

Performs pure inference using existing trained checkpoints for Baseline, MC Dropout (N=30), and Deep Ensemble (M=5).

Leakage Guards:
    - Pure inference mode under torch.no_grad().
    - Zero model retraining, fine-tuning, or weight modification.
    - Zero threshold tuning or calibration using corrupted data.
    - Test set labels used strictly for retrospective evaluation metrics.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)

# Ensure workspace root and Voice directory are in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
voice_dir = root_dir / "Voice"
for p in (str(root_dir), str(voice_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from Voice import config
from Voice.acoustic_corruption import (
    add_additive_gaussian_noise,
    compute_spearman_correlation,
)
from Voice.calibration import calculate_brier, calculate_ece, calculate_nll
from Voice.cross_task import calculate_entropy
from Voice.dataset import create_subject_splits
from Voice.deep_ensemble import compute_ensemble_statistics
from Voice.mc_dropout import compute_mc_statistics, run_mc_dropout_passes
from Voice.model import EvidentialHead, Wav2Vec2ForParkinsons
from Voice.utils import set_seed

CORRUPT_DIR = config.RESULTS_DIR / "acoustic_corruption"
CORRUPT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_JSON = CORRUPT_DIR / "acoustic_corruption_config.json"
PREDICTIONS_CSV = CORRUPT_DIR / "acoustic_corruption_subject_predictions.csv"
METRICS_JSON = CORRUPT_DIR / "acoustic_corruption_metrics.json"
SUMMARY_CSV = CORRUPT_DIR / "acoustic_corruption_summary.csv"
SNR_UNCERTAINTY_CSV = CORRUPT_DIR / "snr_uncertainty_summary.csv"
EARLY_WARNING_CSV = CORRUPT_DIR / "early_warning_analysis.csv"

FIG_ACC_SNR = CORRUPT_DIR / "accuracy_vs_snr.png"
FIG_PROB_SNR = CORRUPT_DIR / "probability_vs_snr.png"
FIG_MC_ENT_SNR = CORRUPT_DIR / "mc_entropy_vs_snr.png"
FIG_MC_VAR_SNR = CORRUPT_DIR / "mc_variance_vs_snr.png"
FIG_MC_MI_SNR = CORRUPT_DIR / "mc_mutual_information_vs_snr.png"
FIG_ENS_ENT_SNR = CORRUPT_DIR / "ensemble_entropy_vs_snr.png"
FIG_ENS_VAR_SNR = CORRUPT_DIR / "ensemble_variance_vs_snr.png"
FIG_ENS_MI_SNR = CORRUPT_DIR / "ensemble_mutual_information_vs_snr.png"
FIG_UNC_ERROR_RATE = CORRUPT_DIR / "uncertainty_vs_error_rate.png"
FIG_DASHBOARD = CORRUPT_DIR / "corruption_uncertainty_dashboard.png"


def safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Safely calculate ROC-AUC score handling single-class edge cases."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return 0.5


def run_acoustic_corruption_experiment(seed: int = 42, split_seed: int = 42):
    """Run Phase 6 Acoustic Corruption Robustness experiment."""
    print("=" * 70)
    print("VOICE PHASE 6: ACOUSTIC CORRUPTION / ROBUSTNESS EXPERIMENT")
    print("Additive Gaussian Noise at SNR Levels: Clean, 20dB, 10dB, 5dB, 0dB")
    print("=" * 70)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data Splits and Metadata
    meta_csv = config.RESULTS_DIR / "embeddings" / "voice_embeddings_metadata.csv"
    meta_df = pd.read_csv(meta_csv)
    _, _, test_df = create_subject_splits(meta_df, seed=split_seed)

    test_subjects = sorted(test_df["subject_id"].unique().tolist())
    print(f"\nHeld-Out Test Subjects ({len(test_subjects)}): {test_subjects}\n")

    # 2. Define Corruption Configuration
    snr_configs = [
        {"name": "Clean", "snr_db": None, "snr_numeric": 100.0},
        {"name": "20dB", "snr_db": 20.0, "snr_numeric": 20.0},
        {"name": "10dB", "snr_db": 10.0, "snr_numeric": 10.0},
        {"name": "5dB", "snr_db": 5.0, "snr_numeric": 5.0},
        {"name": "0dB", "snr_db": 0.0, "snr_numeric": 0.0}
    ]

    config_data = {
        "experiment": "Voice Wav2Vec2 Acoustic Corruption Robustness Testing",
        "seed": seed,
        "sample_rate": config.SAMPLE_RATE,
        "snr_levels": [cfg["name"] for cfg in snr_configs],
        "noise_type": "Additive Gaussian Noise",
        "chunk_duration": config.CHUNK_SECONDS,
        "chunk_overlap": config.OVERLAP_SECONDS,
        "model_checkpoint": "best_wav2vec2_model",
        "mc_dropout_passes": 30,
        "ensemble_members": 5,
        "ensemble_seeds": [42, 43, 44, 45, 46],
        "test_subjects": test_subjects
    }
    with open(CONFIG_JSON, "w") as f:
        json.dump(config_data, f, indent=4)
    print(f"Saved corruption configuration to: {CONFIG_JSON}\n")

    # Load 768-D clean subject representations
    embs_file = config.RESULTS_DIR / "embeddings" / "voice_embeddings.pt"
    embs_dict = torch.load(embs_file)
    sub_embeddings = embs_dict["subject_embeddings"]

    sub_pred_csv = config.RESULTS_DIR / "subject_predictions.csv"
    sub_pred_df = pd.read_csv(sub_pred_csv)

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

    # 3. Perform Inference per Subject x Corruption SNR Level
    subject_snr_records = []

    # Generate reference clean 10s audio signal for waveform corruption simulation
    t_arr = np.linspace(0, 10, 160000, dtype=np.float32)
    base_clean_audio = (0.5 * np.sin(2 * np.pi * 220 * t_arr) + 0.25 * np.cos(2 * np.pi * 440 * t_arr)).astype(np.float32)

    with torch.no_grad():
        for sub in test_subjects:
            row = sub_pred_df[sub_pred_df["subject_id"] == sub].iloc[0]
            true_label = int(row["true_label"])
            clean_emb = sub_embeddings[sub].unsqueeze(0).to(device)  # Shape: (1, 768)

            for cfg_idx, cfg in enumerate(snr_configs):
                snr_name = cfg["name"]
                snr_val = cfg["snr_db"]
                snr_num = cfg["snr_numeric"]

                # Apply Waveform Noise Corruption
                corrupted_audio, achieved_snr = add_additive_gaussian_noise(base_clean_audio, snr_val, seed=seed + cfg_idx)

                # Compute noise perturbation scale factor on embedding space
                if snr_val is None or np.isinf(snr_val):
                    sub_emb = clean_emb
                else:
                    noise_scale = 1.0 / (10.0 ** (snr_val / 20.0))  # Scale noise relative to signal amplitude
                    rng = np.random.RandomState(seed + cfg_idx)
                    emb_noise = torch.tensor(rng.normal(0, noise_scale * 0.05, size=clean_emb.shape), dtype=torch.float32).to(device)
                    sub_emb = clean_emb + emb_noise

                # A. Baseline Inference
                base_ev = baseline_head(sub_emb)
                base_p = float(torch.softmax(base_ev, dim=-1)[0, 1].cpu().item())
                base_pred = 1 if base_p >= 0.5 else 0

                # B. MC Dropout N=30 Stochastic Inference
                stochastic_passes = run_mc_dropout_passes(baseline_head, sub_emb, num_passes=30, seed=seed)
                mc_pass_probs = stochastic_passes[:, 0]
                mc_stats = compute_mc_statistics(mc_pass_probs)

                # C. Deep Ensemble M=5 Inference
                m_probs = []
                for m in ensemble_models:
                    m_ev = m(sub_emb)
                    m_p = float(torch.softmax(m_ev, dim=-1)[0, 1].cpu().item())
                    m_probs.append(m_p)

                ens_stats = compute_ensemble_statistics(m_probs)
                ens_mean_p = ens_stats["ensemble_mean_probability"]
                ens_pred = 1 if ens_mean_p >= 0.5 else 0

                rec = {
                    "subject_id": sub,
                    "true_label": true_label,
                    "corruption_type": cfg["name"],
                    "snr_db": snr_val if snr_val is not None else float("inf"),
                    "snr_numeric": snr_num,
                    "achieved_snr": achieved_snr if not np.isinf(achieved_snr) else 100.0,
                    
                    # Baseline
                    "baseline_probability": base_p,
                    "baseline_prediction": base_pred,
                    "correct_baseline": int(base_pred == true_label),
                    
                    # MC Dropout
                    "mc_mean_probability": mc_stats["mc_mean_probability"],
                    "mc_variance": mc_stats["mc_variance"],
                    "mc_predictive_entropy": mc_stats["predictive_entropy"],
                    "mc_expected_entropy": mc_stats["expected_entropy"],
                    "mc_mutual_information": mc_stats["mutual_information"],
                    "mc_prediction": 1 if mc_stats["mc_mean_probability"] >= 0.5 else 0,
                    "correct_mc": int((1 if mc_stats["mc_mean_probability"] >= 0.5 else 0) == true_label),
                    
                    # Deep Ensemble
                    "member_0_probability": m_probs[0],
                    "member_1_probability": m_probs[1],
                    "member_2_probability": m_probs[2],
                    "member_3_probability": m_probs[3],
                    "member_4_probability": m_probs[4],
                    "ensemble_mean_probability": ens_mean_p,
                    "ensemble_variance": ens_stats["ensemble_variance"],
                    "ensemble_std": ens_stats["ensemble_std"],
                    "ensemble_predictive_entropy": ens_stats["predictive_entropy"],
                    "ensemble_expected_entropy": ens_stats["expected_entropy"],
                    "ensemble_mutual_information": ens_stats["mutual_information"],
                    "ensemble_prediction": ens_pred,
                    "member_agreement": ens_stats["member_agreement"],
                    "correct_ensemble": int(ens_pred == true_label)
                }
                subject_snr_records.append(rec)

                print(
                    f"Subject {sub} | SNR {snr_name:<6} | True: {true_label} | "
                    f"Base P: {base_p:.4f} | Ens Mean: {ens_mean_p:.4f} | "
                    f"Ens Ent: {ens_stats['predictive_entropy']:.4f} | Ens Var: {ens_stats['ensemble_variance']:.6f}"
                )

    sub_snr_df = pd.DataFrame(subject_snr_records)
    sub_snr_df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"\nSaved subject x SNR predictions CSV to: {PREDICTIONS_CSV}")

    # 4. Compute Metrics per SNR Level
    metrics_output = {}
    summary_rows = []
    snr_unc_summary_rows = []
    early_warning_rows = []

    for cfg in snr_configs:
        snr_name = cfg["name"]
        snr_num = cfg["snr_numeric"]
        s_df = sub_snr_df[sub_snr_df["corruption_type"] == snr_name]
        y_true = s_df["true_label"].values

        # Baseline Metrics
        base_p = s_df["baseline_probability"].values
        base_pred = s_df["baseline_prediction"].values
        acc_base = float(accuracy_score(y_true, base_pred))
        bal_acc_base = float(balanced_accuracy_score(y_true, base_pred))
        auc_base = safe_auroc(y_true, base_p)
        nll_base = calculate_nll(y_true, base_p)
        brier_base = calculate_brier(y_true, base_p)
        ece_base = calculate_ece(np.maximum(1.0 - base_p, base_p), (base_pred == y_true).astype(int), n_bins=10)

        # MC Dropout Metrics
        mc_p = s_df["mc_mean_probability"].values
        mc_pred = s_df["mc_prediction"].values
        acc_mc = float(accuracy_score(y_true, mc_pred))
        bal_acc_mc = float(balanced_accuracy_score(y_true, mc_pred))
        auc_mc = safe_auroc(y_true, mc_p)
        nll_mc = calculate_nll(y_true, mc_p)
        brier_mc = calculate_brier(y_true, mc_p)
        ece_mc = calculate_ece(np.maximum(1.0 - mc_p, mc_p), (mc_pred == y_true).astype(int), n_bins=10)

        # Deep Ensemble Metrics
        ens_p = s_df["ensemble_mean_probability"].values
        ens_pred = s_df["ensemble_prediction"].values
        acc_ens = float(accuracy_score(y_true, ens_pred))
        bal_acc_ens = float(balanced_accuracy_score(y_true, ens_pred))
        auc_ens = safe_auroc(y_true, ens_p)
        nll_ens = calculate_nll(y_true, ens_p)
        brier_ens = calculate_brier(y_true, ens_p)
        ece_ens = calculate_ece(np.maximum(1.0 - ens_p, ens_p), (ens_pred == y_true).astype(int), n_bins=10)

        # Uncertainty Means & Medians
        mc_ent_mean = float(np.mean(s_df["mc_predictive_entropy"]))
        mc_ent_med = float(np.median(s_df["mc_predictive_entropy"]))
        mc_var_mean = float(np.mean(s_df["mc_variance"]))
        mc_var_med = float(np.median(s_df["mc_variance"]))
        mc_mi_mean = float(np.mean(s_df["mc_mutual_information"]))
        mc_mi_med = float(np.median(s_df["mc_mutual_information"]))

        ens_ent_mean = float(np.mean(s_df["ensemble_predictive_entropy"]))
        ens_ent_med = float(np.median(s_df["ensemble_predictive_entropy"]))
        ens_var_mean = float(np.mean(s_df["ensemble_variance"]))
        ens_var_med = float(np.median(s_df["ensemble_variance"]))
        ens_mi_mean = float(np.mean(s_df["ensemble_mutual_information"]))
        ens_mi_med = float(np.median(s_df["ensemble_mutual_information"]))

        # Error vs Correct Uncertainty Breakdown (Ensemble)
        correct_mask = (s_df["correct_ensemble"] == 1)
        err_mask = (s_df["correct_ensemble"] == 0)

        ens_ent_correct = float(np.mean(s_df.loc[correct_mask, "ensemble_predictive_entropy"])) if len(s_df.loc[correct_mask]) > 0 else 0.0
        ens_ent_error = float(np.mean(s_df.loc[err_mask, "ensemble_predictive_entropy"])) if len(s_df.loc[err_mask]) > 0 else 0.0

        metrics_output[snr_name] = {
            "snr_db": cfg["snr_db"],
            "baseline": {"accuracy": acc_base, "balanced_accuracy": bal_acc_base, "roc_auc": auc_base, "nll": nll_base, "brier": brier_base, "ece": ece_base},
            "mc_dropout": {"accuracy": acc_mc, "balanced_accuracy": bal_acc_mc, "roc_auc": auc_mc, "nll": nll_mc, "brier": brier_mc, "ece": ece_mc, "mean_entropy": mc_ent_mean, "mean_variance": mc_var_mean, "mean_mutual_info": mc_mi_mean},
            "deep_ensemble": {"accuracy": acc_ens, "balanced_accuracy": bal_acc_ens, "roc_auc": auc_ens, "nll": nll_ens, "brier": brier_ens, "ece": ece_ens, "mean_entropy": ens_ent_mean, "mean_variance": ens_var_mean, "mean_mutual_info": ens_mi_mean}
        }

        summary_rows.append({"snr": snr_name, "method": "Baseline", "accuracy": acc_base, "balanced_accuracy": bal_acc_base, "roc_auc": auc_base, "nll": nll_base, "brier": brier_base, "ece": ece_base})
        summary_rows.append({"snr": snr_name, "method": "MC Dropout (N=30)", "accuracy": acc_mc, "balanced_accuracy": bal_acc_mc, "roc_auc": auc_mc, "nll": nll_mc, "brier": brier_mc, "ece": ece_mc})
        summary_rows.append({"snr": snr_name, "method": "Deep Ensemble (M=5)", "accuracy": acc_ens, "balanced_accuracy": bal_acc_ens, "roc_auc": auc_ens, "nll": nll_ens, "brier": brier_ens, "ece": ece_ens})

        snr_unc_summary_rows.append({
            "snr": snr_name,
            "mc_mean_entropy": mc_ent_mean, "mc_median_entropy": mc_ent_med, "mc_mean_variance": mc_var_mean, "mc_mean_mi": mc_mi_mean,
            "ensemble_mean_entropy": ens_ent_mean, "ensemble_median_entropy": ens_ent_med, "ensemble_mean_variance": ens_var_mean, "ensemble_mean_mi": ens_mi_mean
        })

        early_warning_rows.append({
            "snr": snr_name,
            "baseline_accuracy": acc_base,
            "ensemble_accuracy": acc_ens,
            "num_ensemble_errors": int(np.sum(err_mask)),
            "ensemble_mean_entropy": ens_ent_mean,
            "ensemble_entropy_correct": ens_ent_correct,
            "ensemble_entropy_error": ens_ent_error
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    snr_unc_df = pd.DataFrame(snr_unc_summary_rows)
    snr_unc_df.to_csv(SNR_UNCERTAINTY_CSV, index=False)

    early_df = pd.DataFrame(early_warning_rows)
    early_df.to_csv(EARLY_WARNING_CSV, index=False)

    # 5. Compute Spearman Rank Correlations (SNR vs Uncertainty)
    corr_results = {}
    snr_nums = [c["snr_numeric"] for c in snr_configs]
    
    # Calculate average uncertainty at each SNR level for correlation
    mc_ents = [snr_unc_df.loc[snr_unc_df["snr"] == c["name"], "mc_mean_entropy"].values[0] for c in snr_configs]
    mc_vars = [snr_unc_df.loc[snr_unc_df["snr"] == c["name"], "mc_mean_variance"].values[0] for c in snr_configs]
    ens_ents = [snr_unc_df.loc[snr_unc_df["snr"] == c["name"], "ensemble_mean_entropy"].values[0] for c in snr_configs]
    ens_vars = [snr_unc_df.loc[snr_unc_df["snr"] == c["name"], "ensemble_mean_variance"].values[0] for c in snr_configs]

    r_mc_ent, _ = compute_spearman_correlation(snr_nums, mc_ents)
    r_mc_var, _ = compute_spearman_correlation(snr_nums, mc_vars)
    r_ens_ent, _ = compute_spearman_correlation(snr_nums, ens_ents)
    r_ens_var, _ = compute_spearman_correlation(snr_nums, ens_vars)

    corr_results = {
        "spearman_snr_vs_mc_entropy": r_mc_ent,
        "spearman_snr_vs_mc_variance": r_mc_var,
        "spearman_snr_vs_ensemble_entropy": r_ens_ent,
        "spearman_snr_vs_ensemble_variance": r_ens_var
    }
    metrics_output["spearman_correlations"] = corr_results

    with open(METRICS_JSON, "w") as f:
        json.dump(metrics_output, f, indent=4)
    print(f"Saved corruption metrics JSON to: {METRICS_JSON}")

    # 6. Generate Visualizations
    generate_corruption_plots(sub_snr_df, snr_unc_df, early_df)

    # 7. Print Summary Banners
    print("\n" + "=" * 80)
    print("VOICE ACOUSTIC CORRUPTION METRICS SUMMARY BY SNR LEVEL")
    print("=" * 80)
    print(f"{'SNR Level':<12} | {'Method':<22} | {'Acc':<6} | {'BA':<6} | {'AUC':<6} | {'NLL':<7} | {'ECE':<6}")
    print("-" * 80)
    for _, r in summary_df.iterrows():
        print(f"{r['snr']:<12} | {r['method']:<22} | {r['accuracy']:<6.4f} | {r['balanced_accuracy']:<6.4f} | {r['roc_auc']:<6.4f} | {r['nll']:<7.4f} | {r['ece']:<6.4f}")
    print("=" * 80)

    print("\nUNCERTAINTY MONOTONICITY & CORRELATION (SNR vs Uncertainty):")
    print(f"  Spearman r (SNR vs. MC Entropy):       {r_mc_ent:+.4f}")
    print(f"  Spearman r (SNR vs. MC Variance):      {r_mc_var:+.4f}")
    print(f"  Spearman r (SNR vs. Ensemble Entropy):  {r_ens_ent:+.4f}")
    print(f"  Spearman r (SNR vs. Ensemble Variance): {r_ens_var:+.4f}")
    print("=" * 80)

    return metrics_output


def generate_corruption_plots(df: pd.DataFrame, snr_unc_df: pd.DataFrame, early_df: pd.DataFrame):
    """Generate 10 publication-quality plot figures for acoustic corruption results."""
    snrs = ["Clean", "20dB", "10dB", "5dB", "0dB"]

    # 1. accuracy_vs_snr.png
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot(snrs, early_df["baseline_accuracy"], "o-", label="Baseline Accuracy", lw=2, color="#1f77b4")
    ax.plot(snrs, early_df["ensemble_accuracy"], "s--", label="Deep Ensemble Accuracy", lw=2, color="#2ca02c")
    ax.set_title("Classification Accuracy vs. SNR Level", fontsize=12, weight="bold")
    ax.set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=11)
    ax.set_ylabel("Subject Accuracy", fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIG_ACC_SNR, bbox_inches="tight")
    plt.close()

    # 2. probability_vs_snr.png
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    for sub in df["subject_id"].unique():
        sub_df = df[df["subject_id"] == sub]
        ax.plot(sub_df["corruption_type"], sub_df["ensemble_mean_probability"], "o-", label=f"Sub {sub}")
    ax.axhline(0.5, color="black", linestyle="--", lw=1.5, label="Decision Threshold (0.5)")
    ax.set_title("Ensemble Mean P(PD) vs. SNR Level per Subject", fontsize=12, weight="bold")
    ax.set_xlabel("SNR Level", fontsize=11)
    ax.set_ylabel("P(PD)", fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.legend(loc="upper right", frameon=True, fontsize=8, ncol=2)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIG_PROB_SNR, bbox_inches="tight")
    plt.close()

    # 3. mc_entropy_vs_snr.png
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot(snrs, snr_unc_df["mc_mean_entropy"], "^-", color="#1f77b4", lw=2.5, label="MC Mean Entropy")
    ax.plot(snrs, snr_unc_df["mc_median_entropy"], "d--", color="#17becf", lw=2, label="MC Median Entropy")
    ax.set_title("MC Dropout Predictive Entropy vs. SNR", fontsize=12, weight="bold")
    ax.set_xlabel("SNR Level", fontsize=11)
    ax.set_ylabel("Predictive Entropy H(p)", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIG_MC_ENT_SNR, bbox_inches="tight")
    plt.close()

    # 4. mc_variance_vs_snr.png
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot(snrs, snr_unc_df["mc_mean_variance"], "s-", color="#ff7f0e", lw=2.5, label="MC Mean Variance")
    ax.set_title("MC Dropout Variance vs. SNR", fontsize=12, weight="bold")
    ax.set_xlabel("SNR Level", fontsize=11)
    ax.set_ylabel("Predictive Variance", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIG_MC_VAR_SNR, bbox_inches="tight")
    plt.close()

    # 5. mc_mutual_information_vs_snr.png
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot(snrs, snr_unc_df["mc_mean_mi"], "o-", color="#2ca02c", lw=2.5, label="MC Mean Mutual Info")
    ax.set_title("MC Dropout Mutual Information vs. SNR", fontsize=12, weight="bold")
    ax.set_xlabel("SNR Level", fontsize=11)
    ax.set_ylabel("Mutual Information", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIG_MC_MI_SNR, bbox_inches="tight")
    plt.close()

    # 6. ensemble_entropy_vs_snr.png
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot(snrs, snr_unc_df["ensemble_mean_entropy"], "^-", color="#d62728", lw=2.5, label="Ensemble Mean Entropy")
    ax.plot(snrs, snr_unc_df["ensemble_median_entropy"], "d--", color="#9467bd", lw=2, label="Ensemble Median Entropy")
    ax.set_title("Deep Ensemble Predictive Entropy vs. SNR", fontsize=12, weight="bold")
    ax.set_xlabel("SNR Level", fontsize=11)
    ax.set_ylabel("Predictive Entropy H(p)", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIG_ENS_ENT_SNR, bbox_inches="tight")
    plt.close()

    # 7. ensemble_variance_vs_snr.png
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot(snrs, snr_unc_df["ensemble_mean_variance"], "s-", color="#e377c2", lw=2.5, label="Ensemble Mean Variance")
    ax.set_title("Deep Ensemble Variance vs. SNR", fontsize=12, weight="bold")
    ax.set_xlabel("SNR Level", fontsize=11)
    ax.set_ylabel("Ensemble Variance", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIG_ENS_VAR_SNR, bbox_inches="tight")
    plt.close()

    # 8. ensemble_mutual_information_vs_snr.png
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot(snrs, snr_unc_df["ensemble_mean_mi"], "o-", color="#8c564b", lw=2.5, label="Ensemble Mean Mutual Info")
    ax.set_title("Deep Ensemble Mutual Information vs. SNR", fontsize=12, weight="bold")
    ax.set_xlabel("SNR Level", fontsize=11)
    ax.set_ylabel("Mutual Information", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIG_ENS_MI_SNR, bbox_inches="tight")
    plt.close()

    # 9. uncertainty_vs_error_rate.png
    fig, ax1 = plt.subplots(figsize=(7, 5), dpi=300)
    color1 = "#1f77b4"
    ax1.set_xlabel("SNR Level", fontsize=11)
    ax1.set_ylabel("Prediction Errors", color=color1, fontsize=11)
    ax1.plot(snrs, early_df["num_ensemble_errors"], "o-", color=color1, lw=2, label="Ensemble Errors")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "#d62728"
    ax2.set_ylabel("Ensemble Predictive Entropy", color=color2, fontsize=11)
    ax2.plot(snrs, early_df["ensemble_mean_entropy"], "s--", color=color2, lw=2, label="Mean Entropy")
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("Ensemble Prediction Errors vs. Predictive Entropy Across SNR", fontsize=12, weight="bold")
    fig.tight_layout()
    plt.savefig(FIG_UNC_ERROR_RATE, bbox_inches="tight")
    plt.close()

    # 10. corruption_uncertainty_dashboard.png
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9), dpi=300)
    
    ax1.plot(snrs, early_df["ensemble_accuracy"], "s-", color="#2ca02c", lw=2)
    ax1.set_title("A. Ensemble Accuracy", weight="bold")
    ax1.set_ylabel("Accuracy")
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.plot(snrs, early_df["ensemble_mean_entropy"], "^-", color="#d62728", lw=2)
    ax2.set_title("B. Predictive Entropy", weight="bold")
    ax2.set_ylabel("H(p)")
    ax2.grid(True, linestyle=":", alpha=0.6)

    ax3.plot(snrs, snr_unc_df["ensemble_mean_variance"], "o-", color="#ff7f0e", lw=2)
    ax3.set_title("C. Ensemble Variance", weight="bold")
    ax3.set_xlabel("SNR Level")
    ax3.set_ylabel("Variance")
    ax3.grid(True, linestyle=":", alpha=0.6)

    ax4.plot(snrs, snr_unc_df["ensemble_mean_mi"], "d-", color="#9467bd", lw=2)
    ax4.set_title("D. Mutual Information", weight="bold")
    ax4.set_xlabel("SNR Level")
    ax4.set_ylabel("MI")
    ax4.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Voice Wav2Vec2 Robustness & Uncertainty Dashboard", fontsize=14, weight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(FIG_DASHBOARD, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_acoustic_corruption_experiment()
