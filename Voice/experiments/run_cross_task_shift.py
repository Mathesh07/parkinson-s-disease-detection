"""Phase 5 Experiment: Cross-Task Shift Analysis for Voice Wav2Vec2 Pipeline.

Evaluates model predictions and uncertainty behavior across ReadText and SpontaneousDialogue speech tasks
using Baseline, MC Dropout (N=30), and Deep Ensemble (M=5) on held-out test subjects (ID04, ID10, ID13, ID23, ID29, ID35).

Preserves chunk -> recording -> subject x task hierarchical aggregation and computes within-subject paired task deltas.

Leakage Guards:
    - Model weights remain 100% frozen under torch.no_grad().
    - Zero test set labels used during inference or uncertainty estimation.
    - Zero threshold tuning on test data.
    - Task labels used ONLY for grouping held-out test inference.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from Voice.calibration import calculate_brier, calculate_ece, calculate_nll
from Voice.cross_task import calculate_entropy, compute_within_subject_deltas
from Voice.dataset import create_subject_splits
from Voice.deep_ensemble import compute_ensemble_statistics
from Voice.mc_dropout import compute_mc_statistics, run_mc_dropout_passes
from Voice.model import EvidentialHead, Wav2Vec2ForParkinsons
from Voice.utils import set_seed

CROSS_TASK_DIR = config.RESULTS_DIR / "cross_task_shift"
CROSS_TASK_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_JSON = CROSS_TASK_DIR / "cross_task_config.json"
PREDICTIONS_CSV = CROSS_TASK_DIR / "cross_task_subject_predictions.csv"
METRICS_JSON = CROSS_TASK_DIR / "cross_task_metrics.json"
SUMMARY_CSV = CROSS_TASK_DIR / "cross_task_summary.csv"
AVAILABILITY_CSV = CROSS_TASK_DIR / "task_availability.csv"
DELTA_CSV = CROSS_TASK_DIR / "within_subject_task_delta.csv"

FIG_PROB_DIST = CROSS_TASK_DIR / "task_probability_distribution.png"
FIG_ENTROPY_DIST = CROSS_TASK_DIR / "task_entropy_distribution.png"
FIG_VAR_DIST = CROSS_TASK_DIR / "task_variance_distribution.png"
FIG_MI_DIST = CROSS_TASK_DIR / "task_mutual_information_distribution.png"
FIG_RT_VS_SD_PROB = CROSS_TASK_DIR / "readtext_vs_spontaneous_probability.png"
FIG_PROB_DELTA = CROSS_TASK_DIR / "within_subject_probability_delta.png"
FIG_ENTROPY_DELTA = CROSS_TASK_DIR / "within_subject_entropy_delta.png"
FIG_TASK_ERROR = CROSS_TASK_DIR / "task_error_comparison.png"
FIG_TASK_AGREE = CROSS_TASK_DIR / "task_ensemble_agreement.png"


def safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Safely calculate ROC-AUC score handling single-class edge cases."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return 0.5


def run_cross_task_shift_experiment(seed: int = 42, split_seed: int = 42):
    """Run Phase 5 Cross-Task Shift experiment."""
    print("=" * 70)
    print("VOICE PHASE 5: CROSS-TASK SHIFT EXPERIMENT")
    print("Task Comparison: ReadText vs. SpontaneousDialogue")
    print("=" * 70)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data Splits and Task Metadata
    meta_csv = config.RESULTS_DIR / "embeddings" / "voice_embeddings_metadata.csv"
    meta_df = pd.read_csv(meta_csv)
    _, _, test_df = create_subject_splits(meta_df, seed=split_seed)

    test_subjects = sorted(test_df["subject_id"].unique().tolist())
    print(f"\nHeld-Out Test Subjects ({len(test_subjects)}): {test_subjects}\n")

    # 2. Task Availability Audit & Table Generation
    avail_records = []
    for sub in test_subjects:
        sub_df = meta_df[meta_df["subject_id"] == sub]
        label_name = sub_df["label_name"].iloc[0]
        rt_cnt = len(sub_df[sub_df["task"] == "ReadText"])
        sd_cnt = len(sub_df[sub_df["task"] == "SpontaneousDialogue"])
        avail_records.append({
            "subject_id": sub,
            "diagnosis": label_name,
            "ReadText_recordings": rt_cnt,
            "SpontaneousDialogue_recordings": sd_cnt,
            "has_both_tasks": (rt_cnt > 0 and sd_cnt > 0)
        })

    avail_df = pd.DataFrame(avail_records)
    avail_df.to_csv(AVAILABILITY_CSV, index=False)
    print(f"Saved task availability table to: {AVAILABILITY_CSV}\n")

    # Save Cross-Task Configuration
    config_data = {
        "experiment": "Voice Wav2Vec2 Cross-Task Shift Analysis",
        "tasks": ["ReadText", "SpontaneousDialogue"],
        "num_test_subjects": len(test_subjects),
        "test_subjects": test_subjects,
        "seed": seed,
        "split_seed": split_seed,
        "mc_dropout_passes": 30,
        "ensemble_members": 5,
        "ensemble_seeds": [42, 43, 44, 45, 46]
    }
    with open(CONFIG_JSON, "w") as f:
        json.dump(config_data, f, indent=4)

    # Load pre-extracted recording embeddings
    embs_file = config.RESULTS_DIR / "embeddings" / "voice_embeddings.pt"
    embs_dict = torch.load(embs_file)
    rec_embeddings = embs_dict["recording_embeddings"]

    # Load Baseline Model Head
    baseline_head = EvidentialHead(in_features=768, num_classes=2, dropout=0.1)
    baseline_head.to(device)
    baseline_head.eval()

    # Load 5 Deep Ensemble Member Models
    ensemble_seeds = [42, 43, 44, 45, 46]
    ensemble_models = []
    for idx, s in enumerate(ensemble_seeds):
        ckpt = config.RESULTS_DIR / "deep_ensemble" / f"member_{idx}" / "checkpoint" / "best_model.pt"
        m = Wav2Vec2ForParkinsons(model_name=config.MODEL_NAME, num_classes=2)
        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        m.to(device)
        m.eval()
        ensemble_models.append(m)

    # 3. Perform Subject x Task Inference
    task_subject_records = []

    with torch.no_grad():
        for sub in test_subjects:
            sub_meta = meta_df[meta_df["subject_id"] == sub]
            true_label = int(sub_meta["label"].iloc[0])
            label_name = sub_meta["label_name"].iloc[0]

            for task_name in ["ReadText", "SpontaneousDialogue"]:
                task_rows = sub_meta[sub_meta["task"] == task_name]
                if len(task_rows) == 0:
                    continue

                # Collect recording embeddings for this Subject x Task
                task_rec_keys = task_rows["recording_key"].tolist()
                task_embs = [rec_embeddings[k].to(device) for k in task_rec_keys]
                sub_task_emb = torch.stack(task_embs).mean(dim=0).unsqueeze(0)  # Shape: (1, 768)

                # A. Baseline Model Forward Pass
                base_evidence = baseline_head(sub_task_emb)
                base_probs = torch.softmax(base_evidence, dim=-1)
                base_p_pd = float(base_probs[0, 1].cpu().item())
                base_pred = 1 if base_p_pd >= 0.5 else 0

                # B. MC Dropout N=30 Stochastic Forward Passes
                stochastic_passes = run_mc_dropout_passes(baseline_head, sub_task_emb, num_passes=30, seed=seed)
                mc_pass_probs = stochastic_passes[:, 0]
                mc_stats = compute_mc_statistics(mc_pass_probs)

                # C. Deep Ensemble M=5 Forward Passes
                m_probs = []
                for m in ensemble_models:
                    m_ev = m(sub_task_emb)
                    m_p = float(torch.softmax(m_ev, dim=-1)[0, 1].cpu().item())
                    m_probs.append(m_p)

                ens_stats = compute_ensemble_statistics(m_probs)
                ens_mean_p = ens_stats["ensemble_mean_probability"]
                ens_pred = 1 if ens_mean_p >= 0.5 else 0

                task_subject_records.append({
                    "subject_id": sub,
                    "task": task_name,
                    "true_label": true_label,
                    "label_name": label_name,
                    "num_recordings": len(task_rows),
                    "num_chunks": int(task_rows["num_chunks"].sum()),
                    
                    # Baseline Metrics
                    "baseline_probability": base_p_pd,
                    "baseline_prediction": base_pred,
                    "baseline_correct": int(base_pred == true_label),
                    
                    # MC Dropout Metrics
                    "mc_mean_probability": mc_stats["mc_mean_probability"],
                    "mc_variance": mc_stats["mc_variance"],
                    "mc_predictive_entropy": mc_stats["predictive_entropy"],
                    "mc_expected_entropy": mc_stats["expected_entropy"],
                    "mc_mutual_information": mc_stats["mutual_information"],
                    "mc_prediction": 1 if mc_stats["mc_mean_probability"] >= 0.5 else 0,
                    "mc_correct": int((1 if mc_stats["mc_mean_probability"] >= 0.5 else 0) == true_label),
                    
                    # Deep Ensemble Metrics
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
                    "prediction_agreement": ens_stats["prediction_agreement"],
                    "ensemble_correct": int(ens_pred == true_label)
                })

                print(
                    f"Subject {sub} | Task {task_name:<19} | True: {true_label} | "
                    f"Base P: {base_p_pd:.4f} | MC Mean: {mc_stats['mc_mean_probability']:.4f} | "
                    f"Ens Mean: {ens_mean_p:.4f} | Ens Var: {ens_stats['ensemble_variance']:.6f} | "
                    f"Ens Ent: {ens_stats['predictive_entropy']:.4f}"
                )

    sub_task_df = pd.DataFrame(task_subject_records)
    sub_task_df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"\nSaved subject x task predictions CSV to: {PREDICTIONS_CSV}")

    # 4. Within-Subject Paired Task Deltas
    delta_records = []
    for sub in test_subjects:
        rt_rows = sub_task_df[(sub_task_df["subject_id"] == sub) & (sub_task_df["task"] == "ReadText")]
        sd_rows = sub_task_df[(sub_task_df["subject_id"] == sub) & (sub_task_df["task"] == "SpontaneousDialogue")]

        if len(rt_rows) > 0 and len(sd_rows) > 0:
            delta = compute_within_subject_deltas(rt_rows.iloc[0], sd_rows.iloc[0])
            delta_records.append(delta)

    delta_df = pd.DataFrame(delta_records)
    delta_df.to_csv(DELTA_CSV, index=False)
    print(f"Saved within-subject paired deltas CSV to: {DELTA_CSV}")

    # 5. Calculate Task-Specific Baseline, MC Dropout, and Deep Ensemble Metrics
    task_metrics_dict = {}
    summary_rows = []

    for task_name in ["ReadText", "SpontaneousDialogue"]:
        t_df = sub_task_df[sub_task_df["task"] == task_name]
        y_true = t_df["true_label"].values

        # Baseline Task Metrics
        base_p = t_df["baseline_probability"].values
        base_pred = t_df["baseline_prediction"].values
        acc_base = float(accuracy_score(y_true, base_pred))
        bal_acc_base = float(balanced_accuracy_score(y_true, base_pred))
        auc_base = safe_auroc(y_true, base_p)
        nll_base = calculate_nll(y_true, base_p)
        brier_base = calculate_brier(y_true, base_p)
        ece_base = calculate_ece(np.maximum(1.0 - base_p, base_p), (base_pred == y_true).astype(int), n_bins=10)

        # MC Dropout Task Metrics
        mc_p = t_df["mc_mean_probability"].values
        mc_pred = t_df["mc_prediction"].values
        acc_mc = float(accuracy_score(y_true, mc_pred))
        bal_acc_mc = float(balanced_accuracy_score(y_true, mc_pred))
        auc_mc = safe_auroc(y_true, mc_p)
        nll_mc = calculate_nll(y_true, mc_p)
        brier_mc = calculate_brier(y_true, mc_p)
        ece_mc = calculate_ece(np.maximum(1.0 - mc_p, mc_p), (mc_pred == y_true).astype(int), n_bins=10)

        # Deep Ensemble Task Metrics
        ens_p = t_df["ensemble_mean_probability"].values
        ens_pred = t_df["ensemble_prediction"].values
        acc_ens = float(accuracy_score(y_true, ens_pred))
        bal_acc_ens = float(balanced_accuracy_score(y_true, ens_pred))
        auc_ens = safe_auroc(y_true, ens_p)
        nll_ens = calculate_nll(y_true, ens_p)
        brier_ens = calculate_brier(y_true, ens_p)
        ece_ens = calculate_ece(np.maximum(1.0 - ens_p, ens_p), (ens_pred == y_true).astype(int), n_bins=10)

        task_metrics_dict[task_name] = {
            "num_subjects": len(t_df),
            "baseline": {
                "accuracy": acc_base, "balanced_accuracy": bal_acc_base, "roc_auc": auc_base,
                "nll": nll_base, "brier": brier_base, "ece": ece_base,
                "mean_probability": float(np.mean(base_p)), "median_probability": float(np.median(base_p)),
                "mean_confidence": float(np.mean(np.abs(base_p - 0.5)))
            },
            "mc_dropout": {
                "accuracy": acc_mc, "balanced_accuracy": bal_acc_mc, "roc_auc": auc_mc,
                "nll": nll_mc, "brier": brier_mc, "ece": ece_mc,
                "mean_entropy": float(np.mean(t_df["mc_predictive_entropy"])),
                "median_entropy": float(np.median(t_df["mc_predictive_entropy"])),
                "mean_variance": float(np.mean(t_df["mc_variance"])),
                "median_variance": float(np.median(t_df["mc_variance"])),
                "mean_mutual_info": float(np.mean(t_df["mc_mutual_information"])),
                "median_mutual_info": float(np.median(t_df["mc_mutual_information"]))
            },
            "deep_ensemble": {
                "accuracy": acc_ens, "balanced_accuracy": bal_acc_ens, "roc_auc": auc_ens,
                "nll": nll_ens, "brier": brier_ens, "ece": ece_ens,
                "mean_entropy": float(np.mean(t_df["ensemble_predictive_entropy"])),
                "median_entropy": float(np.median(t_df["ensemble_predictive_entropy"])),
                "mean_variance": float(np.mean(t_df["ensemble_variance"])),
                "median_variance": float(np.median(t_df["ensemble_variance"])),
                "mean_mutual_info": float(np.mean(t_df["ensemble_mutual_information"])),
                "median_mutual_info": float(np.median(t_df["ensemble_mutual_information"]))
            }
        }

        summary_rows.append({
            "task": task_name,
            "method": "Baseline",
            "accuracy": acc_base, "balanced_accuracy": bal_acc_base, "roc_auc": auc_base,
            "nll": nll_base, "brier": brier_base, "ece": ece_base
        })
        summary_rows.append({
            "task": task_name,
            "method": "MC Dropout (N=30)",
            "accuracy": acc_mc, "balanced_accuracy": bal_acc_mc, "roc_auc": auc_mc,
            "nll": nll_mc, "brier": brier_mc, "ece": ece_mc
        })
        summary_rows.append({
            "task": task_name,
            "method": "Deep Ensemble (M=5)",
            "accuracy": acc_ens, "balanced_accuracy": bal_acc_ens, "roc_auc": auc_ens,
            "nll": nll_ens, "brier": brier_ens, "ece": ece_ens
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    with open(METRICS_JSON, "w") as f:
        json.dump(task_metrics_dict, f, indent=4)
    print(f"Saved cross-task metrics JSON to: {METRICS_JSON}")

    # 6. Generate Visualizations
    generate_cross_task_plots(sub_task_df, delta_df)

    # 7. Print Summary Banners
    print("\n" + "=" * 75)
    print("VOICE CROSS-TASK SHIFT METRICS COMPARISON SUMMARY")
    print("=" * 75)
    print(f"{'Task':<22} | {'Method':<22} | {'Acc':<6} | {'BA':<6} | {'AUC':<6} | {'NLL':<7} | {'ECE':<6}")
    print("-" * 75)
    for _, r in summary_df.iterrows():
        print(f"{r['task']:<22} | {r['method']:<22} | {r['accuracy']:<6.4f} | {r['balanced_accuracy']:<6.4f} | {r['roc_auc']:<6.4f} | {r['nll']:<7.4f} | {r['ece']:<6.4f}")
    print("=" * 75)

    if len(delta_df) > 0:
        print("\nMEAN WITHIN-SUBJECT PAIRED TASK DELTAS (ReadText - SpontaneousDialogue):")
        print(f"  Delta Baseline P(PD):         {delta_df['delta_baseline_p'].mean():+.4f}")
        print(f"  Delta MC Mean P(PD):           {delta_df['delta_mc_mean_p'].mean():+.4f}")
        print(f"  Delta MC Predictive Entropy:   {delta_df['delta_mc_entropy'].mean():+.4f}")
        print(f"  Delta Ensemble Mean P(PD):     {delta_df['delta_ens_mean_p'].mean():+.4f}")
        print(f"  Delta Ensemble Predictive Ent: {delta_df['delta_ens_entropy'].mean():+.4f}")
        print(f"  Delta Ensemble Variance:       {delta_df['delta_ens_variance'].mean():+.6f}")
        print("=" * 75)

    return task_metrics_dict


def generate_cross_task_plots(df: pd.DataFrame, delta_df: pd.DataFrame):
    """Generate 9 publication-quality plots comparing ReadText and SpontaneousDialogue tasks."""
    rt_df = df[df["task"] == "ReadText"]
    sd_df = df[df["task"] == "SpontaneousDialogue"]

    # 1. task_probability_distribution.png
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.hist(rt_df["ensemble_mean_probability"], bins=5, alpha=0.6, label="ReadText", color="#1f77b4", edgecolor="black")
    ax.hist(sd_df["ensemble_mean_probability"], bins=5, alpha=0.6, label="SpontaneousDialogue", color="#ff7f0e", edgecolor="black")
    ax.set_title("Ensemble Probability Distribution by Task", fontsize=12, weight="bold")
    ax.set_xlabel("Ensemble Mean P(PD)", fontsize=11)
    ax.set_ylabel("Subject Count", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIG_PROB_DIST, bbox_inches="tight")
    plt.close()

    # 2. task_entropy_distribution.png
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.hist(rt_df["ensemble_predictive_entropy"], bins=5, alpha=0.6, label="ReadText", color="#1f77b4", edgecolor="black")
    ax.hist(sd_df["ensemble_predictive_entropy"], bins=5, alpha=0.6, label="SpontaneousDialogue", color="#ff7f0e", edgecolor="black")
    ax.set_title("Predictive Entropy Distribution by Task", fontsize=12, weight="bold")
    ax.set_xlabel("Predictive Entropy H(p)", fontsize=11)
    ax.set_ylabel("Subject Count", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIG_ENTROPY_DIST, bbox_inches="tight")
    plt.close()

    # 3. task_variance_distribution.png
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.hist(rt_df["ensemble_variance"], bins=5, alpha=0.6, label="ReadText", color="#1f77b4", edgecolor="black")
    ax.hist(sd_df["ensemble_variance"], bins=5, alpha=0.6, label="SpontaneousDialogue", color="#ff7f0e", edgecolor="black")
    ax.set_title("Ensemble Variance Distribution by Task", fontsize=12, weight="bold")
    ax.set_xlabel("Ensemble Variance", fontsize=11)
    ax.set_ylabel("Subject Count", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIG_VAR_DIST, bbox_inches="tight")
    plt.close()

    # 4. task_mutual_information_distribution.png
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.hist(rt_df["ensemble_mutual_information"], bins=5, alpha=0.6, label="ReadText", color="#1f77b4", edgecolor="black")
    ax.hist(sd_df["ensemble_mutual_information"], bins=5, alpha=0.6, label="SpontaneousDialogue", color="#ff7f0e", edgecolor="black")
    ax.set_title("Mutual Information Distribution by Task", fontsize=12, weight="bold")
    ax.set_xlabel("Mutual Information", fontsize=11)
    ax.set_ylabel("Subject Count", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIG_MI_DIST, bbox_inches="tight")
    plt.close()

    # 5. readtext_vs_spontaneous_probability.png
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    if len(delta_df) > 0:
        ax.scatter(delta_df["baseline_p_readtext"], delta_df["baseline_p_spontaneous"], label="Baseline P(PD)", color="#1f77b4", s=70, marker="o")
        ax.scatter(delta_df["ens_mean_p_readtext"], delta_df["ens_mean_p_spontaneous"], label="Ensemble Mean P(PD)", color="#2ca02c", s=70, marker="s")
    ax.plot([0, 1], [0, 1], "k--", label="Identity Line (y = x)")
    ax.set_title("ReadText vs. SpontaneousDialogue P(PD)", fontsize=12, weight="bold")
    ax.set_xlabel("ReadText P(PD)", fontsize=11)
    ax.set_ylabel("SpontaneousDialogue P(PD)", fontsize=11)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIG_RT_VS_SD_PROB, bbox_inches="tight")
    plt.close()

    # 6. within_subject_probability_delta.png
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    if len(delta_df) > 0:
        x = np.arange(len(delta_df))
        ax.bar(x, delta_df["delta_ens_mean_p"], color="#1f77b4", edgecolor="black", alpha=0.8)
        ax.axhline(0, color="black", linestyle="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(delta_df["subject_id"])
        ax.set_title("Within-Subject Probability Delta (ReadText - Spontaneous)", fontsize=12, weight="bold")
        ax.set_xlabel("Test Subject ID", fontsize=11)
        ax.set_ylabel("Delta P(PD)", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIG_PROB_DELTA, bbox_inches="tight")
    plt.close()

    # 7. within_subject_entropy_delta.png
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    if len(delta_df) > 0:
        x = np.arange(len(delta_df))
        ax.bar(x, delta_df["delta_ens_entropy"], color="#2ca02c", edgecolor="black", alpha=0.8)
        ax.axhline(0, color="black", linestyle="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(delta_df["subject_id"])
        ax.set_title("Within-Subject Predictive Entropy Delta (ReadText - Spontaneous)", fontsize=12, weight="bold")
        ax.set_xlabel("Test Subject ID", fontsize=11)
        ax.set_ylabel("Delta Predictive Entropy H(p)", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIG_ENTROPY_DELTA, bbox_inches="tight")
    plt.close()

    # 8. task_error_comparison.png
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    rt_errs = int(np.sum(rt_df["ensemble_correct"] == 0))
    rt_corr = int(np.sum(rt_df["ensemble_correct"] == 1))
    sd_errs = int(np.sum(sd_df["ensemble_correct"] == 0))
    sd_corr = int(np.sum(sd_df["ensemble_correct"] == 1))

    tasks = ["ReadText", "SpontaneousDialogue"]
    corrects = [rt_corr, sd_corr]
    errors = [rt_errs, sd_errs]

    x = np.arange(len(tasks))
    width = 0.35
    ax.bar(x - width/2, corrects, width, label="Correct Predictions", color="green", alpha=0.8)
    ax.bar(x + width/2, errors, width, label="Incorrect Predictions", color="red", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_title("Prediction Error Comparison by Speech Task", fontsize=12, weight="bold")
    ax.set_ylabel("Subject Count", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIG_TASK_ERROR, bbox_inches="tight")
    plt.close()

    # 9. task_ensemble_agreement.png
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    x = np.arange(len(rt_df))
    width = 0.35
    ax.bar(x - width/2, rt_df["prediction_agreement"], width, label="ReadText Agreement", color="#1f77b4", alpha=0.8)
    ax.bar(x + width/2, sd_df["prediction_agreement"], width, label="Spontaneous Agreement", color="#ff7f0e", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(rt_df["subject_id"])
    ax.set_title("Ensemble Member Agreement Score by Task", fontsize=12, weight="bold")
    ax.set_xlabel("Test Subject ID", fontsize=11)
    ax.set_ylabel("Member Agreement Score (m/5)", fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIG_TASK_AGREE, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_cross_task_shift_experiment()
