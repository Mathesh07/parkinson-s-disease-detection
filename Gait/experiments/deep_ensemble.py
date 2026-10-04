"""Deep Ensemble Uncertainty Framework for Gait-Based Parkinson's Detection.

Research Objective:
    Evaluate whether an ensemble of independently initialized and trained neural networks
    (Deep Ensemble) provides superior calibration, error detection, and robustness against
    "silent confidence" under cohort shift compared to single-model baselines (Softmax,
    Temperature Scaling, EDL, and MC Dropout).

Methodology:
    - 5 independent ensemble members trained with random seeds: [42, 43, 44, 45, 46].
    - Independent parameters, random initializations, and separate optimizer states.
    - Zero data leakage: Normalization fitted only on training cohorts; validation split used for model selection/calibration; test cohort strictly held out.
    - Window-level aggregation across M=5 models:
        * Mean predictive probability: p_bar = (1/M) * sum(p_m)
        * Predictive entropy: H(p_bar) = -sum(p_bar * log(p_bar))
        * Prediction variance: Var(p_PD) = (1/M) * sum((p_m,PD - p_bar_PD)^2)
        * Mutual information: I(y, theta | x) = H(p_bar) - (1/M) * sum(H(p_m))
    - Hierarchical aggregation: WINDOW -> RECORDING -> SUBJECT.
    - Head-to-head comparison against Softmax, Temperature Scaling, EDL, and MC Dropout.

Outputs:
    Gait/outputs/deep_ensemble/
        - deep_ensemble_results.csv
        - deep_ensemble_results.json
        - deep_ensemble_subject_predictions.csv
        - deep_ensemble_comparison.csv
        - plots/*.png
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

# Path bootstrapping
_THIS_FILE = Path(__file__).resolve()
_EXPERIMENTS_DIR = _THIS_FILE.parent
_GAIT_DIR = _EXPERIMENTS_DIR.parent
_ROOT_DIR = _GAIT_DIR.parent

for _p in (str(_GAIT_DIR), str(_ROOT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Gait import config
from Gait.dataset import GaitWindowDataset, collate_gait_batch
from Gait.experiments.calibration_baselines import (
    calculate_aurc,
    calculate_ece,
    extract_validation_logits,
    fit_temperature,
)
from Gait.experiments.leave_one_cohort_out import (
    _safe_auroc,
    _safe_brier,
    _safe_nll,
    build_cohort_splits,
    build_dataloaders,
    fit_normalizer_on_train,
    json_default,
    train_loco_model,
)
from Gait.experiments.mc_dropout import compute_window_entropy_metrics, enable_mc_dropout
from Gait.model import GaitCNNBiLSTM
from Gait.preprocessing import build_manifest
from Gait.utils import calculate_metrics, get_device, set_seed
from evidential import dirichlet_from_evidence

DEFAULT_ENSEMBLE_SEEDS = [42, 43, 44, 45, 46]
NUM_ENSEMBLE_MEMBERS = len(DEFAULT_ENSEMBLE_SEEDS)

OUTPUT_DIR = _GAIT_DIR / "outputs" / "deep_ensemble"
CHECKPOINT_DIR = _GAIT_DIR / "checkpoints" / "deep_ensemble"
PLOTS_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = OUTPUT_DIR / "deep_ensemble_results.csv"
RESULTS_JSON = OUTPUT_DIR / "deep_ensemble_results.json"
SUBJECTS_CSV = OUTPUT_DIR / "deep_ensemble_subject_predictions.csv"
COMPARISON_CSV = OUTPUT_DIR / "deep_ensemble_comparison.csv"

LOCO_EXPERIMENTS = [
    {"exp_id": 1, "train_cohorts": ["Ga", "Ju"], "test_cohort": "Si"},
    {"exp_id": 2, "train_cohorts": ["Ga", "Si"], "test_cohort": "Ju"},
    {"exp_id": 3, "train_cohorts": ["Ju", "Si"], "test_cohort": "Ga"},
]


# ===========================================================================
# 1. ENSEMBLE TRAINING & INFERENCE
# ===========================================================================

def train_ensemble_members(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    exp_id: int,
    seeds: List[int] = DEFAULT_ENSEMBLE_SEEDS,
    epochs: int = config.EPOCHS,
    checkpoint_dir: Path = CHECKPOINT_DIR,
) -> List[GaitCNNBiLSTM]:
    """
    Train M independent ensemble models with separate seeds and optimizers.
    Saves and loads checkpoints to support fast re-use.
    """
    models = []
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for idx, seed in enumerate(seeds):
        ckpt_path = checkpoint_dir / f"ensemble_exp{exp_id}_member{idx}_seed{seed}.pt"
        exp_label = f"Ensemble_Exp{exp_id}_M{idx}_Seed{seed}"

        if ckpt_path.exists():
            print(f"  Loading existing checkpoint -> {ckpt_path.name}")
            model = GaitCNNBiLSTM(
                in_channels=config.NUM_CHANNELS,
                cnn_channels=config.CNN_CHANNELS,
                lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
                lstm_num_layers=config.LSTM_NUM_LAYERS,
                embedding_dim=config.EMBEDDING_DIM,
                num_classes=2,
                dropout=config.DROPOUT,
            ).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            model.eval()
            models.append(model)
        else:
            print(f"\n  Training Ensemble Member {idx + 1}/{len(seeds)} (Seed: {seed})...")
            model = train_loco_model(
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=epochs,
                seed=seed,
                exp_label=exp_label,
            )
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved member checkpoint -> {ckpt_path.name}")
            models.append(model)

    return models


@torch.no_grad()
def run_ensemble_inference(
    models: List[GaitCNNBiLSTM],
    test_loader: DataLoader,
    device: torch.device,
    exp_id: int,
    train_cohorts: List[str],
    test_cohort: str,
) -> pd.DataFrame:
    """
    Run multi-model forward pass on test data.
    Computes per-window probabilities for each member, ensemble mean,
    predictive entropy, and variance across member probabilities.
    """
    for m in models:
        m.eval()

    num_members = len(models)
    records = []

    for batch in test_loader:
        x = batch["features"].to(device)
        y = batch["labels"].to(device)
        batch_size = len(y)
        y_np = y.cpu().numpy()

        # Collect probabilities from all members: shape (M, B, 2)
        member_probs = []
        for model in models:
            _, emb = model(x, return_embedding=True)
            _, logits = model.evidential_head(emb, return_raw_logits=True)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            member_probs.append(probs)

        # shape: (M, B, 2)
        member_probs = np.stack(member_probs, axis=0)

        # Ensemble mean probability: shape (B, 2)
        mean_p = np.mean(member_probs, axis=0)
        # Variance across member predictions for PD class: shape (B,)
        var_p_pd = np.var(member_probs[:, :, 1], axis=0)

        # Predictive entropy: -sum(p_bar * log(p_bar))
        eps = 1e-12
        p_clipped = np.clip(mean_p, eps, 1.0)
        pred_entropy = -np.sum(p_clipped * np.log(p_clipped), axis=1)

        # Expected entropy under member distributions
        member_clipped = np.clip(member_probs, eps, 1.0)
        member_entropies = -np.sum(member_clipped * np.log(member_clipped), axis=-1)  # (M, B)
        exp_entropy = np.mean(member_entropies, axis=0)  # (B,)

        # Mutual information (Epistemic uncertainty)
        mutual_info = np.maximum(0.0, pred_entropy - exp_entropy)

        for i in range(batch_size):
            records.append({
                "experiment": exp_id,
                "train_cohorts": "+".join(sorted(train_cohorts)),
                "test_cohort": test_cohort,
                "subject_id": batch["subject_ids"][i],
                "recording_id": batch["recording_ids"][i],
                "study": batch["studies"][i],
                "true_label": int(y_np[i]),
                "P_HC": float(mean_p[i, 0]),
                "P_PD": float(mean_p[i, 1]),
                "predictive_entropy": float(pred_entropy[i]),
                "prediction_variance": float(var_p_pd[i]),
                "mutual_information": float(mutual_info[i]),
            })

    return pd.DataFrame(records)


# ===========================================================================
# 2. HIERARCHICAL AGGREGATION & METRICS
# ===========================================================================

def aggregate_ensemble_to_subject_level(window_df: pd.DataFrame) -> pd.DataFrame:
    """
    Hierarchical aggregation: WINDOW -> RECORDING -> SUBJECT.
    """
    metric_cols = ["P_HC", "P_PD", "predictive_entropy", "prediction_variance", "mutual_information"]
    rec_keys = ["experiment", "train_cohorts", "test_cohort", "subject_id", "recording_id", "study", "true_label"]
    sub_keys = ["experiment", "train_cohorts", "test_cohort", "subject_id", "study", "true_label"]

    rec_df = window_df.groupby(rec_keys)[metric_cols].mean().reset_index()
    sub_df = rec_df.groupby(sub_keys)[metric_cols].mean().reset_index()

    sub_df["predicted_label"] = (sub_df["P_PD"] >= 0.5).astype(int)
    sub_df["is_correct"] = (sub_df["predicted_label"] == sub_df["true_label"]).astype(int)
    sub_df["confidence"] = np.maximum(sub_df["P_HC"], sub_df["P_PD"])
    sub_df["uncertainty"] = sub_df["predictive_entropy"]  # Primary uncertainty = predictive entropy
    return sub_df


def evaluate_subject_predictions(
    sub_df: pd.DataFrame,
    method_name: str,
    uncertainty_col: str = "uncertainty"
) -> Dict[str, float]:
    """Compute complete classification, probabilistic, calibration, and error-detection metrics."""
    y_true = sub_df["true_label"].values.astype(int)
    y_pred = sub_df["predicted_label"].values.astype(int)
    p_pd = sub_df["P_PD"].values
    conf = sub_df["confidence"].values if "confidence" in sub_df.columns else np.maximum(p_pd, 1 - p_pd)
    unc = sub_df[uncertainty_col].values
    is_corr = sub_df["is_correct"].values.astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    sens = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    spec = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = _safe_auroc(y_true, p_pd)

    nll = _safe_nll(y_true, p_pd)
    brier = _safe_brier(y_true, p_pd)
    ece = calculate_ece(conf, is_corr, n_bins=10)

    corr_mask = (is_corr == 1)
    inc_mask = ~corr_mask

    mean_u_corr = float(unc[corr_mask].mean()) if corr_mask.any() else float("nan")
    mean_u_inc = float(unc[inc_mask].mean()) if inc_mask.any() else float("nan")
    err_auroc = _safe_auroc(1 - is_corr, unc)
    aurc = calculate_aurc(unc, is_corr)

    return {
        "method": method_name,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "sensitivity": sens,
        "specificity": spec,
        "f1": f1,
        "roc_auc": float(roc_auc) if roc_auc is not None else float("nan"),
        "nll": nll,
        "brier": brier,
        "ece": ece,
        "error_detection_auroc": float(err_auroc) if err_auroc is not None else float("nan"),
        "aurc": aurc,
        "mean_uncertainty_correct": mean_u_corr,
        "mean_uncertainty_incorrect": mean_u_inc,
    }


# ===========================================================================
# 3. BASELINE COMPARISON EVALUATION
# ===========================================================================

@torch.no_grad()
def evaluate_all_baselines_on_fold(
    reference_model: GaitCNNBiLSTM,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    exp_id: int,
    train_cohorts: List[str],
    test_cohort: str,
    mc_samples: int = 30,
) -> Dict[str, pd.DataFrame]:
    """
    Run Softmax, Temperature Scaling, EDL, and MC Dropout on the same test fold
    to enable exact head-to-head comparison with Deep Ensemble.
    """
    reference_model.eval()

    # 1. Fit temperature on validation logits strictly
    val_logits, val_labels = extract_validation_logits(reference_model, val_loader, device)
    learned_t = fit_temperature(val_logits, val_labels)

    softmax_windows, temp_windows, edl_windows, mc_windows = [], [], [], []

    for batch in test_loader:
        x = batch["features"].to(device)
        y = batch["labels"].to(device)
        batch_size = len(y)
        y_np = y.cpu().numpy()

        # Standard forward pass
        reference_model.eval()
        evidence, emb = reference_model(x, return_embedding=True)
        _, raw_logits = reference_model.evidential_head(emb, return_raw_logits=True)
        alpha, S, edl_probs, u_edl = dirichlet_from_evidence(evidence, num_classes=2)

        probs_softmax = torch.softmax(raw_logits, dim=1).cpu().numpy()
        probs_temp = torch.softmax(raw_logits / learned_t, dim=1).cpu().numpy()
        probs_edl = edl_probs.cpu().numpy()
        u_edl_np = u_edl.squeeze(-1).cpu().numpy()

        for i in range(batch_size):
            meta = {
                "experiment": exp_id,
                "train_cohorts": "+".join(sorted(train_cohorts)),
                "test_cohort": test_cohort,
                "subject_id": batch["subject_ids"][i],
                "recording_id": batch["recording_ids"][i],
                "study": batch["studies"][i],
                "true_label": int(y_np[i]),
            }
            softmax_windows.append({
                **meta, "P_HC": float(probs_softmax[i, 0]), "P_PD": float(probs_softmax[i, 1])
            })
            temp_windows.append({
                **meta, "P_HC": float(probs_temp[i, 0]), "P_PD": float(probs_temp[i, 1])
            })
            edl_windows.append({
                **meta, "P_HC": float(probs_edl[i, 0]), "P_PD": float(probs_edl[i, 1]),
                "uncertainty": float(u_edl_np[i])
            })

        # MC Dropout forward pass
        enable_mc_dropout(reference_model)
        mc_batch_probs = []
        for _ in range(mc_samples):
            _, emb_mc = reference_model(x, return_embedding=True)
            _, logits_mc = reference_model.evidential_head(emb_mc, return_raw_logits=True)
            p_mc = torch.softmax(logits_mc, dim=1).cpu().numpy()
            mc_batch_probs.append(p_mc)
        mc_batch_probs = np.stack(mc_batch_probs, axis=0)  # (N, B, 2)

        for i in range(batch_size):
            win_mc = mc_batch_probs[:, i, :]
            mean_mc_p = win_mc.mean(axis=0)
            pred_ent, _, _ = compute_window_entropy_metrics(win_mc)
            mc_windows.append({
                "experiment": exp_id,
                "train_cohorts": "+".join(sorted(train_cohorts)),
                "test_cohort": test_cohort,
                "subject_id": batch["subject_ids"][i],
                "recording_id": batch["recording_ids"][i],
                "study": batch["studies"][i],
                "true_label": int(y_np[i]),
                "P_HC": float(mean_mc_p[0]),
                "P_PD": float(mean_mc_p[1]),
                "uncertainty": float(pred_ent),
            })

    # Hierarchical aggregation for baselines
    rec_keys = ["experiment", "train_cohorts", "test_cohort", "subject_id", "recording_id", "study", "true_label"]
    sub_keys = ["experiment", "train_cohorts", "test_cohort", "subject_id", "study", "true_label"]

    def _agg(win_list, unc_type):
        wdf = pd.DataFrame(win_list)
        cols = ["P_HC", "P_PD"] + (["uncertainty"] if "uncertainty" in wdf.columns else [])
        rdf = wdf.groupby(rec_keys)[cols].mean().reset_index()
        sdf = rdf.groupby(sub_keys)[cols].mean().reset_index()
        sdf["predicted_label"] = (sdf["P_PD"] >= 0.5).astype(int)
        sdf["is_correct"] = (sdf["predicted_label"] == sdf["true_label"]).astype(int)
        sdf["confidence"] = np.maximum(sdf["P_HC"], sdf["P_PD"])
        if unc_type == "softmax":
            sdf["uncertainty"] = 1.0 - sdf["confidence"]
        return sdf

    return {
        "Softmax": _agg(softmax_windows, "softmax"),
        "Temperature Scaling": _agg(temp_windows, "softmax"),
        "EDL": _agg(edl_windows, "edl"),
        "MC Dropout": _agg(mc_windows, "mc"),
    }


# ===========================================================================
# 4. PLOTS
# ===========================================================================

def generate_ensemble_plots(comparison_df: pd.DataFrame, ensemble_sub_df: pd.DataFrame) -> None:
    """Generate comparative figures for Deep Ensemble vs other uncertainty methods."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: Methods Comparison across Key Uncertainty Metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics_to_plot = ["balanced_accuracy", "ece", "error_detection_auroc"]
    metric_titles = ["Balanced Accuracy (Higher = Better)", "ECE (Lower = Better)", "Error Detection AUROC (Higher = Better)"]

    methods = comparison_df["method"].unique()
    colors = {
        "Softmax": "#4a5568",
        "Temperature Scaling": "#3182ce",
        "EDL": "#805ad5",
        "MC Dropout": "#d69e2e",
        "Deep Ensemble": "#e53e3e",
    }

    for ax, metric, title in zip(axes, metrics_to_plot, metric_titles):
        exp_ids = sorted(comparison_df["experiment"].unique())
        x = np.arange(len(exp_ids))
        width = 0.16
        multiplier = 0

        for method in methods:
            sub = comparison_df[comparison_df["method"] == method]
            vals = [sub[sub["experiment"] == e][metric].iloc[0] if len(sub[sub["experiment"] == e]) > 0 else 0 for e in exp_ids]
            offset = width * multiplier
            rects = ax.bar(x + offset, vals, width, label=method if ax == axes[0] else "", color=colors.get(method, "#718096"), alpha=0.85)
            multiplier += 1

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels([f"Exp {e}" for e in exp_ids])
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.15), ncol=len(methods), frameon=True)
    plt.suptitle("Deep Ensemble vs Single-Model Baselines across LOCO Folds", y=1.05, fontsize=14, fontweight="bold")
    plt.tight_layout()
    p1 = PLOTS_DIR / "deep_ensemble_comparison_bars.png"
    plt.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot -> {p1}")

    # Figure 2: Uncertainty Separation for Deep Ensemble
    fig, ax = plt.subplots(figsize=(8, 5))
    exp_ids = sorted(ensemble_sub_df["experiment"].unique())
    box_data_corr = [ensemble_sub_df[(ensemble_sub_df["experiment"] == e) & (ensemble_sub_df["is_correct"] == 1)]["predictive_entropy"].values for e in exp_ids]
    box_data_inc = [ensemble_sub_df[(ensemble_sub_df["experiment"] == e) & (ensemble_sub_df["is_correct"] == 0)]["predictive_entropy"].values for e in exp_ids]

    pos_corr = np.arange(len(exp_ids)) * 2.0 - 0.35
    pos_inc = np.arange(len(exp_ids)) * 2.0 + 0.35

    ax.boxplot(box_data_corr, positions=pos_corr, widths=0.5, patch_artist=True,
               boxprops=dict(facecolor="#3182ce", alpha=0.7), medianprops=dict(color="#1a365d", linewidth=2))
    ax.boxplot(box_data_inc, positions=pos_inc, widths=0.5, patch_artist=True,
               boxprops=dict(facecolor="#e53e3e", alpha=0.7), medianprops=dict(color="#742a2a", linewidth=2))

    ax.set_xticks(np.arange(len(exp_ids)) * 2.0)
    ax.set_xticklabels([f"Exp {e} (Test: {ensemble_sub_df[ensemble_sub_df['experiment']==e]['test_cohort'].iloc[0]})" for e in exp_ids])
    ax.set_ylabel("Predictive Entropy")
    ax.set_title("Deep Ensemble: Uncertainty Distribution for Correct vs Incorrect Predictions", fontsize=12, fontweight="bold")
    ax.legend([
        plt.Rectangle((0, 0), 1, 1, fc="#3182ce", alpha=0.7),
        plt.Rectangle((0, 0), 1, 1, fc="#e53e3e", alpha=0.7),
    ], ["Correct Predictions", "Incorrect Predictions"], loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    p2 = PLOTS_DIR / "deep_ensemble_uncertainty_separation.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"  Saved plot -> {p2}")


# ===========================================================================
# 5. MAIN EXECUTION PIPELINE
# ===========================================================================

def run_deep_ensemble(
    seeds: List[int] = DEFAULT_ENSEMBLE_SEEDS,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    device_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute complete Deep Ensemble experiment across all 3 LOCO folds."""
    t_start = time.time()
    device = torch.device(device_name) if device_name else get_device()
    print("=" * 60)
    print("DEEP ENSEMBLE LOCO EXPERIMENT")
    print(f"Ensemble Members (M={len(seeds)}): Seeds={seeds}")
    print(f"Epochs per member: {epochs}")
    print(f"Device: {device}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

    manifest_df = build_manifest()
    all_ensemble_results = []
    all_comparison_rows = []
    all_sub_preds = []

    for exp in LOCO_EXPERIMENTS:
        exp_id = exp["exp_id"]
        train_cohorts = exp["train_cohorts"]
        test_cohort = exp["test_cohort"]
        print(f"\n{'='*50}")
        print(f"LOCO FOLD {exp_id}: Train={train_cohorts} -> Test={test_cohort}")
        print(f"{'='*50}")

        # Split data using master seed 42
        train_df, val_df, test_df = build_cohort_splits(
            manifest_df=manifest_df,
            train_cohorts=train_cohorts,
            test_cohort=test_cohort,
            val_fraction=config.VAL_SPLIT_RATIO,
            seed=42,
        )

        normalizer = fit_normalizer_on_train(train_df)
        train_loader, val_loader, test_loader = build_dataloaders(
            train_df=train_df, val_df=val_df, test_df=test_df,
            normalizer=normalizer, batch_size=batch_size,
        )

        # 1. Train or load M ensemble members
        models = train_ensemble_members(
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            exp_id=exp_id,
            seeds=seeds,
            epochs=epochs,
        )

        # 2. Deep Ensemble inference
        print("\n  Running Deep Ensemble inference across all members...")
        win_df = run_ensemble_inference(
            models=models,
            test_loader=test_loader,
            device=device,
            exp_id=exp_id,
            train_cohorts=train_cohorts,
            test_cohort=test_cohort,
        )
        ens_sub_df = aggregate_ensemble_to_subject_level(win_df)
        all_sub_preds.append(ens_sub_df)

        ens_metrics = evaluate_subject_predictions(ens_sub_df, method_name="Deep Ensemble")
        ens_metrics.update({
            "experiment": exp_id,
            "train_cohorts": "+".join(sorted(train_cohorts)),
            "test_cohort": test_cohort,
            "num_members": len(models),
        })
        all_ensemble_results.append(ens_metrics)
        all_comparison_rows.append(ens_metrics)

        print(f"  [Deep Ensemble] Fold {exp_id} -> BalAcc: {ens_metrics['balanced_accuracy']:.4f} | "
              f"ROC-AUC: {ens_metrics['roc_auc']:.4f} | ECE: {ens_metrics['ece']:.4f} | "
              f"Err-AUROC: {ens_metrics['error_detection_auroc']:.4f}")

        # 3. Baselines comparison (using Member 1 / Seed 42 as the reference single model)
        print("  Evaluating single-model baselines on identical test split...")
        baseline_sub_dfs = evaluate_all_baselines_on_fold(
            reference_model=models[0],
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            exp_id=exp_id,
            train_cohorts=train_cohorts,
            test_cohort=test_cohort,
        )

        for b_name, b_sub_df in baseline_sub_dfs.items():
            b_metrics = evaluate_subject_predictions(b_sub_df, method_name=b_name)
            b_metrics.update({
                "experiment": exp_id,
                "train_cohorts": "+".join(sorted(train_cohorts)),
                "test_cohort": test_cohort,
                "num_members": 1,
            })
            all_comparison_rows.append(b_metrics)
            print(f"  [{b_name:20s}] Fold {exp_id} -> BalAcc: {b_metrics['balanced_accuracy']:.4f} | "
                  f"ROC-AUC: {b_metrics['roc_auc']:.4f} | ECE: {b_metrics['ece']:.4f} | "
                  f"Err-AUROC: {b_metrics['error_detection_auroc']:.4f}")

    # Save CSVs
    results_df = pd.DataFrame(all_ensemble_results)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved ensemble results -> {RESULTS_CSV}")

    comparison_df = pd.DataFrame(all_comparison_rows)
    comparison_df.to_csv(COMPARISON_CSV, index=False)
    print(f"Saved comparative results -> {COMPARISON_CSV}")

    combined_subs = pd.concat(all_sub_preds, ignore_index=True)
    combined_subs.to_csv(SUBJECTS_CSV, index=False)
    print(f"Saved subject predictions -> {SUBJECTS_CSV}")

    with open(RESULTS_JSON, "w") as fh:
        json.dump({
            "num_members": len(seeds),
            "seeds": seeds,
            "results": all_ensemble_results,
            "comparison": all_comparison_rows,
        }, fh, indent=4, default=json_default)
    print(f"Saved results JSON -> {RESULTS_JSON}")

    # Generate plots
    generate_ensemble_plots(comparison_df, combined_subs)

    print("\n" + "=" * 60)
    print(f"DEEP ENSEMBLE COMPLETED ({time.time() - t_start:.1f}s)")
    print("=" * 60)
    return results_df, comparison_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep Ensemble for Gait Parkinson's Detection")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Epochs per ensemble member")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_ENSEMBLE_SEEDS, help="Ensemble seeds")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_deep_ensemble(seeds=args.seeds, epochs=args.epochs, batch_size=args.batch_size)
