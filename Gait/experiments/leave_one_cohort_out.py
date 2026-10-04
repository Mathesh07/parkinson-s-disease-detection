"""Leave-One-Cohort-Out (LOCO) Experiment for Gait-Based Parkinson's Detection.

Research Question:
    When a Parkinson's disease classifier trained on some cohorts encounters a previously
    unseen cohort, does predictive uncertainty increase, does it identify errors, and can
    uncertainty help detect distribution shift?

Three Experiments:
    Exp 1: TRAIN = Ga + Ju  |  TEST = Si
    Exp 2: TRAIN = Ga + Si  |  TEST = Ju
    Exp 3: TRAIN = Ju + Si  |  TEST = Ga

Leakage Guards:
    - Normalization fitted exclusively on train-cohort recordings.
    - Held-out cohort subjects never appear in train, norm-fitting, or val sets.
    - Explicit assertion checks after every split construction.
    - Test labels never influence training or model selection.

Usage:
    python -m Gait.experiments.leave_one_cohort_out
    python -m Gait.experiments.leave_one_cohort_out --epochs 25 --seed 42
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader


def json_default(obj):
    """JSON serializer helper for NumPy types."""
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ---------------------------------------------------------------------------
# Path bootstrapping — supports running as module or standalone script
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_EXPERIMENTS_DIR = _THIS_FILE.parent          # Gait/experiments/
_GAIT_DIR = _EXPERIMENTS_DIR.parent           # Gait/
_ROOT_DIR = _GAIT_DIR.parent                  # repo root

for _p in (str(_GAIT_DIR), str(_ROOT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Project imports — reuse every existing component
# ---------------------------------------------------------------------------
from Gait import config
from Gait.dataset import GaitWindowDataset, collate_gait_batch
from Gait.model import GaitCNNBiLSTM
from Gait.preprocessing import GaitNormalizer, build_manifest, load_raw_recording
from Gait.utils import calculate_metrics, get_device, set_seed
from evidential import EvidentialLoss, dirichlet_from_evidence

# ---------------------------------------------------------------------------
# LOCO output directories
# ---------------------------------------------------------------------------
LOCO_DIR = config.OUTPUT_DIR / "loco"
LOCO_PLOTS_DIR = LOCO_DIR / "plots"
LOCO_DIR.mkdir(parents=True, exist_ok=True)
LOCO_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

LOCO_RESULTS_CSV     = LOCO_DIR / "loco_results.csv"
LOCO_SUBJECTS_CSV    = LOCO_DIR / "loco_subject_predictions.csv"
LOCO_SUMMARY_JSON    = LOCO_DIR / "loco_summary.json"

PLOT_UNCERTAINTY_CORRECT_INCORRECT = LOCO_PLOTS_DIR / "uncertainty_correct_vs_incorrect.png"
PLOT_UNCERTAINTY_BY_COHORT         = LOCO_PLOTS_DIR / "uncertainty_by_cohort.png"
PLOT_RELIABILITY_CURVE             = LOCO_PLOTS_DIR / "reliability_curve.png"
PLOT_ROC_ERROR_DETECTION           = LOCO_PLOTS_DIR / "roc_uncertainty_error_detection.png"

# ---------------------------------------------------------------------------
# LOCO experiment definitions
# ---------------------------------------------------------------------------
LOCO_EXPERIMENTS = [
    {"exp_id": 1, "train_cohorts": ["Ga", "Ju"], "test_cohort": "Si"},
    {"exp_id": 2, "train_cohorts": ["Ga", "Si"], "test_cohort": "Ju"},
    {"exp_id": 3, "train_cohorts": ["Ju", "Si"], "test_cohort": "Ga"},
]

EXPECTED_COHORTS = {"Ga", "Ju", "Si"}


# ===========================================================================
# SECTION 1 — DATA SPLIT UTILITIES
# ===========================================================================

def build_cohort_splits(
    manifest_df: pd.DataFrame,
    train_cohorts: List[str],
    test_cohort: str,
    val_fraction: float = 0.15,
    seed: int = config.RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split manifest into train / val / test DataFrames based on cohort labels.

    Leakage contract:
        - test_df  ←  ONLY recordings from test_cohort
        - train_df ←  recordings from train_cohorts minus a val_fraction (subject-level)
        - val_df   ←  a held-out fraction of train-cohort subjects

    The held-out test cohort is untouched until final evaluation.
    """
    # ---------- Separate test cohort FIRST (strict) -------------------------
    test_df = manifest_df[manifest_df["study"] == test_cohort].copy().reset_index(drop=True)
    train_pool_df = manifest_df[manifest_df["study"].isin(train_cohorts)].copy().reset_index(drop=True)

    # ---------- Leakage assertion: cohorts must be disjoint -----------------
    assert test_cohort not in train_cohorts, (
        f"LEAKAGE: test cohort '{test_cohort}' appears in train_cohorts {train_cohorts}"
    )
    assert test_df["subject_id"].nunique() > 0, (
        f"ERROR: No subjects found for test cohort '{test_cohort}'"
    )
    assert train_pool_df["subject_id"].nunique() > 0, (
        f"ERROR: No subjects found for train cohorts {train_cohorts}"
    )

    # ---------- Subject-level train/val split within train cohorts (Stratified) ----------
    sub_label_df = train_pool_df.groupby("subject_id")["label"].first()
    classes = sorted(sub_label_df.unique())

    if len(classes) < 2:
        raise ValueError(
            f"Cannot perform stratified split: train pool contains only 1 class ({classes})"
        )

    val_subjects = set()
    train_subjects = set()
    rng = np.random.RandomState(seed)

    for c in classes:
        c_subs = sorted(sub_label_df[sub_label_df == c].index.tolist())
        if len(c_subs) < 2:
            raise ValueError(
                f"Cannot perform stratified split: Class '{c}' has only {len(c_subs)} subject(s) in training pool. At least 2 subjects per class are required."
            )
        c_shuffled = list(rng.permutation(c_subs))
        c_n_val = max(1, int(round(len(c_shuffled) * val_fraction)))
        if c_n_val >= len(c_shuffled):
            raise ValueError(
                f"Cannot perform stratified split: Class '{c}' has {len(c_shuffled)} subjects, but val_fraction ({val_fraction}) allocated all of them to validation."
            )
        val_subjects.update(c_shuffled[:c_n_val])
        train_subjects.update(c_shuffled[c_n_val:])

    # Guard: train/val must be non-overlapping
    assert len(train_subjects & val_subjects) == 0, (
        "LEAKAGE: Train and val subject sets overlap!"
    )

    train_df = train_pool_df[train_pool_df["subject_id"].isin(train_subjects)].copy().reset_index(drop=True)
    val_df   = train_pool_df[train_pool_df["subject_id"].isin(val_subjects)].copy().reset_index(drop=True)

    # ---------- Final leakage assertions ------------------------------------
    all_test_subjects  = set(test_df["subject_id"].unique())
    all_train_subjects_set = set(train_df["subject_id"].unique())
    all_val_subjects   = set(val_df["subject_id"].unique())

    assert len(all_train_subjects_set & all_test_subjects) == 0, (
        f"LEAKAGE DETECTED: Subjects appear in both train and test!\n"
        f"  Overlap: {all_train_subjects_set & all_test_subjects}"
    )
    assert len(all_val_subjects & all_test_subjects) == 0, (
        f"LEAKAGE DETECTED: Subjects appear in both val and test!\n"
        f"  Overlap: {all_val_subjects & all_test_subjects}"
    )
    assert len(all_train_subjects_set & all_val_subjects) == 0, (
        "LEAKAGE DETECTED: Subjects appear in both train and val!"
    )

    return train_df, val_df, test_df


def fit_normalizer_on_train(train_df: pd.DataFrame) -> GaitNormalizer:
    """
    Fit GaitNormalizer STRICTLY on training-cohort recordings.

    Check 3 — Normalization leakage guard:
        The normalizer sees zero recordings from the test cohort.
    """
    train_recordings = [load_raw_recording(fp) for fp in train_df["filepath"]]
    normalizer = GaitNormalizer()
    normalizer.fit(train_recordings)
    # Explicit confirmation (for audit trail)
    assert normalizer.is_fit, "Normalizer was not fitted correctly."
    return normalizer


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    normalizer: GaitNormalizer,
    batch_size: int = config.BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build DataLoaders for train, val, and test splits."""
    train_ds = GaitWindowDataset(
        train_df, normalizer=normalizer,
        window_size=config.WINDOW_SIZE, step_size=config.STEP_SIZE
    )
    val_ds = GaitWindowDataset(
        val_df, normalizer=normalizer,
        window_size=config.WINDOW_SIZE, step_size=config.STEP_SIZE
    )
    test_ds = GaitWindowDataset(
        test_df, normalizer=normalizer,
        window_size=config.WINDOW_SIZE, step_size=config.STEP_SIZE
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_gait_batch, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_gait_batch, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              collate_fn=collate_gait_batch, drop_last=False)

    return train_loader, val_loader, test_loader


# ===========================================================================
# SECTION 2 — MODEL TRAINING
# ===========================================================================

def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: EvidentialLoss,
    device: torch.device,
    epoch: int,
    max_grad_norm: float = config.MAX_GRAD_NORM,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Tuple[float, float, float]:
    """Train one epoch with the Evidential Loss (ACE + annealed KL)."""
    model.train()
    total_loss = total_ace = total_kl = 0.0
    optimizer.zero_grad()
    use_amp = (device.type == "cuda") and (scaler is not None)

    for batch in loader:
        features = batch["features"].to(device)
        labels   = batch["labels"].to(device)

        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                evidence = model(features)
                loss, ld = criterion(evidence, labels, epoch=epoch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            evidence = model(features)
            loss, ld = criterion(evidence, labels, epoch=epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        optimizer.zero_grad()

        total_loss += ld["loss"]
        total_ace  += ld["ace_loss"]
        total_kl   += ld["kl_loss"]

    n = max(1, len(loader))
    return total_loss / n, total_ace / n, total_kl / n


@torch.no_grad()
def _validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: EvidentialLoss,
    device: torch.device,
    epoch: int,
) -> Tuple[float, Dict]:
    """Evaluate one epoch with subject-level aggregation."""
    model.eval()
    total_loss = 0.0
    records: List[Dict] = []

    for batch in loader:
        features = batch["features"].to(device)
        labels   = batch["labels"].to(device)

        evidence = model(features)
        loss, _  = criterion(evidence, labels, epoch=epoch)
        total_loss += loss.item()

        alpha, S, probs, u = dirichlet_from_evidence(evidence, num_classes=2)

        e_np = evidence.cpu().numpy()
        a_np = alpha.cpu().numpy()
        p_np = probs.cpu().numpy()
        u_np = u.squeeze(-1).cpu().numpy()
        y_np = labels.cpu().numpy()

        for i in range(len(y_np)):
            records.append({
                "subject_id":      batch["subject_ids"][i],
                "recording_id":    batch["recording_ids"][i],
                "study":           batch["studies"][i],
                "true_label":      int(y_np[i]),
                "e_HC":            float(e_np[i, 0]),
                "e_PD":            float(e_np[i, 1]),
                "alpha_HC":        float(a_np[i, 0]),
                "alpha_PD":        float(a_np[i, 1]),
                "P_HC":            float(p_np[i, 0]),
                "P_PD":            float(p_np[i, 1]),
                "uncertainty":     float(u_np[i]),
                "predicted_label": int(p_np[i, 1] >= 0.5),
            })

    df = pd.DataFrame(records)
    avg_loss = total_loss / max(1, len(loader))

    # Subject-level aggregation for model selection
    sub_df = (df.groupby(["subject_id", "true_label"])[["P_PD", "uncertainty"]]
                .mean().reset_index())
    sub_df["predicted_label"] = (sub_df["P_PD"] >= 0.5).astype(int)

    metrics = calculate_metrics(
        sub_df["true_label"].tolist(),
        sub_df["predicted_label"].tolist(),
        sub_df["P_PD"].tolist(),
    )
    return avg_loss, metrics


def train_loco_model(
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device:       torch.device,
    epochs:       int,
    seed:         int,
    exp_label:    str,
    annealing_epochs: int = 10,
    patience:     int = config.PATIENCE,
) -> GaitCNNBiLSTM:
    """
    Train a fresh GaitCNNBiLSTM with Evidential Loss for one LOCO fold.

    Two-stage training mirrors train_evidential.py:
        Stage A: Evidential head warmup (backbone frozen)  — first 20% of epochs
        Stage B: Fine-tune BiLSTM + embedding + head       — remaining epochs

    Check 4 — Test labels NEVER seen:
        val_loader uses train-cohort validation subjects only.
        test_loader is never passed to this function.
    """
    set_seed(seed)
    model = GaitCNNBiLSTM(
        in_channels=config.NUM_CHANNELS,
        cnn_channels=config.CNN_CHANNELS,
        lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
        lstm_num_layers=config.LSTM_NUM_LAYERS,
        embedding_dim=config.EMBEDDING_DIM,
        num_classes=2,
        dropout=config.DROPOUT,
    ).to(device)

    criterion = EvidentialLoss(num_classes=2, annealing_epochs=annealing_epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # --- Stage A epochs: head warmup ----------------------------------------
    if epochs <= 1:
        stage_a_epochs = 1
        stage_b_epochs = 1
    else:
        stage_a_epochs = max(1, epochs // 5)
        stage_b_epochs = max(1, epochs - stage_a_epochs)

    # Stage A: freeze backbone, train only evidential head
    for name, param in model.named_parameters():
        param.requires_grad = ("evidential_head" in name)

    opt_a = torch.optim.AdamW(
        model.evidential_head.parameters(), lr=1e-3, weight_decay=1e-4
    )
    best_f1 = -1.0
    best_state: Optional[Dict] = None
    patience_count = 0

    print(f"  [{exp_label}] Stage A — Head warmup ({stage_a_epochs} epochs)")
    for ep in range(1, stage_a_epochs + 1):
        t_loss, ace, kl = _train_one_epoch(
            model, train_loader, opt_a, criterion, device, ep, scaler=scaler
        )
        v_loss, v_met = _validate_one_epoch(model, val_loader, criterion, device, ep)
        vf1 = v_met["f1_score"]
        print(
            f"    A-{ep:02d}/{stage_a_epochs:02d} | "
            f"TLoss {t_loss:.4f} (ACE {ace:.4f} KL {kl:.4f}) | "
            f"VLoss {v_loss:.4f} | VSubjAcc {v_met['accuracy']:.4f} | VF1 {vf1:.4f}"
        )
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Stage B: unfreeze BiLSTM + embedding + head; keep CNN frozen
    for name, param in model.named_parameters():
        param.requires_grad = ("cnn" not in name)

    opt_b = torch.optim.AdamW([
        {"params": model.lstm.parameters(),            "lr": 1e-4},
        {"params": model.embedding_layer.parameters(), "lr": 2e-4},
        {"params": model.evidential_head.parameters(), "lr": 5e-4},
    ], weight_decay=1e-4)
    sched_b = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_b, T_max=stage_b_epochs, eta_min=1e-6
    )

    print(f"  [{exp_label}] Stage B — Fine-tune ({stage_b_epochs} epochs)")
    for ep in range(1, stage_b_epochs + 1):
        t_loss, ace, kl = _train_one_epoch(
            model, train_loader, opt_b, criterion, device,
            epoch=ep + stage_a_epochs, scaler=scaler
        )
        sched_b.step()
        v_loss, v_met = _validate_one_epoch(
            model, val_loader, criterion, device, epoch=ep + stage_a_epochs
        )
        vf1 = v_met["f1_score"]
        print(
            f"    B-{ep:02d}/{stage_b_epochs:02d} | "
            f"TLoss {t_loss:.4f} | "
            f"VSubjAcc {v_met['accuracy']:.4f} | VF1 {vf1:.4f} | "
            f"VAUC {v_met['roc_auc']:.4f}"
        )
        if vf1 >= best_f1:
            best_f1 = vf1
            patience_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"    Early stopping at Stage-B epoch {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    print(f"  [{exp_label}] Training complete. Best Val F1: {best_f1:.4f}")
    return model


# ===========================================================================
# SECTION 3 — INFERENCE & AGGREGATION
# ===========================================================================

@torch.no_grad()
def run_inference_on_test(
    model:       GaitCNNBiLSTM,
    test_loader: DataLoader,
    device:      torch.device,
    exp_id:      int,
    train_cohorts: List[str],
    test_cohort:   str,
) -> pd.DataFrame:
    """
    Run inference on held-out test cohort.
    Returns a window-level DataFrame with full EDL outputs.
    """
    model.eval()
    window_records: List[Dict] = []

    for batch in test_loader:
        features = batch["features"].to(device)
        labels   = batch["labels"].to(device)

        evidence = model(features)
        alpha, S, probs, u = dirichlet_from_evidence(evidence, num_classes=2)

        e_np = evidence.cpu().numpy()
        a_np = alpha.cpu().numpy()
        p_np = probs.cpu().numpy()
        u_np = u.squeeze(-1).cpu().numpy()
        y_np = labels.cpu().numpy()

        for i in range(len(y_np)):
            pred = int(p_np[i, 1] >= 0.5)
            window_records.append({
                "experiment":     exp_id,
                "train_cohorts":  "+".join(sorted(train_cohorts)),
                "test_cohort":    test_cohort,
                "subject_id":     batch["subject_ids"][i],
                "recording_id":   batch["recording_ids"][i],
                "study":          batch["studies"][i],
                "true_label":     int(y_np[i]),
                "e_HC":           float(e_np[i, 0]),
                "e_PD":           float(e_np[i, 1]),
                "alpha_HC":       float(a_np[i, 0]),
                "alpha_PD":       float(a_np[i, 1]),
                "P_HC":           float(p_np[i, 0]),
                "P_PD":           float(p_np[i, 1]),
                "uncertainty":    float(u_np[i]),
                "predicted_label": pred,
                "is_correct":     int(pred == int(y_np[i])),
            })

    return pd.DataFrame(window_records)


def aggregate_to_subject_level(window_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate window predictions to subject level via hierarchical aggregation:
    WINDOW -> RECORDING -> SUBJECT

    1. Group window predictions by recording_id (and subject/cohort metadata) and compute recording-level means.
    2. Group recording-level means by subject_id and compute subject-level means.
    3. Determine subject-level prediction (P_PD >= 0.5) and correctness.
    Matches the existing project's hierarchical aggregation pattern from utils.py.
    """
    agg_cols = ["e_HC", "e_PD", "alpha_HC", "alpha_PD", "P_HC", "P_PD", "uncertainty"]
    group_keys = [
        "experiment", "train_cohorts", "test_cohort",
        "subject_id", "study", "true_label"
    ]
    rec_group_keys = group_keys + ["recording_id"]

    # Step 1: WINDOW -> RECORDING
    rec_df = (
        window_df.groupby(rec_group_keys)[agg_cols]
        .mean()
        .reset_index()
    )

    # Step 2: RECORDING -> SUBJECT
    sub_df = (
        rec_df.groupby(group_keys)[agg_cols]
        .mean()
        .reset_index()
    )

    # Step 3: Thresholding & correctness at subject level
    sub_df["predicted_label"] = (sub_df["P_PD"] >= 0.5).astype(int)
    sub_df["is_correct"]      = (sub_df["predicted_label"] == sub_df["true_label"]).astype(int)

    return sub_df


# ===========================================================================
# SECTION 4 — METRICS COMPUTATION
# ===========================================================================

def _safe_nll(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-9) -> float:
    """Binary Negative Log-Likelihood (log-loss)."""
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(
        y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)
    ))


def _safe_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier Score for binary classification."""
    return float(np.mean((y_prob - y_true) ** 2))


def _safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    """
    AUROC guard: returns None if either class is absent in y_true.
    Caller must handle None explicitly — never fabricate a value.
    """
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return None


def compute_loco_metrics(sub_df: pd.DataFrame) -> Dict:
    """
    Compute the full suite of LOCO metrics from subject-level predictions.

    Classification:    Accuracy, Balanced Accuracy, Sensitivity, Specificity, F1, ROC-AUC
    Probabilistic:     NLL, Brier Score
    Uncertainty:       Mean/Median uncertainty (correct vs incorrect)
                       AUROC for uncertainty-based error detection
    """
    y_true = sub_df["true_label"].values.astype(int)
    y_pred = sub_df["predicted_label"].values.astype(int)
    p_pd   = sub_df["P_PD"].values
    u      = sub_df["uncertainty"].values
    correct_mask   = sub_df["is_correct"].values.astype(bool)
    incorrect_mask = ~correct_mask

    # --- Standard classification metrics ------------------------------------
    base_metrics = calculate_metrics(y_true.tolist(), y_pred.tolist(), p_pd.tolist())

    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    # --- Probabilistic metrics ----------------------------------------------
    nll   = _safe_nll(y_true, p_pd)
    brier = _safe_brier(y_true, p_pd)

    # --- Uncertainty breakdown ----------------------------------------------
    mean_u_correct   = float(u[correct_mask].mean())   if correct_mask.any()   else float("nan")
    mean_u_incorrect = float(u[incorrect_mask].mean()) if incorrect_mask.any() else float("nan")
    med_u_correct    = float(np.median(u[correct_mask]))   if correct_mask.any()   else float("nan")
    med_u_incorrect  = float(np.median(u[incorrect_mask])) if incorrect_mask.any() else float("nan")

    # --- AUROC for uncertainty → error detection ----------------------------
    # Target: 1 = prediction is INCORRECT, 0 = correct
    # Score:  higher uncertainty → more likely incorrect
    error_target = (~correct_mask).astype(int)
    uncertainty_auroc = _safe_auroc(error_target, u)

    return {
        "accuracy":                  base_metrics["accuracy"],
        "balanced_accuracy":         bal_acc,
        "sensitivity":               base_metrics["recall_sensitivity"],
        "specificity":               base_metrics["specificity"],
        "f1":                        base_metrics["f1_score"],
        "roc_auc":                   base_metrics["roc_auc"],
        "nll":                       nll,
        "brier":                     brier,
        "mean_uncertainty_correct":  mean_u_correct,
        "mean_uncertainty_incorrect":mean_u_incorrect,
        "median_uncertainty_correct":  med_u_correct,
        "median_uncertainty_incorrect":med_u_incorrect,
        "uncertainty_error_auroc":   uncertainty_auroc,
        # raw counts for audit
        "n_correct":   int(correct_mask.sum()),
        "n_incorrect": int(incorrect_mask.sum()),
        "tp": base_metrics["tp"],
        "tn": base_metrics["tn"],
        "fp": base_metrics["fp"],
        "fn": base_metrics["fn"],
    }


# ===========================================================================
# SECTION 5 — VISUALIZATIONS
# ===========================================================================

def _figsize_dpi() -> Tuple[Tuple[int, int], int]:
    return (7, 5), 200


def plot_uncertainty_correct_vs_incorrect(all_sub_df: pd.DataFrame, save_path: Path) -> None:
    """
    Plot 1 — Boxplot: predictive uncertainty for correct vs incorrect predictions.
    Pools subjects across all LOCO experiments.
    """
    fig_size, dpi = _figsize_dpi()
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

    correct_u   = all_sub_df[all_sub_df["is_correct"] == 1]["uncertainty"].values
    incorrect_u = all_sub_df[all_sub_df["is_correct"] == 0]["uncertainty"].values

    data_to_plot = []
    labels = []
    if len(correct_u):
        data_to_plot.append(correct_u)
        labels.append(f"Correct\n(n={len(correct_u)})")
    if len(incorrect_u):
        data_to_plot.append(incorrect_u)
        labels.append(f"Incorrect\n(n={len(incorrect_u)})")

    if not data_to_plot:
        plt.close()
        return

    bp = ax.boxplot(
        data_to_plot,
        tick_labels=labels,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
        widths=0.4,
    )
    colours = ["#2ecc71", "#e74c3c"]
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.7)

    ax.set_title(
        "Predictive Uncertainty: Correct vs Incorrect Predictions\n(Pooled across LOCO folds)",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.set_xlabel("Prediction Outcome", fontsize=10)
    ax.set_ylabel("Predictive Uncertainty  (K/S)", fontsize=10)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_uncertainty_by_cohort(all_sub_df: pd.DataFrame, save_path: Path) -> None:
    """
    Plot 2 — Boxplot: predictive uncertainty for each held-out cohort.
    """
    fig_size, dpi = _figsize_dpi()
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

    cohorts_present = sorted(all_sub_df["test_cohort"].unique())
    data_to_plot = []
    labels = []
    for cohort in cohorts_present:
        u_vals = all_sub_df[all_sub_df["test_cohort"] == cohort]["uncertainty"].values
        if len(u_vals):
            data_to_plot.append(u_vals)
            labels.append(f"{cohort}\n(n={len(u_vals)})")

    if not data_to_plot:
        plt.close()
        return

    bp = ax.boxplot(
        data_to_plot,
        tick_labels=labels,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
        widths=0.4,
    )
    palette = ["#3498db", "#e67e22", "#9b59b6"]
    for patch, colour in zip(bp["boxes"], palette):
        patch.set_facecolor(colour)
        patch.set_alpha(0.7)

    ax.set_title(
        "Predictive Uncertainty by Held-Out Cohort",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.set_xlabel("Held-Out Test Cohort", fontsize=10)
    ax.set_ylabel("Predictive Uncertainty  (K/S)", fontsize=10)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_reliability_diagram(all_sub_df: pd.DataFrame, save_path: Path, n_bins: int = 10) -> None:
    """
    Plot 3 — Reliability diagram (calibration curve).
    X-axis: Mean predicted confidence (P_PD in PD-positive bins, 1-P_PD in HC bins).
    Y-axis: Fraction of positive labels in that bin.
    """
    fig_size, dpi = _figsize_dpi()
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

    p_pd   = all_sub_df["P_PD"].values
    y_true = all_sub_df["true_label"].values

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_means, bin_fracs = [], []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (p_pd >= lo) & (p_pd <= hi)
        else:
            mask = (p_pd >= lo) & (p_pd < hi)
        if mask.sum() == 0:
            continue
        bin_means.append(float(p_pd[mask].mean()))
        bin_fracs.append(float(y_true[mask].mean()))

    if not bin_means:
        plt.close()
        return

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    ax.plot(bin_means, bin_fracs, "o-", color="#1f77b4", lw=2,
            markersize=7, label="Model (LOCO pooled)")

    ax.fill_between(
        [0, 1], [0, 1], [0, 1],
        alpha=0.05, color="gray", label=None
    )

    ax.set_title("Reliability Diagram (Calibration Curve)\n(Pooled across LOCO folds)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Mean Predicted Confidence (P_PD)", fontsize=10)
    ax.set_ylabel("Fraction of PD-Positive Subjects", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_roc_uncertainty_error_detection(all_sub_df: pd.DataFrame, save_path: Path) -> None:
    """
    Plot 4 — ROC curve for uncertainty-based error detection.
    Target:  1 = prediction is incorrect
    Score:   predictive uncertainty (higher → more likely wrong)
    """
    fig_size, dpi = _figsize_dpi()
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

    error_target = (all_sub_df["is_correct"] == 0).astype(int).values
    u_scores     = all_sub_df["uncertainty"].values

    auroc = _safe_auroc(error_target, u_scores)
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance (AUROC = 0.500)")

    if auroc is not None:
        fpr, tpr, _ = roc_curve(error_target, u_scores)
        ax.plot(fpr, tpr, color="#e74c3c", lw=2.5,
                label=f"Uncertainty detector (AUROC = {auroc:.3f})")
    else:
        ax.text(0.5, 0.5, "AUROC undefined\n(only one class in labels)",
                ha="center", va="center", fontsize=10, color="red",
                transform=ax.transAxes)

    ax.set_title("Uncertainty-Based Error Detection ROC Curve\n(Pooled across LOCO folds)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ===========================================================================
# SECTION 6 — RESEARCH SAFETY CHECKS
# ===========================================================================

def run_safety_checks(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    normalizer: GaitNormalizer,
    train_cohorts: List[str],
    test_cohort:   str,
) -> None:
    """
    Explicit research-safety assertions before inference.
    Crash loud rather than silently produce compromised results.
    """
    # Check 1 — Cohort separation
    assert test_cohort not in train_cohorts, (
        f"CHECK 1 FAIL: test cohort '{test_cohort}' is in train cohorts {train_cohorts}"
    )

    # Check 2 — Subject separation
    train_subjs = set(train_df["subject_id"].unique())
    val_subjs   = set(val_df["subject_id"].unique())
    test_subjs  = set(test_df["subject_id"].unique())
    overlap_train_test = train_subjs & test_subjs
    overlap_val_test   = val_subjs   & test_subjs
    assert not overlap_train_test, (
        f"CHECK 2 FAIL: {len(overlap_train_test)} subjects in both train and test: "
        f"{overlap_train_test}"
    )
    assert not overlap_val_test, (
        f"CHECK 2 FAIL: {len(overlap_val_test)} subjects in both val and test: "
        f"{overlap_val_test}"
    )

    # Check 3 — Normalizer fitted only on training data
    assert normalizer.is_fit, "CHECK 3 FAIL: normalizer is not fitted."
    assert normalizer.mean is not None and normalizer.std is not None, (
        "CHECK 3 FAIL: normalizer has no mean/std arrays."
    )
    # (Structural check — runtime proof that fit() used train data only
    # is guaranteed by build_cohort_splits() which returns train_df before
    # the test_df was ever constructed.)

    # Check 4 — Confirmed by function signature: test_loader is NEVER passed
    # to train_loco_model(); the function only receives train_loader + val_loader.

    # Check 5 — Seed set externally; confirmed by set_seed() calls at each entry point.

    print(
        f"  [SAFETY OK] Cohort sep ✓ | Subject sep "
        f"(train∩test={len(overlap_train_test)}, val∩test={len(overlap_val_test)}) ✓ | "
        f"Normalizer fitted ✓"
    )


# ===========================================================================
# SECTION 7 — MAIN LOCO RUNNER
# ===========================================================================

def run_loco(epochs: int = config.EPOCHS, seed: int = config.RANDOM_SEED) -> None:
    """
    Execute all three LOCO experiments sequentially.

    For each fold:
        1. Split manifest by cohort (no test-cohort contamination)
        2. Fit normalizer on train recordings only
        3. Build DataLoaders
        4. Train fresh model (train cohorts only, val from train cohorts only)
        5. Run safety checks
        6. Infer on held-out test cohort
        7. Aggregate to subject level
        8. Compute metrics
    Then pool all predictions and generate plots + CSV/JSON outputs.
    """
    t_start = time.time()
    set_seed(seed)
    device = get_device()

    # ------------------------------------------------------------------
    # Step 0 — Load manifest and verify cohorts
    # ------------------------------------------------------------------
    print("\nLoading Gait dataset manifest...")
    if not config.DATA_DIR.exists():
        print(f"\nERROR: Dataset directory not found.\n  Expected: {config.DATA_DIR}")
        print("  Please download the PhysioNet Gait dataset and place it at the path above.")
        sys.exit(1)

    manifest_df = build_manifest(config.DATA_DIR)

    found_cohorts = set(manifest_df["study"].unique())
    missing = EXPECTED_COHORTS - found_cohorts
    if missing:
        print(f"\nERROR: Expected cohorts not found in manifest.")
        print(f"  Missing: {missing}")
        print(f"  Found:   {found_cohorts}")
        print("  Check that study prefixes Ga/Ju/Si are present in the filenames.")
        sys.exit(1)

    print(f"  Found {len(manifest_df)} recordings | "
          f"Subjects: {manifest_df['subject_id'].nunique()} | "
          f"Cohorts: {sorted(found_cohorts)}")

    # ------------------------------------------------------------------
    # Main LOCO loop
    # ------------------------------------------------------------------
    all_subject_rows: List[pd.DataFrame] = []
    result_rows:      List[Dict]         = []
    exp_summaries:    List[Dict]         = []

    for exp in LOCO_EXPERIMENTS:
        exp_id        = exp["exp_id"]
        train_cohorts = exp["train_cohorts"]
        test_cohort   = exp["test_cohort"]
        exp_label     = f"Exp{exp_id} (train={'+'.join(train_cohorts)} test={test_cohort})"

        print("\n" + "=" * 62)
        print(f"LOCO Experiment {exp_id}/3")
        print(f"  Train cohorts : {' + '.join(train_cohorts)}")
        print(f"  Test cohort   : {test_cohort}")
        print("=" * 62)

        # --- 1. Split manifest by cohort -----------------------------------
        train_df, val_df, test_df = build_cohort_splits(
            manifest_df, train_cohorts, test_cohort, seed=seed
        )
        n_train_subjs = train_df["subject_id"].nunique()
        n_val_subjs   = val_df["subject_id"].nunique()
        n_test_subjs  = test_df["subject_id"].nunique()
        print(f"  Train subjects: {n_train_subjs} | Val subjects: {n_val_subjs} | "
              f"Test subjects: {n_test_subjs}")

        # --- 2. Fit normalizer on train recordings ONLY --------------------
        print(f"  Fitting normalizer on {n_train_subjs} train subjects "
              f"({len(train_df)} recordings)...")
        normalizer = fit_normalizer_on_train(train_df)

        # --- 3. Build DataLoaders ------------------------------------------
        train_loader, val_loader, test_loader = build_dataloaders(
            train_df, val_df, test_df, normalizer
        )
        print(f"  Windows → Train: {len(train_loader.dataset)} | "
              f"Val: {len(val_loader.dataset)} | "
              f"Test: {len(test_loader.dataset)}")

        # --- 4. Train model -----------------------------------------------
        set_seed(seed)   # re-seed before each fold for reproducibility
        model = train_loco_model(
            train_loader, val_loader, device,
            epochs=epochs, seed=seed,
            exp_label=exp_label,
            annealing_epochs=min(10, epochs // 2),
            patience=config.PATIENCE,
        )

        # --- 5. Safety checks BEFORE inference ----------------------------
        run_safety_checks(
            train_df, val_df, test_df,
            normalizer, train_cohorts, test_cohort
        )

        # --- 6. Inference on held-out test cohort -------------------------
        print(f"  Running inference on held-out cohort: {test_cohort}")
        window_df = run_inference_on_test(
            model, test_loader, device,
            exp_id, train_cohorts, test_cohort
        )

        # --- 7. Subject-level aggregation ---------------------------------
        sub_df = aggregate_to_subject_level(window_df)
        all_subject_rows.append(sub_df)

        # --- 8. Compute metrics -------------------------------------------
        metrics = compute_loco_metrics(sub_df)
        u_auroc_str = (
            f"{metrics['uncertainty_error_auroc']:.4f}"
            if metrics["uncertainty_error_auroc"] is not None
            else "N/A"
        )

        # Build result row for CSV
        result_row = {
            "experiment":              exp_id,
            "train_cohorts":           "+".join(sorted(train_cohorts)),
            "test_cohort":             test_cohort,
            "num_train_subjects":      n_train_subjs,
            "num_test_subjects":       n_test_subjs,
            "accuracy":                metrics["accuracy"],
            "balanced_accuracy":       metrics["balanced_accuracy"],
            "sensitivity":             metrics["sensitivity"],
            "specificity":             metrics["specificity"],
            "f1":                      metrics["f1"],
            "roc_auc":                 metrics["roc_auc"],
            "nll":                     metrics["nll"],
            "brier":                   metrics["brier"],
            "mean_uncertainty_correct":   metrics["mean_uncertainty_correct"],
            "mean_uncertainty_incorrect": metrics["mean_uncertainty_incorrect"],
            "median_uncertainty_correct":   metrics["median_uncertainty_correct"],
            "median_uncertainty_incorrect": metrics["median_uncertainty_incorrect"],
            "uncertainty_error_auroc": metrics["uncertainty_error_auroc"],
        }
        result_rows.append(result_row)

        # Build experiment summary for print and JSON
        exp_summary = {
            "experiment":    exp_id,
            "train_cohorts": train_cohorts,
            "test_cohort":   test_cohort,
            "n_train_subjects": n_train_subjs,
            "n_test_subjects":  n_test_subjs,
            "metrics":       metrics,
        }
        exp_summaries.append(exp_summary)

        # Per-experiment console summary
        print(f"\n  Results for Exp {exp_id} (test={test_cohort}):")
        print(f"    Accuracy:          {metrics['accuracy']:.4f}")
        print(f"    Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"    Sensitivity:       {metrics['sensitivity']:.4f}")
        print(f"    Specificity:       {metrics['specificity']:.4f}")
        print(f"    F1:                {metrics['f1']:.4f}")
        print(f"    ROC-AUC:           {metrics['roc_auc']:.4f}")
        print(f"    NLL:               {metrics['nll']:.4f}")
        print(f"    Brier:             {metrics['brier']:.4f}")
        print(f"    Mean U (correct):  {metrics['mean_uncertainty_correct']:.4f}")
        print(f"    Mean U (incorrect):{metrics['mean_uncertainty_incorrect']:.4f}")
        print(f"    Uncertainty AUROC: {u_auroc_str}")

        # Free model memory before next fold
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Post-loop: pool results and save outputs
    # ------------------------------------------------------------------
    all_sub_df = pd.concat(all_subject_rows, ignore_index=True)
    results_df = pd.DataFrame(result_rows)

    # Save CSVs
    all_sub_df.to_csv(LOCO_SUBJECTS_CSV, index=False)
    results_df.to_csv(LOCO_RESULTS_CSV, index=False)
    print(f"\n  Saved subject predictions → {LOCO_SUBJECTS_CSV}")
    print(f"  Saved results CSV         → {LOCO_RESULTS_CSV}")

    # Save JSON summary
    summary = {
        "research_question": (
            "Can Uncertainty Warn Us? Cohort-Shift Reliability of Parkinson's "
            "Disease Classifiers Across Gait"
        ),
        "experiment_type": "Leave-One-Cohort-Out (LOCO)",
        "modality":        "Gait (VGRF, 1D CNN + BiLSTM + EDL)",
        "seed":            seed,
        "epochs_per_fold": epochs,
        "experiments":     exp_summaries,
    }
    with open(LOCO_SUMMARY_JSON, "w") as fh:
        json.dump(summary, fh, indent=4, default=json_default)
    print(f"  Saved summary JSON        → {LOCO_SUMMARY_JSON}")

    # ------------------------------------------------------------------
    # Generate plots
    # ------------------------------------------------------------------
    print("\nGenerating plots...")
    plot_uncertainty_correct_vs_incorrect(all_sub_df, PLOT_UNCERTAINTY_CORRECT_INCORRECT)
    plot_uncertainty_by_cohort(all_sub_df, PLOT_UNCERTAINTY_BY_COHORT)
    plot_reliability_diagram(all_sub_df, PLOT_RELIABILITY_CURVE)
    plot_roc_uncertainty_error_detection(all_sub_df, PLOT_ROC_ERROR_DETECTION)

    # ------------------------------------------------------------------
    # Final research report
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    print("\n")
    print("=" * 50)
    print("Gait Leave-One-Cohort-Out Experiment")
    print("=" * 50)

    for row, exp_s in zip(result_rows, exp_summaries):
        exp_id   = row["experiment"]
        u_auroc  = row["uncertainty_error_auroc"]
        u_auroc_display = f"{u_auroc:.4f}" if u_auroc is not None else "N/A"

        print(f"\nExperiment {exp_id}:")
        print(f"  Train : {row['train_cohorts']}")
        print(f"  Test  : {row['test_cohort']}")
        print(f"\n  Subjects:")
        print(f"    Train : {row['num_train_subjects']}")
        print(f"    Test  : {row['num_test_subjects']}")
        print(f"\n  Balanced Accuracy : {row['balanced_accuracy']:.4f}")
        print(f"  Sensitivity       : {row['sensitivity']:.4f}")
        print(f"  Specificity       : {row['specificity']:.4f}")
        print(f"  NLL               : {row['nll']:.4f}")
        print(f"  Brier             : {row['brier']:.4f}")
        print(f"\n  Mean uncertainty:")
        print(f"    Correct  : {row['mean_uncertainty_correct']:.4f}")
        print(f"    Incorrect: {row['mean_uncertainty_incorrect']:.4f}")
        print(f"\n  Uncertainty Error Detection AUROC: {u_auroc_display}")
        print("-" * 50)

    print(f"\nResults saved to:\n  {LOCO_DIR}")
    print(f"\nTotal elapsed time: {elapsed / 60:.1f} min")
    print("=" * 50)


# ===========================================================================
# SECTION 8 — CLI ENTRY POINT
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gait Leave-One-Cohort-Out (LOCO) Experiment\n"
            "Research: Cohort-shift reliability of Parkinson's classifiers.\n\n"
            "Three experiments:\n"
            "  Exp 1: TRAIN=Ga+Ju  TEST=Si\n"
            "  Exp 2: TRAIN=Ga+Si  TEST=Ju\n"
            "  Exp 3: TRAIN=Ju+Si  TEST=Ga"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--epochs", type=int, default=config.EPOCHS,
        help=f"Training epochs per LOCO fold (default: {config.EPOCHS})"
    )
    parser.add_argument(
        "--seed", type=int, default=config.RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {config.RANDOM_SEED})"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_loco(epochs=args.epochs, seed=args.seed)
