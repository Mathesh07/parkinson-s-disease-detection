"""Evaluation script for Gait Evidential model with per-sample Dirichlet & uncertainty outputs."""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
for p in (str(current_dir), str(parent_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from Gait import config
from Gait.dataset import GaitWindowDataset, collate_gait_batch, create_subject_splits
from Gait.model import GaitCNNBiLSTM
from Gait.preprocessing import GaitNormalizer, build_manifest
from Gait.utils import calculate_metrics, get_device, set_seed
from evidential import dirichlet_from_evidence

EVIDENTIAL_CHECKPOINT = config.CHECKPOINT_DIR / "best_gait_evidential.pt"
EVIDENTIAL_PREDICTIONS_CSV = config.OUTPUT_DIR / "metrics" / "gait_evidential_predictions.csv"
EVIDENTIAL_SUBJECT_CSV = config.OUTPUT_DIR / "metrics" / "gait_evidential_subjects.csv"
EVIDENTIAL_METRICS_JSON = config.OUTPUT_DIR / "metrics" / "gait_evidential_metrics.json"


def evaluate_gait_evidential(
    checkpoint_path: Path = EVIDENTIAL_CHECKPOINT,
    normalization_path: Path = config.NORMALIZATION_STATS_PATH
) -> Dict[str, any]:
    """Run thorough evidential evaluation on held-out test subjects."""
    print("=" * 60)
    print("GAIT EVIDENTIAL EVALUATION (TEST SET)")
    print("=" * 60)

    set_seed(config.RANDOM_SEED)
    device = get_device()

    manifest_df = build_manifest()
    train_df, val_df, test_df = create_subject_splits(manifest_df, seed=config.RANDOM_SEED)

    normalizer = GaitNormalizer.load(normalization_path)
    test_dataset = GaitWindowDataset(test_df, normalizer=normalizer, window_size=config.WINDOW_SIZE, step_size=config.STEP_SIZE)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_gait_batch
    )

    # Load trained evidential model
    model = GaitCNNBiLSTM(
        in_channels=config.NUM_CHANNELS,
        cnn_channels=config.CNN_CHANNELS,
        lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
        lstm_num_layers=config.LSTM_NUM_LAYERS,
        embedding_dim=config.EMBEDDING_DIM,
        num_classes=2,
        dropout=config.DROPOUT
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Evidential checkpoint not found at: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    window_records = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)

            evidence = model(features)  # (B, 2)
            alpha, S, probs, u = dirichlet_from_evidence(evidence, num_classes=2)

            e_np = evidence.cpu().numpy()
            a_np = alpha.cpu().numpy()
            p_np = probs.cpu().numpy()
            u_np = u.squeeze(-1).cpu().numpy()
            y_np = labels.cpu().numpy()

            for i in range(len(y_np)):
                pred_label = int(p_np[i, 1] >= 0.5)
                window_records.append({
                    "sample_id": f"{batch['recording_ids'][i]}_win{i}",
                    "subject_id": batch["subject_ids"][i],
                    "recording_id": batch["recording_ids"][i],
                    "study": batch["studies"][i],
                    "true_label": int(y_np[i]),
                    "predicted_label": pred_label,
                    "is_correct": int(pred_label == int(y_np[i])),
                    "e_HC": float(e_np[i, 0]),
                    "e_PD": float(e_np[i, 1]),
                    "alpha_HC": float(a_np[i, 0]),
                    "alpha_PD": float(a_np[i, 1]),
                    "P_HC": float(p_np[i, 0]),
                    "P_PD": float(p_np[i, 1]),
                    "uncertainty": float(u_np[i])
                })

    window_df = pd.DataFrame(window_records)

    # Subject-level aggregation
    sub_df = window_df.groupby(["subject_id", "true_label"])[["e_HC", "e_PD", "alpha_HC", "alpha_PD", "P_HC", "P_PD", "uncertainty"]].mean().reset_index()
    sub_df["predicted_label"] = (sub_df["P_PD"] >= 0.5).astype(int)
    sub_df["is_correct"] = (sub_df["predicted_label"] == sub_df["true_label"]).astype(int)

    # Save CSV manifests
    window_df.to_csv(EVIDENTIAL_PREDICTIONS_CSV, index=False)
    sub_df.to_csv(EVIDENTIAL_SUBJECT_CSV, index=False)
    print(f"Saved window-level evidential predictions to: {EVIDENTIAL_PREDICTIONS_CSV}")
    print(f"Saved subject-level evidential predictions to: {EVIDENTIAL_SUBJECT_CSV}")

    # Standard metrics at subject level
    sub_metrics = calculate_metrics(
        sub_df["true_label"].tolist(),
        sub_df["predicted_label"].tolist(),
        sub_df["P_PD"].tolist()
    )

    # Uncertainty breakdowns
    correct_u = sub_df[sub_df["is_correct"] == 1]["uncertainty"].mean()
    incorrect_u = sub_df[sub_df["is_correct"] == 0]["uncertainty"].mean() if (sub_df["is_correct"] == 0).any() else 0.0
    hc_u = sub_df[sub_df["true_label"] == 0]["uncertainty"].mean()
    pd_u = sub_df[sub_df["true_label"] == 1]["uncertainty"].mean()

    uncertainty_stats = {
        "mean_uncertainty_all": float(sub_df["uncertainty"].mean()),
        "mean_uncertainty_correct": float(correct_u),
        "mean_uncertainty_incorrect": float(incorrect_u),
        "mean_uncertainty_HC": float(hc_u),
        "mean_uncertainty_PD": float(pd_u)
    }

    full_results = {
        "modality": "Gait",
        "subject_metrics": sub_metrics,
        "uncertainty_stats": uncertainty_stats
    }

    with open(EVIDENTIAL_METRICS_JSON, "w") as f:
        json.dump(full_results, f, indent=4)

    print("\n" + "=" * 50)
    print("GAIT EVIDENTIAL TEST RESULTS (SUBJECT-LEVEL)")
    print("=" * 50)
    print(f"Accuracy:        {sub_metrics['accuracy'] * 100:.2f}%")
    print(f"Precision:       {sub_metrics['precision'] * 100:.2f}%")
    print(f"Sensitivity:     {sub_metrics['recall_sensitivity'] * 100:.2f}%")
    print(f"Specificity:     {sub_metrics['specificity'] * 100:.2f}%")
    print(f"F1-Score:        {sub_metrics['f1_score']:.4f}")
    print(f"ROC-AUC:         {sub_metrics['roc_auc']:.4f}")
    print("-" * 50)
    print("UNCERTAINTY BREAKDOWN:")
    print(f"  Mean Uncertainty (All):       {uncertainty_stats['mean_uncertainty_all']:.4f}")
    print(f"  Mean Uncertainty (Correct):   {uncertainty_stats['mean_uncertainty_correct']:.4f}")
    print(f"  Mean Uncertainty (Incorrect): {uncertainty_stats['mean_uncertainty_incorrect']:.4f}")
    print(f"  Mean Uncertainty (HC):        {uncertainty_stats['mean_uncertainty_HC']:.4f}")
    print(f"  Mean Uncertainty (PD):        {uncertainty_stats['mean_uncertainty_PD']:.4f}")
    print("=" * 50)

    return full_results


if __name__ == "__main__":
    evaluate_gait_evidential()
