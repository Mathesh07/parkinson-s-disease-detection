"""Evaluation script for Handwriting Evidential ViT with per-sample Dirichlet & uncertainty outputs."""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
for p in (str(current_dir), str(parent_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from Handwriting.config import (
    BATCH_SIZE, IMAGE_SIZE, OUTPUT_DIR, RANDOM_SEED
)
from Handwriting.dataset import (
    build_manifest, create_dataloaders,
    create_patient_level_split
)
from Handwriting.model import ViTBinaryClassifier
from evidential import dirichlet_from_evidence

EVIDENTIAL_VIT_CHECKPOINT = OUTPUT_DIR / "best_vit_evidential.pth"
EVIDENTIAL_PREDICTIONS_CSV = OUTPUT_DIR / "handwriting_evidential_predictions.csv"
EVIDENTIAL_METRICS_JSON = OUTPUT_DIR / "handwriting_evidential_metrics.json"


def evaluate_handwriting_evidential(
    checkpoint_path: Path = EVIDENTIAL_VIT_CHECKPOINT
) -> Dict[str, any]:
    """Run evidential evaluation on held-out test handwriting dataset."""
    print("=" * 60)
    print("HANDWRITING ViT EVIDENTIAL EVALUATION (TEST SET)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest, _ = build_manifest()
    manifest = create_patient_level_split(manifest)
    _, _, test_loader = create_dataloaders(
        manifest, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, num_workers=0
    )

    model = ViTBinaryClassifier(freeze_backbone=False).to(device)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Evidential checkpoint not found at: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    records = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            evidence = model(images)  # (B, 2)
            alpha, S, probs, u = dirichlet_from_evidence(evidence, num_classes=2)

            e_np = evidence.cpu().numpy()
            a_np = alpha.cpu().numpy()
            p_np = probs.cpu().numpy()
            u_np = u.squeeze(-1).cpu().numpy()
            y_np = labels.cpu().numpy()

            for i in range(len(y_np)):
                pred = int(p_np[i, 1] >= 0.5)
                records.append({
                    "sample_id": f"test_img_{batch_idx * BATCH_SIZE + i}",
                    "true_label": int(y_np[i]),
                    "predicted_label": pred,
                    "is_correct": int(pred == int(y_np[i])),
                    "e_HC": float(e_np[i, 0]),
                    "e_PD": float(e_np[i, 1]),
                    "alpha_HC": float(a_np[i, 0]),
                    "alpha_PD": float(a_np[i, 1]),
                    "P_HC": float(p_np[i, 0]),
                    "P_PD": float(p_np[i, 1]),
                    "uncertainty": float(u_np[i])
                })

    df = pd.DataFrame(records)
    df.to_csv(EVIDENTIAL_PREDICTIONS_CSV, index=False)
    print(f"Saved evidential predictions to: {EVIDENTIAL_PREDICTIONS_CSV}")

    y_true = df["true_label"].values
    y_pred = df["predicted_label"].values
    y_prob = df["P_PD"].values

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.0

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall_sensitivity": float(rec),
        "specificity": float(spec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }

    correct_u = df[df["is_correct"] == 1]["uncertainty"].mean()
    incorrect_u = df[df["is_correct"] == 0]["uncertainty"].mean() if (df["is_correct"] == 0).any() else 0.0
    hc_u = df[df["true_label"] == 0]["uncertainty"].mean()
    pd_u = df[df["true_label"] == 1]["uncertainty"].mean()

    uncertainty_stats = {
        "mean_uncertainty_all": float(df["uncertainty"].mean()),
        "mean_uncertainty_correct": float(correct_u),
        "mean_uncertainty_incorrect": float(incorrect_u),
        "mean_uncertainty_HC": float(hc_u),
        "mean_uncertainty_PD": float(pd_u)
    }

    full_results = {
        "modality": "Handwriting (ViT)",
        "metrics": metrics,
        "uncertainty_stats": uncertainty_stats
    }

    with open(EVIDENTIAL_METRICS_JSON, "w") as f:
        json.dump(full_results, f, indent=4)

    print("\n" + "=" * 50)
    print("HANDWRITING ViT EVIDENTIAL TEST RESULTS")
    print("=" * 50)
    print(f"Accuracy:        {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision:       {metrics['precision'] * 100:.2f}%")
    print(f"Sensitivity:     {metrics['recall_sensitivity'] * 100:.2f}%")
    print(f"Specificity:     {metrics['specificity'] * 100:.2f}%")
    print(f"F1-Score:        {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:         {metrics['roc_auc']:.4f}")
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
    evaluate_handwriting_evidential()
