"""Evaluation and visualization module for Parkinson's handwriting detection.

Computes metrics on the held-out patient-level test set:
- Accuracy
- Precision
- Recall / Sensitivity
- Specificity
- F1-Score
- ROC-AUC
- Confusion Matrix (TN, FP, FN, TP)

Also plots and saves training and validation loss/accuracy curves.
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from config import (BATCH_SIZE, CNN_CHECKPOINT, CURVES_PLOT_PATH, IMAGE_SIZE,
                    METRICS_OUTPUT_PATH, OUTPUT_DIR, VIT_CHECKPOINT)
from dataset import (build_manifest, create_dataloaders,
                     create_patient_level_split)
from model import CNNBaseline, ViTBinaryClassifier


def compute_metrics(y_true: list, y_pred: list, y_prob: list) -> Dict[str, object]:
    """Compute standard classification metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }
    return metrics


def evaluate_checkpoint(
    model_type: str = "vit"
) -> Tuple[Dict[str, object], float, float]:
    """Load model checkpoint and evaluate on held-out patient-level test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest, _ = build_manifest()
    manifest = create_patient_level_split(manifest)
    _, _, test_loader = create_dataloaders(manifest, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, num_workers=0)

    checkpoint_path = VIT_CHECKPOINT if model_type.lower() == "vit" else CNN_CHECKPOINT
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint for {model_type} not found at {checkpoint_path}")

    if model_type.lower() == "vit":
        model = ViTBinaryClassifier(freeze_backbone=False).to(device)
    else:
        model = CNNBaseline(num_classes=2).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)[:, 1]

            correct += (preds == labels).sum().item()
            total += images.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    test_loss = running_loss / total
    test_acc = correct / total
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    return metrics, test_loss, test_acc


def plot_training_curves() -> None:
    """Plot and save training/validation loss and accuracy curves for ViT and CNN."""
    vit_hist_file = OUTPUT_DIR / "vit_history.json"
    cnn_hist_file = OUTPUT_DIR / "cnn_history.json"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    if vit_hist_file.exists():
        with open(vit_hist_file) as f:
            vit_h = json.load(f)
        epochs_vit = range(1, len(vit_h["train_loss"]) + 1)
        axes[0].plot(epochs_vit, vit_h["train_loss"], "b-", label="ViT Train Loss")
        axes[0].plot(epochs_vit, vit_h["val_loss"], "b--", label="ViT Val Loss")
        axes[1].plot(epochs_vit, vit_h["train_acc"], "b-", label="ViT Train Acc")
        axes[1].plot(epochs_vit, vit_h["val_acc"], "b--", label="ViT Val Acc")

    if cnn_hist_file.exists():
        with open(cnn_hist_file) as f:
            cnn_h = json.load(f)
        epochs_cnn = range(1, len(cnn_h["train_loss"]) + 1)
        axes[0].plot(epochs_cnn, cnn_h["train_loss"], "r-", label="CNN Train Loss")
        axes[0].plot(epochs_cnn, cnn_h["val_loss"], "r--", label="CNN Val Loss")
        axes[1].plot(epochs_cnn, cnn_h["train_acc"], "r-", label="CNN Train Acc")
        axes[1].plot(epochs_cnn, cnn_h["val_acc"], "r--", label="CNN Val Acc")

    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    axes[1].set_title("Training & Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(CURVES_PLOT_PATH, dpi=300)
    plt.close()
    print(f"Saved training curves to {CURVES_PLOT_PATH}")


def main() -> None:
    results = {}
    print("\n==========================================")
    print(" Held-Out Patient-Level Test Evaluation")
    print("==========================================")

    # Evaluate ViT
    if VIT_CHECKPOINT.exists():
        vit_metrics, vit_loss, vit_acc = evaluate_checkpoint("vit")
        results["ViT"] = vit_metrics
        print("\n--- Vision Transformer (ViT) Test Results ---")
        print(f"Test Loss:            {vit_loss:.4f}")
        print(f"Accuracy:             {vit_metrics['accuracy'] * 100:.2f}%")
        print(f"Precision:            {vit_metrics['precision'] * 100:.2f}%")
        print(f"Recall (Sensitivity): {vit_metrics['recall_sensitivity'] * 100:.2f}%")
        print(f"Specificity:          {vit_metrics['specificity'] * 100:.2f}%")
        print(f"F1-Score:             {vit_metrics['f1_score'] * 100:.2f}%")
        print(f"ROC-AUC:              {vit_metrics['roc_auc']:.4f}")
        print(f"Confusion Matrix:     {vit_metrics['confusion_matrix']}")

    # Evaluate CNN Baseline
    if CNN_CHECKPOINT.exists():
        cnn_metrics, cnn_loss, cnn_acc = evaluate_checkpoint("cnn")
        results["CNN_Baseline"] = cnn_metrics
        print("\n--- CNN Baseline Test Results ---")
        print(f"Test Loss:            {cnn_loss:.4f}")
        print(f"Accuracy:             {cnn_metrics['accuracy'] * 100:.2f}%")
        print(f"Precision:            {cnn_metrics['precision'] * 100:.2f}%")
        print(f"Recall (Sensitivity): {cnn_metrics['recall_sensitivity'] * 100:.2f}%")
        print(f"Specificity:          {cnn_metrics['specificity'] * 100:.2f}%")
        print(f"F1-Score:             {cnn_metrics['f1_score'] * 100:.2f}%")
        print(f"ROC-AUC:              {cnn_metrics['roc_auc']:.4f}")
        print(f"Confusion Matrix:     {cnn_metrics['confusion_matrix']}")

    # Save JSON results
    with open(METRICS_OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation metrics to {METRICS_OUTPUT_PATH}")

    # Generate curves plot
    plot_training_curves()


if __name__ == "__main__":
    main()
