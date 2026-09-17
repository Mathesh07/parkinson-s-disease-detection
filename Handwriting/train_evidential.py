"""Two-stage evidential training pipeline for Handwriting Vision Transformer (ViT)."""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Add workspace roots
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
for p in (str(current_dir), str(parent_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from Handwriting.config import (
    BATCH_SIZE, IMAGE_SIZE, LEARNING_RATE_BACKBONE,
    LEARNING_RATE_HEAD, NUM_EPOCHS, OUTPUT_DIR, PATIENCE,
    RANDOM_SEED, VIT_CHECKPOINT, WEIGHT_DECAY
)
from Handwriting.dataset import (
    build_manifest, create_dataloaders,
    create_patient_level_split, print_dataset_summary
)
from Handwriting.model import ViTBinaryClassifier
from evidential import EvidentialLoss, dirichlet_from_evidence

EVIDENTIAL_VIT_CHECKPOINT = OUTPUT_DIR / "best_vit_evidential.pth"
EVIDENTIAL_METRICS_PATH = OUTPUT_DIR / "vit_evidential_metrics.json"
EVIDENTIAL_PREDICTIONS_PATH = OUTPUT_DIR / "handwriting_evidential_predictions.csv"


def set_seed(seed: int = RANDOM_SEED) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_one_epoch_evidential(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: EvidentialLoss,
    epoch: int
) -> Tuple[float, float, float]:
    """Execute one training epoch with evidential loss."""
    model.train()
    total_loss = 0.0
    total_ace = 0.0
    total_kl = 0.0

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        evidence = model(images)  # (B, 2)
        loss, loss_dict = criterion(evidence, labels, epoch=epoch)
        loss.backward()
        optimizer.step()

        total_loss += loss_dict["loss"]
        total_ace += loss_dict["ace_loss"]
        total_kl += loss_dict["kl_loss"]

    n_b = max(1, len(loader))
    return total_loss / n_b, total_ace / n_b, total_kl / n_b


def evaluate_loader_evidential(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: EvidentialLoss,
    epoch: int = 0
) -> Tuple[float, float, pd.DataFrame]:
    """Evaluate performance on validation or test DataLoader."""
    model.eval()
    total_loss = 0.0
    records = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            evidence = model(images)
            loss, _ = criterion(evidence, labels, epoch=epoch)
            total_loss += loss.item()

            alpha, S, probs, u = dirichlet_from_evidence(evidence, num_classes=2)

            e_np = evidence.cpu().numpy()
            a_np = alpha.cpu().numpy()
            p_np = probs.cpu().numpy()
            u_np = u.squeeze(-1).cpu().numpy()
            y_np = labels.cpu().numpy()

            for i in range(len(y_np)):
                pred = int(p_np[i, 1] >= 0.5)
                records.append({
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
    avg_loss = total_loss / max(1, len(loader))
    acc = df["is_correct"].mean() if len(df) > 0 else 0.0

    return avg_loss, float(acc), df


def train_handwriting_evidential(
    baseline_checkpoint: Path = VIT_CHECKPOINT,
    stage_a_epochs: int = 8,
    stage_b_epochs: int = 12,
    annealing_epochs: int = 8
):
    """Execute two-stage evidential training for Vision Transformer on Handwriting."""
    print("=" * 60)
    print("HANDWRITING ViT EVIDENTIAL TRAINING (STAGE A -> STAGE B)")
    print("=" * 60)

    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest, summary = build_manifest()
    manifest = create_patient_level_split(manifest)
    train_loader, val_loader, test_loader = create_dataloaders(
        manifest, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, num_workers=0
    )

    criterion = EvidentialLoss(num_classes=2, annealing_epochs=annealing_epochs)

    # 1. Instantiate ViT model with frozen backbone
    model = ViTBinaryClassifier(freeze_backbone=True).to(device)

    # 2. Load backbone weights from existing baseline checkpoint if available
    if baseline_checkpoint.exists():
        print(f"Loading pretrained ViT backbone from: {baseline_checkpoint}")
        baseline_state = torch.load(baseline_checkpoint, map_location=device, weights_only=True)
        # Exclude old classifier weights and evidential head
        filtered_state = {k: v for k, v in baseline_state.items() if not k.startswith("classifier") and not k.startswith("evidential_head")}
        model.load_state_dict(filtered_state, strict=False)
        print(f"  --> Successfully loaded {len(filtered_state)} backbone parameter tensors.")

    # ---------------------------------------------------------
    # STAGE A: Train Evidential Head Only (Backbone Frozen)
    # ---------------------------------------------------------
    print("\n" + "-" * 50)
    print("STAGE A: Training Evidential Head (ViT Backbone Frozen)")
    print("-" * 50)

    model.freeze_backbone()
    optimizer_a = AdamW(model.evidential_head.parameters(), lr=LEARNING_RATE_HEAD, weight_decay=WEIGHT_DECAY)

    best_val_acc_a = 0.0
    best_state_a = None

    for epoch in range(1, stage_a_epochs + 1):
        t_loss, ace_l, kl_l = train_one_epoch_evidential(model, train_loader, optimizer_a, device, criterion, epoch)
        v_loss, v_acc, _ = evaluate_loader_evidential(model, val_loader, device, criterion, epoch)
        print(f"Stage A Epoch {epoch:02d}/{stage_a_epochs:02d} | Train Loss: {t_loss:.4f} (ACE: {ace_l:.4f}, KL: {kl_l:.4f}) | Val Acc: {v_acc * 100:.2f}%")

        if v_acc > best_val_acc_a:
            best_val_acc_a = v_acc
            best_state_a = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"Stage A Complete. Best Val Accuracy: {best_val_acc_a * 100:.2f}%")
    if best_state_a is not None:
        model.load_state_dict(best_state_a)

    # ---------------------------------------------------------
    # STAGE B: Fine-tune Upper ViT Layers + Evidential Head
    # ---------------------------------------------------------
    print("\n" + "-" * 50)
    print("STAGE B: Fine-tuning Upper ViT Layers (Last 2 Encoder Blocks)")
    print("-" * 50)

    model.unfreeze_backbone(unfreeze_last_n_layers=2)

    optimizer_b = AdamW([
        {"params": model.evidential_head.parameters(), "lr": LEARNING_RATE_HEAD * 0.5},
        {"params": [p for n, p in model.backbone.named_parameters() if p.requires_grad], "lr": LEARNING_RATE_BACKBONE}
    ], weight_decay=WEIGHT_DECAY)

    scheduler_b = CosineAnnealingLR(optimizer_b, T_max=stage_b_epochs)

    best_val_acc_b = best_val_acc_a

    for epoch in range(1, stage_b_epochs + 1):
        t_loss, ace_l, kl_l = train_one_epoch_evidential(model, train_loader, optimizer_b, device, criterion, epoch + stage_a_epochs)
        scheduler_b.step()
        v_loss, v_acc, _ = evaluate_loader_evidential(model, val_loader, device, criterion, epoch + stage_a_epochs)
        print(f"Stage B Epoch {epoch:02d}/{stage_b_epochs:02d} | Train Loss: {t_loss:.4f} | Val Acc: {v_acc * 100:.2f}%")

        if v_acc >= best_val_acc_b:
            best_val_acc_b = v_acc
            print(f"  --> Saved new best evidential ViT checkpoint to: {EVIDENTIAL_VIT_CHECKPOINT}")
            torch.save(model.state_dict(), EVIDENTIAL_VIT_CHECKPOINT)

    print(f"\nHandwriting Evidential Training Complete! Best Validation Accuracy: {best_val_acc_b * 100:.2f}%")
    return model


if __name__ == "__main__":
    train_handwriting_evidential()
