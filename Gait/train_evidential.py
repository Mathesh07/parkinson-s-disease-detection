"""Two-Stage Evidential Training pipeline for Gait-Based Parkinson's Disease classification."""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add workspace roots
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
for p in (str(current_dir), str(parent_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from Gait import config
from Gait.dataset import create_dataloaders
from Gait.model import GaitCNNBiLSTM
from Gait.utils import calculate_metrics, get_device, set_seed
from evidential import EvidentialLoss, dirichlet_from_evidence

EVIDENTIAL_CHECKPOINT = config.CHECKPOINT_DIR / "best_gait_evidential.pt"
EVIDENTIAL_CONFIG_PATH = config.CHECKPOINT_DIR / "evidential_model_config.json"
EVIDENTIAL_METRICS_DIR = config.OUTPUT_DIR / "metrics"
EVIDENTIAL_METRICS_DIR.mkdir(parents=True, exist_ok=True)


def train_one_epoch_evidential(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: EvidentialLoss,
    device: torch.device,
    epoch: int,
    scaler: Optional[torch.amp.GradScaler] = None,
    max_grad_norm: float = config.MAX_GRAD_NORM
) -> Tuple[float, float, float]:
    """Train one epoch with Evidential Loss (ACE + Annealed KL)."""
    model.train()
    total_loss = 0.0
    total_ace = 0.0
    total_kl = 0.0
    optimizer.zero_grad()

    use_cuda = device.type == "cuda"

    for batch in dataloader:
        features = batch["features"].to(device)  # (B, 500, 16)
        labels = batch["labels"].to(device)      # (B,)

        if use_cuda and scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                evidence = model(features)
                loss, loss_dict = criterion(evidence, labels, epoch=epoch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            evidence = model(features)
            loss, loss_dict = criterion(evidence, labels, epoch=epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss_dict["loss"]
        total_ace += loss_dict["ace_loss"]
        total_kl += loss_dict["kl_loss"]

    num_b = max(1, len(dataloader))
    return total_loss / num_b, total_ace / num_b, total_kl / num_b


def validate_evidential(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: EvidentialLoss,
    device: torch.device,
    epoch: int = 0
) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    """Evaluate evidential model on validation set with Dirichlet conversion."""
    model.eval()
    total_loss = 0.0
    records = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)

            evidence = model(features)  # (B, 2)
            loss, _ = criterion(evidence, labels, epoch=epoch)
            total_loss += loss.item()

            alpha, S, probs, u = dirichlet_from_evidence(evidence, num_classes=2)

            e_np = evidence.cpu().numpy()
            a_np = alpha.cpu().numpy()
            p_np = probs.cpu().numpy()
            u_np = u.squeeze(-1).cpu().numpy()
            y_np = labels.cpu().numpy()

            for i in range(len(y_np)):
                records.append({
                    "subject_id": batch["subject_ids"][i],
                    "recording_id": batch["recording_ids"][i],
                    "study": batch["studies"][i],
                    "true_label": int(y_np[i]),
                    "e_HC": float(e_np[i, 0]),
                    "e_PD": float(e_np[i, 1]),
                    "alpha_HC": float(a_np[i, 0]),
                    "alpha_PD": float(a_np[i, 1]),
                    "P_HC": float(p_np[i, 0]),
                    "P_PD": float(p_np[i, 1]),
                    "uncertainty": float(u_np[i]),
                    "predicted_label": int(p_np[i, 1] >= 0.5)
                })

    df = pd.DataFrame(records)
    avg_loss = total_loss / max(1, len(dataloader))

    # Subject-level aggregation
    sub_df = df.groupby(["subject_id", "true_label"])[["P_PD", "uncertainty"]].mean().reset_index()
    sub_df["predicted_label"] = (sub_df["P_PD"] >= 0.5).astype(int)

    metrics = calculate_metrics(
        sub_df["true_label"].tolist(),
        sub_df["predicted_label"].tolist(),
        sub_df["P_PD"].tolist()
    )

    return avg_loss, metrics, df


def train_gait_evidential(
    baseline_checkpoint: Path = config.BEST_MODEL_PATH,
    stage_a_epochs: int = 10,
    stage_b_epochs: int = 15,
    annealing_epochs: int = 10
):
    """Execute two-stage evidential training for Gait CNN+BiLSTM."""
    print("=" * 60)
    print("GAIT EVIDENTIAL TRAINING (STAGE A: HEAD WARMUP -> STAGE B: FINE-TUNE)")
    print("=" * 60)

    set_seed(config.RANDOM_SEED)
    device = get_device()

    # 1. Dataloaders
    train_loader, val_loader, test_loader, normalizer, train_df, val_df, test_df = create_dataloaders(
        batch_size=config.BATCH_SIZE,
        window_size=config.WINDOW_SIZE,
        step_size=config.STEP_SIZE,
        seed=config.RANDOM_SEED
    )

    criterion = EvidentialLoss(num_classes=2, annealing_epochs=annealing_epochs)

    # 2. Instantiate Model and Load Backbone from Baseline Checkpoint
    model = GaitCNNBiLSTM(
        in_channels=config.NUM_CHANNELS,
        cnn_channels=config.CNN_CHANNELS,
        lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
        lstm_num_layers=config.LSTM_NUM_LAYERS,
        embedding_dim=config.EMBEDDING_DIM,
        num_classes=2,
        dropout=config.DROPOUT
    ).to(device)

    if baseline_checkpoint.exists():
        print(f"Loading pretrained backbone weights from: {baseline_checkpoint}")
        baseline_state = torch.load(baseline_checkpoint, map_location=device, weights_only=True)
        # Filter out old 1-dim classifier weights
        filtered_state = {k: v for k, v in baseline_state.items() if not k.startswith("classifier") and not k.startswith("evidential_head")}
        model.load_state_dict(filtered_state, strict=False)
        print(f"  --> Successfully initialized {len(filtered_state)} backbone parameter tensors.")
    else:
        print("No baseline checkpoint found. Training from scratch.")

    # ---------------------------------------------------------
    # STAGE A: Train Only Evidential Head (Backbone Frozen)
    # ---------------------------------------------------------
    print("\n" + "-" * 50)
    print("STAGE A: Training Evidential Head (Backbone Frozen)")
    print("-" * 50)

    for name, param in model.named_parameters():
        if "evidential_head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer_a = torch.optim.AdamW(model.evidential_head.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_f1_a = -1.0
    best_state_a = None

    for epoch in range(1, stage_a_epochs + 1):
        t_loss, ace_l, kl_l = train_one_epoch_evidential(model, train_loader, optimizer_a, criterion, device, epoch)
        v_loss, v_metrics, _ = validate_evidential(model, val_loader, criterion, device, epoch)
        print(f"Stage A Epoch {epoch:02d}/{stage_a_epochs:02d} | Train Loss: {t_loss:.4f} (ACE: {ace_l:.4f}, KL: {kl_l:.4f}) | Val Subj Acc: {v_metrics['accuracy']:.4f} | Val F1: {v_metrics['f1_score']:.4f}")

        if v_metrics["f1_score"] > best_val_f1_a:
            best_val_f1_a = v_metrics["f1_score"]
            best_state_a = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"Stage A Complete. Best Val F1: {best_val_f1_a:.4f}")
    if best_state_a is not None:
        model.load_state_dict(best_state_a)

    # ---------------------------------------------------------
    # STAGE B: Fine-tune BiLSTM + Embedding + Evidential Head
    # ---------------------------------------------------------
    print("\n" + "-" * 50)
    print("STAGE B: Fine-tuning Upper Layers + Evidential Head")
    print("-" * 50)

    # Unfreeze BiLSTM, embedding layer, and head (keep initial CNN low-level filters stable)
    for name, param in model.named_parameters():
        if "cnn" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    optimizer_b = torch.optim.AdamW([
        {"params": model.lstm.parameters(), "lr": 1e-4},
        {"params": model.embedding_layer.parameters(), "lr": 2e-4},
        {"params": model.evidential_head.parameters(), "lr": 5e-4},
    ], weight_decay=1e-4)

    scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_b, T_max=stage_b_epochs, eta_min=1e-6)

    best_val_f1_b = best_val_f1_a
    best_epoch = 0

    for epoch in range(1, stage_b_epochs + 1):
        t_loss, ace_l, kl_l = train_one_epoch_evidential(model, train_loader, optimizer_b, criterion, device, epoch + stage_a_epochs)
        scheduler_b.step()
        v_loss, v_metrics, _ = validate_evidential(model, val_loader, criterion, device, epoch + stage_a_epochs)
        print(f"Stage B Epoch {epoch:02d}/{stage_b_epochs:02d} | Train Loss: {t_loss:.4f} | Val Subj Acc: {v_metrics['accuracy']:.4f} | Val F1: {v_metrics['f1_score']:.4f} | Val AUC: {v_metrics['roc_auc']:.4f}")

        if v_metrics["f1_score"] >= best_val_f1_b:
            best_val_f1_b = v_metrics["f1_score"]
            best_epoch = epoch
            print(f"  --> Saved new best evidential checkpoint to: {EVIDENTIAL_CHECKPOINT}")
            torch.save(model.state_dict(), EVIDENTIAL_CHECKPOINT)

    print(f"\nEvidential Training Complete! Best Validation F1: {best_val_f1_b:.4f}")
    return model


if __name__ == "__main__":
    train_gait_evidential()
