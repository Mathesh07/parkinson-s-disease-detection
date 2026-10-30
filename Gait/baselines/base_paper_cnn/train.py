"""Training script for Gait Base-Paper-Style 2D CNN Baseline.

Implements reproducible training with early stopping, model checkpointing,
and training visualization, strictly isolated within the baseline directory.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

try:
    from . import config
    from .dataset import create_dataloaders
    from .model import GaitBasePaper2DCNN
except ImportError:
    import config
    from dataset import create_dataloaders
    from model import GaitBasePaper2DCNN


def set_seed(seed: int = config.RANDOM_SEED):
    """Set global seeds for deterministic reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Detect available compute device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU for training.")
    return device


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float = config.MAX_GRAD_NORM
) -> Tuple[float, float]:
    """Execute one training epoch over batches."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in dataloader:
        features = batch["features"].to(device)       # (B, 16, 33, 32)
        labels = batch["labels"].to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += loss.item() * len(labels)
        probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy().flatten().astype(int))

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    return avg_loss, acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float, float]:
    """Execute validation over held-out validation batches."""
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device).unsqueeze(1)

            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_targets.extend(labels.cpu().numpy().flatten().astype(int))

    avg_loss = total_loss / len(dataloader.dataset)
    targets_np = np.array(all_targets)
    probs_np = np.array(all_probs)
    preds_np = (probs_np >= 0.5).astype(int)

    acc = accuracy_score(targets_np, preds_np)
    f1 = f1_score(targets_np, preds_np, zero_division=0)
    try:
        auc = roc_auc_score(targets_np, probs_np)
    except Exception:
        auc = 0.5

    return avg_loss, acc, f1, auc


def plot_training_history(history: Dict, save_loss_path: Path, save_acc_path: Path):
    """Plot and save training and validation metrics curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    # 1. Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], "b-o", label="Train Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], "r--s", label="Val Loss", linewidth=2)
    plt.title("Gait Base-Paper 2D CNN: Loss Curve", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("BCE Loss", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_loss_path, dpi=300)
    plt.close()

    # 2. Accuracy & F1 Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], "b-o", label="Train Accuracy", linewidth=2)
    plt.plot(epochs, history["val_acc"], "g--^", label="Val Accuracy", linewidth=2)
    plt.plot(epochs, history["val_f1"], "m-s", label="Val F1-Score", linewidth=2)
    plt.title("Gait Base-Paper 2D CNN: Accuracy & F1 Curve", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Metric Score", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_acc_path, dpi=300)
    plt.close()


def train():
    """Main training routine for the 2D CNN Gait baseline."""
    print("=" * 60)
    print("GAIT BASE-PAPER 2D CNN BASELINE TRAINING")
    print(f"Seed: {config.RANDOM_SEED} | Architecture: 3-Stage 2D CNN on 16-channel STFT")
    print("=" * 60)

    set_seed(config.RANDOM_SEED)
    device = get_device()

    # 1. Build Datasets & Splits
    train_loader, val_loader, test_loader, normalizer, train_df, val_df, test_df = create_dataloaders(
        batch_size=config.BATCH_SIZE,
        seed=config.RANDOM_SEED
    )

    print(f"\nSubject Distribution:")
    print(f"  Train: {train_df['subject_id'].nunique()} subjects ({len(train_loader.dataset)} windows)")
    print(f"  Val:   {val_df['subject_id'].nunique()} subjects ({len(val_loader.dataset)} windows)")
    print(f"  Test:  {test_df['subject_id'].nunique()} subjects ({len(test_loader.dataset)} windows)\n")

    # 2. Initialize Model
    model = GaitBasePaper2DCNN(
        in_channels=config.NUM_CHANNELS,
        conv_channels=config.CONV_CHANNELS,
        dense_units=config.DENSE_UNITS,
        dropout_rate=config.DROPOUT_RATE
    ).to(device)

    # Save model config
    with open(config.MODEL_CONFIG_PATH, "w") as f:
        json.dump(model.get_config(), f, indent=4)
    print(f"Model config saved to: {config.MODEL_CONFIG_PATH}")

    # Compute positive class weight for balanced BCE
    train_labels = [sample["label"] for sample in train_loader.dataset.samples]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-6)], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_auc": []
    }

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    print("Beginning training loop...\n")
    for epoch in range(1, config.EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            max_grad_norm=config.MAX_GRAD_NORM
        )

        val_loss, val_acc, val_f1, val_auc = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["val_f1"].append(float(val_f1))
        history["val_auc"].append(float(val_auc))

        print(
            f"Epoch {epoch:02d}/{config.EPOCHS:02d} [{elapsed:.1f}s] - "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} AUC: {val_auc:.4f}"
        )

        # Early Stopping check
        if val_loss < (best_val_loss - config.MIN_DELTA):
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"  --> Checkpoint saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs (Best Epoch: {best_epoch}).")
                break

    # Save training history JSON
    history_path = config.RESULTS_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    # Plot training curves
    plot_training_history(history, config.LOSS_CURVE_PATH, config.ACCURACY_CURVE_PATH)
    print(f"\nTraining complete. Best checkpoint saved at: {config.BEST_MODEL_PATH}")
    print(f"Curves saved to:\n  - {config.LOSS_CURVE_PATH}\n  - {config.ACCURACY_CURVE_PATH}")


if __name__ == "__main__":
    train()
