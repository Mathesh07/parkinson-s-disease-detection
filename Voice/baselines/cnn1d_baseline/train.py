"""Training script for Voice 1D CNN Baseline on Raw Waveforms."""

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Voice.baselines.cnn1d_baseline import config
from Voice.baselines.cnn1d_baseline.dataset import create_voice_1d_dataloaders
from Voice.baselines.cnn1d_baseline.model import Voice1DCNN


def set_seed(seed: int = config.RANDOM_SEED):
    """Ensure deterministic execution across PyTorch and NumPy."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train for a single epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        features = batch["features"].to(device)  # (B, 1, 48000)
        labels = batch["labels"].to(device)       # (B,)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()

        if config.MAX_GRAD_NORM > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)

        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    epoch_loss = total_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

    epoch_loss = total_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def plot_training_curves(history: Dict[str, List[float]], plots_dir: Path):
    """Plot and save loss and accuracy progression curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    plt.plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2)
    plt.title("Voice 1D CNN: Training & Validation Loss", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Cross-Entropy Loss", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(config.LOSS_CURVE_PATH, dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy", color="#2ca02c", lw=2)
    plt.plot(epochs, history["val_acc"], label="Val Accuracy", color="#d62728", lw=2)
    plt.title("Voice 1D CNN: Training & Validation Accuracy", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(config.ACCURACY_CURVE_PATH, dpi=300)
    plt.close()


def train_voice_1d_cnn() -> Dict:
    """Execute training pipeline for Voice 1D CNN."""
    print("=" * 60)
    print("VOICE 1D CNN BASELINE TRAINING (RAW WAVEFORMS)")
    print("=" * 60)

    set_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, normalizer, train_df, val_df, test_df = create_voice_1d_dataloaders(
        batch_size=config.BATCH_SIZE, seed=config.RANDOM_SEED
    )

    print(f"\nSubjects: Train={train_df['subject_id'].nunique()}, Val={val_df['subject_id'].nunique()}, Test={test_df['subject_id'].nunique()}")
    print(f"Test Subjects ({test_df['subject_id'].nunique()}): {sorted(test_df['subject_id'].unique())}\n")

    model = Voice1DCNN().to(device)
    with open(config.MODEL_CONFIG_PATH, "w") as f:
        json.dump(model.get_config(), f, indent=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0

    print("Beginning training loop...")
    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{config.EPOCHS:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"  --> Saved new best checkpoint to: {config.BEST_MODEL_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs (patience={config.PATIENCE}).")
                break

    with open(config.OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    plot_training_curves(history, config.PLOTS_DIR)
    print(f"Voice 1D CNN training completed. Best Val Loss: {best_val_loss:.4f}")
    return history


if __name__ == "__main__":
    train_voice_1d_cnn()
