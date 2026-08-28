"""Training pipeline for Gait-Based Parkinson's Disease classification using 1D CNN + BiLSTM."""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Support running as a standalone script or as a module
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
for p in (str(current_dir), str(parent_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from Gait import config
    from Gait.dataset import create_dataloaders
    from Gait.model import GaitCNNBiLSTM
    from Gait.utils import (
        aggregate_predictions,
        calculate_metrics,
        get_device,
        plot_training_curves,
        set_seed,
    )
except ImportError:
    import config
    from dataset import create_dataloaders
    from model import GaitCNNBiLSTM
    from utils import (
        aggregate_predictions,
        calculate_metrics,
        get_device,
        plot_training_curves,
        set_seed,
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    max_grad_norm: float = config.MAX_GRAD_NORM
) -> float:
    """Run one epoch of training with mixed precision and gradient clipping."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    use_cuda = device.type == "cuda"

    for batch in dataloader:
        features = batch["features"].to(device)  # (B, 500, 16)
        labels = batch["labels"].to(device)      # (B,)

        if use_cuda and scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(features)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(dataloader) if len(dataloader) > 0 else 0.0


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Evaluate model on validation data and compute metrics at Window, Recording, and Subject levels.
    """
    model.eval()
    total_loss = 0.0
    window_records = []

    use_cuda = device.type == "cuda"

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)

            if use_cuda:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(features)
                    loss = criterion(logits, labels)
            else:
                logits = model(features)
                loss = criterion(logits, labels)

            total_loss += loss.item()

            probs = torch.sigmoid(logits.float()).cpu().numpy()
            targets = labels.cpu().numpy()

            for i in range(len(targets)):
                window_records.append({
                    "subject_id": batch["subject_ids"][i],
                    "recording_id": batch["recording_ids"][i],
                    "study": batch["studies"][i],
                    "true_label": int(targets[i]),
                    "pd_probability": float(probs[i]),
                    "predicted_label": int(probs[i] >= 0.5)
                })

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    window_df = pd.DataFrame(window_records)

    # 1. Window-Level Metrics
    win_metrics = calculate_metrics(
        window_df["true_label"].tolist(),
        window_df["predicted_label"].tolist(),
        window_df["pd_probability"].tolist()
    )

    # 2. Hierarchical Aggregation to Recording and Subject Levels
    rec_df, sub_df = aggregate_predictions(window_df)

    rec_metrics = calculate_metrics(
        rec_df["true_label"].tolist(),
        rec_df["predicted_label"].tolist(),
        rec_df["pd_probability"].tolist()
    )

    sub_metrics = calculate_metrics(
        sub_df["true_label"].tolist(),
        sub_df["predicted_label"].tolist(),
        sub_df["pd_probability"].tolist()
    )

    return avg_loss, win_metrics, rec_metrics, sub_metrics


def save_checkpoint(
    model: nn.Module,
    sub_metrics: Dict[str, float],
    epoch: int,
    save_path: Path = config.BEST_MODEL_PATH,
    config_path: Path = config.MODEL_CONFIG_PATH
) -> None:
    """Save trained model weights and configuration metadata."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)

    meta = {
        "model_architecture": "GaitCNNBiLSTM (1D CNN + BiLSTM)",
        "sampling_rate": config.SAMPLING_RATE,
        "window_duration_sec": config.WINDOW_DURATION,
        "window_size": config.WINDOW_SIZE,
        "overlap_ratio": config.OVERLAP_RATIO,
        "num_channels": config.NUM_CHANNELS,
        "cnn_channels": config.CNN_CHANNELS,
        "lstm_hidden_size": config.LSTM_HIDDEN_SIZE,
        "lstm_num_layers": config.LSTM_NUM_LAYERS,
        "embedding_dim": config.EMBEDDING_DIM,
        "best_epoch": epoch,
        "val_subject_accuracy": sub_metrics["accuracy"],
        "val_subject_precision": sub_metrics["precision"],
        "val_subject_recall": sub_metrics["recall_sensitivity"],
        "val_subject_specificity": sub_metrics["specificity"],
        "val_subject_f1": sub_metrics["f1_score"],
        "val_subject_roc_auc": sub_metrics["roc_auc"],
        "label_map": config.LABEL_MAP,
        "id2label": config.ID2LABEL
    }
    with open(config_path, "w") as f:
        json.dump(meta, f, indent=4)


def train():
    """Main training execution function."""
    print("=" * 60)
    print("GAIT PARKINSON'S CLASSIFICATION (1D CNN + BiLSTM)")
    print("=" * 60)

    set_seed(config.RANDOM_SEED)
    device = get_device()

    # 1. Build DataLoaders & Normalizer
    train_loader, val_loader, test_loader, normalizer, train_df, val_df, test_df = create_dataloaders(
        batch_size=config.BATCH_SIZE,
        window_size=config.WINDOW_SIZE,
        step_size=config.STEP_SIZE,
        seed=config.RANDOM_SEED
    )

    # 2. Print Detailed Subject, Recording, and Window Counts
    train_windows = len(train_loader.dataset)
    val_windows = len(val_loader.dataset)
    test_windows = len(test_loader.dataset)

    print("\n" + "-" * 50)
    print("DATASET SPLIT SUMMARY (ZERO-LEAKAGE)")
    print("-" * 50)
    print(f"Train Subjects:      {train_df['subject_id'].nunique()} (HC: {train_df[train_df['label']==0]['subject_id'].nunique()}, PD: {train_df[train_df['label']==1]['subject_id'].nunique()})")
    print(f"Validation Subjects: {val_df['subject_id'].nunique()} (HC: {val_df[val_df['label']==0]['subject_id'].nunique()}, PD: {val_df[val_df['label']==1]['subject_id'].nunique()})")
    print(f"Test Subjects:       {test_df['subject_id'].nunique()} (HC: {test_df[test_df['label']==0]['subject_id'].nunique()}, PD: {test_df[test_df['label']==1]['subject_id'].nunique()})")
    print()
    print(f"Train Recordings:      {len(train_df)}")
    print(f"Validation Recordings: {len(val_df)}")
    print(f"Test Recordings:       {len(test_df)}")
    print()
    print(f"Train Windows (5s):      {train_windows}")
    print(f"Validation Windows (5s): {val_windows}")
    print(f"Test Windows (5s):       {test_windows}")
    print("-" * 50 + "\n")

    # 3. Class Imbalance Handling (pos_weight in BCEWithLogitsLoss)
    train_labels = [s["label"] for s in train_loader.dataset.samples]
    num_hc = train_labels.count(0)
    num_pd = train_labels.count(1)
    pos_weight_val = num_hc / num_pd if num_pd > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    print(f"Training Class Distribution -> HC (0): {num_hc} windows, PD (1): {num_pd} windows")
    print(f"Class Imbalance pos_weight: {pos_weight_val:.4f}\n")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 4. Initialize Model, Optimizer, Scheduler
    model = GaitCNNBiLSTM(
        in_channels=config.NUM_CHANNELS,
        cnn_channels=config.CNN_CHANNELS,
        lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
        lstm_num_layers=config.LSTM_NUM_LAYERS,
        embedding_dim=config.EMBEDDING_DIM,
        dropout=config.DROPOUT
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS,
        eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # 5. Training Loop
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_subject_acc": [],
        "val_subject_f1": [],
        "val_subject_auc": []
    }

    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    print("Starting Training Loop...\n")

    for epoch in range(1, config.EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            max_grad_norm=config.MAX_GRAD_NORM
        )
        scheduler.step()

        val_loss, val_win_m, val_rec_m, val_sub_m = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )
        elapsed = time.time() - t0

        val_acc = val_sub_m["accuracy"]
        val_f1 = val_sub_m["f1_score"]
        val_auc = val_sub_m["roc_auc"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_subject_acc"].append(val_acc)
        history["val_subject_f1"].append(val_f1)
        history["val_subject_auc"].append(val_auc)

        print(
            f"Epoch {epoch:02d}/{config.EPOCHS:02d} [{elapsed:.1f}s] - "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Subj Acc: {val_acc:.4f} | "
            f"Val Subj F1: {val_f1:.4f} | "
            f"Val Subj AUC: {val_auc:.4f}"
        )

        # Checkpointing based on Validation Subject F1
        if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and val_loss < history["val_loss"][best_epoch - 1]):
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(model, val_sub_m, epoch)
            print(f"  --> Saved new best checkpoint at Epoch {epoch} (Val Subj F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"\nEarly stopping triggered at Epoch {epoch} (no improvement for {config.PATIENCE} epochs).")
                break

    print(f"\nTraining Complete! Best Validation Subject F1: {best_val_f1:.4f} at Epoch {best_epoch}")

    # 6. Save History & Plots
    with open(config.OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    plot_training_curves(history, config.LOSS_CURVE_PATH, config.ACCURACY_CURVE_PATH)
    print(f"Saved loss curves to: {config.LOSS_CURVE_PATH}")
    print(f"Saved accuracy curves to: {config.ACCURACY_CURVE_PATH}")
    print(f"Best model checkpoint saved to: {config.BEST_MODEL_PATH}")


if __name__ == "__main__":
    train()
