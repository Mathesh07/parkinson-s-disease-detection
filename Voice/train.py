"""Training script for Voice-Based Parkinson's Detection using Wav2Vec2."""

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
from transformers import Wav2Vec2Processor

# Support running as a standalone script or as a module
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
for p in (str(current_dir), str(parent_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from Voice import config
    from Voice.dataset import (
        VoiceChunkDataset,
        build_metadata,
        collate_audio_batch,
        create_subject_splits,
    )
    from Voice.model import Wav2Vec2ForParkinsons
    from Voice.utils import (
        aggregate_predictions,
        calculate_metrics,
        get_device,
        plot_training_curves,
        set_seed,
    )
except ImportError:
    import config
    from dataset import (
        VoiceChunkDataset,
        build_metadata,
        collate_audio_batch,
        create_subject_splits,
    )
    from model import Wav2Vec2ForParkinsons
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
    accumulation_steps: int = config.GRADIENT_ACCUMULATION_STEPS,
    max_grad_norm: float = config.MAX_GRAD_NORM
) -> float:
    """Run one epoch of training with mixed precision and gradient accumulation."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    use_cuda = device.type == "cuda"

    for step, batch in enumerate(dataloader):
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if use_cuda and scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(input_values, attention_mask=attention_mask)
                loss = criterion(logits, labels)
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            logits = model(input_values, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss = loss / accumulation_steps
            loss.backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """Evaluate model on validation split and aggregate chunk to subject level."""
    model.eval()
    total_loss = 0.0
    records = []

    use_cuda = device.type == "cuda"

    with torch.no_grad():
        for batch in dataloader:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if use_cuda:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(input_values, attention_mask=attention_mask)
                    loss = criterion(logits, labels)
            else:
                logits = model(input_values, attention_mask=attention_mask)
                loss = criterion(logits, labels)

            total_loss += loss.item()

            probs = torch.softmax(logits.float(), dim=-1)
            pd_probs = probs[:, 1].cpu().numpy()

            for i in range(len(labels)):
                records.append({
                    "recording_id": batch["recording_ids"][i],
                    "subject_id": batch["subject_ids"][i],
                    "task": batch["tasks"][i],
                    "true_label": int(labels[i].cpu().item()),
                    "pd_probability": float(pd_probs[i])
                })

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    chunk_df = pd.DataFrame(records)

    # Aggregate predictions to Recording and Subject levels
    rec_df, sub_df = aggregate_predictions(chunk_df)

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

    return avg_loss, rec_metrics, sub_metrics


def save_checkpoint(
    model: Wav2Vec2ForParkinsons,
    processor: Wav2Vec2Processor,
    val_sub_metrics: Dict[str, float],
    epoch: int,
    save_dir: Path = config.CHECKPOINT_DIR
) -> None:
    """Save best model weights, processor, and metadata."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), save_dir / "best_model.pt")

    # Save processor
    processor.save_pretrained(save_dir)

    # Save metadata & config
    meta = {
        "model_name": config.MODEL_NAME,
        "sample_rate": config.SAMPLE_RATE,
        "chunk_seconds": config.CHUNK_SECONDS,
        "overlap_seconds": config.OVERLAP_SECONDS,
        "best_epoch": epoch,
        "val_subject_f1": val_sub_metrics["f1_score"],
        "val_subject_accuracy": val_sub_metrics["accuracy"],
        "val_subject_roc_auc": val_sub_metrics["roc_auc"],
        "label_map": config.LABEL_MAP,
        "id2label": config.ID2LABEL
    }
    with open(save_dir / "model_config.json", "w") as f:
        json.dump(meta, f, indent=4)


def train():
    """Main training execution function."""
    print("=" * 50)
    print("WAV2VEC2 PARKINSON'S VOICE CLASSIFICATION")
    print("=" * 50)

    set_seed(config.RANDOM_SEED)
    device = get_device()

    # 1. Build metadata and subject splits
    metadata_df = build_metadata()
    train_df, val_df, test_df = create_subject_splits(metadata_df)

    print(f"\nDataset: MDVR-KCL")
    print(f"Model Backbone: {config.MODEL_NAME}\n")
    print("Subjects:")
    print(f"  Train:      {train_df['subject_id'].nunique()} (HC: {train_df[train_df['label_name']=='HC']['subject_id'].nunique()}, PD: {train_df[train_df['label_name']=='PD']['subject_id'].nunique()})")
    print(f"  Validation: {val_df['subject_id'].nunique()} (HC: {val_df[val_df['label_name']=='HC']['subject_id'].nunique()}, PD: {val_df[val_df['label_name']=='PD']['subject_id'].nunique()})")
    print(f"  Test:       {test_df['subject_id'].nunique()} (HC: {test_df[test_df['label_name']=='HC']['subject_id'].nunique()}, PD: {test_df[test_df['label_name']=='PD']['subject_id'].nunique()})\n")
    print("Recordings:")
    print(f"  Train:      {len(train_df)} (HC: {len(train_df[train_df['label_name']=='HC'])}, PD: {len(train_df[train_df['label_name']=='PD'])})")
    print(f"  Validation: {len(val_df)} (HC: {len(val_df[val_df['label_name']=='HC'])}, PD: {len(val_df[val_df['label_name']=='PD'])})")
    print(f"  Test:       {len(test_df)} (HC: {len(test_df[test_df['label_name']=='HC'])}, PD: {len(test_df[test_df['label_name']=='PD'])})\n")

    # 2. Load Processor and Datasets
    print("Loading Wav2Vec2 Processor...")
    processor = Wav2Vec2Processor.from_pretrained(config.MODEL_NAME)

    print("Building Audio Chunk Datasets...")
    train_dataset = VoiceChunkDataset(train_df, processor=processor)
    val_dataset = VoiceChunkDataset(val_df, processor=processor)

    print(f"Audio Chunks: Train={len(train_dataset)}, Validation={len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_audio_batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_audio_batch
    )

    # 3. Initialize Model
    print("Initializing Wav2Vec2 Model for Parkinson's Classification...")
    model = Wav2Vec2ForParkinsons(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        freeze_feature_encoder=config.FREEZE_FEATURE_ENCODER
    ).to(device)

    # Balanced class weights for loss
    train_labels = [item["label"] for item in train_dataset.chunk_items]
    num_hc = train_labels.count(0)
    num_pd = train_labels.count(1)
    total_samples = len(train_labels)
    class_weights = torch.tensor([
        total_samples / (2.0 * num_hc) if num_hc > 0 else 1.0,
        total_samples / (2.0 * num_pd) if num_pd > 0 else 1.0
    ], dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS,
        eta_min=1e-7
    )

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # 4. Training Loop
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
        "val_acc": [],
        "val_auc": []
    }

    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    print("\nStarting training...\n")

    for epoch in range(1, config.EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
            max_grad_norm=config.MAX_GRAD_NORM
        )
        scheduler.step()

        val_loss, val_rec_m, val_sub_m = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )
        elapsed = time.time() - t0

        val_f1 = val_sub_m["f1_score"]
        val_acc = val_sub_m["accuracy"]
        val_auc = val_sub_m["roc_auc"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        print(
            f"Epoch {epoch:02d}/{config.EPOCHS:02d} [{elapsed:.1f}s] - "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Subject Acc: {val_acc:.4f} | "
            f"Val Subject F1: {val_f1:.4f} | "
            f"Val AUC: {val_auc:.4f}"
        )

        # Checkpointing based on Subject-Level Validation F1
        if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and val_loss < history["val_loss"][best_epoch - 1]):
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(model, processor, val_sub_m, epoch)
            print(f"  --> Saved new best checkpoint at Epoch {epoch} (Val F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {config.PATIENCE} epochs).")
                break

    print(f"\nTraining completed! Best Validation Subject F1: {best_val_f1:.4f} at Epoch {best_epoch}")

    # 5. Save History & Plots
    with open(config.TRAINING_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=4)

    plot_training_curves(history, config.TRAINING_CURVES_PATH)
    print(f"Saved training curves to: {config.TRAINING_CURVES_PATH}")
    print(f"Best model checkpoint saved to: {config.CHECKPOINT_DIR}")


if __name__ == "__main__":
    train()
