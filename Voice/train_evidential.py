"""Two-stage evidential training pipeline for Voice Wav2Vec2."""

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

# Add workspace roots
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
for p in (str(current_dir), str(parent_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from Voice import config
from Voice.dataset import (
    VoiceChunkDataset,
    build_metadata,
    collate_audio_batch,
    create_subject_splits,
)
from Voice.model import Wav2Vec2ForParkinsons
from Voice.utils import calculate_metrics, get_device, set_seed
from evidential import EvidentialLoss, dirichlet_from_evidence

EVIDENTIAL_VOICE_CHECKPOINT = config.RESULTS_DIR / "best_wav2vec2_evidential"
EVIDENTIAL_VOICE_CHECKPOINT.mkdir(parents=True, exist_ok=True)
EVIDENTIAL_MODEL_FILE = EVIDENTIAL_VOICE_CHECKPOINT / "best_model.pt"
EVIDENTIAL_METRICS_PATH = config.RESULTS_DIR / "voice_evidential_metrics.json"


def train_one_epoch_evidential(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: EvidentialLoss,
    device: torch.device,
    epoch: int,
    accumulation_steps: int = config.GRADIENT_ACCUMULATION_STEPS,
    scaler: Optional[torch.amp.GradScaler] = None,
    max_grad_norm: float = config.MAX_GRAD_NORM
) -> Tuple[float, float, float]:
    """Train one epoch with evidential loss and gradient accumulation."""
    model.train()
    total_loss = 0.0
    total_ace = 0.0
    total_kl = 0.0
    optimizer.zero_grad()

    use_cuda = device.type == "cuda"

    for step, batch in enumerate(dataloader):
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if use_cuda and scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                evidence = model(input_values, attention_mask=attention_mask)
                loss, loss_dict = criterion(evidence, labels, epoch=epoch)
                scaled_loss = loss / accumulation_steps
            scaler.scale(scaled_loss).backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            evidence = model(input_values, attention_mask=attention_mask)
            loss, loss_dict = criterion(evidence, labels, epoch=epoch)
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss_dict["loss"]
        total_ace += loss_dict["ace_loss"]
        total_kl += loss_dict["kl_loss"]

    n_b = max(1, len(dataloader))
    return total_loss / n_b, total_ace / n_b, total_kl / n_b


def validate_evidential(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: EvidentialLoss,
    device: torch.device,
    epoch: int = 0
) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    """Validate evidential Wav2Vec2 model."""
    model.eval()
    total_loss = 0.0
    records = []

    with torch.no_grad():
        for batch in dataloader:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            evidence = model(input_values, attention_mask=attention_mask)
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
                    "task": batch["tasks"][i],
                    "true_label": int(y_np[i]),
                    "e_HC": float(e_np[i, 0]),
                    "e_PD": float(e_np[i, 1]),
                    "alpha_HC": float(a_np[i, 0]),
                    "alpha_PD": float(a_np[i, 1]),
                    "P_HC": float(p_np[i, 0]),
                    "P_PD": float(p_np[i, 1]),
                    "uncertainty": float(u_np[i])
                })

    df = pd.DataFrame(records)
    avg_loss = total_loss / max(1, len(dataloader))

    # Recording-level -> Subject-level aggregation
    rec_grouped = df.groupby(["recording_id", "subject_id", "true_label"])[["P_PD", "uncertainty"]].mean().reset_index()
    sub_grouped = rec_grouped.groupby(["subject_id", "true_label"])[["P_PD", "uncertainty"]].mean().reset_index()
    sub_grouped["predicted_label"] = (sub_grouped["P_PD"] >= 0.5).astype(int)

    metrics = calculate_metrics(
        sub_grouped["true_label"].tolist(),
        sub_grouped["predicted_label"].tolist(),
        sub_grouped["P_PD"].tolist()
    )

    return avg_loss, metrics, df


def train_voice_evidential(
    stage_a_epochs: int = 5,
    stage_b_epochs: int = 6,
    annealing_epochs: int = 5
):
    """Execute two-stage evidential training for Voice Wav2Vec2."""
    print("=" * 60)
    print("VOICE WAV2VEC2 EVIDENTIAL TRAINING (STAGE A -> STAGE B)")
    print("=" * 60)

    set_seed(config.RANDOM_SEED)
    device = get_device()

    metadata_df = build_metadata()
    train_df, val_df, test_df = create_subject_splits(metadata_df)

    processor = Wav2Vec2Processor.from_pretrained(config.MODEL_NAME)
    train_dataset = VoiceChunkDataset(train_df, processor=processor)
    val_dataset = VoiceChunkDataset(val_df, processor=processor)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_audio_batch)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_audio_batch)

    criterion = EvidentialLoss(num_classes=2, annealing_epochs=annealing_epochs)

    # 1. Instantiate Wav2Vec2 model
    model = Wav2Vec2ForParkinsons(
        model_name=config.MODEL_NAME,
        num_classes=2,
        freeze_feature_encoder=True
    ).to(device)

    # 2. Check if baseline weights exist to initialize backbone
    baseline_path = config.CHECKPOINT_DIR / "best_model.pt"
    if baseline_path.exists():
        print(f"Loading pretrained Wav2Vec2 backbone from: {baseline_path}")
        baseline_state = torch.load(baseline_path, map_location=device, weights_only=True)
        filtered_state = {k: v for k, v in baseline_state.items() if not k.startswith("classifier") and not k.startswith("evidential_head")}
        model.load_state_dict(filtered_state, strict=False)
        print(f"  --> Successfully loaded {len(filtered_state)} backbone parameter tensors.")

    # ---------------------------------------------------------
    # STAGE A: Train Evidential Head (Wav2Vec2 Backbone Frozen)
    # ---------------------------------------------------------
    print("\n" + "-" * 50)
    print("STAGE A: Training Evidential Head (Wav2Vec2 Backbone Frozen)")
    print("-" * 50)

    for name, param in model.named_parameters():
        if "evidential_head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer_a = torch.optim.AdamW(model.evidential_head.parameters(), lr=1e-3, weight_decay=config.WEIGHT_DECAY)

    best_val_f1_a = -1.0
    best_state_a = None

    for epoch in range(1, stage_a_epochs + 1):
        t_loss, ace_l, kl_l = train_one_epoch_evidential(model, train_loader, optimizer_a, criterion, device, epoch)
        v_loss, v_metrics, _ = validate_evidential(model, val_loader, criterion, device, epoch)
        print(f"Stage A Epoch {epoch:02d}/{stage_a_epochs:02d} | Train Loss: {t_loss:.4f} (ACE: {ace_l:.4f}, KL: {kl_l:.4f}) | Val Subj F1: {v_metrics['f1_score']:.4f} | Val Subj Acc: {v_metrics['accuracy']:.4f}")

        if v_metrics["f1_score"] > best_val_f1_a:
            best_val_f1_a = v_metrics["f1_score"]
            best_state_a = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"Stage A Complete. Best Val F1: {best_val_f1_a:.4f}")
    if best_state_a is not None:
        model.load_state_dict(best_state_a)

    # ---------------------------------------------------------
    # STAGE B: Fine-tune Upper Transformer Layers + Evidential Head
    # ---------------------------------------------------------
    print("\n" + "-" * 50)
    print("STAGE B: Fine-tuning Upper Wav2Vec2 Layers + Evidential Head")
    print("-" * 50)

    # Unfreeze only the last 2 transformer layers
    for name, param in model.wav2vec2.encoder.layers[-2:].named_parameters():
        param.requires_grad = True

    optimizer_b = torch.optim.AdamW([
        {"params": model.evidential_head.parameters(), "lr": 5e-4},
        {"params": filter(lambda p: p.requires_grad, model.wav2vec2.parameters()), "lr": 1e-5}
    ], weight_decay=config.WEIGHT_DECAY)

    best_val_f1_b = best_val_f1_a

    for epoch in range(1, stage_b_epochs + 1):
        t_loss, ace_l, kl_l = train_one_epoch_evidential(model, train_loader, optimizer_b, criterion, device, epoch + stage_a_epochs)
        v_loss, v_metrics, _ = validate_evidential(model, val_loader, criterion, device, epoch + stage_a_epochs)
        print(f"Stage B Epoch {epoch:02d}/{stage_b_epochs:02d} | Train Loss: {t_loss:.4f} | Val Subj F1: {v_metrics['f1_score']:.4f} | Val Subj Acc: {v_metrics['accuracy']:.4f}")

        if v_metrics["f1_score"] >= best_val_f1_b:
            best_val_f1_b = v_metrics["f1_score"]
            print(f"  --> Saved new best evidential checkpoint to: {EVIDENTIAL_MODEL_FILE}")
            torch.save(model.state_dict(), EVIDENTIAL_MODEL_FILE)

    print(f"\nVoice Evidential Training Complete! Best Validation F1: {best_val_f1_b:.4f}")
    return model


if __name__ == "__main__":
    train_voice_evidential()
