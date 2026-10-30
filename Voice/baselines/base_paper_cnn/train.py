"""Training script for Voice Base-Paper 2D CNN Baseline.

Implements end-to-end training for:
    1. Primary Continuous STFT Spectrogram 2D CNN
    2. Speech Phonation Onset Transition 2D CNN (Vásquez-Correa et al.)
    3. Speech Phonation Offset Transition 2D CNN (Vásquez-Correa et al.)
"""

import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

try:
    from . import config
    from .dataset import (
        VoiceSTFTChunkDataset,
        build_voice_metadata,
        collate_voice_stft_batch,
        create_subject_splits,
    )
    from .model import VoiceBasePaper2DCNN
    from .preprocessing import (
        VoiceSTFTNormalizer,
        detect_speech_transitions,
        load_and_resample_audio,
    )
except ImportError:
    import config
    from dataset import (
        VoiceSTFTChunkDataset,
        build_voice_metadata,
        collate_voice_stft_batch,
        create_subject_splits,
    )
    from model import VoiceBasePaper2DCNN
    from preprocessing import (
        VoiceSTFTNormalizer,
        detect_speech_transitions,
        load_and_resample_audio,
    )


def ensure_dataset_extracted():
    """Ensure MDVR-KCL dataset zip is unpacked into DATA_DIR."""
    if not config.DATA_DIR.exists() or len(list(config.DATA_DIR.glob("**/*.wav"))) < 10:
        zip_path = config.VOICE_DIR / "26_29_09_2017_KCL.zip"
        if zip_path.exists():
            print(f"Unpacking {zip_path.name} into {config.VOICE_DIR}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(config.VOICE_DIR)
            print("Unpacking complete.")
        else:
            raise FileNotFoundError(f"Voice dataset zip not found at: {zip_path}")


def set_seed(seed: int = config.RANDOM_SEED):
    """Set random seed for deterministic execution."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_classifier_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    save_path: Path,
    epochs: int = config.EPOCHS,
    lr: float = config.LEARNING_RATE,
    patience: int = config.PATIENCE
) -> Dict:
    """Train a 2D CNN model with early stopping."""
    # Balanced class weights
    labels_all = []
    for b in train_loader:
        if isinstance(b, dict):
            labels_all.extend(b["labels"].tolist())
        else:
            labels_all.extend(b[1].tolist())

    n_hc = labels_all.count(0)
    n_pd = labels_all.count(1)
    tot = len(labels_all)
    weights = torch.tensor([
        tot / (2.0 * max(1, n_hc)),
        tot / (2.0 * max(1, n_pd))
    ], dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_loss = float("inf")
    patience_cnt = 0

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss = 0.0
        n_samples = 0

        for batch in train_loader:
            if isinstance(batch, dict):
                x = batch["features"].to(device)
                y = batch["labels"].to(device)
            else:
                x = batch[0].to(device)
                y = batch[1].to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()

            t_loss += loss.item() * len(y)
            n_samples += len(y)

        scheduler.step()
        train_loss = t_loss / max(1, n_samples)

        # Validation
        model.eval()
        v_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    x = batch["features"].to(device)
                    y = batch["labels"].to(device)
                else:
                    x = batch[0].to(device)
                    y = batch[1].to(device)

                logits = model(x)
                loss = criterion(logits, y)
                v_loss += loss.item() * len(y)
                preds = torch.argmax(logits, dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_targets.extend(y.cpu().tolist())

        val_loss = v_loss / max(1, len(all_targets))
        val_acc = accuracy_score(all_targets, all_preds) if all_targets else 0.0
        val_f1 = f1_score(all_targets, all_preds, zero_division=0) if all_targets else 0.0

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["val_f1"].append(float(val_f1))

        if val_loss < best_loss:
            best_loss = val_loss
            patience_cnt = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    return history


def train_voice_baselines():
    """Train primary continuous STFT model and Onset/Offset transition models."""
    print("=" * 60)
    print("VOICE BASE-PAPER 2D CNN BASELINE TRAINING")
    print("=" * 60)

    set_seed(config.RANDOM_SEED)
    ensure_dataset_extracted()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Build metadata and split
    metadata_df = build_voice_metadata()
    train_df, val_df, test_df = create_subject_splits(metadata_df, seed=config.RANDOM_SEED)

    print(f"\nSubjects: Train={train_df['subject_id'].nunique()}, Val={val_df['subject_id'].nunique()}, Test={test_df['subject_id'].nunique()}")
    print(f"Test Subjects ({test_df['subject_id'].nunique()}): {sorted(test_df['subject_id'].unique())}\n")

    # 2. Build Continuous STFT Dataset & Normalizer
    if not config.BEST_MODEL_PATH.exists() or not (config.RESULTS_DIR / "training_history.json").exists():
        print("1) Training Primary Continuous STFT 2D CNN...")
        unnorm_train_ds = VoiceSTFTChunkDataset(train_df, normalizer=None)
        normalizer = VoiceSTFTNormalizer()
        normalizer.fit([s["features"] for s in unnorm_train_ds.samples])
        normalizer.save(config.NORMALIZATION_STATS_PATH)

        train_ds = VoiceSTFTChunkDataset(train_df, normalizer=normalizer)
        val_ds = VoiceSTFTChunkDataset(val_df, normalizer=normalizer)

        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_voice_stft_batch, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_voice_stft_batch)

        model = VoiceBasePaper2DCNN().to(device)
        with open(config.MODEL_CONFIG_PATH, "w") as f:
            json.dump(model.get_config(), f, indent=4)

        history = train_classifier_epochs(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            save_path=config.BEST_MODEL_PATH
        )

        with open(config.RESULTS_DIR / "training_history.json", "w") as f:
            json.dump(history, f, indent=4)

        print(f"Continuous STFT Model saved to: {config.BEST_MODEL_PATH}")
    else:
        print(f"1) Continuous STFT Model already exists at: {config.BEST_MODEL_PATH}")

    # 3. Phonation Transition Models (Onset & Offset)
    print("\n2) Extracting and Training Onset & Offset Transition Models...")
    train_onsets, train_offsets = [], []
    train_on_y, train_off_y = [], []

    for _, row in train_df.iterrows():
        fp = row.get("filepath", "")
        if fp and Path(fp).exists():
            audio = load_and_resample_audio(fp)
            ons, offs = detect_speech_transitions(audio)
            train_onsets.extend(ons)
            train_on_y.extend([row["label"]] * len(ons))
            train_offsets.extend(offs)
            train_off_y.extend([row["label"]] * len(offs))

    val_onsets, val_offsets = [], []
    val_on_y, val_off_y = [], []
    for _, row in val_df.iterrows():
        fp = row.get("filepath", "")
        if fp and Path(fp).exists():
            audio = load_and_resample_audio(fp)
            ons, offs = detect_speech_transitions(audio)
            val_onsets.extend(ons)
            val_on_y.extend([row["label"]] * len(ons))
            val_offsets.extend(offs)
            val_off_y.extend([row["label"]] * len(offs))

    print(f"Extracted Transitions -> Onsets: Train={len(train_onsets)}, Val={len(val_onsets)} | Offsets: Train={len(train_offsets)}, Val={len(val_offsets)}")

    if len(train_onsets) > config.BATCH_SIZE and len(val_onsets) > 0:
        on_norm = VoiceSTFTNormalizer()
        on_norm.fit(train_onsets)
        train_on_norm = [on_norm.transform(s) for s in train_onsets]
        val_on_norm = [on_norm.transform(s) for s in val_onsets]

        on_train_loader = DataLoader(
            TensorDataset(torch.stack(train_on_norm), torch.tensor(train_on_y, dtype=torch.long)),
            batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True
        )
        on_val_loader = DataLoader(
            TensorDataset(torch.stack(val_on_norm), torch.tensor(val_on_y, dtype=torch.long)),
            batch_size=config.BATCH_SIZE, shuffle=False
        )

        onset_model = VoiceBasePaper2DCNN().to(device)
        train_classifier_epochs(onset_model, on_train_loader, on_val_loader, device, config.ONSET_MODEL_PATH)
        print(f"Onset Model saved to: {config.ONSET_MODEL_PATH}")

    if len(train_offsets) > config.BATCH_SIZE and len(val_offsets) > 0:
        off_norm = VoiceSTFTNormalizer()
        off_norm.fit(train_offsets)
        train_off_norm = [off_norm.transform(s) for s in train_offsets]
        val_off_norm = [off_norm.transform(s) for s in val_offsets]

        off_train_loader = DataLoader(
            TensorDataset(torch.stack(train_off_norm), torch.tensor(train_off_y, dtype=torch.long)),
            batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True
        )
        off_val_loader = DataLoader(
            TensorDataset(torch.stack(val_off_norm), torch.tensor(val_off_y, dtype=torch.long)),
            batch_size=config.BATCH_SIZE, shuffle=False
        )

        offset_model = VoiceBasePaper2DCNN().to(device)
        train_classifier_epochs(offset_model, off_train_loader, off_val_loader, device, config.OFFSET_MODEL_PATH)
        print(f"Offset Model saved to: {config.OFFSET_MODEL_PATH}")

    print("\nAll Voice baseline models trained successfully.")


if __name__ == "__main__":
    train_voice_baselines()
