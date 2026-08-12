"""Training pipeline for Parkinson's handwriting classification.

This module provides:
1. Two-stage Vision Transformer (ViT) training (frozen head training -> upper layer fine-tuning).
2. CNN Baseline model training for Phase 6 comparison.
3. Class-weighted CrossEntropyLoss to address dataset balance.
4. Early stopping and model checkpointing.
5. Saving training and validation loss/accuracy history.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from config import (BATCH_SIZE, CNN_CHECKPOINT, IMAGE_SIZE,
                    LEARNING_RATE_BACKBONE, LEARNING_RATE_HEAD, MIN_DELTA,
                    NUM_EPOCHS, OUTPUT_DIR, PATIENCE, RANDOM_SEED,
                    VIT_CHECKPOINT, WEIGHT_DECAY)
from dataset import (build_manifest, create_dataloaders,
                     create_patient_level_split, print_dataset_summary)
from evaluate import compute_metrics
from model import CNNBaseline, ViTBinaryClassifier


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Ensure reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module
) -> Tuple[float, float]:
    """Execute one training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    num_batches = len(loader)
    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        if batch_idx % 5 == 0 or batch_idx == num_batches:
            current_acc = (preds == labels).float().mean().item()
            print(f"  --> Batch [{batch_idx:02d}/{num_batches:02d}] Loss: {loss.item():.4f} Acc: {current_acc:.2f}")


    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module
) -> Tuple[float, float, list, list, list]:
    """Evaluate model performance on val or test loader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
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

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc, all_labels, all_preds, all_probs


def train_model(
    model_type: str = "vit",
    epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE
) -> Dict[str, list]:
    """Train ViT or CNN baseline model with early stopping and checkpointing."""
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==========================================")
    print(f" Starting Training: {model_type.upper()} Model")
    print(f" Device: {device}")
    print(f" Max Epochs: {epochs} | Early Stopping Patience: {PATIENCE}")
    print(f"==========================================")

    # Step 1: Load dataset & patient-isolated splits
    manifest, summary = build_manifest()
    print_dataset_summary(summary, manifest)
    manifest = create_patient_level_split(manifest)
    train_loader, val_loader, test_loader = create_dataloaders(
        manifest, batch_size=batch_size, image_size=IMAGE_SIZE, num_workers=0
    )

    # Step 2: Compute class weights to handle slight class imbalance
    # Target 0: Healthy, Target 1: Parkinson
    train_targets = manifest[manifest["split"] == "train"]["target"].values
    n_class0 = (train_targets == 0).sum()
    n_class1 = (train_targets == 1).sum()
    total_train = n_class0 + n_class1
    w0 = total_train / (2.0 * n_class0) if n_class0 > 0 else 1.0
    w1 = total_train / (2.0 * n_class1) if n_class1 > 0 else 1.0
    class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)
    print(f"Class weights (0=Healthy: {w0:.3f}, 1=Parkinson: {w1:.3f})")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Step 3: Instantiate model & optimizer
    if model_type.lower() == "vit":
        model = ViTBinaryClassifier(freeze_backbone=True).to(device)
        checkpoint_path = VIT_CHECKPOINT
        # Warmup head training
        optimizer = AdamW(model.classifier.parameters(), lr=LEARNING_RATE_HEAD, weight_decay=WEIGHT_DECAY)
    else:
        model = CNNBaseline(num_classes=2).to(device)
        checkpoint_path = CNN_CHECKPOINT
        optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=WEIGHT_DECAY)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Stage 2 for ViT: unfreeze upper backbone layers early in fine-tuning (epoch 4)
        if model_type.lower() == "vit" and epoch == 4:
            print("\n--> Stage 2: Unfreezing upper ViT encoder layers for fine-tuning...")
            model.unfreeze_backbone(unfreeze_last_n_layers=2)
            # Use lower learning rate for backbone parameters
            backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and "classifier" not in n]
            head_params = [p for n, p in model.named_parameters() if p.requires_grad and "classifier" in n]
            optimizer = AdamW([
                {"params": head_params, "lr": LEARNING_RATE_HEAD * 0.5},
                {"params": backbone_params, "lr": LEARNING_RATE_BACKBONE}
            ], weight_decay=WEIGHT_DECAY)
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs - 3)

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_loss, val_acc, _, _, _ = evaluate_epoch(model, val_loader, device, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Early Stopping check: require at least MIN_DELTA improvement
        if val_loss < (best_val_loss - MIN_DELTA):
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc
            }, checkpoint_path)
            print(f"  [+] Saved best checkpoint to {checkpoint_path.name} (Val Loss improved to {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  [-] Val loss did not improve significantly ({patience_counter}/{PATIENCE} patience steps)")
            if patience_counter >= PATIENCE:
                print(f"\n[!] Early stopping triggered at epoch {epoch} (no validation improvement for {PATIENCE} consecutive epochs).")
                break

    # Save training history
    history_file = OUTPUT_DIR / f"{model_type}_history.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {history_file.name}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Parkinson's handwriting classifier.")
    parser.add_argument("--model", type=str, default="vit", choices=["vit", "cnn"], help="Model type: vit or cnn")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs")
    args = parser.parse_args()

    train_model(model_type=args.model, epochs=args.epochs)
