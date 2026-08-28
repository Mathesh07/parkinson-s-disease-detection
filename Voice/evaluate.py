"""Evaluation script for Voice-Based Parkinson's Detection using Wav2Vec2."""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
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
        plot_confusion_matrix,
        plot_roc_curve,
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
        plot_confusion_matrix,
        plot_roc_curve,
        set_seed,
    )


def load_trained_model(
    checkpoint_dir: Path = config.CHECKPOINT_DIR,
    device: torch.device = torch.device("cpu")
) -> Tuple[Wav2Vec2ForParkinsons, Wav2Vec2Processor]:
    """Load trained model weights and processor from checkpoint directory."""
    weights_path = checkpoint_dir / "best_model.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {weights_path}. Please train the model first.")

    processor = Wav2Vec2Processor.from_pretrained(str(checkpoint_dir))
    model = Wav2Vec2ForParkinsons(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        freeze_feature_encoder=config.FREEZE_FEATURE_ENCODER
    )
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, processor


def evaluate(checkpoint_dir: Path = config.CHECKPOINT_DIR):
    """Run thorough evaluation on held-out test subjects."""
    print("=" * 50)
    print("WAV2VEC2 PARKINSON'S VOICE EVALUATION")
    print("=" * 50)

    set_seed(config.RANDOM_SEED)
    device = get_device()

    # 1. Load Data Splits
    metadata_df = build_metadata()
    _, _, test_df = create_subject_splits(metadata_df)

    test_subjects = sorted(test_df["subject_id"].unique().tolist())
    print(f"\nHeld-Out Test Subjects ({len(test_subjects)}): {test_subjects}")
    print(f"  HC Subjects: {test_df[test_df['label_name']=='HC']['subject_id'].nunique()} | PD Subjects: {test_df[test_df['label_name']=='PD']['subject_id'].nunique()}")
    print(f"  Total Test Recordings: {len(test_df)} (ReadText: {len(test_df[test_df['task']=='ReadText'])}, Spontaneous: {len(test_df[test_df['task']=='SpontaneousDialogue'])})\n")

    # 2. Load Model & Processor
    print(f"Loading checkpoint from: {checkpoint_dir}")
    model, processor = load_trained_model(checkpoint_dir, device)

    # 3. Create Dataset & DataLoader
    test_dataset = VoiceChunkDataset(test_df, processor=processor)
    print(f"Generated {len(test_dataset)} test audio chunks.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_audio_batch
    )

    # 4. Inference on Test Set
    chunk_records = []
    use_cuda = device.type == "cuda"

    with torch.no_grad():
        for batch in test_loader:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if use_cuda:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(input_values, attention_mask=attention_mask)
            else:
                logits = model(input_values, attention_mask=attention_mask)

            probs = torch.softmax(logits.float(), dim=-1)
            pd_probs = probs[:, 1].cpu().numpy()

            for i in range(len(labels)):
                chunk_records.append({
                    "recording_id": batch["recording_ids"][i],
                    "subject_id": batch["subject_ids"][i],
                    "task": batch["tasks"][i],
                    "true_label": int(labels[i].cpu().item()),
                    "pd_probability": float(pd_probs[i])
                })

    chunk_df = pd.DataFrame(chunk_records)

    # 5. Aggregation
    rec_df, sub_df = aggregate_predictions(chunk_df)

    # Save Prediction CSVs
    rec_df.to_csv(config.TEST_PREDICTIONS_PATH, index=False)
    sub_df.to_csv(config.SUBJECT_PREDICTIONS_PATH, index=False)
    print(f"Saved recording-level predictions to: {config.TEST_PREDICTIONS_PATH}")
    print(f"Saved subject-level predictions to: {config.SUBJECT_PREDICTIONS_PATH}")

    # 6. Compute Metrics
    # Subject-Level Overall Metrics
    overall_sub_metrics = calculate_metrics(
        sub_df["true_label"].tolist(),
        sub_df["predicted_label"].tolist(),
        sub_df["pd_probability"].tolist()
    )

    # Recording-Level Overall Metrics
    overall_rec_metrics = calculate_metrics(
        rec_df["true_label"].tolist(),
        rec_df["predicted_label"].tolist(),
        rec_df["pd_probability"].tolist()
    )

    # Task Breakdowns (ReadText vs SpontaneousDialogue at recording level)
    readtext_df = rec_df[rec_df["task"] == "ReadText"]
    spont_df = rec_df[rec_df["task"] == "SpontaneousDialogue"]

    readtext_metrics = calculate_metrics(
        readtext_df["true_label"].tolist(),
        readtext_df["predicted_label"].tolist(),
        readtext_df["pd_probability"].tolist()
    ) if len(readtext_df) > 0 else {}

    spont_metrics = calculate_metrics(
        spont_df["true_label"].tolist(),
        spont_df["predicted_label"].tolist(),
        spont_df["pd_probability"].tolist()
    ) if len(spont_df) > 0 else {}

    metrics_output = {
        "subject_level_overall": overall_sub_metrics,
        "recording_level_overall": overall_rec_metrics,
        "task_breakdown": {
            "ReadText": readtext_metrics,
            "SpontaneousDialogue": spont_metrics
        }
    }

    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics_output, f, indent=4)
    print(f"Saved metrics to: {config.METRICS_PATH}")

    # 7. Generate Plots
    plot_confusion_matrix(
        sub_df["true_label"].tolist(),
        sub_df["predicted_label"].tolist(),
        class_names=["HC", "PD"],
        save_path=config.CONFUSION_MATRIX_PATH,
        title="Test Confusion Matrix (Subject-Level)"
    )
    print(f"Saved confusion matrix plot to: {config.CONFUSION_MATRIX_PATH}")

    plot_roc_curve(
        sub_df["true_label"].tolist(),
        sub_df["pd_probability"].tolist(),
        save_path=config.ROC_CURVE_PATH,
        title="Test ROC Curve (Subject-Level)"
    )
    print(f"Saved ROC curve plot to: {config.ROC_CURVE_PATH}")

    # 8. Display Formatted Metrics Summary
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS SUMMARY (TEST SET)")
    print("=" * 50)
    print(f"{'Metric':<25} | {'Subject-Level':<15} | {'Recording-Level':<15}")
    print("-" * 60)
    print(f"{'Accuracy':<25} | {overall_sub_metrics['accuracy']:<15.4f} | {overall_rec_metrics['accuracy']:<15.4f}")
    print(f"{'Precision':<25} | {overall_sub_metrics['precision']:<15.4f} | {overall_rec_metrics['precision']:<15.4f}")
    print(f"{'Recall (Sensitivity)':<25} | {overall_sub_metrics['recall']:<15.4f} | {overall_rec_metrics['recall']:<15.4f}")
    print(f"{'Specificity':<25} | {overall_sub_metrics['specificity']:<15.4f} | {overall_rec_metrics['specificity']:<15.4f}")
    print(f"{'F1-Score':<25} | {overall_sub_metrics['f1_score']:<15.4f} | {overall_rec_metrics['f1_score']:<15.4f}")
    print(f"{'ROC-AUC':<25} | {overall_sub_metrics['roc_auc']:<15.4f} | {overall_rec_metrics['roc_auc']:<15.4f}")
    print("=" * 50)

    print("\n--- Speech Task Breakdown (Recording-Level) ---")
    print(f"{'Task':<25} | {'Accuracy':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
    print("-" * 60)
    if readtext_metrics:
        print(f"{'ReadText':<25} | {readtext_metrics['accuracy']:<10.4f} | {readtext_metrics['f1_score']:<10.4f} | {readtext_metrics['roc_auc']:<10.4f}")
    if spont_metrics:
        print(f"{'SpontaneousDialogue':<25} | {spont_metrics['accuracy']:<10.4f} | {spont_metrics['f1_score']:<10.4f} | {spont_metrics['roc_auc']:<10.4f}")
    print("=" * 50)


if __name__ == "__main__":
    evaluate()
