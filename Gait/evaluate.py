"""Independent evaluation script for Gait-Based Parkinson's Disease detection on held-out test subjects."""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Support running as a standalone script or as a module
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
for p in (str(current_dir), str(parent_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from Gait import config
    from Gait.dataset import (
        GaitWindowDataset,
        collate_gait_batch,
        create_subject_splits,
    )
    from Gait.model import GaitCNNBiLSTM
    from Gait.preprocessing import GaitNormalizer, build_manifest
    from Gait.utils import (
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
        GaitWindowDataset,
        collate_gait_batch,
        create_subject_splits,
    )
    from model import GaitCNNBiLSTM
    from preprocessing import GaitNormalizer, build_manifest
    from utils import (
        aggregate_predictions,
        calculate_metrics,
        get_device,
        plot_confusion_matrix,
        plot_roc_curve,
        set_seed,
    )


def load_trained_model(
    checkpoint_path: Path = config.BEST_MODEL_PATH,
    device: torch.device = torch.device("cpu")
) -> GaitCNNBiLSTM:
    """Load trained model weights from checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Trained checkpoint not found at: {checkpoint_path}. Please train the model first.")

    model = GaitCNNBiLSTM(
        in_channels=config.NUM_CHANNELS,
        cnn_channels=config.CNN_CHANNELS,
        lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
        lstm_num_layers=config.LSTM_NUM_LAYERS,
        embedding_dim=config.EMBEDDING_DIM,
        dropout=config.DROPOUT
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def evaluate(
    checkpoint_path: Path = config.BEST_MODEL_PATH,
    normalization_path: Path = config.NORMALIZATION_STATS_PATH
):
    """Run thorough evaluation on held-out test subjects."""
    print("=" * 60)
    print("GAIT PARKINSON'S EVALUATION (HELD-OUT TEST SET)")
    print("=" * 60)

    set_seed(config.RANDOM_SEED)
    device = get_device()

    # 1. Load Data Splits and Pre-fitted Normalizer
    manifest_df = build_manifest()
    train_df, val_df, test_df = create_subject_splits(manifest_df, seed=config.RANDOM_SEED)

    print(f"Loading normalization statistics from: {normalization_path}")
    normalizer = GaitNormalizer.load(normalization_path)

    # 2. Build Test Dataset & DataLoader
    test_dataset = GaitWindowDataset(test_df, normalizer=normalizer, window_size=config.WINDOW_SIZE, step_size=config.STEP_SIZE)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_gait_batch
    )

    test_subjects = sorted(test_df["subject_id"].unique().tolist())
    print(f"\nHeld-Out Test Subjects ({len(test_subjects)}): {test_subjects}")
    print(f"  HC Subjects: {test_df[test_df['label']==0]['subject_id'].nunique()} | PD Subjects: {test_df[test_df['label']==1]['subject_id'].nunique()}")
    print(f"  Total Test Recordings: {len(test_df)}")
    print(f"  Total Test 5s Windows: {len(test_dataset)}\n")

    # 3. Load Trained Model
    print(f"Loading checkpoint from: {checkpoint_path}")
    model = load_trained_model(checkpoint_path, device)

    # 4. Inference on Test Set
    window_records = []
    use_cuda = device.type == "cuda"

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)

            if use_cuda:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(features)
            else:
                logits = model(features)

            probs = torch.sigmoid(logits.float()).cpu().numpy()
            targets = labels.cpu().numpy()

            for i in range(len(targets)):
                window_records.append({
                    "subject_id": batch["subject_ids"][i],
                    "recording_id": batch["recording_ids"][i],
                    "study": batch["studies"][i],
                    "true_label": int(targets[i]),
                    "pd_probability": float(probs[i]),
                    "hc_probability": float(1.0 - probs[i]),
                    "predicted_label": int(probs[i] >= 0.5),
                    "predicted_name": config.ID2LABEL[int(probs[i] >= 0.5)]
                })

    window_df = pd.DataFrame(window_records)

    # 5. Hierarchical Aggregation (Window -> Recording -> Subject)
    rec_df, sub_df = aggregate_predictions(window_df)

    # Save Predictions to CSV
    window_df.to_csv(config.WINDOW_PREDICTIONS_PATH, index=False)
    rec_df.to_csv(config.RECORDING_PREDICTIONS_PATH, index=False)
    sub_df.to_csv(config.SUBJECT_PREDICTIONS_PATH, index=False)
    print(f"Saved window-level predictions to: {config.WINDOW_PREDICTIONS_PATH}")
    print(f"Saved recording-level predictions to: {config.RECORDING_PREDICTIONS_PATH}")
    print(f"Saved subject-level predictions to: {config.SUBJECT_PREDICTIONS_PATH}")

    # 6. Calculate Metrics
    win_metrics = calculate_metrics(
        window_df["true_label"].tolist(),
        window_df["predicted_label"].tolist(),
        window_df["pd_probability"].tolist()
    )

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

    # Breakdown by sub-study (Ga, Ju, Si) at subject level
    study_breakdowns = {}
    for study_name in ["Ga", "Ju", "Si"]:
        study_sub = sub_df[sub_df["study"] == study_name]
        if len(study_sub) > 0:
            study_breakdowns[study_name] = calculate_metrics(
                study_sub["true_label"].tolist(),
                study_sub["predicted_label"].tolist(),
                study_sub["pd_probability"].tolist()
            )

    metrics_output = {
        "subject_level": sub_metrics,
        "recording_level": rec_metrics,
        "window_level": win_metrics,
        "study_breakdown": study_breakdowns,
        "split_counts": {
            "train_subjects": train_df["subject_id"].nunique(),
            "val_subjects": val_df["subject_id"].nunique(),
            "test_subjects": test_df["subject_id"].nunique(),
            "train_recordings": len(train_df),
            "val_recordings": len(val_df),
            "test_recordings": len(test_df),
            "test_windows": len(test_dataset)
        }
    }

    with open(config.TEST_METRICS_PATH, "w") as f:
        json.dump(metrics_output, f, indent=4)
    print(f"Saved complete metrics report to: {config.TEST_METRICS_PATH}")

    # 7. Generate Evaluation Plots
    plot_confusion_matrix(
        sub_df["true_label"].tolist(),
        sub_df["predicted_label"].tolist(),
        class_names=["HC", "PD"],
        save_path=config.CONFUSION_MATRIX_PATH,
        title="Gait Test Confusion Matrix (Subject-Level)"
    )
    print(f"Saved confusion matrix plot to: {config.CONFUSION_MATRIX_PATH}")

    plot_roc_curve(
        sub_df["true_label"].tolist(),
        sub_df["pd_probability"].tolist(),
        save_path=config.ROC_CURVE_PATH,
        title="Gait Test ROC Curve (Subject-Level)"
    )
    print(f"Saved ROC curve plot to: {config.ROC_CURVE_PATH}")

    # 8. Print Formatted Test Results
    print("\n" + "=" * 60)
    print("===== TEST RESULTS =====")
    print("=" * 60)
    print(f"Subject-level Accuracy:    {sub_metrics['accuracy']:.4f}")
    print(f"Subject-level Precision:   {sub_metrics['precision']:.4f}")
    print(f"Subject-level Recall:      {sub_metrics['recall_sensitivity']:.4f} (Sensitivity)")
    print(f"Subject-level Specificity: {sub_metrics['specificity']:.4f}")
    print(f"Subject-level F1:          {sub_metrics['f1_score']:.4f}")
    print(f"Subject-level ROC-AUC:     {sub_metrics['roc_auc']:.4f}")
    print("-" * 60)
    print(f"{'Level':<18} | {'Accuracy':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
    print("-" * 60)
    print(f"{'Subject-Level':<18} | {sub_metrics['accuracy']:<10.4f} | {sub_metrics['f1_score']:<10.4f} | {sub_metrics['roc_auc']:<10.4f}")
    print(f"{'Recording-Level':<18} | {rec_metrics['accuracy']:<10.4f} | {rec_metrics['f1_score']:<10.4f} | {rec_metrics['roc_auc']:<10.4f}")
    print(f"{'Window-Level':<18} | {win_metrics['accuracy']:<10.4f} | {win_metrics['f1_score']:<10.4f} | {win_metrics['roc_auc']:<10.4f}")
    print("=" * 60)
    print("\nDataset Counts:")
    print(f"Train subjects:       {train_df['subject_id'].nunique()}")
    print(f"Validation subjects:  {val_df['subject_id'].nunique()}")
    print(f"Test subjects:        {test_df['subject_id'].nunique()}")
    print()
    print(f"Train recordings:     {len(train_df)}")
    print(f"Validation recordings:{len(val_df)}")
    print(f"Test recordings:      {len(test_df)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    evaluate()
