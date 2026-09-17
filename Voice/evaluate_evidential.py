"""Evaluation script for Voice Evidential Wav2Vec2 with per-sample Dirichlet & uncertainty outputs."""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor

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
from evidential import dirichlet_from_evidence

EVIDENTIAL_MODEL_FILE = config.RESULTS_DIR / "best_wav2vec2_evidential" / "best_model.pt"
EVIDENTIAL_PREDICTIONS_CSV = config.RESULTS_DIR / "voice_evidential_predictions.csv"
EVIDENTIAL_SUBJECT_CSV = config.RESULTS_DIR / "voice_evidential_subjects.csv"
EVIDENTIAL_METRICS_JSON = config.RESULTS_DIR / "voice_evidential_metrics.json"


def evaluate_voice_evidential(
    checkpoint_path: Path = EVIDENTIAL_MODEL_FILE
) -> Dict[str, any]:
    """Run evidential evaluation on held-out test audio dataset."""
    print("=" * 60)
    print("VOICE WAV2VEC2 EVIDENTIAL EVALUATION (TEST SET)")
    print("=" * 60)

    set_seed(config.RANDOM_SEED)
    device = get_device()

    metadata_df = build_metadata()
    train_df, val_df, test_df = create_subject_splits(metadata_df)

    processor = Wav2Vec2Processor.from_pretrained(config.MODEL_NAME)
    test_dataset = VoiceChunkDataset(test_df, processor=processor)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_audio_batch)

    model = Wav2Vec2ForParkinsons(model_name=config.MODEL_NAME, num_classes=2).to(device)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Evidential checkpoint not found at: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    chunk_records = []

    with torch.no_grad():
        for batch in test_loader:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            evidence = model(input_values, attention_mask=attention_mask)
            alpha, S, probs, u = dirichlet_from_evidence(evidence, num_classes=2)

            e_np = evidence.cpu().numpy()
            a_np = alpha.cpu().numpy()
            p_np = probs.cpu().numpy()
            u_np = u.squeeze(-1).cpu().numpy()
            y_np = labels.cpu().numpy()

            for i in range(len(y_np)):
                pred = int(p_np[i, 1] >= 0.5)
                chunk_records.append({
                    "sample_id": f"{batch['recording_ids'][i]}_chk{i}",
                    "subject_id": batch["subject_ids"][i],
                    "recording_id": batch["recording_ids"][i],
                    "task": batch["tasks"][i],
                    "true_label": int(y_np[i]),
                    "predicted_label": pred,
                    "is_correct": int(pred == int(y_np[i])),
                    "e_HC": float(e_np[i, 0]),
                    "e_PD": float(e_np[i, 1]),
                    "alpha_HC": float(a_np[i, 0]),
                    "alpha_PD": float(a_np[i, 1]),
                    "P_HC": float(p_np[i, 0]),
                    "P_PD": float(p_np[i, 1]),
                    "uncertainty": float(u_np[i])
                })

    chunk_df = pd.DataFrame(chunk_records)
    chunk_df.to_csv(EVIDENTIAL_PREDICTIONS_CSV, index=False)

    # Subject-level aggregation
    rec_df = chunk_df.groupby(["recording_id", "subject_id", "true_label"])[["e_HC", "e_PD", "alpha_HC", "alpha_PD", "P_HC", "P_PD", "uncertainty"]].mean().reset_index()
    sub_df = rec_df.groupby(["subject_id", "true_label"])[["e_HC", "e_PD", "alpha_HC", "alpha_PD", "P_HC", "P_PD", "uncertainty"]].mean().reset_index()
    sub_df["predicted_label"] = (sub_df["P_PD"] >= 0.5).astype(int)
    sub_df["is_correct"] = (sub_df["predicted_label"] == sub_df["true_label"]).astype(int)

    sub_df.to_csv(EVIDENTIAL_SUBJECT_CSV, index=False)
    print(f"Saved evidential chunk predictions to: {EVIDENTIAL_PREDICTIONS_CSV}")
    print(f"Saved evidential subject predictions to: {EVIDENTIAL_SUBJECT_CSV}")

    metrics = calculate_metrics(
        sub_df["true_label"].tolist(),
        sub_df["predicted_label"].tolist(),
        sub_df["P_PD"].tolist()
    )

    correct_u = sub_df[sub_df["is_correct"] == 1]["uncertainty"].mean() if (sub_df["is_correct"] == 1).any() else 0.0
    incorrect_u = sub_df[sub_df["is_correct"] == 0]["uncertainty"].mean() if (sub_df["is_correct"] == 0).any() else 0.0
    hc_u = sub_df[sub_df["true_label"] == 0]["uncertainty"].mean()
    pd_u = sub_df[sub_df["true_label"] == 1]["uncertainty"].mean()

    uncertainty_stats = {
        "mean_uncertainty_all": float(sub_df["uncertainty"].mean()),
        "mean_uncertainty_correct": float(correct_u),
        "mean_uncertainty_incorrect": float(incorrect_u),
        "mean_uncertainty_HC": float(hc_u),
        "mean_uncertainty_PD": float(pd_u)
    }

    full_results = {
        "modality": "Voice (Wav2Vec2)",
        "subject_metrics": metrics,
        "uncertainty_stats": uncertainty_stats
    }

    with open(EVIDENTIAL_METRICS_JSON, "w") as f:
        json.dump(full_results, f, indent=4)

    print("\n" + "=" * 50)
    print("VOICE WAV2VEC2 EVIDENTIAL TEST RESULTS (SUBJECT-LEVEL)")
    print("=" * 50)
    print(f"Accuracy:        {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision:       {metrics['precision'] * 100:.2f}%")
    print(f"Sensitivity:     {metrics['sensitivity'] * 100:.2f}%")
    print(f"Specificity:     {metrics['specificity'] * 100:.2f}%")
    print(f"F1-Score:        {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:         {metrics['roc_auc']:.4f}")
    print("-" * 50)
    print("UNCERTAINTY BREAKDOWN:")
    print(f"  Mean Uncertainty (All):       {uncertainty_stats['mean_uncertainty_all']:.4f}")
    print(f"  Mean Uncertainty (Correct):   {uncertainty_stats['mean_uncertainty_correct']:.4f}")
    print(f"  Mean Uncertainty (Incorrect): {uncertainty_stats['mean_uncertainty_incorrect']:.4f}")
    print(f"  Mean Uncertainty (HC):        {uncertainty_stats['mean_uncertainty_HC']:.4f}")
    print(f"  Mean Uncertainty (PD):        {uncertainty_stats['mean_uncertainty_PD']:.4f}")
    print("=" * 50)

    return full_results


if __name__ == "__main__":
    evaluate_voice_evidential()
