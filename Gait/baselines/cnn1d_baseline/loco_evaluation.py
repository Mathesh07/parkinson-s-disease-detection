"""Leave-One-Cohort-Out (LOCO) Robustness Evaluation for Gait 1D CNN Baseline."""

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Gait.baselines.cnn1d_baseline import config
from Gait.baselines.cnn1d_baseline.dataset import Gait1DCNNWindowDataset, collate_gait_1d_batch
from Gait.baselines.cnn1d_baseline.evaluate import calculate_metrics_dict
from Gait.baselines.cnn1d_baseline.model import Gait1DCNN
from Gait.baselines.cnn1d_baseline.train import set_seed, train_one_epoch
from Gait.preprocessing import GaitNormalizer, build_manifest, load_raw_recording


def run_loco_benchmark(epochs_per_cohort: int = 20) -> Dict:
    """
    Run Leave-One-Cohort-Out across Ga, Ju, and Si studies.
    
    In each fold:
        - Train on 2 cohorts (fitted normalizer on training cohorts only).
        - Evaluate on the held-out cohort.
    """
    print("=" * 60)
    print("GAIT 1D CNN: LEAVE-ONE-COHORT-OUT (LOCO) BENCHMARK")
    print("=" * 60)

    set_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest_df = build_manifest(config.DATA_DIR)
    studies = ["Ga", "Ju", "Si"]
    loco_results = {}

    for held_out_study in studies:
        print(f"\n--- LOCO Fold: Held-Out Study = {held_out_study} ---")

        train_df = manifest_df[manifest_df["study"] != held_out_study].copy().reset_index(drop=True)
        test_df = manifest_df[manifest_df["study"] == held_out_study].copy().reset_index(drop=True)

        print(f"Train Cohorts: {[s for s in studies if s != held_out_study]} (Subjects: {train_df['subject_id'].nunique()})")
        print(f"Test Cohort:   {held_out_study} (Subjects: {test_df['subject_id'].nunique()})")

        # Fit normalizer strictly on training cohorts
        train_raws = [load_raw_recording(fp) for fp in train_df["filepath"]]
        norm = GaitNormalizer().fit(train_raws)

        train_ds = Gait1DCNNWindowDataset(train_df, normalizer=norm)
        test_ds = Gait1DCNNWindowDataset(test_df, normalizer=norm)

        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_gait_1d_batch, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_gait_1d_batch)

        model = Gait1DCNN().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

        for ep in range(1, epochs_per_cohort + 1):
            train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate on held-out cohort
        model.eval()
        test_records = []
        with torch.no_grad():
            for batch in test_loader:
                feats = batch["features"].to(device)
                logits = model(feats)
                probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
                labels = batch["labels"].cpu().numpy()

                for i in range(len(labels)):
                    test_records.append({
                        "subject_id": batch["subject_ids"][i],
                        "recording_id": batch["recording_ids"][i],
                        "true_label": int(labels[i]),
                        "pd_probability": float(probs[i])
                    })

        test_df_res = pd.DataFrame(test_records)
        sub_grouped = test_df_res.groupby(["subject_id", "true_label"]).agg(
            pd_probability=("pd_probability", "mean")
        ).reset_index()

        metrics = calculate_metrics_dict(sub_grouped["true_label"].values, sub_grouped["pd_probability"].values)
        metrics["subject_count"] = len(sub_grouped)
        metrics["hc_count"] = int((sub_grouped["true_label"] == 0).sum())
        metrics["pd_count"] = int((sub_grouped["true_label"] == 1).sum())

        loco_results[held_out_study] = metrics
        print(f"Result for {held_out_study}: Accuracy = {metrics['accuracy']:.4f}, Bal Acc = {metrics['balanced_accuracy']:.4f}, F1 = {metrics['f1_score']:.4f}, AUC = {metrics['roc_auc']:.4f}")

    # Macro averages
    macro_acc = np.mean([loco_results[s]["accuracy"] for s in studies])
    macro_bal_acc = np.mean([loco_results[s]["balanced_accuracy"] for s in studies])
    macro_f1 = np.mean([loco_results[s]["f1_score"] for s in studies])
    macro_auc = np.mean([loco_results[s]["roc_auc"] for s in studies])

    loco_summary = {
        "benchmark_name": "Gait1DCNN_Leave_One_Cohort_Out",
        "cohorts": loco_results,
        "macro_average": {
            "accuracy": float(macro_acc),
            "balanced_accuracy": float(macro_bal_acc),
            "f1_score": float(macro_f1),
            "roc_auc": float(macro_auc)
        }
    }

    with open(config.LOCO_METRICS_PATH, "w") as f:
        json.dump(loco_summary, f, indent=4)

    print("\n" + "=" * 60)
    print("GAIT 1D CNN LOCO SUMMARY (MACRO AVERAGE):")
    print(f"  Macro Accuracy:     {macro_acc:.4f}")
    print(f"  Macro Bal. Acc:     {macro_bal_acc:.4f}")
    print(f"  Macro F1-Score:     {macro_f1:.4f}")
    print(f"  Macro ROC-AUC:      {macro_auc:.4f}")
    print("=" * 60)

    return loco_summary


if __name__ == "__main__":
    run_loco_benchmark(epochs_per_cohort=20)
