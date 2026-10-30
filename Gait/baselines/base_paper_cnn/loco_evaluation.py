"""Leave-One-Cohort-Out (LOCO) Evaluation for Gait Base-Paper 2D CNN Baseline.

Evaluates cohort-shift generalization across the three clinical studies:
    - Hold-out Cohort 'Ga' (Galit et al.)
    - Hold-out Cohort 'Ju' (Hausdorff et al.)
    - Hold-out Cohort 'Si' (Frenkel-Toledo et al.)
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

try:
    from . import config
    from .dataset import GaitSTFTDataset, collate_gait_stft_batch
    from .model import GaitBasePaper2DCNN
    from .preprocessing import GaitSTFTNormalizer, build_manifest
    from .train import get_device, set_seed, train_one_epoch, validate
except ImportError:
    import config
    from dataset import GaitSTFTDataset, collate_gait_stft_batch
    from model import GaitBasePaper2DCNN
    from preprocessing import GaitSTFTNormalizer, build_manifest
    from train import get_device, set_seed, train_one_epoch, validate


def run_loco_evaluation() -> Dict:
    """Run Leave-One-Cohort-Out cross-study evaluation."""
    print("=" * 60)
    print("GAIT BASE-PAPER 2D CNN: LEAVE-ONE-COHORT-OUT (LOCO) EVALUATION")
    print("=" * 60)

    set_seed(config.RANDOM_SEED)
    device = get_device()
    manifest_df = build_manifest()

    cohorts = ["Ga", "Ju", "Si"]
    loco_results = {}

    for held_out_cohort in cohorts:
        print(f"\n>>> Running LOCO Split: Held-Out Cohort = '{held_out_cohort}' <<<")
        train_val_df = manifest_df[manifest_df["study"] != held_out_cohort].reset_index(drop=True)
        test_df = manifest_df[manifest_df["study"] == held_out_cohort].reset_index(drop=True)

        # 85% train, 15% val from remaining two cohorts
        rng = np.random.RandomState(config.RANDOM_SEED)
        subjects = list(train_val_df["subject_id"].unique())
        rng.shuffle(subjects)
        n_val = max(1, int(round(0.15 * len(subjects))))
        val_subs = set(subjects[:n_val])
        train_subs = set(subjects[n_val:])

        train_split = train_val_df[train_val_df["subject_id"].isin(train_subs)].reset_index(drop=True)
        val_split = train_val_df[train_val_df["subject_id"].isin(val_subs)].reset_index(drop=True)

        # Build train dataset
        train_ds = GaitSTFTDataset(train_split, normalizer=None)
        normalizer = GaitSTFTNormalizer()
        normalizer.fit([s["features"] for s in train_ds.samples])

        for s in train_ds.samples:
            s["features"] = normalizer.transform(s["features"])
        train_ds.normalizer = normalizer

        val_ds = GaitSTFTDataset(val_split, normalizer=normalizer)
        test_ds = GaitSTFTDataset(test_df, normalizer=normalizer)

        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_gait_stft_batch)
        val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_gait_stft_batch)
        test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_gait_stft_batch)

        # Model and loss
        model = GaitBasePaper2DCNN().to(device)
        train_labels = [s["label"] for s in train_ds.samples]
        n_pos = sum(train_labels)
        n_neg = len(train_labels) - n_pos
        pos_weight = torch.tensor([n_neg / (n_pos + 1e-6)], dtype=torch.float32).to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(1, 10):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, val_f1, val_auc = validate(model, val_loader, criterion, device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        # Evaluate on held-out cohort
        model.eval()
        window_preds = []
        with torch.no_grad():
            for batch in test_loader:
                feats = batch["features"].to(device)
                logits = model(feats)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                labels = batch["labels"].cpu().numpy().flatten()

                for i in range(len(probs)):
                    window_preds.append({
                        "subject_id": batch["subject_ids"][i],
                        "true_label": int(labels[i]),
                        "pd_probability": float(probs[i])
                    })

        wdf = pd.DataFrame(window_preds)
        sub_df = wdf.groupby(["subject_id", "true_label"]).agg(pd_probability=("pd_probability", "mean")).reset_index()
        y_true = sub_df["true_label"].values
        y_prob = sub_df["pd_probability"].values
        y_pred = (y_prob >= 0.5).astype(int)

        acc = float(accuracy_score(y_true, y_pred))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = 0.5

        print(f"Cohort '{held_out_cohort}' Results -> Subjects: {len(y_true)}, Acc: {acc:.4f}, Bal Acc: {bal_acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

        loco_results[held_out_cohort] = {
            "held_out_cohort": held_out_cohort,
            "num_test_subjects": len(y_true),
            "num_test_windows": len(wdf),
            "subject_accuracy": acc,
            "subject_balanced_accuracy": bal_acc,
            "subject_f1_score": f1,
            "subject_roc_auc": auc
        }

    loco_results["macro_average"] = {
        "subject_accuracy": float(np.mean([loco_results[c]["subject_accuracy"] for c in cohorts])),
        "subject_balanced_accuracy": float(np.mean([loco_results[c]["subject_balanced_accuracy"] for c in cohorts])),
        "subject_f1_score": float(np.mean([loco_results[c]["subject_f1_score"] for c in cohorts])),
        "subject_roc_auc": float(np.mean([loco_results[c]["subject_roc_auc"] for c in cohorts]))
    }

    with open(config.LOCO_METRICS_PATH, "w") as f:
        json.dump(loco_results, f, indent=4)
    print(f"\nSaved LOCO cohort metrics to: {config.LOCO_METRICS_PATH}")

    return loco_results


if __name__ == "__main__":
    run_loco_evaluation()
