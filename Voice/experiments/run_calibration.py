"""Phase 2 Experiment: Post-Hoc Temperature Scaling for Voice Parkinson's Classifier.

Executes validation-only scalar Temperature Scaling (Guo et al., ICML 2017)
and evaluates calibration metrics (NLL, Brier, ECE) on the held-out test set.

Leakage Guards:
    - Temperature parameter T > 0 is fitted ONLY on validation subjects (5 subjects).
    - Test set subjects (6 subjects) and test labels are completely isolated until final evaluation.
    - Pretrained model weights remain strictly frozen.
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

# Ensure workspace root and Voice directory are in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
voice_dir = root_dir / "Voice"
for p in (str(root_dir), str(voice_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from Voice import config
from Voice.calibration import (
    TemperatureScaler,
    calculate_brier,
    calculate_ece,
    calculate_nll,
    plot_calibration_comparison,
    plot_reliability_diagram,
)
from Voice.dataset import create_subject_splits
from Voice.model import EvidentialHead
from Voice.utils import calculate_metrics, set_seed

CALIB_OUTPUT_DIR = config.RESULTS_DIR / "calibration"
CALIB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEMP_JSON_PATH = CALIB_OUTPUT_DIR / "temperature.json"
CALIB_RESULTS_CSV = CALIB_OUTPUT_DIR / "calibration_results.csv"
CALIB_RESULTS_JSON = CALIB_OUTPUT_DIR / "calibration_results.json"
RELIABILITY_PLOT_PATH = CALIB_OUTPUT_DIR / "reliability_diagram.png"
METRICS_PLOT_PATH = CALIB_OUTPUT_DIR / "calibration_metrics.png"


def run_temperature_scaling_experiment(seed: int = config.RANDOM_SEED):
    """Run validation-only temperature scaling experiment for Voice baseline."""
    print("=" * 65)
    print("VOICE PHASE 2: POST-HOC TEMPERATURE SCALING EXPERIMENT")
    print("=" * 65)

    set_seed(seed)

    # 1. Load metadata and create subject splits
    meta_csv = config.RESULTS_DIR / "embeddings" / "voice_embeddings_metadata.csv"
    if meta_csv.exists():
        meta_df = pd.read_csv(meta_csv)
    else:
        raise FileNotFoundError(f"Metadata file not found at: {meta_csv}")

    train_df, val_df, test_df = create_subject_splits(meta_df, seed=seed)

    val_subjects = sorted(val_df["subject_id"].unique().tolist())
    test_subjects = sorted(test_df["subject_id"].unique().tolist())

    print(f"\nSubjects Breakdown:")
    print(f"  Train subjects ({len(train_df['subject_id'].unique())}): {sorted(train_df['subject_id'].unique().tolist())}")
    print(f"  Val subjects   ({len(val_subjects)}): {val_subjects}")
    print(f"  Test subjects  ({len(test_subjects)}): {test_subjects}\n")

    # Load 768-D embeddings and baseline EvidentialHead
    embs_file = config.RESULTS_DIR / "embeddings" / "voice_embeddings.pt"
    if not embs_file.exists():
        raise FileNotFoundError(f"Embeddings file not found at: {embs_file}")

    embs_dict = torch.load(embs_file)
    sub_embeddings = embs_dict["subject_embeddings"]

    head = EvidentialHead(in_features=768, num_classes=2, dropout=0.1)
    head.eval()

    # 2. Extract Validation Set Logits and Labels
    val_logits_list = []
    val_labels_list = []

    for sub_id in val_subjects:
        sub_rows = val_df[val_df["subject_id"] == sub_id]
        true_label = int(sub_rows["label"].iloc[0])
        emb = sub_embeddings[sub_id].unsqueeze(0)

        with torch.no_grad():
            evidence = head(emb)  # Shape (1, 2) non-negative evidence/logits

        val_logits_list.append(evidence.squeeze(0).numpy())
        val_labels_list.append(true_label)

    val_logits_np = np.array(val_logits_list)
    val_labels_np = np.array(val_labels_list)

    print(f"Extracted {len(val_logits_np)} Validation subject logits.")

    # 3. Fit Temperature Scaler T strictly on Validation data
    scaler = TemperatureScaler()
    learned_T = scaler.fit(val_logits_np, val_labels_np)

    print(f"\n  --> LEARNED VALIDATION TEMPERATURE T = {learned_T:.6f}")
    print(f"  --> Fitted strictly on {len(val_subjects)} Validation subjects with zero test label usage.")

    # Save temperature.json
    temp_info = {
        "temperature": learned_T,
        "fitted_on": "validation",
        "evaluation_on": "test",
        "num_validation_subjects": len(val_subjects),
        "num_test_subjects": len(test_subjects),
        "seed": seed,
    }
    with open(TEMP_JSON_PATH, "w") as f:
        json.dump(temp_info, f, indent=4)
    print(f"Saved temperature config to: {TEMP_JSON_PATH}")

    # 4. Extract Test Set Logits and Probabilities
    # Load baseline test predictions for baseline probabilities
    sub_pred_csv = config.RESULTS_DIR / "subject_predictions.csv"
    if sub_pred_csv.exists():
        sub_pred_df = pd.read_csv(sub_pred_csv)
    else:
        raise FileNotFoundError(f"Baseline predictions not found at: {sub_pred_csv}")

    test_records = []
    for sub_id in test_subjects:
        row = sub_pred_df[sub_pred_df["subject_id"] == sub_id].iloc[0]
        true_label = int(row["true_label"])
        p_pd_uncal = float(row["pd_probability"])
        p_hc_uncal = 1.0 - p_pd_uncal

        # Reconstruct baseline test logit difference Delta e = log(P_PD / P_HC)
        eps = 1e-12
        delta_e = float(np.log(np.clip(p_pd_uncal, eps, 1.0 - eps) / np.clip(p_hc_uncal, eps, 1.0 - eps)))
        test_logits_2d = np.array([[0.0, delta_e]])

        # Uncalibrated confidence & correctness
        conf_uncal = max(p_hc_uncal, p_pd_uncal)
        pred_uncal = 1 if p_pd_uncal >= 0.5 else 0
        correct_uncal = int(pred_uncal == true_label)

        # Temperature-scaled probabilities: P_cal = sigmoid(Delta e / learned_T)
        probs_cal_2d = scaler.calibrate_probabilities(test_logits_2d, T=learned_T)
        p_hc_cal = float(probs_cal_2d[0, 0])
        p_pd_cal = float(probs_cal_2d[0, 1])

        conf_cal = max(p_hc_cal, p_pd_cal)
        pred_cal = 1 if p_pd_cal >= 0.5 else 0
        correct_cal = int(pred_cal == true_label)

        test_records.append({
            "subject_id": sub_id,
            "true_label": true_label,
            "uncalibrated_p_hc": p_hc_uncal,
            "uncalibrated_p_pd": p_pd_uncal,
            "uncalibrated_confidence": conf_uncal,
            "uncalibrated_prediction": pred_uncal,
            "uncalibrated_is_correct": correct_uncal,
            "temperature": learned_T,
            "calibrated_p_hc": p_hc_cal,
            "calibrated_p_pd": p_pd_cal,
            "calibrated_confidence": conf_cal,
            "calibrated_prediction": pred_cal,
            "calibrated_is_correct": correct_cal,
        })

    sub_test_df = pd.DataFrame(test_records)
    sub_test_df.to_csv(CALIB_RESULTS_CSV, index=False)
    print(f"Saved calibration test predictions to: {CALIB_RESULTS_CSV}")

    # 5. Compute Test Set Evaluation Metrics BEFORE vs. AFTER Calibration
    y_test_true = sub_test_df["true_label"].values
    uncal_p_pd = sub_test_df["uncalibrated_p_pd"].values
    cal_p_pd = sub_test_df["calibrated_p_pd"].values

    uncal_preds = sub_test_df["uncalibrated_prediction"].values
    cal_preds = sub_test_df["calibrated_prediction"].values

    # Uncalibrated Test Metrics (T = 1.0)
    acc_uncal = float(accuracy_score(y_test_true, uncal_preds))
    bal_acc_uncal = float(balanced_accuracy_score(y_test_true, uncal_preds))
    auc_uncal = float(roc_auc_score(y_test_true, uncal_p_pd))
    nll_uncal = calculate_nll(y_test_true, uncal_p_pd)
    brier_uncal = calculate_brier(y_test_true, uncal_p_pd)
    ece_uncal = calculate_ece(sub_test_df["uncalibrated_confidence"].values, sub_test_df["uncalibrated_is_correct"].values, n_bins=10)

    # Calibrated Test Metrics (T = learned_T)
    acc_cal = float(accuracy_score(y_test_true, cal_preds))
    bal_acc_cal = float(balanced_accuracy_score(y_test_true, cal_preds))
    auc_cal = float(roc_auc_score(y_test_true, cal_p_pd))
    nll_cal = calculate_nll(y_test_true, cal_p_pd)
    brier_cal = calculate_brier(y_test_true, cal_p_pd)
    ece_cal = calculate_ece(sub_test_df["calibrated_confidence"].values, sub_test_df["calibrated_is_correct"].values, n_bins=10)

    uncal_metrics = {
        "accuracy": acc_uncal,
        "balanced_accuracy": bal_acc_uncal,
        "roc_auc": auc_uncal,
        "nll": nll_uncal,
        "brier": brier_uncal,
        "ece": ece_uncal,
    }

    cal_metrics = {
        "accuracy": acc_cal,
        "balanced_accuracy": bal_acc_cal,
        "roc_auc": auc_cal,
        "nll": nll_cal,
        "brier": brier_cal,
        "ece": ece_cal,
    }

    full_calib_summary = {
        "experiment": "Voice Wav2Vec2 Post-Hoc Temperature Scaling",
        "modality": "Voice",
        "seed": seed,
        "temperature": learned_T,
        "num_validation_subjects": len(val_subjects),
        "num_test_subjects": len(test_subjects),
        "uncalibrated_metrics_t1": uncal_metrics,
        "calibrated_metrics_t_learned": cal_metrics,
    }

    with open(CALIB_RESULTS_JSON, "w") as f:
        json.dump(full_calib_summary, f, indent=4)
    print(f"Saved calibration summary JSON to: {CALIB_RESULTS_JSON}")

    # 6. Generate Reliability Diagram & Comparison Plots
    plot_reliability_diagram(
        sub_test_df,
        save_path=RELIABILITY_PLOT_PATH,
        title=f"Voice Test Reliability Diagram\n(Uncalibrated T=1.0 vs. Calibrated T={learned_T:.4f})"
    )
    print(f"Saved Reliability Diagram to: {RELIABILITY_PLOT_PATH}")

    plot_calibration_comparison(
        uncal_metrics,
        cal_metrics,
        save_path=METRICS_PLOT_PATH,
        title=f"Voice Calibration Metrics (T={learned_T:.4f})"
    )
    print(f"Saved Calibration Metrics Chart to: {METRICS_PLOT_PATH}")

    # 7. Print Results Summary
    print("\n" + "=" * 65)
    print("VOICE TEMPERATURE SCALING TEST EVALUATION SUMMARY")
    print("=" * 65)
    print(f"{'Metric':<25} | {'Uncalibrated (T=1.0)':<20} | {'Calibrated (T=' + f'{learned_T:.4f})':<20}")
    print("-" * 70)
    print(f"{'Accuracy':<25} | {acc_uncal:<20.4f} | {acc_cal:<20.4f}")
    print(f"{'Balanced Accuracy':<25} | {bal_acc_uncal:<20.4f} | {bal_acc_cal:<20.4f}")
    print(f"{'ROC-AUC':<25} | {auc_uncal:<20.4f} | {auc_cal:<20.4f}")
    print(f"{'NLL (Cross-Entropy)':<25} | {nll_uncal:<20.4f} | {nll_cal:<20.4f}")
    print(f"{'Brier Score':<25} | {brier_uncal:<20.4f} | {brier_cal:<20.4f}")
    print(f"{'ECE (10 Bins)':<25} | {ece_uncal:<20.4f} | {ece_cal:<20.4f}")
    print("=" * 65)

    return full_calib_summary


if __name__ == "__main__":
    run_temperature_scaling_experiment()
