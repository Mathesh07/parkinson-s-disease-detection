"""Tests for Chunk -> Recording -> Subject hierarchical prediction aggregation."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure workspace root and Voice directory are in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
voice_dir = root_dir / "Voice"
for p in (str(root_dir), str(voice_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from Voice import config
from Voice.utils import aggregate_predictions, calculate_metrics


def test_chunk_to_recording_and_subject_aggregation():
    """
    Verify exact hierarchical aggregation:
      Chunk probabilities -> mean -> Recording probability
      Recording probabilities -> mean -> Subject probability
      Decision rule: P(PD) >= 0.5 -> 1 (PD), else 0 (HC)
    """
    # Create synthetic chunk predictions for 2 subjects, each having 2 recordings
    # Subject ID01 (HC, true_label=0):
    #   Recording 1 (ReadText): chunks = [0.2, 0.4, 0.6] -> mean = 0.4
    #   Recording 2 (Spontaneous): chunks = [0.1, 0.3] -> mean = 0.2
    #   Subject ID01 aggregated P(PD) = (0.4 + 0.2) / 2 = 0.3 -> predicted_label = 0 (HC)
    #
    # Subject ID02 (PD, true_label=1):
    #   Recording 1 (ReadText): chunks = [0.8, 0.9, 0.7] -> mean = 0.8
    #   Recording 2 (Spontaneous): chunks = [0.6, 0.6] -> mean = 0.6
    #   Subject ID02 aggregated P(PD) = (0.8 + 0.6) / 2 = 0.7 -> predicted_label = 1 (PD)

    chunk_data = [
        # Subject ID01 - Recording 1
        {"recording_id": "ID01_rec1.wav", "subject_id": "ID01", "task": "ReadText", "true_label": 0, "pd_probability": 0.2},
        {"recording_id": "ID01_rec1.wav", "subject_id": "ID01", "task": "ReadText", "true_label": 0, "pd_probability": 0.4},
        {"recording_id": "ID01_rec1.wav", "subject_id": "ID01", "task": "ReadText", "true_label": 0, "pd_probability": 0.6},
        # Subject ID01 - Recording 2
        {"recording_id": "ID01_rec2.wav", "subject_id": "ID01", "task": "SpontaneousDialogue", "true_label": 0, "pd_probability": 0.1},
        {"recording_id": "ID01_rec2.wav", "subject_id": "ID01", "task": "SpontaneousDialogue", "true_label": 0, "pd_probability": 0.3},
        # Subject ID02 - Recording 1
        {"recording_id": "ID02_rec1.wav", "subject_id": "ID02", "task": "ReadText", "true_label": 1, "pd_probability": 0.8},
        {"recording_id": "ID02_rec1.wav", "subject_id": "ID02", "task": "ReadText", "true_label": 1, "pd_probability": 0.9},
        {"recording_id": "ID02_rec1.wav", "subject_id": "ID02", "task": "ReadText", "true_label": 1, "pd_probability": 0.7},
        # Subject ID02 - Recording 2
        {"recording_id": "ID02_rec2.wav", "subject_id": "ID02", "task": "SpontaneousDialogue", "true_label": 1, "pd_probability": 0.6},
        {"recording_id": "ID02_rec2.wav", "subject_id": "ID02", "task": "SpontaneousDialogue", "true_label": 1, "pd_probability": 0.6},
    ]

    chunk_df = pd.DataFrame(chunk_data)
    rec_df, sub_df = aggregate_predictions(chunk_df)

    # 1. Check Recording-level predictions
    rec1_id01 = rec_df[rec_df["recording_id"] == "ID01_rec1.wav"].iloc[0]
    rec2_id01 = rec_df[rec_df["recording_id"] == "ID01_rec2.wav"].iloc[0]
    assert np.isclose(rec1_id01["pd_probability"], 0.4)
    assert rec1_id01["predicted_label"] == 0
    assert np.isclose(rec2_id01["pd_probability"], 0.2)
    assert rec2_id01["predicted_label"] == 0

    rec1_id02 = rec_df[rec_df["recording_id"] == "ID02_rec1.wav"].iloc[0]
    rec2_id02 = rec_df[rec_df["recording_id"] == "ID02_rec2.wav"].iloc[0]
    assert np.isclose(rec1_id02["pd_probability"], 0.8)
    assert rec1_id02["predicted_label"] == 1
    assert np.isclose(rec2_id02["pd_probability"], 0.6)
    assert rec2_id02["predicted_label"] == 1

    # 2. Check Subject-level predictions
    sub_id01 = sub_df[sub_df["subject_id"] == "ID01"].iloc[0]
    sub_id02 = sub_df[sub_df["subject_id"] == "ID02"].iloc[0]

    assert np.isclose(sub_id01["pd_probability"], 0.3)
    assert sub_id01["predicted_label"] == 0
    assert sub_id01["predicted_name"] == "Healthy Control (HC)"

    assert np.isclose(sub_id02["pd_probability"], 0.7)
    assert sub_id02["predicted_label"] == 1
    assert sub_id02["predicted_name"] == "Parkinson's Disease (PD)"


def test_calculate_metrics_functionality():
    """Verify metric calculations for accuracy, precision, recall, F1, sensitivity, specificity, and ROC-AUC."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    y_scores = [0.1, 0.6, 0.8, 0.9]

    metrics = calculate_metrics(y_true, y_pred, y_scores)

    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 0.5
    assert metrics["f1_score"] == 0.8
    assert metrics["roc_auc"] == 1.0
    assert metrics["tp"] == 2
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 0
