"""Unit tests for Handwriting dataset processing and zero-leakage patient grouping."""

import pytest
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure paths
root_dir = Path(__file__).resolve().parent.parent.parent
hw_dir = root_dir / "Handwriting"
for p in (str(root_dir), str(hw_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from Handwriting.dataset import create_patient_level_split, summarize_dataset


def test_summarize_dataset_synthetic():
    """Verify summarize_dataset correctly calculates counts and patient statistics."""
    # Synthetic manifest with 4 patients (2 HC, 2 PD)
    data = {
        "ID_PATIENT": [101, 101, 102, 201, 202, 202],
        "IMAGE_NAME": ["sp_1.jpg", "me_1.jpg", "sp_2.jpg", "sp_3.jpg", "me_2.jpg", "sp_4.jpg"],
        "CLASS_TYPE": [1, 1, 1, 2, 2, 2],
        "image_exists": [True, True, True, True, True, True],
        "target": [0, 0, 0, 1, 1, 1]
    }
    df = pd.DataFrame(data)
    summary = summarize_dataset(df)

    assert summary["unique_patients"] == 4
    assert summary["total_images"] == 6
    assert summary["unique_classes"] == [1, 2]
    assert summary["missing_images"] == 0
    assert summary["patients_with_multiple_images"] == 2


def test_zero_patient_leakage_in_splits():
    """Verify that patient IDs never overlap across train, val, and test splits."""
    # Create 20 synthetic patients
    records = []
    for pid in range(1, 21):
        label = 2 if pid > 10 else 1
        target = 1 if label == 2 else 0
        # 2 images per patient
        records.append({"ID_PATIENT": pid, "CLASS_TYPE": label, "target": target, "image_name": f"img_{pid}_1"})
        records.append({"ID_PATIENT": pid, "CLASS_TYPE": label, "target": target, "image_name": f"img_{pid}_2"})

    manifest = pd.DataFrame(records)
    split_df = create_patient_level_split(manifest, test_fold=0, val_fold=1)

    train_patients = set(split_df[split_df["split"] == "train"]["ID_PATIENT"])
    val_patients = set(split_df[split_df["split"] == "val"]["ID_PATIENT"])
    test_patients = set(split_df[split_df["split"] == "test"]["ID_PATIENT"])

    # Verify zero intersection
    assert len(train_patients.intersection(val_patients)) == 0, "Patient leakage between Train and Val"
    assert len(train_patients.intersection(test_patients)) == 0, "Patient leakage between Train and Test"
    assert len(val_patients.intersection(test_patients)) == 0, "Patient leakage between Val and Test"
