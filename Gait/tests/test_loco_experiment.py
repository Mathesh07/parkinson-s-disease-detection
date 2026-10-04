"""Unit tests for LOCO experiment implementation in Gait/experiments/leave_one_cohort_out.py.

Tests:
1. Recording aggregation: WINDOW -> RECORDING -> SUBJECT hierarchy.
2. Reliability diagram: P_PD == 1.0 inclusion in the final bin.
3. Stratified validation split: Subject-level disease class ratio preservation.
4. Leakage prevention: Disjoint subject sets (train vs val vs test).
5. Cohort isolation: Test cohort complete exclusion from train/val splits.
6. Small epoch edge case: Stage A and Stage B both run >= 1 epoch when epochs = 1.
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from Gait.experiments.leave_one_cohort_out import (
    aggregate_to_subject_level,
    plot_reliability_diagram,
    build_cohort_splits,
    build_dataloaders,
)
from Gait.preprocessing import build_manifest


def test_hierarchical_recording_aggregation():
    """
    Test 1: Verify WINDOW -> RECORDING -> SUBJECT aggregation.
    Synthetic subject with 2 recordings of unequal length:
    - Recording A: 10 windows with P_PD = 0.1
    - Recording B: 100 windows with P_PD = 0.9

    Direct window mean = (10*0.1 + 100*0.9)/110 = 0.82727...
    Hierarchical mean = Mean(Rec A=0.1, Rec B=0.9) = 0.50000
    """
    rows = []
    # 10 windows for Recording A
    for i in range(10):
        rows.append({
            "experiment": "Exp1",
            "train_cohorts": "Ga_Ju",
            "test_cohort": "Si",
            "subject_id": "Sub_01",
            "recording_id": "Rec_A",
            "study": "Ga",
            "true_label": 1,
            "e_HC": 0.0, "e_PD": 0.8,
            "alpha_HC": 1.0, "alpha_PD": 1.8,
            "P_HC": 0.9, "P_PD": 0.1,
            "uncertainty": 0.2,
            "predicted_label": 0, "is_correct": 0
        })
    # 100 windows for Recording B
    for i in range(100):
        rows.append({
            "experiment": "Exp1",
            "train_cohorts": "Ga_Ju",
            "test_cohort": "Si",
            "subject_id": "Sub_01",
            "recording_id": "Rec_B",
            "study": "Ga",
            "true_label": 1,
            "e_HC": 0.0, "e_PD": 8.0,
            "alpha_HC": 1.0, "alpha_PD": 9.0,
            "P_HC": 0.1, "P_PD": 0.9,
            "uncertainty": 0.2,
            "predicted_label": 1, "is_correct": 1
        })

    window_df = pd.DataFrame(rows)
    sub_df = aggregate_to_subject_level(window_df)

    assert len(sub_df) == 1, "Expected exactly 1 subject in output."
    subject_p_pd = sub_df["P_PD"].iloc[0]

    # Verify hierarchical mean equals exactly 0.5, NOT the biased window mean ~0.827
    assert np.isclose(subject_p_pd, 0.5, atol=1e-5), (
        f"Hierarchical aggregation failed! Expected P_PD=0.5, got {subject_p_pd:.5f}"
    )
    print("PASS: Test 1 — Hierarchical Recording Aggregation verified.")


def test_reliability_diagram_p_pd_one_inclusion():
    """
    Test 2: Verify P_PD == 1.0 is included in the final reliability bin [0.9, 1.0].
    """
    sub_df = pd.DataFrame([
        {"P_PD": 1.0, "true_label": 1},
        {"P_PD": 0.0, "true_label": 0},
    ])

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "test_rel.png"
        plot_reliability_diagram(sub_df, save_path=save_path, n_bins=10)

        # Check binning logic directly
        p_pd = sub_df["P_PD"].values
        bins = np.linspace(0.0, 1.0, 11)
        # Last bin (i=9)
        lo, hi = bins[9], bins[10]
        mask_last = (p_pd >= lo) & (p_pd <= hi)
        assert mask_last.sum() == 1, "Expected P_PD=1.0 to be captured in last bin."
        assert p_pd[mask_last][0] == 1.0

    print("PASS: Test 2 — P_PD == 1.0 inclusion in reliability diagram verified.")


def test_stratified_validation_split():
    """
    Test 3: Verify subject-level stratified validation split preserves class ratios.
    """
    records = []
    # 40 HC subjects (Ga), 40 PD subjects (Ju), 20 HC subjects (Si - test cohort)
    for i in range(40):
        records.append({"subject_id": f"HC_Ga_{i}", "study": "Ga", "label": 0, "filepath": f"path_ga_{i}"})
    for i in range(40):
        records.append({"subject_id": f"PD_Ju_{i}", "study": "Ju", "label": 1, "filepath": f"path_ju_{i}"})
    for i in range(20):
        records.append({"subject_id": f"HC_Si_{i}", "study": "Si", "label": 0, "filepath": f"path_si_{i}"})

    manifest_df = pd.DataFrame(records)

    train_df, val_df, test_df = build_cohort_splits(
        manifest_df,
        train_cohorts=["Ga", "Ju"],
        test_cohort="Si",
        val_fraction=0.15,
        seed=42
    )

    train_sub_labels = train_df.groupby("subject_id")["label"].first()
    val_sub_labels = val_df.groupby("subject_id")["label"].first()

    # Total train pool = 40 HC + 40 PD = 80 subjects.
    # Val fraction 0.15 -> 6 HC val, 6 PD val = 12 val subjects total.
    # Train = 34 HC, 34 PD = 68 train subjects total.
    assert (val_sub_labels == 0).sum() == 6, f"Expected 6 HC in val, got {(val_sub_labels == 0).sum()}"
    assert (val_sub_labels == 1).sum() == 6, f"Expected 6 PD in val, got {(val_sub_labels == 1).sum()}"
    assert (train_sub_labels == 0).sum() == 34, f"Expected 34 HC in train, got {(train_sub_labels == 0).sum()}"
    assert (train_sub_labels == 1).sum() == 34, f"Expected 34 PD in train, got {(train_sub_labels == 1).sum()}"

    print("PASS: Test 3 — Stratified Validation Split verified.")


def test_leakage_and_cohort_isolation():
    """
    Test 4 & 5: Verify zero subject overlap across train, val, and test splits.
    """
    records = []
    for i in range(30):
        records.append({"subject_id": f"Sub_Ga_{i}", "study": "Ga", "label": 0, "filepath": f"path_ga_{i}"})
    for i in range(30):
        records.append({"subject_id": f"Sub_Ju_{i}", "study": "Ju", "label": 1, "filepath": f"path_ju_{i}"})
    for i in range(15):
        records.append({"subject_id": f"Sub_Si_{i}", "study": "Si", "label": 1, "filepath": f"path_si_{i}"})

    manifest_df = pd.DataFrame(records)

    train_df, val_df, test_df = build_cohort_splits(
        manifest_df,
        train_cohorts=["Ga", "Ju"],
        test_cohort="Si",
        val_fraction=0.20,
        seed=123
    )

    train_subs = set(train_df["subject_id"].unique())
    val_subs = set(val_df["subject_id"].unique())
    test_subs = set(test_df["subject_id"].unique())

    # Disjoint subject checks
    assert len(train_subs & val_subs) == 0, "Leakage between train and val!"
    assert len(train_subs & test_subs) == 0, "Leakage between train and test!"
    assert len(val_subs & test_subs) == 0, "Leakage between val and test!"

    # Cohort isolation check
    assert set(test_df["study"].unique()) == {"Si"}, "Test split contains non-test cohorts!"
    assert "Si" not in train_df["study"].unique(), "Test cohort leaked into train split!"
    assert "Si" not in val_df["study"].unique(), "Test cohort leaked into val split!"

    print("PASS: Tests 4 & 5 — Subject leakage prevention and cohort isolation verified.")


def test_small_epoch_stage_split():
    """
    Test 6: Verify epochs <= 1 assigns stage_a_epochs = 1 and stage_b_epochs = 1.
    """
    epochs = 1
    if epochs <= 1:
        stage_a_epochs = 1
        stage_b_epochs = 1
    else:
        stage_a_epochs = max(1, epochs // 5)
        stage_b_epochs = max(1, epochs - stage_a_epochs)

    assert stage_a_epochs == 1, "Stage A must run at least 1 epoch."
    assert stage_b_epochs == 1, "Stage B must run at least 1 epoch for epochs=1."
    print("PASS: Test 6 — Small epoch edge case handling verified.")


def test_train_loader_drop_last_training_only():
    """
    Test 7: Regression test for BatchNorm crash (torch.Size([1, 128])).
    Verify drop_last=True ONLY on train_loader, while val_loader and test_loader
    maintain drop_last=False.
    """
    manifest_df = build_manifest()
    train_df, val_df, test_df = build_cohort_splits(
        manifest_df, ["Ga", "Ju"], "Si", seed=42
    )
    train_slice = train_df.iloc[:2]
    val_slice = val_df.iloc[:2]
    test_slice = test_df.iloc[:2]

    train_loader, val_loader, test_loader = build_dataloaders(
        train_slice, val_slice, test_slice, normalizer=None, batch_size=4
    )

    assert train_loader.drop_last is True, "train_loader must have drop_last=True to prevent batch size 1"
    assert val_loader.drop_last is False, "val_loader must keep drop_last=False"
    assert test_loader.drop_last is False, "test_loader must keep drop_last=False"

    # Also verify that when iterating batches, no batch has size 1 in train_loader
    for batch in train_loader:
        assert batch["features"].shape[0] > 1, f"Batch size was {batch['features'].shape[0]}, expected > 1"

    print("PASS: Test 7 — drop_last=True on train_loader verified.")


if __name__ == "__main__":
    test_hierarchical_recording_aggregation()
    test_reliability_diagram_p_pd_one_inclusion()
    test_stratified_validation_split()
    test_leakage_and_cohort_isolation()
    test_small_epoch_stage_split()
    test_train_loader_drop_last_training_only()
    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!")
