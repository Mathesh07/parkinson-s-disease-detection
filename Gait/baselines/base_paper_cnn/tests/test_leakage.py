"""Unit tests verifying zero subject leakage across train, val, and test splits."""

import pytest

from Gait.baselines.base_paper_cnn import config
from Gait.baselines.base_paper_cnn.dataset import create_subject_splits
from Gait.baselines.base_paper_cnn.preprocessing import build_manifest


def test_zero_subject_leakage():
    """Verify disjoint subject IDs across all splits."""
    manifest_df = build_manifest()
    train_df, val_df, test_df = create_subject_splits(manifest_df, seed=config.RANDOM_SEED)

    train_subs = set(train_df["subject_id"].unique())
    val_subs = set(val_df["subject_id"].unique())
    test_subs = set(test_df["subject_id"].unique())

    assert len(train_subs & val_subs) == 0, "Train and Val share subjects!"
    assert len(train_subs & test_subs) == 0, "Train and Test share subjects!"
    assert len(val_subs & test_subs) == 0, "Val and Test share subjects!"

    total_subs = manifest_df["subject_id"].nunique()
    assert len(train_subs) + len(val_subs) + len(test_subs) == total_subs
