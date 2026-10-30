"""Unit tests for Voice baseline subject leakage prevention."""

import pytest

from Voice.baselines.base_paper_cnn import config
from Voice.baselines.base_paper_cnn.dataset import build_voice_metadata, create_subject_splits


def test_zero_voice_subject_leakage():
    """Verify zero subject overlap across Train, Val, and Test sets."""
    df = build_voice_metadata()
    train_df, val_df, test_df = create_subject_splits(df, seed=config.RANDOM_SEED)

    train_subs = set(train_df["subject_id"].unique())
    val_subs = set(val_df["subject_id"].unique())
    test_subs = set(test_df["subject_id"].unique())

    assert len(train_subs & val_subs) == 0, "Voice Train and Val share subjects!"
    assert len(train_subs & test_subs) == 0, "Voice Train and Test share subjects!"
    assert len(val_subs & test_subs) == 0, "Voice Val and Test share subjects!"

    # Verify held-out test subjects match existing Wav2Vec2 test set exactly
    expected_test_subs = {"ID04", "ID10", "ID13", "ID23", "ID29", "ID35"}
    assert test_subs == expected_test_subs, f"Expected {expected_test_subs}, got {test_subs}"
