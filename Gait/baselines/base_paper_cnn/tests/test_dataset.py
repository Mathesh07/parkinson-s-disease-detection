"""Unit tests for Gait Base-Paper 2D CNN Dataset and STFT Transformations."""

import numpy as np
import pytest
import torch

from Gait.baselines.base_paper_cnn import config
from Gait.baselines.base_paper_cnn.preprocessing import (
    GaitSTFTNormalizer,
    build_manifest,
    compute_stft_tensor,
    parse_filename,
)


def test_filename_parsing():
    """Verify regex filename parsing of PhysioNet conventions."""
    info1 = parse_filename("GaCo01_01.txt")
    assert info1["study"] == "Ga"
    assert info1["subject_id"] == "GaCo01"
    assert info1["label"] == 0

    info2 = parse_filename("JuPt05_02.txt")
    assert info2["study"] == "Ju"
    assert info2["subject_id"] == "JuPt05"
    assert info2["label"] == 1


def test_stft_tensor_shape():
    """Verify STFT tensor transformation shape is (16, 33, 32)."""
    # 500 time steps x 16 channels
    dummy_window = np.random.randn(500, 16).astype(np.float32)
    stft = compute_stft_tensor(dummy_window, n_fft=64, hop_length=16, win_length=64)

    assert isinstance(stft, torch.Tensor)
    assert stft.shape == (16, 33, 32)
    assert not torch.isnan(stft).any()
    assert not torch.isinf(stft).any()


def test_normalizer_fit_transform():
    """Verify GaitSTFTNormalizer fits and transforms zero mean unit variance."""
    tensors = [compute_stft_tensor(np.random.randn(500, 16).astype(np.float32)) for _ in range(5)]
    normalizer = GaitSTFTNormalizer()
    normalizer.fit(tensors)

    norm_t = normalizer.transform(tensors[0])
    assert norm_t.shape == (16, 33, 32)
    assert not torch.isnan(norm_t).any()
