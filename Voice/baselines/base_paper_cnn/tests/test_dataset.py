"""Unit tests for Voice Base-Paper 2D CNN Dataset and STFT Transformations."""

import numpy as np
import pytest
import torch

from Voice.baselines.base_paper_cnn import config
from Voice.baselines.base_paper_cnn.dataset import extract_subject_id
from Voice.baselines.base_paper_cnn.preprocessing import (
    VoiceSTFTNormalizer,
    compute_stft_spectrogram,
    detect_speech_transitions,
)


def test_subject_id_extraction():
    """Verify regex subject ID extraction."""
    assert extract_subject_id("ID04_pd_2_0_1.wav") == "ID04"
    assert extract_subject_id("ID10_hc_0_0_0.wav") == "ID10"
    assert extract_subject_id("ReadText_ID23_hc.wav") == "ID23"


def test_stft_spectrogram_dimensions():
    """Verify continuous audio STFT shape on 3.0s chunk."""
    dummy_chunk = np.random.randn(config.CHUNK_SAMPLES).astype(np.float32)
    spec = compute_stft_spectrogram(dummy_chunk)

    assert isinstance(spec, torch.Tensor)
    assert spec.shape[0] == 1
    assert spec.shape[1] == config.FREQ_BINS  # 257
    assert spec.shape[2] == config.TIME_FRAMES  # 301
    assert not torch.isnan(spec).any()


def test_transition_detection():
    """Verify speech transition detection produces onset/offset tensors."""
    sr = config.SAMPLE_RATE
    # Generate 1.0s synthetic signal: silence (unvoiced) followed by sine wave (voiced)
    silence = np.zeros(int(sr * 0.5), dtype=np.float32)
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    signal = np.concatenate([silence, tone])

    onsets, offsets = detect_speech_transitions(signal, sr=sr)
    assert isinstance(onsets, list)
    assert isinstance(offsets, list)
    if onsets:
        assert onsets[0].shape == (1, config.TRANSITION_FREQ_BINS, config.TRANSITION_TIME_FRAMES)
