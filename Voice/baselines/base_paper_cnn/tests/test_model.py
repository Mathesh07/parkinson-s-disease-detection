"""Unit tests for Voice Base-Paper 2D CNN Model Architecture."""

import pytest
import torch

from Voice.baselines.base_paper_cnn import config
from Voice.baselines.base_paper_cnn.model import VoiceBasePaper2DCNN


def test_voice_model_forward():
    """Verify forward pass on (B, 1, 257, 301)."""
    model = VoiceBasePaper2DCNN()
    x = torch.randn(2, 1, config.FREQ_BINS, config.TIME_FRAMES)
    logits = model(x)

    assert logits.shape == (2, 2)
    probs = model.predict_proba(x)
    assert probs.shape == (2, 2)
    assert torch.allclose(torch.sum(probs, dim=-1), torch.ones(2), atol=1e-5)


def test_voice_model_transitions():
    """Verify forward pass on transition input shape (B, 1, 129, 33)."""
    model = VoiceBasePaper2DCNN()
    x = torch.randn(2, 1, config.TRANSITION_FREQ_BINS, config.TRANSITION_TIME_FRAMES)
    logits = model(x)
    assert logits.shape == (2, 2)
