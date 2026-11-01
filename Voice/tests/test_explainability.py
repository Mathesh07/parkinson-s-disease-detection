"""Unit tests for Voice Explainability (2D Grad-CAM & Integrated Gradients)."""

import pytest
import torch
from pathlib import Path

from Voice.baselines.base_paper_cnn import config as cnn_config
from Voice.baselines.base_paper_cnn.model import VoiceBasePaper2DCNN
from Voice.explainability import (
    GradCAM2DSpectrogram,
    compute_voice_integrated_gradients,
    summarize_acoustic_frequency_bands,
    visualize_voice_explainability,
)


@pytest.fixture
def voice_model():
    model = VoiceBasePaper2DCNN()
    model.eval()
    return model


@pytest.fixture
def sample_spectrogram():
    # Shape: (1, 1, 257, 301)
    return torch.randn(1, 1, cnn_config.FREQ_BINS, cnn_config.TIME_FRAMES)


def test_voice_gradcam_2d_shape_and_bounds(voice_model, sample_spectrogram):
    """Verify 2D Grad-CAM produces activation map matching spectrogram dimensions."""
    cam_engine = GradCAM2DSpectrogram(voice_model)
    cam = cam_engine.generate(sample_spectrogram, target_class=1)
    assert cam.shape == (1, cnn_config.FREQ_BINS, cnn_config.TIME_FRAMES)
    assert not torch.isnan(cam).any()
    assert cam.min().item() >= -1e-6
    assert cam.max().item() <= 1.0 + 1e-6


def test_voice_integrated_gradients(voice_model, sample_spectrogram):
    """Verify Voice Integrated Gradients attribution matches input shape."""
    attributions = compute_voice_integrated_gradients(voice_model, sample_spectrogram, target_class=1, steps=5)
    assert attributions.shape == sample_spectrogram.shape
    assert not torch.isnan(attributions).any()
    assert (attributions != 0).any()


def test_summarize_acoustic_frequency_bands(sample_spectrogram):
    """Verify frequency band decomposition metrics sum to 100%."""
    dummy_cam = torch.rand(cnn_config.FREQ_BINS, cnn_config.TIME_FRAMES).numpy()
    summary = summarize_acoustic_frequency_bands(dummy_cam, sample_rate=cnn_config.SAMPLE_RATE, n_fft=cnn_config.N_FFT)
    assert "F0_Pitch_Band_Pct" in summary
    assert "F1_F2_Formant_Band_Pct" in summary
    assert "HighFreq_Breathiness_Band_Pct" in summary
    total_pct = summary["F0_Pitch_Band_Pct"] + summary["F1_F2_Formant_Band_Pct"] + summary["HighFreq_Breathiness_Band_Pct"]
    assert abs(total_pct - 100.0) < 0.1


def test_visualize_voice_explainability(voice_model, sample_spectrogram, tmp_path):
    """Verify end-to-end voice explainability visualization dashboard."""
    out_file = tmp_path / "voice_test_explainability.png"
    device = torch.device("cpu")
    fig, metrics = visualize_voice_explainability(
        sample_spectrogram, voice_model, device, sample_id="TestVoiceSample", save_path=out_file
    )
    assert out_file.exists()
    assert "p_pd" in metrics
    assert "dominant_freq_hz" in metrics
    assert "formant_pct" in metrics
