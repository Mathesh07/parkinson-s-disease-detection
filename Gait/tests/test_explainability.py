"""Unit tests for Gait Explainability (Integrated Gradients & 1D Grad-CAM)."""

import pytest
import torch
from pathlib import Path

from Gait import config
from Gait.model import GaitCNNBiLSTM
from Gait.explainability import (
    compute_integrated_gradients,
    GradCAM1D,
    summarize_sensor_attribution,
    visualize_gait_explainability,
)


@pytest.fixture
def gait_model():
    model = GaitCNNBiLSTM()
    model.eval()
    return model


@pytest.fixture
def sample_vgrf_window():
    # Shape: (1, 500, 16)
    return torch.randn(1, config.WINDOW_SIZE, config.NUM_CHANNELS)


def test_integrated_gradients_shape_and_attribution(gait_model, sample_vgrf_window):
    """Verify Integrated Gradients preserves shape and outputs non-trivial attributions."""
    attributions = compute_integrated_gradients(gait_model, sample_vgrf_window, target_class=1, steps=5)
    assert attributions.shape == sample_vgrf_window.shape
    assert not torch.isnan(attributions).any()
    assert (attributions != 0).any()


def test_gradcam_1d_shape_and_bounds(gait_model, sample_vgrf_window):
    """Verify 1D Grad-CAM produces activation map matching sequence length in [0, 1]."""
    gradcam = GradCAM1D(gait_model)
    cam = gradcam.generate(sample_vgrf_window, target_class=1)
    assert cam.shape == (1, config.WINDOW_SIZE)
    assert not torch.isnan(cam).any()
    assert cam.min().item() >= -1e-6
    assert cam.max().item() <= 1.0 + 1e-6


def test_summarize_sensor_attribution(sample_vgrf_window):
    """Verify bilateral sensor scores and asymmetry index calculation."""
    dummy_attr = torch.randn_like(sample_vgrf_window)
    summary = summarize_sensor_attribution(dummy_attr)
    assert "sensor_scores" in summary
    assert len(summary["sensor_scores"]) == 16
    assert "asymmetry_index" in summary
    assert 0.0 <= summary["asymmetry_index"] <= 1.0
    assert summary["dominant_sensor"] in config.SENSOR_NAMES


def test_visualize_gait_explainability(gait_model, sample_vgrf_window, tmp_path):
    """Verify end-to-end visualization dashboard generation and saving."""
    out_file = tmp_path / "gait_test_explainability.png"
    device = torch.device("cpu")
    fig, metrics = visualize_gait_explainability(
        sample_vgrf_window, gait_model, device, sample_id="TestSample", save_path=out_file
    )
    assert out_file.exists()
    assert "p_pd" in metrics
    assert "vacuity" in metrics
    assert "asymmetry_index" in metrics
