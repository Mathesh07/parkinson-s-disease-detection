"""Unit tests for Voice Temperature Scaling, calibration metrics (NLL, Brier, ECE), and leakage isolation."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

# Ensure workspace root and Voice directory are in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
voice_dir = root_dir / "Voice"
for p in (str(root_dir), str(voice_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from Voice.calibration import (
    TemperatureScaler,
    calculate_brier,
    calculate_ece,
    calculate_nll,
)
from Voice.model import Wav2Vec2ForParkinsons


def test_temperature_remains_positive():
    """Requirement 1: Verify Temperature T is always strictly positive (> 0)."""
    scaler = TemperatureScaler(temperature=1.5)
    assert scaler.temperature > 0.0

    # Test fitting enforcing T > 0
    val_logits = np.array([[2.0, 1.0], [0.5, 3.0]])
    val_labels = np.array([0, 1])
    fitted_t = scaler.fit(val_logits, val_labels)
    assert fitted_t > 0.0
    assert scaler.temperature > 0.0


def test_logits_temperature_division():
    """Requirement 2: Verify logits / T calculation is mathematically exact."""
    scaler = TemperatureScaler(temperature=2.0)
    logits = np.array([[4.0, 2.0], [1.0, 3.0]])
    scaled = scaler.calibrate_logits(logits)
    assert np.allclose(scaled, np.array([[2.0, 1.0], [0.5, 1.5]]))

    # PyTorch Tensor check
    logits_t = torch.tensor([[4.0, 2.0], [1.0, 3.0]])
    scaled_t = scaler.calibrate_logits(logits_t)
    assert torch.allclose(scaled_t, torch.tensor([[2.0, 1.0], [0.5, 1.5]]))


def test_temperature_fitting_on_synthetic_validation_logits():
    """Requirement 3: Verify T can be fitted on synthetic validation set logits."""
    scaler = TemperatureScaler()
    # Overconfident validation logits
    val_logits = np.array([
        [10.0, 0.0],
        [8.0, 1.0],
        [0.0, 9.0],
        [1.0, 11.0],
    ])
    val_labels = np.array([0, 0, 1, 1])

    best_t = scaler.fit(val_logits, val_labels)
    assert isinstance(best_t, float)
    assert best_t > 0.0


def test_calibration_does_not_modify_model_parameters():
    """Requirement 4: Verify Temperature Scaling does not mutate model weights/parameters."""
    model = Wav2Vec2ForParkinsons(model_name="facebook/wav2vec2-base", num_classes=2)
    model.eval()

    params_before = [p.clone() for p in model.parameters()]

    # Run temperature scaling on model outputs
    dummy_input = torch.randn(2, 160000)
    with torch.no_grad():
        evidence = model(dummy_input)

    scaler = TemperatureScaler()
    val_labels = torch.tensor([0, 1])
    scaler.fit(evidence, val_labels)

    params_after = [p.clone() for p in model.parameters()]

    for p_before, p_after in zip(params_before, params_after):
        assert torch.equal(p_before, p_after), "Model parameter mutated during temperature scaling!"


def test_test_labels_not_used_during_fitting():
    """Requirement 5: Leakage Audit — Verify test set labels are never passed or used to fit T."""
    scaler = TemperatureScaler()
    val_logits = np.array([[2.0, 0.5], [0.1, 3.0]])
    val_labels = np.array([0, 1])

    # Fit strictly on validation data
    scaler.fit(val_logits, val_labels)
    fitted_t = scaler.temperature

    # Verify scaler fitting signature does not accept test arguments
    import inspect
    sig = inspect.signature(scaler.fit)
    params = list(sig.parameters.keys())
    assert "test_logits" not in params
    assert "test_labels" not in params


def test_output_probability_dimensions():
    """Requirement 6: Verify calibrated probability output dimensions and normalization sum=1."""
    scaler = TemperatureScaler(temperature=1.5)
    logits = np.random.randn(5, 2)
    probs = scaler.calibrate_probabilities(logits)

    assert probs.shape == (5, 2)
    assert np.allclose(probs.sum(axis=-1), np.ones(5))
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_ece_calculation_bounds_and_correctness():
    """Requirement 7: Verify ECE calculation produces valid float in [0, 1]."""
    # Perfect calibration -> ECE = 0.0
    confidences = np.array([0.8, 0.8, 0.8, 0.8, 0.8])
    is_correct = np.array([1, 1, 1, 1, 0])  # 4/5 = 0.8 accuracy
    ece_perf = calculate_ece(confidences, is_correct, n_bins=10)
    assert np.isclose(ece_perf, 0.0)

    # Completely uncalibrated -> non-zero ECE
    conf_bad = np.array([0.99, 0.99, 0.99])
    correct_bad = np.array([0, 0, 0])  # Accuracy 0.0 vs confidence 0.99
    ece_bad = calculate_ece(conf_bad, correct_bad, n_bins=10)
    assert np.isclose(ece_bad, 0.99)
    assert 0.0 <= ece_bad <= 1.0


def test_nll_calculation():
    """Requirement 8: Verify NLL (Negative Log Likelihood) calculation."""
    y_true = np.array([0, 1])
    # Perfect probabilities [1, 0] and [0, 1] -> NLL approx 0
    y_probs_perfect = np.array([[0.999999, 0.000001], [0.000001, 0.999999]])
    nll_perf = calculate_nll(y_true, y_probs_perfect)
    assert nll_perf < 1e-4

    # Uncertain probabilities [0.5, 0.5] -> NLL = -log(0.5) = 0.6931
    y_probs_unc = np.array([[0.5, 0.5], [0.5, 0.5]])
    nll_unc = calculate_nll(y_true, y_probs_unc)
    assert np.isclose(nll_unc, -np.log(0.5), atol=1e-4)


def test_brier_score_calculation():
    """Requirement 9: Verify Brier score calculation (MSE between one-hot label and P(PD))."""
    y_true = np.array([0, 1])
    # Perfect predictions -> Brier = 0.0
    y_probs_perf = np.array([0.0, 1.0])
    assert np.isclose(calculate_brier(y_true, y_probs_perf), 0.0)

    # Worst predictions -> Brier = 1.0
    y_probs_worst = np.array([1.0, 0.0])
    assert np.isclose(calculate_brier(y_true, y_probs_worst), 1.0)


def test_reliability_diagram_data_validity():
    """Requirement 10: Verify reliability diagram plot generation produces non-empty files."""
    from Voice.calibration import plot_reliability_diagram, plot_calibration_comparison
    import tempfile

    sub_df = pd.DataFrame({
        "uncalibrated_confidence": [0.6, 0.7, 0.8, 0.9],
        "uncalibrated_is_correct": [1, 1, 0, 1],
        "calibrated_confidence": [0.55, 0.65, 0.75, 0.85],
        "calibrated_is_correct": [1, 1, 0, 1],
        "temperature": [1.2, 1.2, 1.2, 1.2],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        plot_path = Path(tmpdir) / "reliability_test.png"
        plot_reliability_diagram(sub_df, save_path=plot_path)
        assert plot_path.exists()
        assert plot_path.stat().st_size > 1000

        metrics_path = Path(tmpdir) / "metrics_test.png"
        plot_calibration_comparison(
            {"nll": 0.5, "brier": 0.2, "ece": 0.1},
            {"nll": 0.4, "brier": 0.18, "ece": 0.05},
            save_path=metrics_path
        )
        assert metrics_path.exists()
        assert metrics_path.stat().st_size > 1000
