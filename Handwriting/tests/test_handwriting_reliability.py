"""Unit tests for Handwriting Reliability Engine.

Tests:
- Conformal prediction set construction (singleton, ambiguous, empty)
- Dirichlet uncertainty / vacuity filtering
- Decision logic: ACCEPT (HIGH) vs FLAG (MEDIUM / LOW)
- Borderline probability identification
- Serialization and schema compliance
"""

import pytest
import sys
from pathlib import Path

# Ensure paths
root_dir = Path(__file__).resolve().parent.parent.parent
hw_dir = root_dir / "Handwriting"
for p in (str(root_dir), str(hw_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from Handwriting.reliability_engine import HandwritingReliabilityEngine


def test_handwriting_reliability_accept_high():
    """Verify strong singleton prediction with low uncertainty yields ACCEPT (HIGH)."""
    engine = HandwritingReliabilityEngine(q_hat=0.45, uncertainty_threshold=0.45)
    
    # Confident PD: p_pd = 0.88, p_hc = 0.12, low uncertainty = 0.25
    result = engine.evaluate_reliability(
        p_pd=0.88,
        uncertainty=0.25,
        evidence=(0.2, 7.8)
    )

    assert result["modality"] == "Handwriting"
    assert result["prediction"] == "Parkinson's Disease"
    assert result["predicted_label"] == 1
    assert result["conformal_prediction_set"] == ["Parkinson's Disease"]
    assert result["conformal_set_size"] == 1
    assert result["reliability"] == "ACCEPT"
    assert result["reliability_level"] == "HIGH"
    assert "CONSISTENT_SINGLETON_PREDICTION" in result["reliability_reasons"]


def test_handwriting_reliability_flag_vacuity():
    """Verify elevated Dirichlet vacuity triggers UNCERTAINTY_EXCEEDS_THRESHOLD flag."""
    engine = HandwritingReliabilityEngine(q_hat=0.45, uncertainty_threshold=0.45)
    
    # High uncertainty: u = 0.65
    result = engine.evaluate_reliability(
        p_pd=0.85,
        uncertainty=0.65,
        evidence=(0.1, 0.9)
    )

    assert result["reliability"] == "FLAG"
    assert (
        "HIGH_EVIDENTIAL_UNCERTAINTY" in result["reliability_reasons"]
        or "UNCERTAINTY_EXCEEDS_THRESHOLD" in result["reliability_reasons"]
    )
    assert result["reliability_level"] in ["MEDIUM", "LOW"]


def test_handwriting_reliability_flag_empty_conformal_set():
    """Verify empty conformal set triggers EMPTY_CONFORMAL_SET flag."""
    # When q_hat = 0.40, threshold is 1 - 0.40 = 0.60
    # Both p_pd = 0.52 and p_hc = 0.48 are below 0.60 -> empty set
    engine = HandwritingReliabilityEngine(q_hat=0.40, uncertainty_threshold=0.50)

    result = engine.evaluate_reliability(
        p_pd=0.52,
        uncertainty=0.30,
        evidence=(1.5, 1.7)
    )

    assert result["conformal_prediction_set"] == []
    assert result["conformal_set_size"] == 0
    assert result["reliability"] == "FLAG"
    assert "EMPTY_CONFORMAL_SET" in result["reliability_reasons"]


def test_handwriting_reliability_flag_ambiguous_conformal_set():
    """Verify ambiguous conformal set (size=2) triggers AMBIGUOUS_CONFORMAL_SET."""
    # When q_hat = 0.65, threshold is 1 - 0.65 = 0.35
    # Both p_pd = 0.52 and p_hc = 0.48 are >= 0.35 -> conformal set contains both
    engine = HandwritingReliabilityEngine(q_hat=0.65, uncertainty_threshold=0.50)

    result = engine.evaluate_reliability(
        p_pd=0.52,
        uncertainty=0.30,
        evidence=(1.5, 1.7)
    )

    assert set(result["conformal_prediction_set"]) == {"Healthy Control", "Parkinson's Disease"}
    assert result["conformal_set_size"] == 2
    assert result["reliability"] == "FLAG"
    assert "AMBIGUOUS_CONFORMAL_SET" in result["reliability_reasons"]


def test_handwriting_reliability_flag_borderline():
    """Verify prediction near 0.5 decision boundary triggers BORDERLINE_PROBABILITY."""
    engine = HandwritingReliabilityEngine(q_hat=0.45, borderline_margin=0.10)

    # 0.52 falls within [0.40, 0.60]
    result = engine.evaluate_reliability(
        p_pd=0.52,
        uncertainty=0.30
    )

    assert result["reliability"] == "FLAG"
    assert "BORDERLINE_PROBABILITY" in result["reliability_reasons"]
