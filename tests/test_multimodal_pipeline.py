"""Unit and integration tests for Unified Multimodal Pipeline."""

import pytest
import sys
from pathlib import Path

# Ensure root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from multimodal_pipeline import MultimodalPredictor


def test_multimodal_unanimous_agreement():
    """Verify 3 agreeing confident modalities trigger CASE_A and ACCEPT."""
    predictor = MultimodalPredictor()

    gait = {
        "prediction": "Parkinson's Disease", "predicted_label": 1,
        "p_pd": 0.88, "uncertainty": 0.30,
        "conformal_prediction_set": ["Parkinson's Disease"],
        "reliability": "ACCEPT", "reliability_level": "HIGH"
    }
    voice = {
        "prediction": "Parkinson's Disease", "predicted_label": 1,
        "p_pd": 0.82, "uncertainty": 0.35,
        "conformal_prediction_set": ["Parkinson's Disease"],
        "reliability": "ACCEPT", "reliability_level": "HIGH"
    }
    hw = {
        "prediction": "Parkinson's Disease", "predicted_label": 1,
        "p_pd": 0.85, "uncertainty": 0.28,
        "conformal_prediction_set": ["Parkinson's Disease"],
        "reliability": "ACCEPT", "reliability_level": "HIGH"
    }

    result = predictor.predict(gait=gait, voice=voice, handwriting=hw)

    assert result["decision_case"] == "CASE_A_TRI_MODAL_AGREEMENT"
    assert result["final_prediction"] == "Parkinson's Disease"
    assert result["final_reliability"] == "ACCEPT"
    assert result["active_modality_count"] == 3


def test_multimodal_conflict_flag():
    """Verify 2 vs 1 conflict triggers CASE_C_MODALITY_DISAGREEMENT and FLAG."""
    predictor = MultimodalPredictor()

    gait = {
        "prediction": "Parkinson's Disease", "predicted_label": 1,
        "p_pd": 0.85, "uncertainty": 0.30,
        "conformal_prediction_set": ["Parkinson's Disease"],
        "reliability": "ACCEPT", "reliability_level": "HIGH"
    }
    voice = {
        "prediction": "Parkinson's Disease", "predicted_label": 1,
        "p_pd": 0.80, "uncertainty": 0.32,
        "conformal_prediction_set": ["Parkinson's Disease"],
        "reliability": "ACCEPT", "reliability_level": "HIGH"
    }
    hw = {
        "prediction": "Healthy Control", "predicted_label": 0,
        "p_pd": 0.22, "uncertainty": 0.28,
        "conformal_prediction_set": ["Healthy Control"],
        "reliability": "ACCEPT", "reliability_level": "HIGH"
    }

    result = predictor.predict(gait=gait, voice=voice, handwriting=hw)

    assert result["decision_case"] == "CASE_C_MODALITY_DISAGREEMENT"
    assert result["final_reliability"] == "FLAG"
    assert result["final_reliability_level"] == "LOW"


def test_multimodal_single_modality_fallback():
    """Verify single modality input works and inherits modality status."""
    predictor = MultimodalPredictor()

    voice = {
        "prediction": "Parkinson's Disease", "predicted_label": 1,
        "p_pd": 0.82, "uncertainty": 0.35,
        "conformal_prediction_set": ["Parkinson's Disease"],
        "reliability": "ACCEPT", "reliability_level": "HIGH"
    }

    result = predictor.predict(voice=voice)
    assert result["decision_case"] == "CASE_E_SINGLE_MODALITY"
    assert result["active_modality_count"] == 1
    assert result["active_modalities"] == ["Voice"]
    assert result["final_prediction"] == "Parkinson's Disease"


def test_multimodal_empty_input_raises_error():
    """Verify calling predict with no inputs raises ValueError."""
    predictor = MultimodalPredictor()
    with pytest.raises(ValueError):
        predictor.predict()
