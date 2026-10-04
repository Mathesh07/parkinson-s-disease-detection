"""Unit test suite for Cross-Modal Reliability Fusion (Phase 9).

Verifies cross-modal decision policies, agreement/conflict detection, conformal set alignment,
single-modality fallbacks, JSON serializability, and deterministic behavior.
"""

import json
import pytest

from cross_modal_reliability import CrossModalReliabilityEngine, fuse_cross_modal_reliability


def make_sample_gait(pred="Parkinson's Disease", label=1, p_pd=0.88, rel="ACCEPT", level="HIGH", conf_set=None):
    if conf_set is None:
        conf_set = ["Parkinson's Disease"] if label == 1 else ["Healthy Control"]
    return {
        "prediction": pred,
        "predicted_label": label,
        "p_pd": p_pd,
        "p_healthy": 1.0 - p_pd,
        "uncertainty": 0.35,
        "conformal_prediction_set": conf_set,
        "reliability": rel,
        "reliability_level": level,
        "reliability_reasons": ["CONSISTENT_SINGLETON_PREDICTION"] if rel == "ACCEPT" else ["HIGH_UNCERTAINTY"]
    }


def make_sample_voice(pred="Parkinson's Disease", label=1, p_pd=0.82, rel="ACCEPT", level="HIGH", conf_set=None):
    if conf_set is None:
        conf_set = ["Parkinson's Disease"] if label == 1 else ["Healthy Control"]
    return {
        "point_prediction": pred,
        "predicted_label": label,
        "p_pd": p_pd,
        "p_healthy": 1.0 - p_pd,
        "predictive_entropy": 0.40,
        "conformal_prediction_set": conf_set,
        "reliability": rel,
        "reliability_level": level,
        "reliability_reasons": ["CONSISTENT_SINGLETON_PREDICTION"] if rel == "ACCEPT" else ["HIGH_UNCERTAINTY"]
    }


def test_case_a_strong_agreement_pd():
    """Case A: Both modalities agree on PD and both are reliable."""
    gait = make_sample_gait(pred="Parkinson's Disease", label=1, p_pd=0.89)
    voice = make_sample_voice(pred="Parkinson's Disease", label=1, p_pd=0.79)
    
    fused = fuse_cross_modal_reliability(gait, voice)
    
    assert fused["prediction_agreement"] is True
    assert fused["conformal_agreement_status"] == "CONFORMAL_AGREEMENT"
    assert fused["cross_modal_status"] == "CONSISTENT"
    assert fused["cross_modal_reliability"] == "HIGH"
    assert fused["final_decision"] == "ACCEPT"
    assert fused["reason"] == "CROSS_MODAL_AGREEMENT"


def test_case_a_strong_agreement_healthy():
    """Case A: Both modalities agree on Healthy Control and both are reliable."""
    gait = make_sample_gait(pred="Healthy Control", label=0, p_pd=0.12)
    voice = make_sample_voice(pred="Healthy Control", label=0, p_pd=0.15)
    
    fused = fuse_cross_modal_reliability(gait, voice)
    
    assert fused["prediction_agreement"] is True
    assert fused["cross_modal_status"] == "CONSISTENT"
    assert fused["final_decision"] == "ACCEPT"


def test_case_c_conflict_gait_pd_voice_healthy():
    """Case C: Gait predicts PD while Voice predicts Healthy -> Direct Conflict."""
    gait = make_sample_gait(pred="Parkinson's Disease", label=1, p_pd=0.85)
    voice = make_sample_voice(pred="Healthy Control", label=0, p_pd=0.18)
    
    fused = fuse_cross_modal_reliability(gait, voice)
    
    assert fused["prediction_agreement"] is False
    assert fused["conformal_agreement_status"] == "CONFORMAL_CONFLICT"
    assert fused["cross_modal_status"] == "MODALITY_CONFLICT"
    assert fused["cross_modal_reliability"] == "LOW"
    assert fused["final_decision"] == "FLAG"
    assert fused["reason"] == "GAIT_VOICE_CONFLICT"


def test_case_c_conflict_gait_healthy_voice_pd():
    """Case C: Gait predicts Healthy while Voice predicts PD -> Direct Conflict."""
    gait = make_sample_gait(pred="Healthy Control", label=0, p_pd=0.10)
    voice = make_sample_voice(pred="Parkinson's Disease", label=1, p_pd=0.88)
    
    fused = fuse_cross_modal_reliability(gait, voice)
    
    assert fused["prediction_agreement"] is False
    assert fused["cross_modal_status"] == "MODALITY_CONFLICT"
    assert fused["final_decision"] == "FLAG"


def test_case_b_agreement_one_modality_uncertain():
    """Case B: Predictions agree but Voice is flagged / CAUTION -> PARTIAL_SUPPORT."""
    gait = make_sample_gait(pred="Parkinson's Disease", label=1, p_pd=0.89, rel="ACCEPT", level="HIGH")
    voice = make_sample_voice(pred="Parkinson's Disease", label=1, p_pd=0.56, rel="FLAG", level="MEDIUM", conf_set=[])
    
    fused = fuse_cross_modal_reliability(gait, voice)
    
    assert fused["prediction_agreement"] is True
    assert fused["cross_modal_status"] == "PARTIAL_SUPPORT"
    assert fused["cross_modal_reliability"] == "MEDIUM"
    assert fused["final_decision"] == "FLAG"
    assert fused["reason"] == "ONE_MODALITY_UNCERTAIN"


def test_case_d_both_modalities_unreliable():
    """Case D: Both modalities predict same class but both are UNRELIABLE -> INSUFFICIENT_EVIDENCE."""
    gait = make_sample_gait(pred="Parkinson's Disease", label=1, p_pd=0.52, rel="FLAG", level="LOW")
    voice = make_sample_voice(pred="Parkinson's Disease", label=1, p_pd=0.54, rel="FLAG", level="LOW")
    
    fused = fuse_cross_modal_reliability(gait, voice)
    
    assert fused["prediction_agreement"] is True
    assert fused["cross_modal_status"] == "INSUFFICIENT_EVIDENCE"
    assert fused["cross_modal_reliability"] == "LOW"
    assert fused["final_decision"] == "FLAG"
    assert fused["reason"] == "BOTH_MODALITIES_UNRELIABLE"


def test_case_e_gait_only():
    """Case E: Gait available, Voice unavailable -> SINGLE_MODALITY."""
    gait = make_sample_gait(pred="Parkinson's Disease", label=1, p_pd=0.88, rel="ACCEPT", level="HIGH")
    
    fused = fuse_cross_modal_reliability(gait_output=gait, voice_output=None)
    
    assert fused["cross_modal_status"] == "SINGLE_MODALITY"
    assert fused["final_decision"] == "ACCEPT"
    assert fused["reason"] == "SINGLE_MODALITY_EVIDENCE"
    assert fused["voice"] is None


def test_case_e_voice_only():
    """Case E: Voice available, Gait unavailable -> SINGLE_MODALITY."""
    voice = make_sample_voice(pred="Healthy Control", label=0, p_pd=0.15, rel="ACCEPT", level="HIGH")
    
    fused = fuse_cross_modal_reliability(gait_output=None, voice_output=voice)
    
    assert fused["cross_modal_status"] == "SINGLE_MODALITY"
    assert fused["final_decision"] == "ACCEPT"
    assert fused["reason"] == "SINGLE_MODALITY_EVIDENCE"
    assert fused["gait"] is None


def test_conformal_agreement_types():
    """Test conformal agreement classification logic."""
    engine = CrossModalReliabilityEngine()
    
    # 1. Ambiguous conformal set in one modality
    gait = make_sample_gait(conf_set=["Healthy Control", "Parkinson's Disease"])
    voice = make_sample_voice(conf_set=["Parkinson's Disease"])
    fused = engine.fuse_modalities(gait, voice)
    assert fused["conformal_agreement_status"] == "PARTIAL_CONFORMAL_AGREEMENT"

    # 2. Empty conformal set
    gait = make_sample_gait(conf_set=[])
    voice = make_sample_voice(conf_set=["Parkinson's Disease"])
    fused = engine.fuse_modalities(gait, voice)
    assert fused["conformal_agreement_status"] == "UNCERTAIN_CONFORMAL_EVIDENCE"


def test_json_serialization():
    """Verify 100% JSON serializability of cross-modal result."""
    gait = make_sample_gait()
    voice = make_sample_voice()
    fused = fuse_cross_modal_reliability(gait, voice)

    json_str = json.dumps(fused, indent=2)
    loaded = json.loads(json_str)
    assert loaded["cross_modal_status"] == "CONSISTENT"


def test_deterministic_output():
    """Verify that identical inputs produce 100% identical outputs."""
    gait = make_sample_gait()
    voice = make_sample_voice()
    f1 = fuse_cross_modal_reliability(gait, voice)
    f2 = fuse_cross_modal_reliability(gait, voice)

    assert f1 == f2


def test_no_ground_truth_dependency():
    """Verify that fusion engine requires zero ground truth labels."""
    gait = make_sample_gait()
    voice = make_sample_voice()
    fused = fuse_cross_modal_reliability(gait, voice)

    assert "true_label" not in fused
    assert "ground_truth" not in fused
