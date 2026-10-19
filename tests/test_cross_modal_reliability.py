"""Unit test suite for Cross-Modal Reliability Fusion (Phase 9 & Tri-Modal Fusion).

Verifies cross-modal decision policies, agreement/conflict detection, conformal set alignment,
tri-modal fusion (Gait + Voice + Handwriting), single-modality fallbacks, JSON serializability,
and deterministic behavior.
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


def make_sample_handwriting(pred="Parkinson's Disease", label=1, p_pd=0.86, rel="ACCEPT", level="HIGH", conf_set=None):
    if conf_set is None:
        conf_set = ["Parkinson's Disease"] if label == 1 else ["Healthy Control"]
    return {
        "modality": "Handwriting",
        "prediction": pred,
        "predicted_label": label,
        "p_pd": p_pd,
        "p_healthy": 1.0 - p_pd,
        "uncertainty": 0.32,
        "conformal_prediction_set": conf_set,
        "reliability": rel,
        "reliability_level": level,
        "reliability_reasons": ["CONSISTENT_SINGLETON_PREDICTION"] if rel == "ACCEPT" else ["HIGH_EVIDENTIAL_UNCERTAINTY"]
    }


# =====================================================================
# 2-MODALITY BACKWARDS COMPATIBILITY TESTS
# =====================================================================

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
    assert fused["fused_prediction"] == "Parkinson's Disease"


def test_case_a_strong_agreement_healthy():
    """Case A: Both modalities agree on Healthy Control and both are reliable."""
    gait = make_sample_gait(pred="Healthy Control", label=0, p_pd=0.12)
    voice = make_sample_voice(pred="Healthy Control", label=0, p_pd=0.15)
    
    fused = fuse_cross_modal_reliability(gait, voice)
    
    assert fused["prediction_agreement"] is True
    assert fused["cross_modal_status"] == "CONSISTENT"
    assert fused["final_decision"] == "ACCEPT"
    assert fused["fused_prediction"] == "Healthy Control"


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


# =====================================================================
# TRI-MODAL FUSION TESTS (Gait + Voice + Handwriting)
# =====================================================================

def test_tri_modal_strong_agreement_pd():
    """Tri-Modal Case A: All 3 modalities agree on PD and all 3 are reliable -> ACCEPT."""
    gait = make_sample_gait(pred="Parkinson's Disease", label=1, p_pd=0.89)
    voice = make_sample_voice(pred="Parkinson's Disease", label=1, p_pd=0.82)
    handwriting = make_sample_handwriting(pred="Parkinson's Disease", label=1, p_pd=0.87)
    
    fused = fuse_cross_modal_reliability(gait, voice, handwriting)
    
    assert fused["num_modalities"] == 3
    assert set(fused["active_modalities"]) == {"Gait", "Voice", "Handwriting"}
    assert fused["prediction_agreement"] is True
    assert fused["majority_agreement"] is True
    assert fused["fused_prediction"] == "Parkinson's Disease"
    assert fused["predicted_label"] == 1
    assert fused["fused_p_pd"] > 0.80
    assert fused["conformal_agreement_status"] == "CONFORMAL_AGREEMENT"
    assert fused["cross_modal_status"] == "CONSISTENT"
    assert fused["cross_modal_reliability"] == "HIGH"
    assert fused["final_decision"] == "ACCEPT"
    assert fused["reason"] == "TRI_MODAL_AGREEMENT"


def test_tri_modal_strong_agreement_healthy():
    """Tri-Modal Case A: All 3 modalities agree on Healthy Control -> ACCEPT."""
    gait = make_sample_gait(pred="Healthy Control", label=0, p_pd=0.12)
    voice = make_sample_voice(pred="Healthy Control", label=0, p_pd=0.15)
    handwriting = make_sample_handwriting(pred="Healthy Control", label=0, p_pd=0.14)
    
    fused = fuse_cross_modal_reliability(gait, voice, handwriting)
    
    assert fused["num_modalities"] == 3
    assert fused["prediction_agreement"] is True
    assert fused["fused_prediction"] == "Healthy Control"
    assert fused["predicted_label"] == 0
    assert fused["fused_p_pd"] < 0.20
    assert fused["final_decision"] == "ACCEPT"
    assert fused["cross_modal_status"] == "CONSISTENT"


def test_tri_modal_conflict_handwriting_dissents():
    """Tri-Modal Case C: Gait & Voice predict PD, Handwriting predicts Healthy -> 2 vs 1 Disagreement."""
    gait = make_sample_gait(pred="Parkinson's Disease", label=1, p_pd=0.88)
    voice = make_sample_voice(pred="Parkinson's Disease", label=1, p_pd=0.81)
    handwriting = make_sample_handwriting(pred="Healthy Control", label=0, p_pd=0.18)
    
    fused = fuse_cross_modal_reliability(gait, voice, handwriting)
    
    assert fused["num_modalities"] == 3
    assert fused["prediction_agreement"] is False
    assert fused["majority_agreement"] is True
    # Majority of 3 points to PD (2/3)
    assert fused["fused_prediction"] == "Parkinson's Disease"
    assert fused["predicted_label"] == 1
    # But because of conflict, safety engine FLAGS decision
    assert fused["cross_modal_status"] == "MODALITY_CONFLICT"
    assert fused["final_decision"] == "FLAG"
    assert fused["reason"] == "MODALITY_DISAGREEMENT_2_VS_1"


def test_tri_modal_conflict_voice_dissents():
    """Tri-Modal Case C: Gait & Handwriting predict PD, Voice predicts Healthy -> 2 vs 1 Disagreement."""
    gait = make_sample_gait(pred="Parkinson's Disease", label=1, p_pd=0.85)
    voice = make_sample_voice(pred="Healthy Control", label=0, p_pd=0.20)
    handwriting = make_sample_handwriting(pred="Parkinson's Disease", label=1, p_pd=0.89)
    
    fused = fuse_cross_modal_reliability(gait, voice, handwriting)
    
    assert fused["fused_prediction"] == "Parkinson's Disease"
    assert fused["final_decision"] == "FLAG"
    assert fused["cross_modal_status"] == "MODALITY_CONFLICT"


def test_tri_modal_agreement_one_modality_uncertain():
    """Tri-Modal Case B: All 3 predict PD, but Handwriting reports elevated uncertainty -> FLAG / PARTIAL_SUPPORT."""
    gait = make_sample_gait(pred="Parkinson's Disease", label=1, p_pd=0.89, rel="ACCEPT", level="HIGH")
    voice = make_sample_voice(pred="Parkinson's Disease", label=1, p_pd=0.82, rel="ACCEPT", level="HIGH")
    handwriting = make_sample_handwriting(
        pred="Parkinson's Disease", label=1, p_pd=0.55, rel="FLAG", level="MEDIUM", conf_set=[]
    )
    
    fused = fuse_cross_modal_reliability(gait, voice, handwriting)
    
    assert fused["num_modalities"] == 3
    assert fused["prediction_agreement"] is True
    assert fused["fused_prediction"] == "Parkinson's Disease"
    assert fused["cross_modal_status"] == "PARTIAL_SUPPORT"
    assert fused["cross_modal_reliability"] == "MEDIUM"
    assert fused["final_decision"] == "FLAG"
    assert fused["reason"] == "ONE_OR_MORE_MODALITIES_UNCERTAIN"


def test_tri_modal_all_unreliable():
    """Tri-Modal Case D: All 3 predict PD, but all 3 are UNRELIABLE -> INSUFFICIENT_EVIDENCE."""
    gait = make_sample_gait(pred="Parkinson's Disease", label=1, p_pd=0.52, rel="FLAG", level="LOW")
    voice = make_sample_voice(pred="Parkinson's Disease", label=1, p_pd=0.53, rel="FLAG", level="LOW")
    handwriting = make_sample_handwriting(pred="Parkinson's Disease", label=1, p_pd=0.54, rel="FLAG", level="LOW")
    
    fused = fuse_cross_modal_reliability(gait, voice, handwriting)
    
    assert fused["prediction_agreement"] is True
    assert fused["cross_modal_status"] == "INSUFFICIENT_EVIDENCE"
    assert fused["cross_modal_reliability"] == "LOW"
    assert fused["final_decision"] == "FLAG"
    assert fused["reason"] == "ALL_MODALITIES_UNRELIABLE"


def test_case_e_handwriting_only():
    """Single Modality: Only handwriting available."""
    handwriting = make_sample_handwriting(pred="Parkinson's Disease", label=1, p_pd=0.91, rel="ACCEPT", level="HIGH")
    
    fused = fuse_cross_modal_reliability(handwriting_output=handwriting)
    
    assert fused["num_modalities"] == 1
    assert fused["active_modalities"] == ["Handwriting"]
    assert fused["cross_modal_status"] == "SINGLE_MODALITY"
    assert fused["final_decision"] == "ACCEPT"
    assert fused["fused_prediction"] == "Parkinson's Disease"
    assert fused["handwriting"] is not None
    assert fused["gait"] is None
    assert fused["voice"] is None


def test_conformal_agreement_types():
    """Test conformal agreement classification logic across modalities."""
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


def test_json_serialization_tri_modal():
    """Verify 100% JSON serializability of tri-modal fused results."""
    gait = make_sample_gait()
    voice = make_sample_voice()
    handwriting = make_sample_handwriting()
    fused = fuse_cross_modal_reliability(gait, voice, handwriting)

    json_str = json.dumps(fused, indent=2)
    loaded = json.loads(json_str)
    assert loaded["cross_modal_status"] == "CONSISTENT"
    assert loaded["num_modalities"] == 3
    assert loaded["fused_prediction"] == "Parkinson's Disease"


def test_deterministic_output():
    """Verify that identical inputs produce 100% identical outputs."""
    gait = make_sample_gait()
    voice = make_sample_voice()
    handwriting = make_sample_handwriting()
    f1 = fuse_cross_modal_reliability(gait, voice, handwriting)
    f2 = fuse_cross_modal_reliability(gait, voice, handwriting)

    assert f1 == f2
