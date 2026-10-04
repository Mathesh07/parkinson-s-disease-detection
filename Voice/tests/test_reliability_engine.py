"""Unit test suite for Voice Reliability Engine (Phase 8).

Verifies multi-signal reliability decision policy, conformal set flags, ensemble agreement flags,
entropy flags, borderline probability flags, JSON serializability, and deterministic behavior.
"""

import json
import math
import numpy as np
import pytest

from Voice.reliability_engine import VoiceReliabilityEngine, evaluate_subject_reliability


def test_reliability_accept_high():
    """Test strong singleton prediction with high agreement resulting in ACCEPT / HIGH."""
    # Member probs: 5/5 agree PD with high probability (P_mean ≈ 0.86)
    member_probs = [0.88, 0.85, 0.87, 0.84, 0.86]
    res = evaluate_subject_reliability(member_probs, q_hat=0.45)

    assert res["reliability"] == "ACCEPT"
    assert res["reliability_level"] == "HIGH"
    assert res["reliability_reasons"] == ["CONSISTENT_SINGLETON_PREDICTION"]
    assert res["conformal_set_size"] == 1
    assert res["ensemble_agreement_score"] == 1.0


def test_reliability_flag_empty_conformal_set():
    """Test empty conformal set resulting in FLAG / MEDIUM or LOW."""
    # Member probs hover around 0.52 (within [0.40, 0.60] borderline and producing empty set for q_hat=0.40)
    member_probs = [0.55, 0.52, 0.51, 0.53, 0.50]
    res = evaluate_subject_reliability(member_probs, q_hat=0.40)

    assert res["reliability"] == "FLAG"
    assert "EMPTY_CONFORMAL_SET" in res["reliability_reasons"]
    assert res["conformal_set_size"] == 0


def test_reliability_flag_ambiguous_conformal_set():
    """Test ambiguous conformal set (size=2) resulting in FLAG."""
    # Large q_hat = 0.60 (threshold = 0.40). Probabilities near 0.5 (both >= 0.40)
    member_probs = [0.52, 0.51, 0.53, 0.50, 0.54]
    res = evaluate_subject_reliability(member_probs, q_hat=0.60)

    assert res["reliability"] == "FLAG"
    assert "AMBIGUOUS_CONFORMAL_SET" in res["reliability_reasons"]
    assert res["conformal_set_size"] == 2


def test_reliability_flag_ensemble_disagreement():
    """Test 3/5 split ensemble disagreement resulting in FLAG."""
    # 3 members predict PD (~0.58), 2 predict HC (~0.42)
    member_probs = [0.58, 0.57, 0.59, 0.41, 0.42]
    res = evaluate_subject_reliability(member_probs, q_hat=0.45)

    assert res["reliability"] == "FLAG"
    assert "HIGH_ENSEMBLE_DISAGREEMENT" in res["reliability_reasons"]
    assert res["ensemble_agreement_score"] == 0.6  # 3/5


def test_reliability_flag_high_entropy():
    """Test high predictive entropy resulting in FLAG."""
    member_probs = [0.58, 0.55, 0.56, 0.54, 0.57]  # Mean P=0.56 -> H(0.56) = 0.686 >= 0.65
    res = evaluate_subject_reliability(member_probs, q_hat=0.45)

    assert res["reliability"] == "FLAG"
    assert "HIGH_UNCERTAINTY" in res["reliability_reasons"]
    assert res["predictive_entropy"] >= 0.65


def test_reliability_flag_borderline_probability():
    """Test borderline probability within [0.40, 0.60] resulting in FLAG."""
    member_probs = [0.58, 0.57, 0.56, 0.55, 0.54]
    res = evaluate_subject_reliability(member_probs, q_hat=0.45)

    assert res["reliability"] == "FLAG"
    assert "BORDERLINE_PROBABILITY" in res["reliability_reasons"]


def test_reliability_multiple_flag_reasons():
    """Test multiple flags triggering LOW reliability level."""
    # 3/5 split (disagreement), P_mean=0.52 (borderline), high entropy -> Multiple flags
    member_probs = [0.58, 0.57, 0.56, 0.43, 0.44]
    res = evaluate_subject_reliability(member_probs, q_hat=0.40)

    assert res["reliability"] == "FLAG"
    assert res["reliability_level"] == "LOW"
    assert len(res["reliability_reasons"]) >= 2


def test_json_serializability():
    """Verify that reliability engine result dict is 100% JSON serializable."""
    member_probs = [0.88, 0.85, 0.87, 0.84, 0.86]
    audio_meta = {"filename": "test.wav", "duration_sec": 10.0, "num_chunks": 1}
    res = evaluate_subject_reliability(member_probs, audio_meta=audio_meta)

    json_str = json.dumps(res, indent=2)
    loaded = json.loads(json_str)
    assert loaded["filename"] == "test.wav"
    assert loaded["reliability"] == "ACCEPT"


def test_deterministic_output():
    """Verify that identical inputs produce 100% identical reliability decisions."""
    member_probs = [0.75, 0.72, 0.74, 0.73, 0.76]
    res1 = evaluate_subject_reliability(member_probs)
    res2 = evaluate_subject_reliability(member_probs)

    assert res1 == res2


def test_no_test_label_dependency():
    """Verify that decision logic takes zero ground-truth label arguments."""
    engine = VoiceReliabilityEngine()
    # evaluate_reliability takes only member_probabilities and audio_meta
    res = engine.evaluate_reliability([0.8, 0.8, 0.8, 0.8, 0.8])
    assert "true_label" not in res
    assert "ground_truth" not in res


def test_binary_class_handling():
    """Verify correct class names and complementary P(Healthy) calculation."""
    member_probs = [0.9, 0.9, 0.9, 0.9, 0.9]
    res = evaluate_subject_reliability(member_probs)

    assert math.isclose(res["p_pd"] + res["p_healthy"], 1.0, abs_tol=1e-6)
    assert res["point_prediction"] in ["Healthy Control", "Parkinson's Disease"]
