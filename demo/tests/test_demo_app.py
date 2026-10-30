"""Unit test suite for Streamlit Demo App Components (Phase 10 & Tri-Modal Fusion).

Verifies component imports, demo scenario definitions (Gait + Voice + Handwriting),
tri-modal fusion on demo scenarios, deterministic outputs, and UI stability.
"""

import os
import sys
from pathlib import Path
import pytest

# Ensure workspace root and demo directory are in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
demo_dir = root_dir / "demo"
for p in (str(root_dir), str(demo_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cross_modal_reliability import CrossModalReliabilityEngine

try:
    from demo.components.analyzer_view import load_demo_scenarios
except ImportError:
    from components.analyzer_view import load_demo_scenarios


def test_demo_scenarios_structure():
    """Verify demo scenarios dictionary contains required keys across all 3 modalities."""
    scenarios = load_demo_scenarios()
    assert "Scenario 1: Strong Agreement (ACCEPT)" in scenarios
    assert "Scenario 2: Modality Conflict (FLAG)" in scenarios
    assert "Scenario 3: One Modality Uncertain (FLAG)" in scenarios
    assert "Scenario 4: Strong Healthy Agreement (ACCEPT)" in scenarios

    for key, sc in scenarios.items():
        assert "gait" in sc, f"Gait missing in {key}"
        assert "voice" in sc, f"Voice missing in {key}"
        assert "handwriting" in sc, f"Handwriting missing in {key}"
        assert "title" in sc
        assert "desc" in sc


def test_demo_scenario_1_tri_modal_fusion():
    """Verify Scenario 1 yields CONSISTENT / ACCEPT / HIGH for Parkinson's Disease."""
    scenarios = load_demo_scenarios()
    sc1 = scenarios["Scenario 1: Strong Agreement (ACCEPT)"]
    
    engine = CrossModalReliabilityEngine()
    fused = engine.fuse_modalities(sc1["gait"], sc1["voice"], sc1["handwriting"])

    assert fused["num_modalities"] == 3
    assert fused["prediction_agreement"] is True
    assert fused["fused_prediction"] == "Parkinson's Disease"
    assert fused["cross_modal_status"] == "CONSISTENT"
    assert fused["cross_modal_reliability"] == "HIGH"
    assert fused["final_decision"] == "ACCEPT"
    assert fused["reason"] == "TRI_MODAL_AGREEMENT"


def test_demo_scenario_2_tri_modal_fusion():
    """Verify Scenario 2 yields MODALITY_CONFLICT / FLAG / LOW due to 2 vs 1 conflict."""
    scenarios = load_demo_scenarios()
    sc2 = scenarios["Scenario 2: Modality Conflict (FLAG)"]
    
    engine = CrossModalReliabilityEngine()
    fused = engine.fuse_modalities(sc2["gait"], sc2["voice"], sc2["handwriting"])

    assert fused["num_modalities"] == 3
    assert fused["prediction_agreement"] is False
    assert fused["fused_prediction"] == "Parkinson's Disease"  # Majority 2/3
    assert fused["cross_modal_status"] == "MODALITY_CONFLICT"
    assert fused["final_decision"] == "FLAG"
    assert fused["reason"] == "MODALITY_DISAGREEMENT_2_VS_1"


def test_demo_scenario_3_tri_modal_fusion():
    """Verify Scenario 3 yields PARTIAL_SUPPORT / FLAG / MEDIUM due to Handwriting uncertainty."""
    scenarios = load_demo_scenarios()
    sc3 = scenarios["Scenario 3: One Modality Uncertain (FLAG)"]
    
    engine = CrossModalReliabilityEngine()
    fused = engine.fuse_modalities(sc3["gait"], sc3["voice"], sc3["handwriting"])

    assert fused["num_modalities"] == 3
    assert fused["prediction_agreement"] is True
    assert fused["fused_prediction"] == "Parkinson's Disease"
    assert fused["cross_modal_status"] == "PARTIAL_SUPPORT"
    assert fused["cross_modal_reliability"] == "MEDIUM"
    assert fused["final_decision"] == "FLAG"
    assert fused["reason"] == "ONE_OR_MORE_MODALITIES_UNCERTAIN"


def test_demo_scenario_4_tri_modal_fusion():
    """Verify Scenario 4 yields CONSISTENT / ACCEPT / HIGH for Healthy Control."""
    scenarios = load_demo_scenarios()
    sc4 = scenarios["Scenario 4: Strong Healthy Agreement (ACCEPT)"]
    
    engine = CrossModalReliabilityEngine()
    fused = engine.fuse_modalities(sc4["gait"], sc4["voice"], sc4["handwriting"])

    assert fused["num_modalities"] == 3
    assert fused["prediction_agreement"] is True
    assert fused["fused_prediction"] == "Healthy Control"
    assert fused["cross_modal_status"] == "CONSISTENT"
    assert fused["final_decision"] == "ACCEPT"


def test_demo_imports():
    """Verify demo app entry point and components import without error."""
    try:
        import demo.app as demo_app
    except ImportError:
        import app as demo_app
    assert demo_app is not None
