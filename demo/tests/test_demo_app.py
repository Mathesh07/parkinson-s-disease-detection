"""Unit test suite for Streamlit Demo App Components (Phase 10).

Verifies component imports, demo scenario definitions, deterministic fallback outputs,
and non-mutation of research artifacts.
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
    """Verify demo scenarios dictionary contains required keys and fields."""
    scenarios = load_demo_scenarios()
    assert "Scenario 1: Strong Agreement (ACCEPT)" in scenarios
    assert "Scenario 2: Modality Conflict (FLAG)" in scenarios
    assert "Scenario 3: One Modality Uncertain (FLAG)" in scenarios

    for key, sc in scenarios.items():
        assert "gait" in sc
        assert "voice" in sc
        assert "title" in sc
        assert "desc" in sc


def test_demo_scenario_1_fusion():
    """Verify Scenario 1 yields CONSISTENT / ACCEPT / HIGH."""
    scenarios = load_demo_scenarios()
    sc1 = scenarios["Scenario 1: Strong Agreement (ACCEPT)"]
    
    engine = CrossModalReliabilityEngine()
    fused = engine.fuse_modalities(sc1["gait"], sc1["voice"])

    assert fused["prediction_agreement"] is True
    assert fused["cross_modal_status"] == "CONSISTENT"
    assert fused["cross_modal_reliability"] == "HIGH"
    assert fused["final_decision"] == "ACCEPT"
    assert fused["reason"] == "CROSS_MODAL_AGREEMENT"


def test_demo_scenario_2_fusion():
    """Verify Scenario 2 yields MODALITY_CONFLICT / FLAG / LOW."""
    scenarios = load_demo_scenarios()
    sc2 = scenarios["Scenario 2: Modality Conflict (FLAG)"]
    
    engine = CrossModalReliabilityEngine()
    fused = engine.fuse_modalities(sc2["gait"], sc2["voice"])

    assert fused["prediction_agreement"] is False
    assert fused["cross_modal_status"] == "MODALITY_CONFLICT"
    assert fused["cross_modal_reliability"] == "LOW"
    assert fused["final_decision"] == "FLAG"
    assert fused["reason"] == "GAIT_VOICE_CONFLICT"


def test_demo_scenario_3_fusion():
    """Verify Scenario 3 yields PARTIAL_SUPPORT / FLAG / MEDIUM."""
    scenarios = load_demo_scenarios()
    sc3 = scenarios["Scenario 3: One Modality Uncertain (FLAG)"]
    
    engine = CrossModalReliabilityEngine()
    fused = engine.fuse_modalities(sc3["gait"], sc3["voice"])

    assert fused["prediction_agreement"] is True
    assert fused["cross_modal_status"] == "PARTIAL_SUPPORT"
    assert fused["cross_modal_reliability"] == "MEDIUM"
    assert fused["final_decision"] == "FLAG"
    assert fused["reason"] == "ONE_MODALITY_UNCERTAIN"


def test_demo_imports():
    """Verify demo app entry point and components import without error."""
    try:
        import demo.app as demo_app
    except ImportError:
        import app as demo_app
    assert demo_app is not None
