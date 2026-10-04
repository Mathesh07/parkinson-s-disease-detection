"""Main Analyzer View Component for Streamlit Demo App.

Renders the Parkinson's Reliability Analyzer dashboard:
    - Input controls (Audio/Gait file uploaders & Pre-set Demo Scenarios)
    - Modality evidence cards (Gait & Voice)
    - Visually dominant Cross-Modal Decision Banner (ACCEPT vs FLAG)
    - Transparent evidence explanation boxes
    - Expandable evidence detail inspection panels
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import streamlit as st

# Ensure workspace root is in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
demo_dir = root_dir / "demo"
for p in (str(root_dir), str(demo_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cross_modal_reliability import CrossModalReliabilityEngine


def load_demo_scenarios() -> Dict[str, Dict]:
    """Load pre-set demonstration scenarios from Phase 9 outputs or deterministic definitions."""
    return {
        "Scenario 1: Strong Agreement (ACCEPT)": {
            "title": "Scenario 1: Strong Cross-Modal Agreement",
            "desc": "Both Gait and Voice independently predict Parkinson's Disease with high probability, strong ensemble agreement, and matching singleton conformal prediction sets.",
            "gait": {
                "prediction": "Parkinson's Disease", "predicted_label": 1,
                "p_pd": 0.890, "p_healthy": 0.110, "uncertainty": 0.350,
                "conformal_prediction_set": ["Parkinson's Disease"],
                "reliability": "ACCEPT", "reliability_level": "HIGH",
                "reliability_reasons": ["CONSISTENT_SINGLETON_PREDICTION"],
                "ensemble_agreement": "5/5 PD, 0/5 Healthy", "variance": 0.005
            },
            "voice": {
                "prediction": "Parkinson's Disease", "predicted_label": 1,
                "p_pd": 0.790, "p_healthy": 0.210, "uncertainty": 0.400,
                "conformal_prediction_set": ["Parkinson's Disease"],
                "reliability": "ACCEPT", "reliability_level": "HIGH",
                "reliability_reasons": ["CONSISTENT_SINGLETON_PREDICTION"],
                "ensemble_agreement": "5/5 PD, 0/5 Healthy", "variance": 0.008
            }
        },
        "Scenario 2: Modality Conflict (FLAG)": {
            "title": "Scenario 2: Direct Modality Conflict (Gait PD vs Voice Healthy)",
            "desc": "Gait predicts Parkinson's Disease (85.0%), whereas Voice predicts Healthy Control (65.0%). The system explicitly flags the conflict rather than silently averaging conflicting probabilities.",
            "gait": {
                "prediction": "Parkinson's Disease", "predicted_label": 1,
                "p_pd": 0.850, "p_healthy": 0.150, "uncertainty": 0.320,
                "conformal_prediction_set": ["Parkinson's Disease"],
                "reliability": "ACCEPT", "reliability_level": "HIGH",
                "reliability_reasons": ["CONSISTENT_SINGLETON_PREDICTION"],
                "ensemble_agreement": "5/5 PD, 0/5 Healthy", "variance": 0.006
            },
            "voice": {
                "prediction": "Healthy Control", "predicted_label": 0,
                "p_pd": 0.350, "p_healthy": 0.650, "uncertainty": 0.450,
                "conformal_prediction_set": ["Healthy Control"],
                "reliability": "ACCEPT", "reliability_level": "HIGH",
                "reliability_reasons": ["CONSISTENT_SINGLETON_PREDICTION"],
                "ensemble_agreement": "4/5 Healthy, 1/5 PD", "variance": 0.012
            }
        },
        "Scenario 3: One Modality Uncertain (FLAG)": {
            "title": "Scenario 3: Agreement but Voice Modality Flagged",
            "desc": "Both modalities predict Parkinson's Disease, but Voice exhibits high uncertainty, borderline probability (56.9%), and an empty conformal prediction set.",
            "gait": {
                "prediction": "Parkinson's Disease", "predicted_label": 1,
                "p_pd": 0.880, "p_healthy": 0.120, "uncertainty": 0.300,
                "conformal_prediction_set": ["Parkinson's Disease"],
                "reliability": "ACCEPT", "reliability_level": "HIGH",
                "reliability_reasons": ["CONSISTENT_SINGLETON_PREDICTION"],
                "ensemble_agreement": "5/5 PD, 0/5 Healthy", "variance": 0.004
            },
            "voice": {
                "prediction": "Parkinson's Disease", "predicted_label": 1,
                "p_pd": 0.569, "p_healthy": 0.431, "uncertainty": 0.6835,
                "conformal_prediction_set": [],
                "reliability": "FLAG", "reliability_level": "LOW",
                "reliability_reasons": ["EMPTY_CONFORMAL_SET", "HIGH_UNCERTAINTY", "BORDERLINE_PROBABILITY"],
                "ensemble_agreement": "4/5 PD, 1/5 Healthy", "variance": 0.01304
            }
        }
    }


def render_analyzer_view():
    """Render the main Parkinson's Reliability Analyzer page."""
    st.markdown("<h1 style='text-align: center; color: #f0f2f6;'>PARKINSON'S RELIABILITY ANALYZER</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8b949e; font-size: 1.15rem; font-style: italic; margin-bottom: 2rem;'>\"Prediction is only the beginning. Can the model's prediction be trusted?\"</p>", unsafe_allow_html=True)

    # 1. INPUT AREA
    st.markdown("### 📥 Input & Demo Scenarios")
    
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        uploaded_gait = st.file_uploader("Upload Gait Data (.csv / .txt)", type=["csv", "txt"], key="gait_uploader")
        if uploaded_gait:
            st.success(f"Loaded Gait file: `{uploaded_gait.name}`")

    with col_input2:
        uploaded_voice = st.file_uploader("Upload Voice Audio (.wav)", type=["wav"], key="voice_uploader")
        if uploaded_voice:
            st.audio(uploaded_voice, format="audio/wav")
            st.success(f"Loaded Audio file: `{uploaded_voice.name}`")

    # Pre-set Demo Scenarios
    scenarios = load_demo_scenarios()
    selected_scenario_key = st.selectbox(
        "🎯 Select Pre-set Demonstration Scenario (Saved Research Outputs):",
        options=list(scenarios.keys()),
        index=0
    )
    
    current_scenario = scenarios[selected_scenario_key]
    st.info(f"**{current_scenario['title']}** — {current_scenario['desc']}\n\n*Note: Demonstration scenarios use frozen Phase 8 & 9 research outputs to guarantee panel presentation stability on any hardware.*")

    gait_data = current_scenario["gait"]
    voice_data = current_scenario["voice"]

    # 2. RUN CROSS-MODAL FUSION ENGINE
    engine = CrossModalReliabilityEngine()
    fusion_result = engine.fuse_modalities(gait_data, voice_data)

    st.markdown("---")

    # 3. MODALITY CARDS
    st.markdown("### 🔬 Modality Evidence Cards")
    col_gait, col_voice = st.columns(2)

    with col_gait:
        st.markdown("<div style='background-color: #161b22; padding: 1.25rem; border-radius: 8px; border: 1px solid #30363d;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #58a6ff; margin-top:0;'>🏃 GAIT MODALITY</h4>", unsafe_allow_html=True)
        st.write(f"**Prediction:** {gait_data['prediction']}")
        st.write(f"**P(PD):** `{gait_data['p_pd']*100:.1f}%` | **P(Healthy):** `{gait_data['p_healthy']*100:.1f}%`")
        st.write(f"**Ensemble Agreement:** `{gait_data.get('ensemble_agreement', '5/5')}`")
        st.write(f"**Uncertainty Score:** `{gait_data['uncertainty']:.3f}`")
        
        c_set_str = "{" + ", ".join(gait_data['conformal_prediction_set']) + "}" if gait_data['conformal_prediction_set'] else "{}"
        st.write(f"**Conformal Set:** `{c_set_str}`")

        rel_status = gait_data['reliability']
        rel_color = "#2ea043" if rel_status == "ACCEPT" else "#da3633"
        st.markdown(f"**Reliability:** <span style='color: {rel_color}; font-weight: bold;'>[{rel_status}] ({gait_data['reliability_level']})</span>", unsafe_allow_html=True)
        st.write(f"**Reason:** `{', '.join(gait_data['reliability_reasons'])}`")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_voice:
        st.markdown("<div style='background-color: #161b22; padding: 1.25rem; border-radius: 8px; border: 1px solid #30363d;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #d2a8ff; margin-top:0;'>🎙️ VOICE MODALITY</h4>", unsafe_allow_html=True)
        st.write(f"**Prediction:** {voice_data['prediction']}")
        st.write(f"**P(PD):** `{voice_data['p_pd']*100:.1f}%` | **P(Healthy):** `{voice_data['p_healthy']*100:.1f}%`")
        st.write(f"**Ensemble Agreement:** `{voice_data.get('ensemble_agreement', '4/5 PD, 1/5 Healthy')}`")
        st.write(f"**Predictive Entropy:** `{voice_data['uncertainty']:.3f}`")
        
        v_c_set_str = "{" + ", ".join(voice_data['conformal_prediction_set']) + "}" if voice_data['conformal_prediction_set'] else "{}"
        st.write(f"**Conformal Set:** `{v_c_set_str}`")

        v_rel_status = voice_data['reliability']
        v_rel_color = "#2ea043" if v_rel_status == "ACCEPT" else "#da3633"
        st.markdown(f"**Reliability:** <span style='color: {v_rel_color}; font-weight: bold;'>[{v_rel_status}] ({voice_data['reliability_level']})</span>", unsafe_allow_html=True)
        st.write(f"**Reason:** `{', '.join(voice_data['reliability_reasons'])}`")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # 4. VISUALLY DOMINANT FINAL CROSS-MODAL DECISION BANNER
    st.markdown("### ⚖️ Cross-Modal Evidence & Final Decision")
    
    decision = fusion_result["final_decision"]
    status = fusion_result["cross_modal_status"]
    reason = fusion_result["reason"]
    level = fusion_result["cross_modal_reliability"]

    banner_bg = "#0d1117"
    banner_border = "#2ea043" if decision == "ACCEPT" else "#da3633"
    decision_color = "#3fb950" if decision == "ACCEPT" else "#f85149"
    badge_icon = "🟢" if decision == "ACCEPT" else ("🔴" if status == "MODALITY_CONFLICT" else "🟠")

    st.markdown(f"""
    <div style='background-color: {banner_bg}; padding: 1.75rem; border-radius: 12px; border: 3px solid {banner_border}; text-align: center;'>
        <h3 style='margin-top: 0; color: #8b949e;'>CROSS-MODAL EVIDENCE SYNTHESIS</h3>
        <div style='display: flex; justify-content: space-around; margin: 1.25rem 0;'>
            <div>
                <span style='color: #8b949e; font-size: 0.9rem;'>Prediction Agreement</span><br>
                <span style='font-size: 1.2rem; font-weight: bold;'>{"YES" if fusion_result["prediction_agreement"] else "NO"}</span>
            </div>
            <div>
                <span style='color: #8b949e; font-size: 0.9rem;'>Conformal Status</span><br>
                <span style='font-size: 1.1rem; font-weight: bold;'>{fusion_result["conformal_agreement_status"]}</span>
            </div>
            <div>
                <span style='color: #8b949e; font-size: 0.9rem;'>Evidence Status</span><br>
                <span style='font-size: 1.2rem; font-weight: bold;'>{badge_icon} {status}</span>
            </div>
        </div>
        <hr style='border-color: #30363d; margin: 1rem 0;'>
        <div style='margin-top: 0.75rem;'>
            <span style='color: #8b949e; font-size: 1.1rem;'>FINAL SYSTEM DECISION:</span><br>
            <span style='font-size: 2.8rem; font-weight: 900; color: {decision_color}; letter-spacing: 2px;'>[{decision}]</span><br>
            <span style='font-size: 1.1rem; color: #c9d1d9; font-weight: 600;'>Reliability Level: {level} | Policy Reason: {reason}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Why ACCEPT / Why FLAG explanation box
    if decision == "ACCEPT":
        st.success(
            "**WHY ACCEPT?**\n\n"
            "✓ Both Gait and Voice modalities independently support the same class prediction.\n\n"
            "✓ Conformal prediction sets concur on a single class.\n\n"
            "✓ Both modalities report HIGH reliability with low predictive uncertainty."
        )
    else:
        st.error(
            "**WHY FLAG?**\n\n"
            f"⚠ **Primary Reason:** `{reason}`\n\n"
            + "\n\n".join([f"• {item}" for item in fusion_result["evidence_summary"]]) +
            "\n\n*The reliability engine explicitly flags the decision rather than forcing an unsupported diagnosis.*"
        )

    # 5. EXPANDABLE EVIDENCE INSPECTION
    with st.expander("🔍 Inspect Gait Evidence Details"):
        st.json(gait_data)

    with st.expander("🔍 Inspect Voice Evidence Details"):
        st.json(voice_data)

    with st.expander("🔍 Inspect Cross-Modal Decision Logic Details"):
        st.json(fusion_result)
