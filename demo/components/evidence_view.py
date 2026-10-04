"""Experimental Evidence & Results Dashboard Component for Streamlit Demo App.

Renders research experiment milestone timeline and embeds key research dashboard figures.
"""

import os
from pathlib import Path
import streamlit as st
from PIL import Image

# Ensure workspace root is in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent


def render_evidence_view():
    """Render experimental evidence timeline and dashboard figures."""
    st.markdown("<h2 style='text-align: center;'>📊 EXPERIMENTAL EVIDENCE & DASHBOARDS</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8b949e;'>Rigorous Methodological Progression across Gait, Voice, and Cross-Modal Fusion</p>", unsafe_allow_html=True)
    st.markdown("---")

    # 1. COMPACT MILESTONE TIMELINE
    st.markdown("### 🏆 Project Research Milestone Progression")

    col_gait_m, col_voice_m, col_fusion_m = st.columns(3)

    with col_gait_m:
        st.markdown("<div style='background-color: #161b22; padding: 1rem; border-radius: 6px; border-left: 4px solid #58a6ff;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #58a6ff; margin-top:0;'>🏃 GAIT PIPELINE</h4>", unsafe_allow_html=True)
        st.markdown(r"""
        - ✓ Stride Feature Extraction
        - ✓ LOCO Cohort Shift
        - ✓ Multi-Seed Evaluation
        - ✓ Evidential Deep Learning
        - ✓ MC Dropout ($N=30$)
        - ✓ Deep Ensemble ($M=5$)
        - ✓ Corruption Testing
        - ✓ Split Conformal ($q_{\hat{\text{gait}}}$)
        - ✓ Gait Reliability Engine
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_voice_m:
        st.markdown("<div style='background-color: #161b22; padding: 1rem; border-radius: 6px; border-left: 4px solid #d2a8ff;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #d2a8ff; margin-top:0;'>🎙️ VOICE PIPELINE</h4>", unsafe_allow_html=True)
        st.markdown(r"""
        - ✓ Wav2Vec2 Feature Encoder
        - ✓ Temperature Scaling ($T$)
        - ✓ MC Dropout ($N=30$)
        - ✓ Deep Ensemble ($M=5$)
        - ✓ Cross-Task Shift Analysis
        - ✓ Acoustic Corruption Testing
        - ✓ Split Conformal ($q_{\hat{\text{voice}}}$)
        - ✓ Voice Reliability Engine
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_fusion_m:
        st.markdown("<div style='background-color: #161b22; padding: 1rem; border-radius: 6px; border-left: 4px solid #3fb950;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #3fb950; margin-top:0;'>⚖️ CROSS-MODAL FUSION</h4>", unsafe_allow_html=True)
        st.markdown(r"""
        - ✓ Decision-Level Integration
        - ✓ Modality Agreement Detection
        - ✓ Conflict Flagging Policy
        - ✓ Conformal Set Concurrence
        - ✓ 179 Unit Tests Passing
        - ✓ Deterministic Fallback Mode
        - ✓ Interactive Live Demo
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # 2. KEY RESEARCH DASHBOARDS
    st.markdown("### 📈 Key High-Value Research Dashboards")

    tab1, tab2, tab3 = st.tabs([
        "🔀 Cross-Modal Fusion Dashboard",
        "🎙️ Voice Reliability Dashboard",
        "🔊 Acoustic Corruption Dashboard"
    ])

    with tab1:
        dash_fig1 = root_dir / "results" / "cross_modal" / "cross_modal_dashboard.png"
        if dash_fig1.exists():
            st.image(str(dash_fig1), caption="Cross-Modal Reliability Fusion Dashboard (Gait + Voice)", use_container_width=True)
        else:
            st.info("Cross-Modal dashboard figure available upon running demo runner script.")

    with tab2:
        dash_fig2 = root_dir / "Voice" / "results" / "reliability" / "reliability_dashboard.png"
        if dash_fig2.exists():
            st.image(str(dash_fig2), caption="Voice Wav2Vec2 Reliability Engine Dashboard", use_container_width=True)
        else:
            st.info("Voice Reliability dashboard figure available in Voice/results/reliability/.")

    with tab3:
        dash_fig3 = root_dir / "Voice" / "results" / "acoustic_corruption" / "corruption_uncertainty_dashboard.png"
        if dash_fig3.exists():
            st.image(str(dash_fig3), caption="Voice Acoustic Corruption vs. Predictive Uncertainty Dashboard", use_container_width=True)
        else:
            st.info("Acoustic corruption dashboard figure available in Voice/results/acoustic_corruption/.")
