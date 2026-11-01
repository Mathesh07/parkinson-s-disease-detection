"""Main Analyzer View Component for Streamlit Demo App.

Renders the Tri-Modal Parkinson's Reliability Analyzer dashboard:
    - Input controls (Gait, Voice WAV, and Handwriting drawing uploaders & Pre-set Demo Scenarios)
    - Modality evidence cards (Gait, Voice, and Handwriting)
    - Visually dominant Tri-Modal Decision Banner (Final Prediction, Consensus, ACCEPT vs FLAG)
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

import tempfile
from cross_modal_reliability import CrossModalReliabilityEngine
from multimodal_pipeline import MultimodalPredictor


def load_demo_scenarios() -> Dict[str, Dict]:
    """Load pre-set demonstration scenarios spanning all three modalities (Gait, Voice, Handwriting)."""
    return {
        "Scenario 1: Strong Agreement (ACCEPT)": {
            "title": "Scenario 1: Strong Tri-Modal Agreement (All 3 Predict PD)",
            "desc": "Gait, Voice, and Handwriting all independently predict Parkinson's Disease with high probability, low uncertainty, and matching singleton conformal prediction sets.",
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
            },
            "handwriting": {
                "prediction": "Parkinson's Disease", "predicted_label": 1,
                "p_pd": 0.865, "p_healthy": 0.135, "uncertainty": 0.315,
                "conformal_prediction_set": ["Parkinson's Disease"],
                "reliability": "ACCEPT", "reliability_level": "HIGH",
                "reliability_reasons": ["CONSISTENT_SINGLETON_PREDICTION"],
                "evidence": [0.15, 1.85], "model": "ViT (Evidential Head)"
            }
        },
        "Scenario 2: Modality Conflict (FLAG)": {
            "title": "Scenario 2: Tri-Modal Conflict (Gait & Voice PD vs Handwriting Healthy)",
            "desc": "Gait (85%) and Voice (80%) predict Parkinson's Disease, but Handwriting predicts Healthy Control (78%). The system flags the 2 vs 1 conflict rather than silently masking disagreement.",
            "gait": {
                "prediction": "Parkinson's Disease", "predicted_label": 1,
                "p_pd": 0.850, "p_healthy": 0.150, "uncertainty": 0.320,
                "conformal_prediction_set": ["Parkinson's Disease"],
                "reliability": "ACCEPT", "reliability_level": "HIGH",
                "reliability_reasons": ["CONSISTENT_SINGLETON_PREDICTION"],
                "ensemble_agreement": "5/5 PD, 0/5 Healthy", "variance": 0.006
            },
            "voice": {
                "prediction": "Parkinson's Disease", "predicted_label": 1,
                "p_pd": 0.800, "p_healthy": 0.200, "uncertainty": 0.390,
                "conformal_prediction_set": ["Parkinson's Disease"],
                "reliability": "ACCEPT", "reliability_level": "HIGH",
                "reliability_reasons": ["CONSISTENT_SINGLETON_PREDICTION"],
                "ensemble_agreement": "4/5 PD, 1/5 Healthy", "variance": 0.010
            },
            "handwriting": {
                "prediction": "Healthy Control", "predicted_label": 0,
                "p_pd": 0.220, "p_healthy": 0.780, "uncertainty": 0.340,
                "conformal_prediction_set": ["Healthy Control"],
                "reliability": "ACCEPT", "reliability_level": "HIGH",
                "reliability_reasons": ["CONSISTENT_SINGLETON_PREDICTION"],
                "evidence": [2.40, 0.18], "model": "ViT (Evidential Head)"
            }
        },
        "Scenario 3: One Modality Uncertain (FLAG)": {
            "title": "Scenario 3: Unanimous Prediction but Handwriting Modality Flagged",
            "desc": "All three modalities predict Parkinson's Disease, but Handwriting exhibits elevated Dirichlet uncertainty (0.582), borderline probability (54.5%), and an empty conformal set.",
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
                "p_pd": 0.810, "p_healthy": 0.190, "uncertainty": 0.380,
                "conformal_prediction_set": ["Parkinson's Disease"],
                "reliability": "ACCEPT", "reliability_level": "HIGH",
                "reliability_reasons": ["CONSISTENT_SINGLETON_PREDICTION"],
                "ensemble_agreement": "5/5 PD, 0/5 Healthy", "variance": 0.007
            },
            "handwriting": {
                "prediction": "Parkinson's Disease", "predicted_label": 1,
                "p_pd": 0.545, "p_healthy": 0.455, "uncertainty": 0.582,
                "conformal_prediction_set": [],
                "reliability": "FLAG", "reliability_level": "MEDIUM",
                "reliability_reasons": ["EMPTY_CONFORMAL_SET", "HIGH_EVIDENTIAL_UNCERTAINTY", "BORDERLINE_PROBABILITY"],
                "evidence": [0.42, 0.58], "model": "ViT (Evidential Head)"
            }
        },
        "Scenario 4: Strong Healthy Agreement (ACCEPT)": {
            "title": "Scenario 4: Strong Tri-Modal Healthy Agreement",
            "desc": "All 3 modalities independently predict Healthy Control with high probability and singleton conformal sets.",
            "gait": {
                "prediction": "Healthy Control", "predicted_label": 0,
                "p_pd": 0.110, "p_healthy": 0.890, "uncertainty": 0.280,
                "conformal_prediction_set": ["Healthy Control"],
                "reliability": "ACCEPT", "reliability_level": "HIGH",
                "reliability_reasons": ["CONSISTENT_SINGLETON_PREDICTION"],
                "ensemble_agreement": "5/5 Healthy, 0/5 PD", "variance": 0.003
            },
            "voice": {
                "prediction": "Healthy Control", "predicted_label": 0,
                "p_pd": 0.140, "p_healthy": 0.860, "uncertainty": 0.330,
                "conformal_prediction_set": ["Healthy Control"],
                "reliability": "ACCEPT", "reliability_level": "HIGH",
                "reliability_reasons": ["CONSISTENT_SINGLETON_PREDICTION"],
                "ensemble_agreement": "5/5 Healthy, 0/5 PD", "variance": 0.005
            },
            "handwriting": {
                "prediction": "Healthy Control", "predicted_label": 0,
                "p_pd": 0.135, "p_healthy": 0.865, "uncertainty": 0.290,
                "conformal_prediction_set": ["Healthy Control"],
                "reliability": "ACCEPT", "reliability_level": "HIGH",
                "reliability_reasons": ["CONSISTENT_SINGLETON_PREDICTION"],
                "evidence": [3.10, 0.12], "model": "ViT (Evidential Head)"
            }
        }
    }


def render_analyzer_view():
    """Render the main Tri-Modal Parkinson's Reliability Analyzer page."""
    st.markdown("<h1 style='text-align: center; color: #f0f2f6;'>TRI-MODAL PARKINSON'S RELIABILITY ANALYZER</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8b949e; font-size: 1.15rem; font-style: italic; margin-bottom: 2rem;'>\"Unified Decision-Level Evidence Synthesis Across Gait, Voice, and Handwriting\"</p>", unsafe_allow_html=True)

    # 1. INPUT AREA (3 Modalities)
    st.markdown("### 📥 Multimodal Input Controls")
    
    col_input1, col_input2, col_input3 = st.columns(3)
    with col_input1:
        uploaded_gait = st.file_uploader("🏃 Gait Data (.csv / .txt)", type=["csv", "txt"], key="gait_uploader")
        if uploaded_gait:
            st.success(f"Gait: `{uploaded_gait.name}`")

    with col_input2:
        uploaded_voice = st.file_uploader("🎙️ Voice Audio (.wav)", type=["wav"], key="voice_uploader")
        if uploaded_voice:
            st.audio(uploaded_voice, format="audio/wav")
            st.success(f"Audio: `{uploaded_voice.name}`")

    with col_input3:
        uploaded_hw = st.file_uploader("✍️ Handwriting (.png / .jpg)", type=["png", "jpg", "jpeg"], key="hw_uploader")
        if uploaded_hw:
            st.image(uploaded_hw, width=150, caption=f"Drawing: {uploaded_hw.name}")
            st.success(f"Drawing: `{uploaded_hw.name}`")

    # Mode Selection: Live Uploads vs Pre-set Scenarios
    has_uploaded = uploaded_gait is not None or uploaded_voice is not None or uploaded_hw is not None
    mode = "Pre-set Research Scenarios"
    if has_uploaded:
        mode = st.radio(
            "⚡ Select Analysis Source:",
            options=["🎯 Pre-set Research Scenarios", "🚀 Live Patient File Uploads (Active)"],
            horizontal=True
        )

    predictor = MultimodalPredictor()

    if has_uploaded and "Live" in mode:
        st.success("🔬 **Processing live multimodal patient inputs via MultimodalPredictor pipeline...**")
        
        # Save temp files for uploaded modalities
        gait_temp = None
        if uploaded_gait:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_gait.name}") as f:
                f.write(uploaded_gait.getbuffer())
                gait_temp = f.name
        
        voice_temp = None
        if uploaded_voice:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_voice.name}") as f:
                f.write(uploaded_voice.getbuffer())
                voice_temp = f.name

        hw_temp = None
        if uploaded_hw:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_hw.name}") as f:
                f.write(uploaded_hw.getbuffer())
                hw_temp = f.name

        try:
            gait_data = predictor.process_gait(gait_temp) if gait_temp else None
            voice_data = predictor.process_voice(voice_temp) if voice_temp else None
            hw_data = predictor.process_handwriting(hw_temp) if hw_temp else None
            
            fusion_result = predictor.predict(gait=gait_data, voice=voice_data, handwriting=hw_data)
        finally:
            # Cleanup temp files
            for temp_p in (gait_temp, voice_temp, hw_temp):
                if temp_p and os.path.exists(temp_p):
                    try:
                        os.remove(temp_p)
                    except Exception:
                        pass
    else:
        # Pre-set Demo Scenarios
        scenarios = load_demo_scenarios()
        selected_scenario_key = st.selectbox(
            "🎯 Select Pre-set Demonstration Scenario (Saved Research Outputs):",
            options=list(scenarios.keys()),
            index=0
        )
        
        current_scenario = scenarios[selected_scenario_key]
        st.info(f"**{current_scenario['title']}** — {current_scenario['desc']}\n\n*Note: Demonstration scenarios use frozen research outputs across all 3 modalities to guarantee presentation stability.*")

        gait_data = current_scenario["gait"]
        voice_data = current_scenario["voice"]
        hw_data = current_scenario["handwriting"]

        # RUN TRI-MODAL CROSS-MODAL FUSION ENGINE
        engine = CrossModalReliabilityEngine()
        fusion_result = engine.fuse_modalities(
            gait_output=gait_data,
            voice_output=voice_data,
            handwriting_output=hw_data
        )

    st.markdown("---")

    # 3. MODALITY EVIDENCE CARDS (3 Columns)
    st.markdown("### 🔬 Unimodal Evidence Cards")
    col_gait, col_voice, col_hw = st.columns(3)

    with col_gait:
        st.markdown("<div style='background-color: #161b22; padding: 1.25rem; border-radius: 8px; border: 1px solid #30363d;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #58a6ff; margin-top:0;'>🏃 GAIT MODALITY</h4>", unsafe_allow_html=True)
        if gait_data is not None:
            st.write(f"**Model:** `1D-CNN + BiLSTM (M=5)`")
            st.write(f"**Prediction:** {gait_data['prediction']}")
            st.write(f"**P(PD):** `{gait_data['p_pd']*100:.1f}%` | **P(HC):** `{gait_data['p_healthy']*100:.1f}%`")
            st.write(f"**Ensemble Agreement:** `{gait_data.get('ensemble_agreement', '5/5')}`")
            st.write(f"**Uncertainty Score:** `{gait_data['uncertainty']:.3f}`")
            
            c_set_str = "{" + ", ".join(gait_data['conformal_prediction_set']) + "}" if gait_data['conformal_prediction_set'] else "{}"
            st.write(f"**Conformal Set:** `{c_set_str}`")

            rel_status = gait_data['reliability']
            rel_color = "#2ea043" if rel_status == "ACCEPT" else "#da3633"
            st.markdown(f"**Reliability:** <span style='color: {rel_color}; font-weight: bold;'>[{rel_status}] ({gait_data['reliability_level']})</span>", unsafe_allow_html=True)
            st.write(f"**Reason:** `{', '.join(gait_data['reliability_reasons'])}`")
        else:
            st.info("ℹ️ Modality not uploaded. Skipped from cross-modal synthesis.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_voice:
        st.markdown("<div style='background-color: #161b22; padding: 1.25rem; border-radius: 8px; border: 1px solid #30363d;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #d2a8ff; margin-top:0;'>🎙️ VOICE MODALITY</h4>", unsafe_allow_html=True)
        if voice_data is not None:
            st.write(f"**Model:** `Wav2Vec2 Ensemble (M=5)`")
            st.write(f"**Prediction:** {voice_data['prediction']}")
            st.write(f"**P(PD):** `{voice_data['p_pd']*100:.1f}%` | **P(HC):** `{voice_data['p_healthy']*100:.1f}%`")
            st.write(f"**Ensemble Agreement:** `{voice_data.get('ensemble_agreement', '5/5')}`")
            st.write(f"**Predictive Entropy:** `{voice_data['uncertainty']:.3f}`")
            
            v_c_set_str = "{" + ", ".join(voice_data['conformal_prediction_set']) + "}" if voice_data['conformal_prediction_set'] else "{}"
            st.write(f"**Conformal Set:** `{v_c_set_str}`")

            v_rel_status = voice_data['reliability']
            v_rel_color = "#2ea043" if v_rel_status == "ACCEPT" else "#da3633"
            st.markdown(f"**Reliability:** <span style='color: {v_rel_color}; font-weight: bold;'>[{v_rel_status}] ({voice_data['reliability_level']})</span>", unsafe_allow_html=True)
            st.write(f"**Reason:** `{', '.join(voice_data['reliability_reasons'])}`")
        else:
            st.info("ℹ️ Modality not uploaded. Skipped from cross-modal synthesis.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_hw:
        st.markdown("<div style='background-color: #161b22; padding: 1.25rem; border-radius: 8px; border: 1px solid #30363d;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #7ee787; margin-top:0;'>✍️ HANDWRITING MODALITY</h4>", unsafe_allow_html=True)
        if hw_data is not None:
            st.write(f"**Model:** `Vision Transformer (ViT EDL)`")
            st.write(f"**Prediction:** {hw_data['prediction']}")
            st.write(f"**P(PD):** `{hw_data['p_pd']*100:.1f}%` | **P(HC):** `{hw_data['p_healthy']*100:.1f}%`")
            st.write(f"**Evidence Head:** `{hw_data.get('evidence', 'N/A')}`")
            st.write(f"**Dirichlet Vacuity:** `{hw_data['uncertainty']:.3f}`")
            
            hw_c_set_str = "{" + ", ".join(hw_data['conformal_prediction_set']) + "}" if hw_data['conformal_prediction_set'] else "{}"
            st.write(f"**Conformal Set:** `{hw_c_set_str}`")

            hw_rel_status = hw_data['reliability']
            hw_rel_color = "#2ea043" if hw_rel_status == "ACCEPT" else "#da3633"
            st.markdown(f"**Reliability:** <span style='color: {hw_rel_color}; font-weight: bold;'>[{hw_rel_status}] ({hw_data['reliability_level']})</span>", unsafe_allow_html=True)
            st.write(f"**Reason:** `{', '.join(hw_data['reliability_reasons'])}`")
        else:
            st.info("ℹ️ Modality not uploaded. Skipped from cross-modal synthesis.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # 4. VISUALLY DOMINANT FINAL TRI-MODAL DECISION BANNER
    st.markdown("### ⚖️ Tri-Modal Evidence Synthesis & Final Prediction")
    
    decision = fusion_result["final_decision"]
    status = fusion_result["cross_modal_status"]
    reason = fusion_result["reason"]
    level = fusion_result["cross_modal_reliability"]
    fused_pred = fusion_result["fused_prediction"]
    fused_p_pd = fusion_result["fused_p_pd"]
    fused_p_hc = fusion_result["fused_p_healthy"]

    banner_bg = "#0d1117"
    banner_border = "#2ea043" if decision == "ACCEPT" else "#da3633"
    decision_color = "#3fb950" if decision == "ACCEPT" else "#f85149"
    badge_icon = "🟢" if decision == "ACCEPT" else ("🔴" if status == "MODALITY_CONFLICT" else "🟠")

    st.markdown(f"""
    <div style='background-color: {banner_bg}; padding: 1.75rem; border-radius: 12px; border: 3px solid {banner_border}; text-align: center;'>
        <h3 style='margin-top: 0; color: #8b949e;'>TRI-MODAL EVIDENCE SYNTHESIS</h3>
        <div style='display: flex; justify-content: space-around; margin: 1.25rem 0;'>
            <div>
                <span style='color: #8b949e; font-size: 0.9rem;'>Tri-Modal Consensus</span><br>
                <span style='font-size: 1.2rem; font-weight: bold;'>{"UNANIMOUS (3/3)" if fusion_result["prediction_agreement"] else "CONFLICT (2 vs 1)"}</span>
            </div>
            <div>
                <span style='color: #8b949e; font-size: 0.9rem;'>Conformal Status</span><br>
                <span style='font-size: 1.1rem; font-weight: bold;'>{fusion_result["conformal_agreement_status"]}</span>
            </div>
            <div>
                <span style='color: #8b949e; font-size: 0.9rem;'>Cross-Modal Status</span><br>
                <span style='font-size: 1.2rem; font-weight: bold;'>{badge_icon} {status}</span>
            </div>
        </div>
        <hr style='border-color: #30363d; margin: 1rem 0;'>
        <div style='margin-top: 0.75rem;'>
            <span style='color: #8b949e; font-size: 1.05rem;'>FINAL TRI-MODAL PREDICTION (BASED ON ALL 3 MODALITIES):</span><br>
            <span style='font-size: 2.2rem; font-weight: 800; color: #58a6ff;'>{fused_pred}</span><br>
            <span style='color: #8b949e; font-size: 1.0rem;'>Fused P(PD): <b>{fused_p_pd*100:.1f}%</b> | Fused P(HC): <b>{fused_p_hc*100:.1f}%</b></span><br><br>
            <span style='color: #8b949e; font-size: 1.1rem;'>SYSTEM RELIABILITY DECISION:</span><br>
            <span style='font-size: 2.8rem; font-weight: 900; color: {decision_color}; letter-spacing: 2px;'>[{decision}]</span><br>
            <span style='font-size: 1.1rem; color: #c9d1d9; font-weight: 600;'>Reliability Level: {level} | Policy Reason: {reason}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Why ACCEPT / Why FLAG explanation box
    if decision == "ACCEPT":
        st.success(
            f"**WHY ACCEPT? (Final Prediction: {fused_pred})**\n\n"
            "✓ All 3 modalities (Gait, Voice, and Handwriting) independently support this prediction.\n\n"
            "✓ Split Conformal prediction sets concur across modalities.\n\n"
            "✓ All 3 modalities report HIGH reliability with low predictive uncertainty."
        )
    else:
        st.error(
            f"**WHY FLAG? (Majority Vote: {fused_pred})**\n\n"
            f"⚠ **Primary Reason:** `{reason}`\n\n"
            + "\n\n".join([f"• {item}" for item in fusion_result["evidence_summary"]]) +
            "\n\n*The reliability engine explicitly flags the decision for clinical verification rather than forcing an unsupported diagnosis.*"
        )

    # 5. EXPANDABLE EVIDENCE INSPECTION (All 3 Modalities)
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        with st.expander("🔍 Inspect Gait Evidence Details"):
            st.json(gait_data)
        with st.expander("🔍 Inspect Voice Evidence Details"):
            st.json(voice_data)
    with col_exp2:
        with st.expander("🔍 Inspect Handwriting Evidence Details"):
            st.json(hw_data)
        with st.expander("🔍 Inspect Tri-Modal Fusion Logic Details"):
            st.json(fusion_result)

    # 6. EXPLAINABILITY & CLINICAL ATTRIBUTION MAPS
    st.markdown("---")
    st.markdown("### 🧠 Clinical Explainability & Feature Attribution Maps")
    st.markdown(
        "<p style='color: #8b949e;'>Inspect neural network attributions across continuous VGRF sensor signals, vocal spectrograms, and kinematic drawing strokes.</p>",
        unsafe_allow_html=True
    )

    exp_tab_gait, exp_tab_voice, exp_tab_hw = st.tabs([
        "🏃 Gait (1D Grad-CAM & IG)",
        "🎙️ Voice (2D Grad-CAM & Formants)",
        "✍️ Handwriting (ViT Attention Rollout)"
    ])

    with exp_tab_gait:
        gait_img = root_dir / "results" / "explanations" / "Patient_Sample_gait_explainability.png"
        if not gait_img.exists():
            gait_img = root_dir / "Gait" / "outputs" / "plots" / "test_gait_explainability.png"
        if gait_img.exists():
            st.image(str(gait_img), caption="Gait Modality: 1D Grad-CAM Temporal Saliency & Integrated Gradients Bilateral Asymmetry", use_container_width=True)
        else:
            st.info("Upload gait time-series or run inference with `--explain` to generate live gait attribution maps.")

    with exp_tab_voice:
        voice_img = root_dir / "Voice" / "results" / "voice_explainability.png"
        if not voice_img.exists():
            voice_img = root_dir / "results" / "explanations" / "Patient_Sample_voice_explainability.png"
        if voice_img.exists():
            st.image(str(voice_img), caption="Voice Modality: 2D Grad-CAM Spectrogram Saliency & Marginal Frequency Band Breakdown", use_container_width=True)
        else:
            st.info("Upload voice audio or run inference with `--explain` to generate live acoustic saliency maps.")

    with exp_tab_hw:
        hw_img = root_dir / "Handwriting" / "outputs" / "vit_explainability.png"
        if not hw_img.exists():
            hw_img = root_dir / "results" / "explanations" / "Patient_Sample_hw_explainability.png"
        if hw_img.exists():
            st.image(str(hw_img), caption="Handwriting Modality: Vision Transformer (ViT) Attention Rollout Saliency Overlay", use_container_width=True)
        else:
            st.info("Upload drawing image or run inference with `--explain` to generate live handwriting attention maps.")

