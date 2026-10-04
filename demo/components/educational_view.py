"""Educational View Component ("Why Reliability?") for Streamlit Demo App.

Explains the conceptual shift from Traditional ML to Reliability-Aware Systems.
"""

import streamlit as st


def render_educational_view():
    """Render the "Why Reliability?" educational page."""
    st.markdown("<h2 style='text-align: center;'>💡 WHY RELIABILITY?</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8b949e;'>The Shift from Black-Box Accuracy to Trustworthy AI Decisions</p>", unsafe_allow_html=True)
    st.markdown("---")

    col_trad, col_our = st.columns(2)

    with col_trad:
        st.markdown("""
        <div style='background-color: #161b22; padding: 1.5rem; border-radius: 8px; border: 1px solid #da3633;'>
            <h3 style='color: #f85149; margin-top: 0;'>❌ TRADITIONAL ML PIPELINE</h3>
            <p>Standard classification models output a raw probability and force a label regardless of model confidence or data quality:</p>
            <div style='background-color: #0d1117; padding: 1rem; border-radius: 6px; text-align: center; font-family: monospace;'>
                Input Data<br>↓<br>Model<br>↓<br>Prediction (Label)
            </div>
            <br>
            <p style='color: #8b949e;'><b>Flaws:</b></p>
            <ul>
                <li>Overconfident on out-of-distribution noise</li>
                <li>Silent failure on borderline predictions (e.g., P=51%)</li>
                <li>Hides disagreement between sensors/modalities</li>
                <li>Forces binary label even when data is ambiguous</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_our:
        st.markdown("""
        <div style='background-color: #161b22; padding: 1.5rem; border-radius: 8px; border: 1px solid #2ea043;'>
            <h3 style='color: #3fb950; margin-top: 0;'>✓ OUR RELIABILITY SYSTEM</h3>
            <p>Our pipeline evaluates multiple evidence layers before deciding whether to ACCEPT or FLAG a prediction:</p>
            <div style='background-color: #0d1117; padding: 1rem; border-radius: 6px; text-align: center; font-family: monospace;'>
                Input Data<br>↓<br>Prediction<br>↓<br>Uncertainty (Entropy & Variance)<br>↓<br>Conformal Evidence Set<br>↓<br>Reliability Engine Policy<br>↓<br>Cross-Modal Consistency<br>↓<br><span style='color:#3fb950; font-weight:bold;'>ACCEPT / FLAG</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 🎯 Core Conceptual Distinction")
    col_q1, col_q2 = st.columns(2)

    with col_q1:
        st.info("### 🎯 Accuracy Answers:\n\n**\"Was the model correct on past test labels?\"**\n\nMeasures retrospective performance across a dataset.")

    with col_q2:
        st.success("### 🛡️ Reliability Asks:\n\n**\"Should THIS specific prediction be trusted right now?\"**\n\nEvaluates live inference confidence and evidence coherence.")

    st.markdown("---")
    st.markdown("### 🔍 Key Safety Principles")
    st.markdown(r"""
    1. **Never Average Conflicting Modalities:** If Gait predicts PD (85%) and Voice predicts Healthy (65%), averaging them gives P=60% (PD), hiding the conflict. Our system flags `MODALITY_CONFLICT`.
    2. **Conformal Guarantees:** Prediction sets $C(X)$ bound error rates at nominal $90\%$ target levels.
    3. **Human-in-the-Loop:** Flagged predictions are deferred for clinician review rather than giving false diagnostic certainty.
    """)
