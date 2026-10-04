"""Architecture View Component for Streamlit Demo App.

Renders the end-to-end Parkinson's Reliability System dataflow architecture.
"""

import streamlit as st


def render_architecture_view():
    """Render system architecture diagram and component explanation."""
    st.markdown("<h2 style='text-align: center;'>🏗️ SYSTEM ARCHITECTURE</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8b949e;'>End-to-End Decision-Level Multimodal Reliability System Dataflow</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.code("""
                                 INPUT DATA
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
     GAIT RECORDING                                      VOICE AUDIO
   (Foot Force Sensors)                                (16 kHz WAV)
           │                                                   │
           ▼                                                   ▼
  Feature Extraction (19D)                             Wav2Vec2 Backbone
           │                                                   │
   ┌───────┴───────┐                                   ┌───────┴───────┐
   ▼               ▼                                   ▼               ▼
Ensemble M=5    Evidential Head                     Ensemble M=5    MC Dropout N=30
   │               │                                   │               │
   ▼               ▼                                   ▼               ▼
Gait Probability  Uncertainty                       Voice Prob      Uncertainty
   │               │                                   │               │
   └───────┬───────┘                                   └───────┬───────┘
           ▼                                                   ▼
  Gait Conformal Set                                 Voice Conformal Set
   (q_hat = 0.8147)                                   (q_hat = 0.4075)
           │                                                   │
           ▼                                                   ▼
 Gait Reliability Engine                            Voice Reliability Engine
           │                                                   │
           └─────────────────────────┬─────────────────────────┘
                                     ▼
                      CROSS-MODAL RELIABILITY ENGINE
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           ▼                         ▼                         ▼
     CASE A: CONSISTENT       CASE B: PARTIAL          CASE C: CONFLICT
      [ACCEPT] (HIGH)          [FLAG] (MEDIUM)          [FLAG] (LOW)
    """, language="text")

    st.markdown("### 🧩 Core Component Responsibilities")

    st.markdown(r"""
    1. **Preprocessing & Feature Extraction:**
       - **Gait:** 19 spatial-temporal gait parameters extracted per stride (stride duration, stance phase %, swing phase %, double support time).
       - **Voice:** Raw 16 kHz audio segmented into 10-second chunks with 2-second overlap, passed through pre-trained Wav2Vec2 backbone.

    2. **Primary Predictions & Deep Ensembles ($M=5$):**
       - 5 independently trained model members combined via prediction mean $P(\text{PD})$ and variance $\sigma^2$.

    3. **Uncertainty Quantification:**
       - **Gait:** Evidential Deep Learning (Dirichlet distribution parameters $\alpha_{\text{HC}}, \alpha_{\text{PD}}$) & MC Dropout.
       - **Voice:** Predictive Entropy $H(p) = -p \log(p) - (1-p) \log(1-p)$ & Mutual Information $\text{MI} = H(\bar{p}) - \mathbb{E}[H(p_m)]$.

    4. **Split Conformal Prediction Sets ($C(X)$):**
       - Guarantees finite-sample coverage at nominal $90\%$ target level ($\alpha = 0.10$).
       - Produces prediction sets: $\{\text{Healthy Control}\}$, $\{\text{Parkinson's Disease}\}$, $\{\text{Healthy}, \text{PD}\}$, or $\emptyset$.

    5. **Individual Reliability Engines (Gait & Voice):**
       - Evaluates prediction set size, ensemble agreement, entropy thresholds, and decision boundary margins.
       - Assigns modality decisions: `ACCEPT` (HIGH) vs `FLAG` (MEDIUM/LOW).

    6. **Cross-Modal Fusion Layer:**
       - Decision-level evidence integration without raw feature concatenation or silent probability averaging.
       - Detects cross-modal consistency vs direct modality conflict.
    """)
