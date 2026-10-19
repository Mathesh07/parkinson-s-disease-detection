"""Architecture View Component for Streamlit Demo App.

Renders the end-to-end Tri-Modal Parkinson's Reliability System dataflow architecture.
"""

import streamlit as st


def render_architecture_view():
    """Render system architecture diagram and component explanation."""
    st.markdown("<h2 style='text-align: center;'>🏗️ SYSTEM ARCHITECTURE</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8b949e;'>End-to-End Decision-Level Tri-Modal Reliability System Dataflow</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.code("""
                                           PATIENT MULTIMODAL DATA
                                                      │
         ┌────────────────────────────────────────────┼────────────────────────────────────────────┐
         ▼                                            ▼                                            ▼
   GAIT RECORDING                                VOICE AUDIO                               HANDWRITING DRAWINGS
 (VGRF Force Sensors)                           (16 kHz WAV)                             (Spiral & Meander Images)
         │                                            │                                            │
         ▼                                            ▼                                            ▼
1D-CNN + BiLSTM                              Wav2Vec2 Backbone                           Vision Transformer (ViT)
         │                                            │                                            │
  ┌──────┴──────┐                              ┌──────┴──────┐                              ┌──────┴──────┐
  ▼             ▼                              ▼             ▼                              ▼             ▼
Ensemble M=5  Evidential Head                Ensemble M=5  MC Dropout N=30                Evidential    Attention
  │             │                              │             │                            Head (EDL)     Rollout
  ▼             ▼                              ▼             ▼                              │             │
Gait Prob   Uncertainty                      Voice Prob   Uncertainty                       ▼             ▼
  │             │                              │             │                          Handwriting   Explainability
  └──────┬──────┘                              └──────┬──────┘                          Prob & Vacuity   Heatmap
         ▼                                            ▼                                     │             │
Gait Conformal Set                           Voice Conformal Set                            └──────┬──────┘
 (q_hat = 0.8147)                             (q_hat = 0.4075)                                     ▼
         │                                            │                                Handwriting Conformal Set
         ▼                                            ▼                                     (q_hat = 0.4500)
Gait Reliability                             Voice Reliability                                     │
     Engine                                       Engine                                           ▼
         │                                            │                                  Handwriting Reliability
         │                                            │                                          Engine
         └────────────────────────────────────┬───────┴────────────────────────────────────────────┘
                                              ▼
                             TRI-MODAL CROSS-MODAL FUSION ENGINE
                          (Consensus, Voting & Uncertainty Weighting)
                                              │
         ┌────────────────────────────────────┼────────────────────────────────────┐
         ▼                                    ▼                                    ▼
  CASE A: TRI-MODAL CONSENSUS          CASE B: UNCERTAINTY DETECTED          CASE C: 2 vs 1 CONFLICT
    (All 3 Agree, Low Uncertainty)       (Unanimous, but 1-2 Uncertain)        (2 Predict X, 1 Predicts Y)
       FINAL DECISION: [ACCEPT]             FINAL DECISION: [FLAG]                FINAL DECISION: [FLAG]
       Reliability: HIGH (Safe)             Reliability: MEDIUM (Review)          Reliability: LOW (Conflict)
    """, language="text")

    st.markdown("### 🧩 Core Component Responsibilities")

    st.markdown(r"""
    1. **Preprocessing & Feature Extraction:**
       - **Gait:** 19 spatial-temporal gait parameters per stride (stride duration, stance/swing %, double support) from VGRF sensors.
       - **Voice:** Raw 16 kHz audio segmented into 10-second chunks with 2-second overlap, passed through pre-trained `facebook/wav2vec2-base`.
       - **Handwriting:** Archimedean spirals and meanders resized to 224×224, normalized and tokenized into 14×14 patches via `google/vit-base-patch16-224`.

    2. **Primary Predictions & Uncertainty Quantification:**
       - **Gait:** Deep Ensemble ($M=5$) + Evidential Head producing Dirichlet concentration parameters $(\alpha_{\text{HC}}, \alpha_{\text{PD}})$.
       - **Voice:** Deep Ensemble ($M=5$) + MC Dropout ($N=30$) evaluating predictive entropy $H(p)$ and mutual information.
       - **Handwriting:** Evidential Deep Learning (EDL) transforming pooled ViT representations into non-negative evidence ($e_{\text{HC}}, e_{\text{PD}}$) and vacuity uncertainty $u = K/S$.

    3. **Split Conformal Prediction Sets ($C(X)$):**
       - Guarantees finite-sample coverage at nominal $90\%$ target level ($\alpha = 0.10$).
       - Produces prediction sets across modalities: $\{\text{Healthy Control}\}$, $\{\text{Parkinson's Disease}\}$, or ambiguous $\{\text{Healthy}, \text{PD}\}$.

    4. **Unimodal Reliability Engines (Gait, Voice, Handwriting):**
       - Evaluates prediction set size, ensemble agreement, vacuity/entropy thresholds, and decision boundary proximity.
       - Assigns modality decisions: `ACCEPT` (HIGH) vs `FLAG` (MEDIUM/LOW).

    5. **Tri-Modal Cross-Modal Fusion Engine:**
       - Synthesizes evidence across all three modalities without raw feature leakage.
       - Determines the **Final Clinical Prediction** via majority voting and uncertainty-weighted probability synthesis.
       - Enforces safety abstention: if any modality directly conflicts (2 vs 1) or reports excessive uncertainty, the decision is flagged for human review.
    """)
