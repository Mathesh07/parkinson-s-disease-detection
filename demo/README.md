# Streamlit Demo Application: Parkinson's Reliability Analyzer

This interactive Streamlit application provides the final live demonstration system for the **Parkinson's Disease Detection + Reliability-Aware Analysis** project.

---

## 🚀 How to Launch the Web Application

From the project root directory (`D:\projects\parkinson-s-disease-detection`), execute:

```bash
streamlit run demo/app.py
```

The application will launch locally at `http://localhost:8501`.

---

## 📱 Application Views & Pages

1. **🔍 Parkinson's Reliability Analyzer (Main Screen):**
   - File uploaders for Gait datasets and Voice WAV audio.
   - Pre-set Demo Scenarios (Strong Agreement, Modality Conflict, One Modality Uncertain).
   - Side-by-side Gait and Voice evidence cards.
   - Visually dominant Cross-Modal Decision Banner (`[ACCEPT]` vs `[FLAG]`).
   - "Why ACCEPT?" / "Why FLAG?" explanation boxes.
   - Expandable raw evidence detail panels.

2. **🏗️ System Architecture:**
   - Visual end-to-end dataflow diagram from sensor inputs through feature extractors, Deep Ensembles ($M=5$), MC Dropout ($N=30$), Split Conformal layers, individual Reliability Engines, up to the Cross-Modal Fusion layer.

3. **💡 Why Reliability?:**
   - Educational comparison between Traditional ML ($\text{Input} \rightarrow \text{Model} \rightarrow \text{Prediction}$) and Reliability-Aware Systems ($\text{Input} \rightarrow \text{Prediction} \rightarrow \text{Uncertainty} \rightarrow \text{Conformal} \rightarrow \text{Reliability} \rightarrow \text{ACCEPT/FLAG}$).
   - Highlights: *"Accuracy answers: Was the model correct? Reliability asks: Should this prediction be trusted?"*

4. **📊 Experimental Evidence & Dashboards:**
   - Completed research milestone timeline across Gait, Voice, and Cross-Modal phases.
   - Embedded high-value dashboard visualizations.

---

## 🛡️ Deterministic Fallback Demo Mode
If PyTorch model loading is unavailable or hardware acceleration is absent, the demo scenarios fallback seamlessly to pre-calculated saved outputs from Phase 8 & 9. This guarantees 100% presentation stability during live panel demonstrations on any laptop.

---

## 🔬 Medical Disclaimer
> **Research Prototype** — Model Reliability & Evidence Consistency Analysis Only. Not a clinical diagnostic system. Does not perform medical diagnosis.
