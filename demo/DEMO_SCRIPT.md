# 3-Minute Live Panel Presentation Script
**Parkinson's Disease Detection & Reliability-Aware Analysis**

---

### **0:00 – Introduction & The Core Problem**
> *"Good morning members of the panel. Today, we present our final year project on Parkinson's Disease Detection. Most traditional machine learning systems simply output a raw classification label and force a decision regardless of model confidence or data quality.*
> 
> *Our project introduces a fundamental shift: **Prediction is only the beginning. Can the model's prediction actually be trusted?** We evaluate whether a prediction is supported by uncertainty, ensemble agreement, split conformal sets, and cross-modal consistency before accepting it."*

---

### **0:30 – Demo Scenario 1: Strong Cross-Modal Agreement**
> *(Action: Select 'Scenario 1: Strong Agreement' on the main Analyzer screen)*
> 
> *"Here in Scenario 1, both our Gait pipeline (foot force sensor temporal features) and Voice pipeline (Wav2Vec2 deep audio representations) independently predict Parkinson's Disease with high probabilities (89% and 79%).*
> 
> *Both modalities report strong ensemble agreement (5/5), low predictive entropy, and matching singleton conformal prediction sets (`{Parkinson's Disease}`). The Cross-Modal Reliability Engine recognizes this as **CONSISTENT**, returning a high-confidence **ACCEPT** decision."*

---

### **1:10 – Demo Scenario 2: Direct Modality Conflict (The Key Innovation)**
> *(Action: Select 'Scenario 2: Modality Conflict' on the main Analyzer screen)*
> 
> *"Now, let's look at Scenario 2, which highlights our key methodological contribution. Here, Gait predicts Parkinson's Disease with 85% probability, but Voice predicts Healthy Control with 65% probability.*
> 
> *A traditional machine learning system would average these probabilities to $60\%$ and output 'Parkinson's Disease', completely hiding the sensor disagreement. Our system **never silently averages conflicting modalities**. Instead, it detects the direct prediction conflict, outputs **MODALITY_CONFLICT**, and flags the case with a **FLAG [LOW]** status for clinician review."*

---

### **1:50 – Demo Scenario 3: Agreement but One Modality Uncertain**
> *(Action: Select 'Scenario 3: One Modality Uncertain' on the main Analyzer screen)*
> 
> *"In Scenario 3, both modalities predict Parkinson's Disease. However, the Voice model probability is borderline at $56.9\%$, exhibits high entropy ($0.68$), and yields an empty conformal set.*
> 
> *Because one modality lacks statistical evidence, our fusion policy assigns **PARTIAL_SUPPORT** and flags the prediction rather than outputting false confidence."*

---

### **2:20 – Expandable Evidence Inspection & Transparency**
> *(Action: Expand the 'Inspect Gait Details' and 'Inspect Voice Details' panels)*
> 
> *"The system is completely transparent. At any point, clinicians or researchers can inspect the underlying Deep Ensemble variance, MC Dropout entropy, and exact conformal quantile bounds ($\hat{q}$), ensuring decisions are fully explainable."*

---

### **2:40 – System Architecture Overview**
> *(Action: Switch to the 'System Architecture' tab)*
> 
> *"As shown in our System Architecture diagram, input data flows through dedicated deep learning backbones, Evidential layers, Deep Ensembles ($M=5$), and Split Conformal calibrators into modality reliability engines, concluding at our decision-level Cross-Modal Fusion layer."*

---

### **3:00 – Conclusion & Summary**
> *"In conclusion, our system demonstrates that when machine learning models face noisy, conflicting, or out-of-distribution inputs, **the system does not force an unsupported diagnosis**. It explicitly flags the case and defers to medical experts. Thank you, and we welcome your questions."*
