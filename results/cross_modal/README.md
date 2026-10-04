# Cross-Modal Reliability Fusion Engine (Gait + Voice)

The **Cross-Modal Reliability Fusion Engine** provides a decision-level evidence integration layer for Parkinson's Disease detection. Rather than combining raw neural network embeddings or averaging conflicting probabilities, the engine evaluates whether independently trained **Gait** and **Voice** reliability pipelines yield consistent and reliable predictive evidence.

---

## 1. System Architecture

```
         GAIT SENSOR DATA                         VOICE AUDIO WAVEFORM
                │                                          │
                ▼                                          ▼
       Gait Feature Pipeline                   Wav2Vec2 Voice Pipeline
                │                                          │
                ▼                                          ▼
     Gait Reliability Engine                    Voice Reliability Engine
    (Decision, Set, Uncertainty)              (Decision, Set, Uncertainty)
                │                                          │
                └────────────────────┬─────────────────────┘
                                     ▼
                   CROSS-MODAL RELIABILITY ENGINE
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           ▼                         ▼                         ▼
    CASE A: CONSISTENT       CASE B: PARTIAL          CASE C: CONFLICT
    (ACCEPT / HIGH)           (FLAG / MEDIUM)          (FLAG / LOW)
```

---

## 2. Inputs from Gait & Voice Modalities

Each modality supplies a standardized reliability output dictionary:
- `prediction`: Point prediction label ("Healthy Control" or "Parkinson's Disease").
- `predicted_label`: Binary label (0=HC, 1=PD).
- `p_pd`: Model probability for PD.
- `p_healthy`: Model probability for Healthy ($1 - P(\text{PD})$).
- `uncertainty`: Modality predictive entropy / uncertainty score.
- `conformal_prediction_set`: Split conformal prediction set $C(X)$.
- `reliability`: Modality decision (`ACCEPT` or `FLAG`).
- `reliability_level`: Categorical level (`HIGH`, `MEDIUM`, `LOW`).

---

## 3. Cross-Modal Agreement & Conformal Comparison

- **Prediction Agreement:** Evaluates whether `gait.predicted_label == voice.predicted_label`.
- **Conformal Set Comparison:**
  - `CONFORMAL_AGREEMENT`: Both prediction sets are singletons supporting the identical class.
  - `CONFORMAL_CONFLICT`: Disjoint singletons ($\{\text{PD}\}$ vs $\{\text{Healthy}\}$).
  - `PARTIAL_CONFORMAL_AGREEMENT`: Intersection is non-empty, but one set is ambiguous ($\{|C|=2\}$).
  - `UNCERTAINTY_CONFORMAL_EVIDENCE`: One or both sets are empty ($\{|C|=0\}$).

---

## 4. Cross-Modal Reliability Policy Rules

- **CASE A — STRONG AGREEMENT:**
  - `gait.predicted_label == voice.predicted_label`
  - Both modalities report `RELIABLE` status (`ACCEPT` with `HIGH` level)
  - Conformal sets concur
  - **Status:** `CONSISTENT` | **Level:** `HIGH` | **Decision:** `ACCEPT` | **Reason:** `CROSS_MODAL_AGREEMENT`

- **CASE B — AGREEMENT BUT ONE MODALITY UNCERTAIN:**
  - Predictions agree, but at least one modality reports `CAUTION` or `UNRELIABLE` (`FLAG`)
  - **Status:** `PARTIAL_SUPPORT` | **Level:** `MEDIUM` | **Decision:** `FLAG` | **Reason:** `ONE_MODALITY_UNCERTAIN`

- **CASE C — DIRECT MODALITY CONFLICT:**
  - Gait prediction != Voice prediction
  - **Status:** `MODALITY_CONFLICT` | **Level:** `LOW` | **Decision:** `FLAG` | **Reason:** `GAIT_VOICE_CONFLICT`

- **CASE D — BOTH MODALITIES UNCERTAIN:**
  - Both modalities report `UNRELIABLE`
  - **Status:** `INSUFFICIENT_EVIDENCE` | **Level:** `LOW` | **Decision:** `FLAG` | **Reason:** `BOTH_MODALITIES_UNRELIABLE`

- **CASE E — SINGLE MODALITY AVAILABLE:**
  - Gait only or Voice only available
  - **Status:** `SINGLE_MODALITY` | **Level:** Inherited | **Decision:** Inherited | **Reason:** `SINGLE_MODALITY_EVIDENCE`

---

## 5. Conflict & Probability Averaging Policy

- **No Silent Probability Averaging:** The engine does **NOT** average conflicting probabilities (e.g. $(P_{\text{gait}} + P_{\text{voice}})/2$). Silent averaging masks genuine cross-modal disagreement.
- **Explicit Conflict Flagging:** Modality disagreement is an essential diagnostic signal. When Gait and Voice disagree, the system outputs `MODALITY_CONFLICT` and `FLAG`s the prediction for clinical review.

---

## 6. Dataset Cohort & Synthetic Pairing Note

- Gait data (PhysioNet Ga/Ju/Si cohorts) and Voice data (MDVR-KCL dataset) originate from independent clinical databases.
- **No true paired subjects exist across datasets.**
- Demonstration cases use synthetic pairings of existing modality outputs strictly for evaluating decision fusion logic. Synthetic pairings are never presented as real patient evaluations.

---

## 7. Limitations & Medical Disclaimer

- Thresholds are pre-declared engineering heuristics, not fitted on test labels.
- Fusion operates at the decision level, not the feature representation level.
- **Medical Disclaimer:**
  > **The Cross-Modal Reliability Fusion Engine estimates evidence consistency across machine learning model predictions. It does not provide medical diagnosis or determine whether a patient has Parkinson's disease.**
