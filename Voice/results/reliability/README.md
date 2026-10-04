# Voice Reliability Decision Engine

The **Voice Reliability Engine** provides an inference-time reliability assessment for Parkinson's Disease voice classification. Rather than outputting a raw model probability alone, the engine integrates multiple independent evidence sources to evaluate whether the prediction is sufficiently supported by uncertainty and conformal evidence.

---

## 1. System Pipeline Architecture

```
                    AUDIO WAVEFORM
                          │
                          ▼
                    PREPROCESSING (16 kHz, Mono, 10s chunks)
                          │
                          ▼
                WAV2VEC2 FEATURE EXTRACTOR
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
  Deep Ensemble (M=5)          MC Dropout (N=30)
            │                           │
            ▼                           ▼
  Ensemble Distribution         Stochastic Uncertainty
            │                           │
            └─────────────┬─────────────┘
                          ▼
               Split Conformal Layer (q_hat = 0.407552)
                          │
                          ▼
            Prediction Set C(X) = {c | 1 - P(c) <= q_hat}
                          │
                          ▼
                RELIABILITY ENGINE POLICY
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
          ✓ ACCEPT (HIGH)     ⚠ FLAG (CAUTION / UNRELIABLE)
```

---

## 2. Multi-Source Reliability Signals

The engine evaluates four independent signals:
1. **Split Conformal Prediction Set ($C(X)$):** Derived from calibration on validation subjects ($n=5, \alpha=0.10$). Singleton sets ($\{|C|=1\}$) provide strong evidence, whereas empty sets ($\{|C|=0\}$) or ambiguous sets ($\{|C|=2\}$) trigger a reliability flag.
2. **Ensemble Prediction Agreement:** Measures binary prediction agreement across $M=5$ independently trained ensemble members ($\text{Agreement} = \max(\text{num\_PD}, \text{num\_HC}) / 5$). Disagreement ($< 4/5$) triggers a reliability flag.
3. **Predictive Entropy ($H(p)$):** Measures total predictive uncertainty $H(p) = -p \log(p) - (1-p) \log(1-p)$. High entropy ($H(p) \ge 0.65$) triggers a reliability flag.
4. **Decision Boundary Proximity:** Probabilities within the borderline region $P(\text{PD}) \in [0.40, 0.60]$ indicate decision boundary ambiguity and trigger a flag.

---

## 3. Decision Policy Rules

- **`ACCEPT` (HIGH RELIABILITY):**
  - Conformal set is a singleton ($|C|=1$).
  - Ensemble agreement is strong ($\ge 4/5$).
  - Predictive entropy is low to moderate ($H(p) < 0.65$).
  - Mean probability is outside borderline region ($P(\text{PD}) < 0.40$ or $P(\text{PD}) > 0.60$).
  - Categorical Level: `HIGH`. Reason: `CONSISTENT_SINGLETON_PREDICTION`.

- **`FLAG` (CAUTION / UNRELIABLE):**
  Triggered if ANY of the following flag reasons occur:
  - `EMPTY_CONFORMAL_SET`: Prediction set is empty ($\{|C|=0\}$).
  - `AMBIGUOUS_CONFORMAL_SET`: Prediction set contains both classes ($\{|C|=2\}$).
  - `HIGH_ENSEMBLE_DISAGREEMENT`: Ensemble agreement $< 4/5$ ($3/5$ split).
  - `HIGH_UNCERTAINTY`: Predictive entropy $H(p) \ge 0.65$.
  - `BORDERLINE_PROBABILITY`: Mean probability $P(\text{PD}) \in [0.40, 0.60]$.
  - Categorical Level: `MEDIUM` (1 flag reason) or `LOW` ($\ge 2$ flag reasons).

---

## 4. Conformal Integration

- Uses the finite-sample split-conformal quantile calibrated on the 5 validation subjects ($\hat{q}_{\text{ensemble}} = 0.407552$).
- Zero ground-truth test labels are used during inference set construction.
- Nonconformity score: $s = 1 - P(Y_{\text{true}} \mid X)$.

---

## 5. Why Reliability Differs From Classification Accuracy

- **Accuracy** measures whether point predictions match retrospective ground-truth labels.
- **Reliability** measures whether the model's confidence and predictive evidence are sufficiently coherent to trust the decision in live deployment, **before** ground-truth labels are known.
- A high-accuracy model can be unreliable on specific edge cases (e.g. producing $P(\text{PD}) = 0.52$ with high disagreement). The Reliability Engine detects and flags these ambiguous cases.

---

## 6. Limitations

- Calibration sample size $n=5$ yields a coarse finite-sample conformal quantile ($\hat{q} = \max(s_i)$).
- Thresholds are heuristic engineering bounds declared prior to evaluation, not optimized against test labels.

---

## 7. Medical Disclaimer

> **The reliability engine estimates whether the model's prediction is sufficiently supported by the implemented uncertainty and conformal signals. It does not determine whether a patient has Parkinson's disease.**
