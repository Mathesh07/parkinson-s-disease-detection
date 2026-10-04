# Unified Reliability & Abstention Layer for Gait Parkinson's Detection

## 1. Overview
The goal of this layer is to prove empirically that **nominal prediction confidence is not equivalent to prediction reliability**. Under cohort shift and signal corruptions, models frequently output high probabilities on incorrect predictions ("silent confidence").

The unified reliability layer combines multiple orthogonal signals:
1. **Calibrated Predictive Probability**: Post-hoc Temperature Scaled logits ($T > 0$ fit on validation set).
2. **Epistemic / Evidential Uncertainty**: Vacuity ($u = K/S$ from EDL) or Predictive Entropy from Ensembles.
3. **Inductive Conformal Prediction Set**: Finite-sample confidence set guarantees under exchangeability.

---

## 2. Thresholding Methodology (Zero Test Leakage)
All decision boundaries are computed **exclusively on validation subjects from training cohorts**:
- **Confidence threshold (`tau_conf`)**: 10th percentile of confidence among correct validation subjects.
- **Uncertainty threshold (`tau_unc`)**: 80th percentile of uncertainty across validation subjects.
- **Conformal threshold (`q_hat`)**: $(1 - \alpha)$-quantile of nonconformity scores on validation subjects (nominal $\alpha = 0.10$).

Test cohort data and labels are strictly isolated and never accessed during threshold determination.

---

## 3. Decision Rules
For each subject:
- **`ACCEPT`**:
  - $Confidence \ge \tau_{conf}$
  - $Uncertainty \le \tau_{unc}$
  - Conformal set is a consistent singleton ($\{PD\}$ for PD, $\{HC\}$ for HC).
- **`FLAG`**:
  - High uncertainty ($u > \tau_{unc}$), OR
  - Low confidence ($conf < \tau_{conf}$), OR
  - Ambiguous conformal set $\{Healthy, PD\}$, OR
  - Empty conformal set $\{\}$, OR
  - Conformal set contradicts point prediction.

---

## 4. Key Results Summary
Filtering out `FLAGGED` predictions substantially increases the accuracy of accepted predictions across all LOCO folds, demonstrating that multi-faceted uncertainty successfully identifies untrustworthy classifications without access to test ground truth.

*(Note: This module is an exploratory research component and is not intended for clinical diagnostic decision-making.)*
