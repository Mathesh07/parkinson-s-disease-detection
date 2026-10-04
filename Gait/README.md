# Gait-Based Parkinson's Disease Detection: Uncertainty & Cohort-Shift Reliability Framework

## Research Focus
> **"Can Uncertainty Warn Us? Cohort-Shift Reliability of Parkinson's Disease Classifiers Across Voice, Handwriting and Gait"**

This module provides the complete, self-contained **Gait modality** implementation. The core research objective is to determine whether modern uncertainty quantification methods (Softmax Confidence, Temperature Scaling, Evidential Deep Learning, Monte Carlo Dropout, Deep Ensembles, and Conformal Prediction) reliably identify untrustworthy predictions under unseen cohort shift and controlled signal corruption.

---

## Table of Contents
1. [Dataset & Cohort Characteristics](#1-dataset--cohort-characteristics)
2. [Input Signal Specification (16-Channel VGRF)](#2-input-signal-specification-16-channel-vgrf)
3. [Sampling Rate & Preprocessing](#3-sampling-rate--preprocessing)
4. [Temporal Windowing (5.0s Windows, 50% Overlap)](#4-temporal-windowing-50s-windows-50-overlap)
5. [Subject-Level Splitting & Zero-Leakage Protocol](#5-subject-level-splitting--zero-leakage-protocol)
6. [1D-CNN Backbone Architecture](#6-1d-cnn-backbone-architecture)
7. [2-Layer Bidirectional LSTM](#7-2-layer-bidirectional-lstm)
8. [128-Dimensional Bottleneck Gait Embeddings](#8-128-dimensional-bottleneck-gait-embeddings)
9. [Classification Heads & Representation Flow](#9-classification-heads--representation-flow)
10. [Evidential Deep Learning (EDL)](#10-evidential-deep-learning-edl)
11. [Softmax Baseline ($T=1.0$)](#11-softmax-baseline-t10)
12. [Post-Hoc Temperature Scaling](#12-post-hoc-temperature-scaling)
13. [Monte Carlo Dropout (MC Dropout)](#13-monte-carlo-dropout-mc-dropout)
14. [Deep Ensemble Framework](#14-deep-ensemble-framework)
15. [Inductive Conformal Prediction](#15-inductive-conformal-prediction)
16. [Leave-One-Cohort-Out (LOCO) Evaluation](#16-leave-one-cohort-out-loco-evaluation)
17. [Cohort Shift Dynamics](#17-cohort-shift-dynamics)
18. [Controlled Signal Corruption Suite](#18-controlled-signal-corruption-suite)
19. [Probabilistic Calibration Metrics (ECE & Reliability Diagrams)](#19-probabilistic-calibration-metrics-ece--reliability-diagrams)
20. [Risk-Coverage Analysis & Error Detection (AURC)](#20-risk-coverage-analysis--error-detection-aurc)
21. [The "Silent Confidence" Phenomenon](#21-the-silent-confidence-phenomenon)
22. [Unified Reliability & Abstention Layer](#22-unified-reliability--abstention-layer)
23. [Methodological Limitations & Constraints](#23-methodological-limitations--constraints)

---

## 1. Dataset & Cohort Characteristics
Experiments are conducted on the PhysioNet **Gait in Parkinson's Disease** benchmark database (Goldberger et al.; Hausdorff et al.):
- **Total recordings**: 306 walking trials.
- **Total subjects**: 165 unique participants (93 Parkinson's Disease patients, 72 Healthy Controls).
- **Three independent clinical studies (Cohorts)**:
  1. **Ga (`Ga` - Galit et al.)**: 113 recordings, 47 subjects (29 PD, 18 HC).
  2. **Ju (`Ju` - Hausdorff et al.)**: 129 recordings, 54 subjects (29 PD, 25 HC).
  3. **Si (`Si` - Frenkel-Toledo et al.)**: 64 recordings, 64 subjects (35 PD, 29 HC).

---

## 2. Input Signal Specification (16-Channel VGRF)
Each walking trial records continuous Vertical Ground Reaction Force (VGRF in Newtons) collected through instrumented insole force sensors:
- **Left Foot Sensors (8 channels)**: `L1`, `L2`, `L3`, `L4`, `L5`, `L6`, `L7`, `L8` (covering heel to toe).
- **Right Foot Sensors (8 channels)**: `R1`, `R2`, `R3`, `R4`, `R5`, `R6`, `R7`, `R8`.
- **Total channels**: $C = 16$.

---

## 3. Sampling Rate & Preprocessing
- **Sampling rate**: Exactly $100\text{ Hz}$ ($10\text{ ms}$ interval between successive samples).
- **Channel Normalization**: Channel-wise Z-score standard scaling:
  $$\tilde{x}_{t, c} = \frac{x_{t, c} - \mu_{c}^{\text{train}}}{\sigma_{c}^{\text{train}} + \epsilon}$$
- **Strict Leakage Prevention**: Normalization parameters $\mu_c^{\text{train}}, \sigma_c^{\text{train}}$ are fitted **strictly on training cohort recordings**. Held-out validation and test cohorts are normalized using frozen training statistics.

---

## 4. Temporal Windowing (5.0s Windows, 50% Overlap)
Continuous walking signals are sliced into uniform temporal windows:
- **Window duration**: $5.0\text{ seconds} = 500\text{ time steps}$ at $100\text{ Hz}$.
- **Step size**: $2.5\text{ seconds} = 250\text{ time steps}$ ($50\%$ overlap).
- **Tensor dimension per window**: $(B, 500, 16)$, where $B$ is the batch size.

---

## 5. Subject-Level Splitting & Zero-Leakage Protocol
1. **Subject Disjointness**: Slicing occurs *after* subject-level splitting. All recordings and windows from any subject appear exclusively in Train, Validation, or Test.
2. **Stratified Split**: Within training cohorts, subjects are partitioned with disease-class stratification ($70\%$ train, $15\%$ validation).
3. **Cohort Isolation**: In LOCO experiments, the held-out cohort is completely excluded until final evaluation.

---

## 6. 1D-CNN Backbone Architecture
The temporal feature extractor applies 1D convolutions along the temporal axis ($T=500$):
- **Conv Block 1**: $\text{Conv1D}(16 \to 64\text{ channels}, \text{kernel}=5, \text{padding}=2) \to \text{BatchNorm1D} \to \text{ReLU} \to \text{MaxPool1D}(\text{kernel}=2, \text{stride}=2)$. Output: $(B, 64, 250)$.
- **Conv Block 2**: $\text{Conv1D}(64 \to 128\text{ channels}, \text{kernel}=5, \text{padding}=2) \to \text{BatchNorm1D} \to \text{ReLU} \to \text{MaxPool1D}(\text{kernel}=2, \text{stride}=2)$. Output: $(B, 128, 125)$.

---

## 7. 2-Layer Bidirectional LSTM
Captures bidirectional gait dynamics and stride periodicity:
- **Input dimension**: $128$ features per step.
- **Hidden size**: $128$ units per direction $\to 256$ concatenated outputs per step.
- **Layers**: 2 stacked BiLSTM layers with recurrent dropout ($p=0.3$).
- **Temporal Pooling**: Global temporal mean pooling over sequence length ($125$ steps) yields a 256-dimensional summary vector.

---

## 8. 128-Dimensional Bottleneck Gait Embeddings
The pooled BiLSTM output passes through a linear embedding projection:
$$\mathbf{z} = \text{Dropout}_{0.3}(\text{ReLU}(\text{BatchNorm1D}(\mathbf{W}_{\text{emb}} \mathbf{h}_{\text{pooled}} + \mathbf{b}_{\text{emb}})))$$
- Yields a fixed **128-dimensional embedding vector** designed for modular multimodal fusion and latent representation analysis.

---

## 9. Classification Heads & Representation Flow
The model features two parallel heads operating on the 128-D embedding:
1. **Standard Linear Head**: Computes unconstrained raw class logits $z_{\text{HC}}, z_{\text{PD}}$.
2. **Evidential Dirichlet Head**: Applies a linear layer followed by $\text{Softplus}$ activation to ensure non-negative evidence $e_k \ge 0$.

---

## 10. Evidential Deep Learning (EDL)
Parameterizes a Dirichlet distribution $\text{Dir}(\boldsymbol{\alpha})$ over class probabilities (Sensoy et al.):
$$\alpha_k = e_k + 1, \quad S = \sum_{k=1}^K \alpha_k, \quad \hat{p}_k = \frac{\alpha_k}{S}$$
- **Epistemic Vacuity (Uncertainty)**:
  $$u = \frac{K}{S} = \frac{2}{\alpha_{\text{HC}} + \alpha_{\text{PD}}} \in (0, 1]$$
- **Training Loss**: Expected Cross-Entropy (ACE) with annealed Kullback-Leibler divergence penalty regularizing evidence towards a uniform prior for incorrect predictions.

---

## 11. Softmax Baseline ($T=1.0$)
- Standard uncalibrated probability: $\hat{p}_k = \frac{\exp(z_k)}{\sum_j \exp(z_j)}$.
- Confidence: $conf = \max(\hat{p}_{\text{HC}}, \hat{p}_{\text{PD}})$.
- Uncertainty: $u_{\text{softmax}} = 1.0 - conf$.

---

## 12. Post-Hoc Temperature Scaling
- Calibrated probability: $\hat{p}_k(T) = \frac{\exp(z_k / T)}{\sum_j \exp(z_j / T)}$.
- **Leakage Guard**: Parameter $T > 0$ is optimized exclusively on validation set logits using bounded scalar minimization of NLL. Test logits are strictly isolated.

---

## 13. Monte Carlo Dropout (MC Dropout)
- Activates dropout layers ($p=0.3$) at test time across $N=30$ stochastic forward passes.
- Computes:
  - **Predictive Mean**: $\bar{p} = \frac{1}{N}\sum_{n=1}^N p^{(n)}$.
  - **Predictive Entropy**: $H(\bar{p}) = -\sum_k \bar{p}_k \log \bar{p}_k$ (total uncertainty).
  - **Mutual Information**: $I(y, \theta | x) = H(\bar{p}) - \frac{1}{N}\sum_{n=1}^N H(p^{(n)})$ (epistemic uncertainty).

---

## 14. Deep Ensemble Framework
- Trains $M=5$ stochastically independent models initialized with distinct random seeds ($42, 43, 44, 45, 46$).
- Non-shared weights, non-shared optimizer states, and independent batch orderings.
- Window-level ensemble mean probability $\bar{p} = \frac{1}{M}\sum_{m=1}^M p_m$ and predictive entropy $H(\bar{p})$.
- Prediction variance across members: $\text{Var}(p_{\text{PD}}) = \frac{1}{M}\sum_{m=1}^M (p_{m,\text{PD}} - \bar{p}_{\text{PD}})^2$.

---

## 15. Inductive Conformal Prediction
- Constructs finite-sample prediction sets $C(x) \subseteq \{\text{Healthy}, \text{PD}\}$ with nominal marginal coverage guarantee $1 - \alpha$.
- Nonconformity scores $s_i = 1 - P(Y = y_i | x_i)$ calibrated exclusively on validation subjects.
- Evaluates empirical coverage and prediction set size under distribution shift, demonstrating that **exchangeability violations** degrade theoretical coverage guarantees.

---

## 16. Leave-One-Cohort-Out (LOCO) Evaluation
Three complete cross-cohort experiments evaluating transferability across clinical centers:
- **Exp 1**: Train on `Ga` + `Ju` $\to$ Test on `Si` (64 test subjects).
- **Exp 2**: Train on `Ga` + `Si` $\to$ Test on `Ju` (54 test subjects).
- **Exp 3**: Train on `Ju` + `Si` $\to$ Test on `Ga` (47 test subjects).
- Hierarchical aggregation: $\text{Window} \to \text{Recording} \to \text{Subject}$.

---

## 17. Cohort Shift Dynamics
- When models encounter a new clinical cohort with differing patient demographics or walking protocols, balanced accuracy varies between $0.47$ and $0.67$.
- Probabilistic calibration degrades significantly (NLL spikes up to $>1.0$, ECE increases to $0.20-0.30$).

---

## 18. Controlled Signal Corruption Suite
Five synthetic corruption transformations applied at 5 severity levels ($0 = \text{clean}, 1, 2, 3, 4$):
1. **Gaussian Noise**: Additive sensor thermal noise ($\sigma \in [0.0, 0.50]$).
2. **Amplitude Scaling**: Body weight / sensor calibration shift (scaling factor $\in [1.0, 0.40]$).
3. **Channel Dropout**: Hardware sensor failure ($0$ to $6$ channels set to zero).
4. **Temporal Masking**: Intermittent data loss / occlusion ($0$ to $200$ consecutive time steps zeroed).
5. **Moving-Average Smoothing**: Low-pass temporal filtering / sensor dampening (kernel sizes $1$ to $9$).

---

## 19. Probabilistic Calibration Metrics (ECE & Reliability Diagrams)
- **Expected Calibration Error (ECE)** with 10 bins:
  $$\text{ECE} = \sum_{m=1}^{10} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
- Edge-inclusive binning ensures full coverage of $conf \in [0, 1]$.
- Reliability diagrams plot nominal confidence vs. empirical accuracy.

---

## 20. Risk-Coverage Analysis & Error Detection (AURC)
- **Error Detection AUROC**: Ability of uncertainty metric to rank misclassifications higher than correct classifications.
- **Area Under Risk-Coverage (AURC)**: Measures error rate reduction when the model is permitted to abstain on high-uncertainty instances.

---

## 21. The "Silent Confidence" Phenomenon
A central empirical finding across all 300 experimental conditions:
> **As signal corruption increases and classification error rises, predictive uncertainty DOES NOT reliably increase. In fact, deep models frequently become more overconfident in their misclassifications.**

Nominal confidence alone is therefore an unsafe safeguard against distribution shift.

---

## 22. Unified Reliability & Abstention Layer
Combines calibrated probabilities, epistemic uncertainty, and conformal prediction sets to output actionable decisions:
- **`ACCEPT`**: High confidence, low uncertainty, and consistent singleton conformal prediction set.
- **`FLAG`**: High uncertainty, marginal confidence, ambiguous conformal set ($\{\text{Healthy}, \text{PD}\}$), or prediction contradiction.
- All decision thresholds are tuned strictly on validation data with zero test label leakage.

---

## 23. Methodological Limitations & Constraints
1. **Computational Platform**: All experiments are designed and benchmarked for CPU execution.
2. **Cross-Modality Association**: Gait, handwriting, and speech datasets come from independent clinical studies; multimodal analysis operates at cohort/population shift levels rather than paired patient-level fusion.
3. **Distribution Shift Violations**: Conformal coverage guarantees depend fundamentally on exchangeability. Under severe cohort shift or corruption, conformal guarantees no longer hold.
4. **Clinical Scope**: This framework is an academic exploration of uncertainty and distribution shift reliability; it is not validated as a standalone medical diagnostic device.

---

## Reproduction Commands

### Run Full Test Suite
```bash
python -m pytest Gait/tests/ -v
```

### Multi-Seed LOCO Experiment
```bash
python -m Gait.experiments.multiseed_loco --epochs 25 --seeds 42 43 44 45 46
```

### Deep Ensemble Experiment
```bash
python -m Gait.experiments.deep_ensemble --epochs 25 --seeds 42 43 44 45 46
```

### Conformal Prediction Experiment
```bash
python -m Gait.experiments.conformal_prediction --epochs 25
```

### Unified Reliability Analysis
```bash
python -m Gait.experiments.reliability_analysis --epochs 25
```

### Final Comparison Synthesis
```bash
python -m Gait.experiments.final_comparison
```

### Interactive CLI Demonstration
```bash
python -m Gait.demo --plot
```
