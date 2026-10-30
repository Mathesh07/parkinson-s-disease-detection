# Gait Base-Paper-Style 2D CNN Baseline

## 1. Executive Summary & Overview
This directory contains an independent, reproducible implementation of a **2D Convolutional Neural Network (CNN)** baseline inspired by the reference paper:
> **"Multimodal assessment of Parkinson’s disease: a deep learning approach"**  
> *J. C. Vásquez-Correa, J. R. Orozco-Arroyave, et al., IEEE Journal of Biomedical and Health Informatics, Vol. 23, No. 4, July 2019.*

The reference paper introduced multimodal deep learning for Parkinson's disease (PD) classification across speech, handwriting, and gait. For gait analysis, the paper transformed multichannel sensor time series into **Short-Time Fourier Transform (STFT)** time-frequency representations and processed them using 2D CNN architectures.

This implementation adapts the base paper's 2D CNN time-frequency methodology to the project's **PhysioNet Gait in Parkinson's Disease** benchmark dataset to establish a rigorous, fair, and isolated empirical comparison against the proposed **1D CNN + BiLSTM** architecture.

---

## 2. Methodology & Relation to the Reference Paper

### Reference Paper Design:
- **Sensors:** 12-channel kinematic data (3D accelerometers + 3D gyroscopes attached to both shoes) from the eGaIT system, sampled at 102.4 Hz.
- **Input Representation:** Short-Time Fourier Transform (STFT) computed per sensor channel, converting motion dynamics into 2D time-frequency spectrograms.
- **Architecture:** 2D CNN with convolutional layers, ReLU non-linearities, max-pooling, dropout, and fully connected dense layers producing binary classification outputs.
- **Transition Modeling:** Separate evaluation of gait transitions (onset of walking, offset of walking) and steady-state gait cycles.

### Adapted Reproduction in this Repository:
- **Sensors:** 16-channel Vertical Ground Reaction Force (VGRF in Newtons) collected from 8 sensors on the left foot (`L1`..`L8`) and 8 sensors on the right foot (`R1`..`R8`), sampled at 100 Hz.
- **Input Representation:** STFT computed independently across all 16 VGRF channels:
  - FFT size: $N_{\text{fft}} = 64$ (Frequency resolution: $1.56\text{ Hz/bin}$, covering $0 - 50\text{ Hz}$).
  - Analysis window: $W = 64$ samples ($0.64\text{ s}$, Hann window).
  - Hop size: $H = 16$ samples ($0.16\text{ s}$).
  - Spectrogram log-magnitude: $\log(1 + |\text{STFT}|)$.
  - Input tensor shape: $(16, 33, 32)$ corresponding to $(\text{Channels}, \text{Frequency Bins}, \text{Time Frames})$.
- **Zero-Leakage Preprocessing:** Standard scaling $(\mu, \sigma)$ fitted **strictly on training subjects** and frozen during validation and test evaluation.

---

## 3. Key Methodological Differences from Original Paper

| Aspect | Reference Paper (Vásquez-Correa et al.) | Adapted Baseline in this Repo |
| :--- | :--- | :--- |
| **Dataset** | eGaIT gait dataset (88 participants) | PhysioNet Gait in PD benchmark (165 participants) |
| **Sensor Modality** | 12-channel IMU (Accelerometers + Gyroscopes) | 16-channel insole force sensors (VGRF) |
| **Sampling Rate** | 102.4 Hz | 100 Hz |
| **Transition Annotations** | Manual/algorithmic stride & initiation events | Proxy trial boundaries (onset = window 0, offset = window $N-1$) |
| **Model Structure** | 2D CNN on spectrograms | 2D CNN on 16-channel spectrograms |
| **Splits** | 10-fold cross-validation | Strict subject-stratified split (70% train, 15% val, 15% test) |

> **Scientific Transparency Note:** Because the PhysioNet dataset consists of continuous walking trials without footstep-level transition annotations, we do not claim an exact duplication of the paper's annotated stride transitions. Instead, we evaluate both continuous fixed-window spectrograms and trial boundary proxies (initial window vs terminal window) under identical subject splits.

---

## 4. Model Architecture: `GaitBasePaper2DCNN`

```
INPUT: (B, 16, 33, 32)  [16 VGRF Channels x 33 Frequency Bins x 32 Time Frames]
  │
  ▼
[Conv Block 1]
  Conv2d(16 -> 32, kernel=3x3, pad=1) -> BatchNorm2d(32) -> ReLU -> MaxPool2d(2x2) -> Dropout2d(0.2)
  Output: (B, 32, 16, 16)
  │
  ▼
[Conv Block 2]
  Conv2d(32 -> 64, kernel=3x3, pad=1) -> BatchNorm2d(64) -> ReLU -> MaxPool2d(2x2) -> Dropout2d(0.2)
  Output: (B, 64, 8, 8)
  │
  ▼
[Conv Block 3]
  Conv2d(64 -> 128, kernel=3x3, pad=1) -> BatchNorm2d(128) -> ReLU -> MaxPool2d(2x2) -> Dropout2d(0.3)
  Output: (B, 128, 4, 4)
  │
  ▼
[Adaptive Pooling & Bottleneck]
  AdaptiveAvgPool2d((2, 2)) -> Flatten (512)
  Linear(512 -> 128) -> BatchNorm1d(128) -> ReLU -> Dropout(0.3)
  Output: (B, 128)  [128-dim dense gait embedding]
  │
  ▼
[Classifier Head]
  Linear(128 -> 1) -> Binary Logit for BCEWithLogitsLoss
```

---

## 5. Experimental Protocol & Splits

To ensure an uncompromised, zero-leakage, fair comparison with the proposed model:
- **Random Seed:** 42.
- **Subject Stratification:** 70% Train (115 subjects), 15% Validation (25 subjects), 15% Test (25 subjects).
- **Subject Overlap:** Exactly 0 subjects shared across any split partition ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).
- **Held-Out Test Subjects (25 total):**
  - **Parkinson's Disease (14 subjects):** `GaPt06, GaPt12, GaPt17, GaPt20, GaPt25, JuPt04, JuPt08, JuPt21, SiPt04, SiPt08, SiPt12, SiPt16, SiPt20, SiPt28`
  - **Healthy Control (11 subjects):** `GaCo03, GaCo08, GaCo12, GaCo17, JuCo03, JuCo07, JuCo20, JuCo24, SiCo04, SiCo12, SiCo20`

---

## 6. Empirical Results & Direct Model Comparison

### Primary Test Split (25 Held-Out Test Subjects):

| Evaluation Metric | Base-Paper 2D CNN (STFT) | Proposed 1D CNN + BiLSTM | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Subject Accuracy** | **0.7600** (76.0%) | 0.7200 (72.0%) | **+4.00%** |
| **Balanced Accuracy** | **0.7662** (76.6%) | 0.7110 (71.1%) | **+5.52%** |
| **Subject Precision** | **0.8333** (83.3%) | 0.7333 (73.3%) | **+10.00%** |
| **Sensitivity (Recall)** | 0.7143 (71.4%) | **0.7857** (78.6%) | -7.14% |
| **Specificity** | **0.8182** (81.8%) | 0.6364 (63.6%) | **+18.18%** |
| **Subject F1-Score** | **0.7692** | 0.7586 | **+0.0106** |
| **Subject ROC-AUC** | 0.7143 | **0.7532** | -0.0389 |
| **Brier Score** | **0.2189** | 0.2241 | **-0.0052 (Better calibrated)** |
| **Recording Accuracy** | **0.8500** (85.0%) | 0.7000 (70.0%) | **+15.00%** |
| **Recording F1-Score** | **0.8750** | 0.7600 | **+0.1150** |
| **Window Accuracy** | **0.7893** (78.9%) | 0.6995 (69.9%) | **+8.98%** |

### Breakdown by Clinical Cohort (Test Subjects):
- **Cohort Ga (Galit et al., 8 subjects):**
  - Accuracy: **1.0000 (100.0%)** | F1: **1.0000** | ROC-AUC: **1.0000**
- **Cohort Ju (Hausdorff et al., 6 subjects):**
  - Accuracy: **0.6667 (66.7%)** | F1: **0.6667** | ROC-AUC: **0.7778**
- **Cohort Si (Frenkel-Toledo et al., 11 subjects):**
  - Accuracy: **0.6364 (63.6%)** | F1: **0.6000** | ROC-AUC: **0.4667**

### Transition Analysis (Proxy Initiation vs Termination):
- **Onset Windows (Walking Initiation Proxy):** Window Acc: 0.7000 | F1: 0.7391 | ROC-AUC: 0.8000
- **Offset Windows (Walking Termination Proxy):** Window Acc: 0.7500 | F1: 0.7917 | ROC-AUC: 0.8640
- **Steady-State Windows (Intermediate Gait):** Window Acc: 0.7922 | F1: 0.8263 | ROC-AUC: 0.8384

### Cross-Cohort Generalization: Leave-One-Cohort-Out (LOCO)

| Held-Out Cohort | Subjects Evaluated | Subject Accuracy | Balanced Accuracy | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Cohort Ga Held Out** | 47 | 0.6596 | 0.6504 | 0.7143 | 0.6954 |
| **Cohort Ju Held Out** | 54 | 0.7222 | 0.7055 | 0.7826 | 0.8124 |
| **Cohort Si Held Out** | 64 | 0.5469 | 0.5473 | 0.5672 | 0.5567 |
| **Macro Average** | **165** | **0.6429** | **0.6344** | **0.6880** | **0.6882** |

---

## 7. Key Findings & Discussion
1. **Spectral Patterns vs Temporal Recurrence:**  
   The Base-Paper 2D CNN on STFT representations achieves higher subject accuracy (76.0% vs 72.0%) and markedly higher specificity (81.8% vs 63.6%), reducing false positive diagnoses. The 2D CNN filters effectively extract harmonic power distributions across heel and toe pressure sensors.
2. **Recording-Level Superiority:**  
   At the recording level, the 2D CNN baseline achieves **85.0% accuracy** and an **F1-score of 0.8750**, compared to 70.0% and 0.7600 for the 1D CNN + BiLSTM model.
3. **Cohort-Shift Vulnerability:**  
   Under unseen clinical cohort shift (LOCO), both models experience performance degradation on Cohort Si (54.7% for the 2D CNN), confirming the fundamental finding that inter-cohort center shifts represent the primary clinical vulnerability in automated Parkinson's gait assessment.

---

## 8. Reproducibility Instructions

### Run Unit Tests:
```bash
python -m pytest Gait/baselines/base_paper_cnn/tests -v
```

### Train Baseline Model:
```bash
python Gait/baselines/base_paper_cnn/train.py
```

### Run Primary Held-Out Test Evaluation:
```bash
python Gait/baselines/base_paper_cnn/evaluate.py
```

### Run Cross-Cohort LOCO Evaluation:
```bash
python Gait/baselines/base_paper_cnn/loco_evaluation.py
```
