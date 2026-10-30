# Voice 2D CNN Baseline (Base-Paper Inspired)

Reproducible baseline implementation inspired by the convolutional neural network architecture in:
> **"Multimodal assessment of Parkinson’s disease: a deep learning approach"**  
> *J. C. Vásquez-Correa, J. R. Orozco-Arroyave, et al., IEEE Journal of Biomedical and Health Informatics (JBHI), 2019.*

---

## 1. Executive Summary & Objective

This module implements an isolated, reproducible **2D CNN baseline** on Short-Time Fourier Transform (STFT) time-frequency acoustic representations to evaluate against the repository's proposed **Wav2Vec2-based model**.

All code, checkpoints, preprocessing artifacts, metrics, and plots reside strictly in `Voice/baselines/base_paper_cnn/` without altering the existing Wav2Vec2 implementation, checkpoints, or web demo.

---

## 2. Comparison: Base-Paper Architecture vs. Current Dataset Adaptation

| Dimension | Reference Paper (Vásquez-Correa et al., 2019) | Adapted Voice 2D CNN Baseline (This Repository) |
| :--- | :--- | :--- |
| **Acoustic Dataset** | PC-GITA Spanish database (100 subjects: 50 PD, 50 HC) | MDVR-KCL English database (37 subjects: 22 PD, 15 HC) |
| **Speech Tasks** | Monologue, read text, vowels, sentences | Read Text & Spontaneous Dialogue |
| **Input Representation** | STFT log-magnitude spectrograms of phonation transitions & sentences | 2D STFT log-magnitude spectrograms (25 ms window, 10 ms step, 257 freq bins × 301 time frames per 3.0s chunk) + Phonation Transitions (129 × 33) |
| **Transition Detection** | Praat-aligned phonetic onset/offset boundaries (40ms unvoiced + 40ms voiced) | Algorithmic energy & zero-crossing rate phonation boundaries (80ms windows: 40ms unvoiced + 40ms voiced) |
| **Architecture** | 3–4 Conv2D blocks (Conv, ReLU, MaxPool, Dropout) + Dense + Softmax | 3 Conv2D blocks (32, 64, 128 channels) with BatchNorm, ReLU, MaxPool, Dropout + Dense bottleneck (128 units) + 2-class Softmax |
| **Evaluation Split** | 10-fold Cross-Validation (subject-independent) | Strict 70/15/15 subject-level holdout (26 train, 5 val, 6 test), identical test subjects as the proposed Wav2Vec2 model |
| **Leakage Controls** | Subject-independent splits | Frequency-wise normalization statistics fit **strictly on training subjects**; zero subject overlap |

---

## 3. Architecture Specification

```
Input Spectrogram: (Batch, 1, 257 Freq Bins, 301 Time Frames)
  │
  ├── ConvBlock 1: Conv2D(1 -> 32, 3x3) -> BatchNorm2D -> ReLU -> MaxPool(2x2) -> Dropout2D(0.20)
  ├── ConvBlock 2: Conv2D(32 -> 64, 3x3) -> BatchNorm2D -> ReLU -> MaxPool(2x2) -> Dropout2D(0.20)
  ├── ConvBlock 3: Conv2D(64 -> 128, 3x3) -> BatchNorm2D -> ReLU -> MaxPool(2x2) -> Dropout2D(0.30)
  ├── AdaptiveAvgPool2D((2, 2)) -> Flatten (512 dimensions)
  ├── Dense Bottleneck: Linear(512 -> 128) -> BatchNorm1D -> ReLU -> Dropout(0.30)
  └── Classifier Head: Linear(128 -> 2) -> Cross-Entropy Logits [HC, PD]
```

### Preprocessing Parameters
- **Sampling Rate:** 16,000 Hz
- **Chunk Duration:** 3.0 seconds (48,000 samples) with 50% overlap (1.5s step)
- **STFT Window:** 25 ms (400 samples), Hann window
- **Hop Length:** 10 ms (160 samples)
- **FFT Size:** 512 points (giving $N_{\text{fft}}/2 + 1 = 257$ frequency bins)
- **Time Frames:** 301 frames
- **Log Scaling:** $\log(1.0 + |X|)$

---

## 4. Empirical Evaluation Results

Evaluated on the **identical 6 held-out test subjects** (`ID04`, `ID10`, `ID13`, `ID23`, `ID29`, `ID35`):

### A. Primary Test Set Comparison (Subject-Level)

| Metric | Base-Paper 2D CNN (Continuous STFT) | Base-Paper Onset Transition CNN | Base-Paper Offset Transition CNN | Proposed Wav2Vec2 Model |
| :--- | :---: | :---: | :---: | :---: |
| **Subject Accuracy** | **0.8333 (5/6)** | 0.5000 | 0.5000 | **0.6667 (4/6)** |
| **Balanced Accuracy**| **0.8333** | 0.5000 | 0.5000 | **0.6667** |
| **Precision** | **0.7500** | 0.0000 | 0.0000 | **0.6667** |
| **Sensitivity (Recall)**| **1.0000 (3/3)** | 0.0000 | 0.0000 | **0.6667 (2/3)** |
| **Specificity** | **0.6667 (2/3)** | 1.0000 | 1.0000 | **0.6667 (2/3)** |
| **F1-Score** | **0.8571** | 0.0000 | 0.0000 | **0.6667** |
| **ROC-AUC** | **1.0000** | 0.6111 | 0.6111 | **0.7778** |
| **Brier Score** | **0.1076** | 0.2494 | 0.2486 | N/A |
| **Negative Log-Likelihood** | **0.3817** | 0.6920 | 0.6904 | N/A |

### B. Hierarchical Granularity (Continuous STFT 2D CNN)

| Granularity | Sample Count | Accuracy | Balanced Accuracy | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Subject Level** | 6 subjects | **0.8333** | 0.8333 | 0.8571 | 1.0000 |
| **Recording Level** | 12 recordings | **0.9167** | 0.9167 | 0.9231 | 0.9722 |
| **Chunk Level (3.0s)** | 1,161 chunks | **0.7941** | 0.7953 | 0.8010 | 0.8889 |

### C. Task Breakdown (Recording Level)

- **ReadText Task:** Accuracy **0.8333**, Sensitivity **1.0000**, Specificity **0.6667**, AUC **1.0000**
- **SpontaneousDialogue Task:** Accuracy **1.0000**, Sensitivity **1.0000**, Specificity **1.0000**, AUC **1.0000**

---

## 5. Key Findings & Discussion

1. **Continuous STFT vs. Phonation Transitions:**
   - The continuous STFT 2D CNN achieved strong classification on the MDVR-KCL dataset (83.3% subject accuracy, 1.000 AUC).
   - In contrast, algorithmic onset/offset transition models achieved an AUC of 0.611. In the original paper, Spanish recordings benefited from manual Praat phonetic alignments and vowel repetition tasks. In unconstrained continuous English speech without phoneme-level alignments, broader 3-second spectral temporal context captures dysarthric vocal tremor, dysphonia, and pause patterns more reliably.

2. **Base-Paper 2D CNN vs. Proposed Wav2Vec2 Model:**
   - The adapted 2D CNN baseline outperforms the fine-tuned Wav2Vec2 model on this specific small held-out test split (Subject Acc: 83.3% vs. 66.7%, F1: 0.857 vs. 0.667).
   - While Wav2Vec2 learns rich self-supervised representations from 100k+ hours of general audio, 2D CNN spectrogram representations directly exploit fundamental frequency disruptions and harmonic-to-noise degradations characteristic of Parkinsonian speech without requiring large-scale transformer fine-tuning parameters.

---

## 6. How to Reproduce

### Run Unit Tests
```bash
pytest Voice/baselines/base_paper_cnn/tests/ -v
```

### Train All Models
```bash
python Voice/baselines/base_paper_cnn/train.py
```

### Run Evaluation & Generate Plots
```bash
python Voice/baselines/base_paper_cnn/evaluate.py
```

### Run Predictor
```bash
python Voice/baselines/base_paper_cnn/predict.py --audio "Voice/26-29_09_2017_KCL/ReadText/PD/ID04_ReadText_PD.wav"
```
