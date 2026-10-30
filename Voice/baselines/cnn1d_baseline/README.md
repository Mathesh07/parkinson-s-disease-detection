# Voice 1D CNN Baseline (Raw Audio Waveforms)

A reproducible **conventional 1D CNN baseline** operating directly on raw 16 kHz audio waveforms to compare against our proposed **Wav2Vec2-based model**.

> **Proposed Model:** Pretrained Wav2Vec2 Feature Encoder + Classification Head  
> **Baseline Model:** Conventional 1D CNN on Raw Waveforms (without pretrained embeddings or STFT)

---

## 1. Objective & Design

This baseline evaluates whether a purely feedforward 1D convolutional neural network operating directly on raw time-domain acoustic waveforms ($16\text{ kHz}$) can distinguish Parkinsonian speech from healthy controls on the MDVR-KCL dataset, in comparison to representations learned from self-supervised pretrained speech models (Wav2Vec2).

> **Note on Baseline Status:**  
> This conventional 1D CNN serves as the active raw-waveform baseline. The 2D CNN on STFT representations resides in `Voice/baselines/base_paper_cnn/` as an archived reference experiment.

---

## 2. Model Architecture

```
Raw Speech Waveform: (Batch, 1, 48000 samples) [3.0 seconds at 16 kHz]
  │
  ├── ConvBlock 1: Conv1D(1 -> 32, k=64, s=16, p=32) -> BatchNorm1D -> ReLU -> MaxPool1D(4)
  ├── ConvBlock 2: Conv1D(32 -> 64, k=16, s=4, p=8) -> BatchNorm1D -> ReLU -> MaxPool1D(4)
  ├── ConvBlock 3: Conv1D(64 -> 128, k=8, s=2, p=4) -> BatchNorm1D -> ReLU -> MaxPool1D(2)
  ├── ConvBlock 4: Conv1D(128 -> 256, k=5, s=1, p=2) -> BatchNorm1D -> ReLU -> AdaptiveAvgPool1D(1)
  ├── Dense Bottleneck: Linear(256 -> 128) -> BatchNorm1D -> ReLU -> Dropout(0.3)
  └── Classifier Head: Linear(128 -> 2) -> Softmax Logits [HC, PD]
```

- **Parameters:** ~302,000 trainable weights.
- **Input:** 3.0-second normalized acoustic chunks ($48,000$ samples) with $50\%$ overlap.

---

## 3. Dataset & Split Details

- **Dataset:** MDVR-KCL (King's College London, 37 subjects: 22 PD, 15 HC).
- **Tasks:** Read Text & Spontaneous Dialogue.
- **Subject-Level Split (Seed = 42):**
  - **Train:** 26 subjects (51 recordings)
  - **Val:** 5 subjects (10 recordings)
  - **Test (Held-Out):** 6 subjects (12 recordings: 3 HC [`ID10`, `ID23`, `ID35`], 3 PD [`ID04`, `ID13`, `ID29`])
- **Zero Leakage:**
  - Waveform normalization stats fitted strictly on training subjects.
  - Zero subject overlap across partitions.

---

## 4. Empirical Evaluation Results

Evaluated on the **identical 6 held-out test subjects**:

| Metric | Voice 1D CNN Baseline (Raw Waveform) | Proposed Wav2Vec2 Model | Archived 2D CNN (STFT Spectrogram) |
| :--- | :---: | :---: | :---: |
| **Subject Accuracy** | **0.6667 (4/6)** | **0.6667 (4/6)** | 0.8333 (5/6) |
| **Balanced Accuracy**| **0.6667** | **0.6667** | 0.8333 |
| **Sensitivity (Recall)**| **1.0000 (3/3)** | 0.6667 (2/3) | 1.0000 (3/3) |
| **Specificity** | 0.3333 (1/3) | **0.6667 (2/3)** | 0.6667 (2/3) |
| **Precision** | 0.6000 (3/5) | **0.6667 (2/3)** | 0.7500 (3/4) |
| **F1-Score** | **0.7500** | 0.6667 | 0.8571 |
| **ROC-AUC** | **1.0000** | 0.7778 | 1.0000 |
| **Brier Score** | **0.1441** | — | 0.1076 |
| **Negative Log-Likelihood** | **0.4482** | — | 0.3817 |

---

## 5. Methodological Discussion & Caveats

1. **Raw Waveform 1D CNN vs. Pretrained Wav2Vec2:**
   - Both models achieve identical subject accuracy ($66.7\%$, $4/6$ subjects correct) on this held-out test split.
   - The 1D CNN achieves higher sensitivity ($100\%$ vs. $66.7\%$), correctly identifying all 3 PD subjects, but suffers from lower specificity ($33.3\%$, $2$ false alarms).
   - In contrast, Wav2Vec2 provides balanced sensitivity and specificity ($66.7\%$ each).
2. **Small Test Sample Size Limitation:**
   - Because MDVR-KCL contains 37 total subjects, a 15% holdout comprises only $N=6$ subjects. A single subject change corresponds to a $\pm 16.7\%$ swing in accuracy.
   - Consequently, the bootstrap confidence intervals on voice metrics are wide. This uncertainty must be transparently acknowledged when presenting comparative conclusions.

---

## 6. How to Reproduce

### Run Unit Tests
```bash
python -m pytest Voice/baselines/cnn1d_baseline/tests/ -v
```

### Train Baseline Model
```bash
python Voice/baselines/cnn1d_baseline/train.py
```

### Run Evaluation
```bash
python Voice/baselines/cnn1d_baseline/evaluate.py
```

### Run CLI Prediction
```bash
python Voice/baselines/cnn1d_baseline/predict.py --audio "Voice/26-29_09_2017_KCL/ReadText/PD/ID04_ReadText_PD.wav"
```
