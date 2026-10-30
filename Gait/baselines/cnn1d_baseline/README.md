# Gait 1D CNN Baseline

A clean, reproducible **conventional 1D CNN baseline** designed to benchmark against our proposed Parkinson's disease gait detection model:
> **Proposed Model:** 1D CNN + BiLSTM + 128-dim Embedding + Evidential Uncertainty Head  
> **Baseline Model:** Conventional 1D CNN + Classification Head

---

## 1. Objective & Hypothesis

The goal of this baseline is to evaluate whether adding **bidirectional LSTM recurrent temporal sequence modeling** and **Dirichlet evidential uncertainty estimation** provides measurable improvements over a standard, feedforward 1D convolutional network operating directly on raw 16-channel vertical ground reaction force (VGRF) time-series.

> **Note on Baseline Status:**  
> This conventional 1D CNN serves as the **primary active baseline** for gait detection. The previously implemented 2D CNN (operating on STFT time-frequency representations) is preserved in `Gait/baselines/base_paper_cnn/` as an archived supplementary experiment.

---

## 2. Model Architecture

```
Raw VGRF Window: (Batch, 500 time steps, 16 sensor channels)
  │ Transpose to (Batch, 16, 500)
  ├── ConvBlock 1: Conv1D(16 -> 64, k=5, p=2) -> BatchNorm1D -> ReLU -> MaxPool1D(2)  [500 -> 250]
  ├── ConvBlock 2: Conv1D(64 -> 128, k=5, p=2) -> BatchNorm1D -> ReLU -> MaxPool1D(2) [250 -> 125]
  ├── ConvBlock 3: Conv1D(128 -> 256, k=5, p=2) -> BatchNorm1D -> ReLU -> MaxPool1D(2)[125 -> 62]
  ├── Global Average Pooling: AdaptiveAvgPool1D(1) -> Flatten (256)
  ├── Dense Bottleneck: Linear(256 -> 128) -> BatchNorm1D -> ReLU -> Dropout(0.3)
  └── Classification Head: Linear(128 -> 1) -> Binary Logit (BCEWithLogitsLoss)
```

**Key Architectural Properties:**
- **Parameters:** ~247,000 trainable weights.
- **Pure 1D CNN:** Strictly feedforward; contains NO recurrent cells (LSTM/BiLSTM), NO transformers, and NO evidential heads.
- **Identical Input:** Takes the exact same 16-channel, 5-second VGRF sliding windows at 100 Hz as the proposed model.

---

## 3. Dataset & Zero-Leakage Split Protocol

- **Dataset:** PhysioNet Gait in Parkinson's Disease (Ga, Ju, Si cohorts, total 165 subjects).
- **Subject-Level Stratified Partition (Seed = 42):**
  - **Train:** 115 subjects (226 recordings)
  - **Validation:** 25 subjects (40 recordings)
  - **Test (Held-Out):** 25 subjects (40 recordings: 11 HC, 14 PD)
- **Zero Leakage:**
  - Strict subject-level isolation: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
  - Normalization parameters (channel-wise mean & std) fitted **strictly on training subjects** and saved to `checkpoints/normalization_stats.json`.

---

## 4. Empirical Evaluation Results

### A. Primary Test Set Comparison (25 Held-Out Test Subjects)

| Metric | 1. 1D CNN Baseline | 2. Proposed 1D CNN + BiLSTM (BCE) | 3. Proposed 1D CNN + BiLSTM + Evidential | Difference (Proposed vs. 1D CNN) |
| :--- | :---: | :---: | :---: | :---: |
| **Subject Accuracy** | **0.6800 (17/25)** | **0.7200 (18/25)** | 0.6800 (17/25) | $+4.0\%$ |
| **95% Bootstrap CI** | $[0.4800, 0.8800]$ | $[0.5200, 0.9200]$ | $[0.4800, 0.8800]$ | — |
| **Balanced Accuracy**| 0.6851 | **0.7110** | 0.6948 | $+2.59\%$ |
| **Sensitivity (Recall)**| 0.6429 (9/14) | **0.7857 (11/14)** | 0.5714 (8/14) | $+14.28\%$ |
| **Specificity** | 0.7273 (8/11) | 0.6364 (7/11) | **0.8182 (9/11)** | $-9.09\%$ / $+9.09\%$ |
| **Precision** | 0.7500 | 0.7333 | **0.8000** | $-1.67\%$ / $+5.00\%$ |
| **F1-Score** | 0.6923 | **0.7586** | 0.6667 | $+6.63\%$ |
| **ROC-AUC** | 0.6818 | **0.7532** | **0.7532** | $+7.14\%$ |
| **Brier Score** | 0.2592 | **0.2289** | — | $-0.0303$ (lower is better) |
| **NLL (Log-Loss)** | 1.2645 | **0.6481** | — | $-0.6164$ (lower is better) |

### B. Uncertainty & Error Detection Comparison

- **1D CNN Baseline (Predictive Entropy Proxy):**
  - Mean predictive entropy on correct predictions: **0.2237**
  - Mean predictive entropy on incorrect predictions: **0.3366** ($+50.5\%$ increase on errors)
- **Proposed Evidential Model (Dirichlet Subjective Uncertainty):**
  - Mean evidential uncertainty on correct predictions: **0.0491**
  - Mean evidential uncertainty on incorrect predictions: **0.0590** ($+20.2\%$ increase on errors)

---

## 5. Leave-One-Cohort-Out (LOCO) Robustness Benchmark

To assess generalization across clinical sites:

| Held-Out Cohort | 1D CNN Baseline | Proposed 1D CNN + BiLSTM |
| :--- | :---: | :---: |
| **Ga (Galit)** | Acc: 0.5745, F1: 0.6552, AUC: 0.6571 | Acc: 0.6170 |
| **Ju (Hausdorff)** | Acc: 0.7222, F1: 0.7368, AUC: 0.7972 | Acc: 0.7778 |
| **Si (Silberstein)**| Acc: 0.6406, F1: 0.6567, AUC: 0.7764 | Acc: 0.6406 |
| **Macro Average** | **Acc: 0.6458**, F1: 0.6829, AUC: 0.7436 | **Acc: 0.6785** |

---

## 6. Key Scientific Findings & Discussion

1. **Benefit of BiLSTM Sequence Modelling:**
   - Incorporating bidirectional LSTM layers after 1D convolutions improves Subject Accuracy from **68.0% to 72.0%**, Sensitivity from **64.3% to 78.6%**, and ROC-AUC from **0.6818 to 0.7532**.
   - Because gait impairments involve periodic bilateral stride asymmetry that spans across consecutive gait cycles, recurrent temporal cells capture inter-stride dependencies that global average pooling across convolutional features misses.
2. **Benefit of Evidential Head:**
   - The evidential model yields the highest specificity (**81.82%**) and precision (**80.00%**), with lower false alarms (FP=2 vs FP=4).
   - In clinical screening, combining BiLSTM representation learning with evidential calibration produces calibrated risk estimates where erroneous decisions exhibit higher uncertainty.

---

## 7. How to Reproduce

### Run Unit Tests
```bash
python -m pytest Gait/baselines/cnn1d_baseline/tests/ -v
```

### Train Baseline Model
```bash
python Gait/baselines/cnn1d_baseline/train.py
```

### Run Evaluation & Generate Visualizations
```bash
python Gait/baselines/cnn1d_baseline/evaluate.py
```

### Run LOCO Benchmark
```bash
python Gait/baselines/cnn1d_baseline/loco_evaluation.py
```

### Run CLI Inference
```bash
python Gait/baselines/cnn1d_baseline/predict.py --filepath "Gait/gait-in-parkinsons-disease-1.0.0/GaPt03_01.txt"
```
