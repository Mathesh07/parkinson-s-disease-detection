# Gait-Based Parkinson's Disease Detection (1D CNN + BiLSTM)

This module implements the Gait analysis modality for our multimodal Parkinson's Disease detection framework using the PhysioNet **Gait in Parkinson's Disease** dataset.

---

## 1. Modality & Architecture Overview

The pipeline processes continuous Vertical Ground Reaction Force (VGRF) time-series data from 16 insole sensors (8 on the left foot, 8 on the right foot) sampled at 100 Hz.

```
Raw 16-Channel VGRF Signals
           ↓
Zero-Leakage Training Normalization (Z-score)
           ↓
Fixed-Length Window Segmentation (5.0s = 500 samples @ 100 Hz, 50% overlap)
           ↓
1D CNN Block 1 (16 → 64 channels, kernel=5, MaxPool /2)
           ↓
1D CNN Block 2 (64 → 128 channels, kernel=5, MaxPool /2)
           ↓
2-Layer Bidirectional LSTM (hidden_size=128, dropout=0.3 → 256 dim)
           ↓
Global Temporal Mean Pooling (over time steps)
           ↓
Dense Embedding Layer (256 → 128 dim + BatchNorm + ReLU + Dropout)
           ↓
Fixed 128-Dimensional Gait Feature Vector (for Multimodal Fusion)
           ↓
Binary Classification Head (Dropout + Linear → Logit)
```

---

## 2. Directory Structure

```
Gait/
├── gait-in-parkinsons-disease-1.0.0/   # Raw PhysioNet dataset (306 text files & demographics)
├── checkpoints/                        # Model weights (.pt) and normalization JSON
│   ├── best_gait_model.pt
│   ├── model_config.json
│   └── normalization_stats.json
├── outputs/
│   ├── metrics/                        # test_metrics.json, predictions CSVs
│   ├── plots/                          # Loss curves, accuracy curves, confusion matrix, ROC
│   └── features/                       # Extracted 128-d embeddings (.npy, .pt, .csv)
├── __init__.py                         # Package initialization
├── config.py                           # Central configuration & hyperparameters
├── preprocessing.py                   # Parsing, 16-channel extraction, GaitNormalizer
├── dataset.py                          # Subject-stratified splitting, windowing, PyTorch Dataset
├── model.py                            # 1D CNN + BiLSTM + 128-d Embedding + Classifier
├── utils.py                            # Metrics calculation, hierarchical aggregation, plotting
├── train.py                            # Training pipeline with mixed precision (AMP)
├── evaluate.py                         # Standalone held-out test evaluation
├── feature_extraction.py               # Feature extractor for dynamic multimodal fusion
└── README.md
```

---

## 3. Strict Zero-Leakage Protocol

1. **Stratified Subject-Level Split**:
   - 70% Train (~115 subjects)
   - 15% Validation (~25 subjects)
   - 15% Test (~25 subjects)
   - Multiple recordings from the same patient always remain inside the same split.
2. **Independent Normalization**:
   - Channel-wise mean and standard deviation are computed **strictly on the training subjects**.
   - The same transform is saved and applied to validation and test data.
3. **Hierarchical Aggregation**:
   - Windows $\to$ Mean Recording Probability $\to$ Mean Subject Probability.
   - Medical evaluation metrics are reported at the **Subject Level**.

---

## 4. Usage

### Training
```bash
python Gait/train.py
```

### Evaluation on Held-Out Test Subjects
```bash
python Gait/evaluate.py
```

### Feature Extraction for Multimodal Fusion
```bash
python Gait/feature_extraction.py
```

Or programmatically in Python:
```python
from Gait.feature_extraction import extract_gait_features

# Extract 128-dimensional embedding from a recording
embedding_128d = extract_gait_features("path/to/recording.txt")
print(embedding_128d.shape)  # (128,)
```
