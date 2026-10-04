# Gait Visualizations

**Project:** Can Uncertainty Warn Us? Cohort-Shift Reliability of Parkinson's Disease Classifiers

All figures are generated from existing experiment outputs in `Gait/outputs/`.  
**No data is fabricated.** All numbers come from saved CSV/JSON files.

---

## How to Regenerate

```bash
python Gait/visualizations/generate_all_visualizations.py
```

Figures are saved as **PNG (300 DPI)** and **SVG** (for diagram figures).

---

## Figure Index

### `architecture/`

| Figure | Type | Data Source | Purpose |
|--------|------|-------------|---------|
| `gait_model_architecture.png/.svg` | Diagram | Architecture knowledge | Layer-by-layer GaitCNNBiLSTM block diagram with shapes and sizes |
| `final_system_overview.png/.svg` | Diagram | Architecture knowledge | Complete system from raw signal → ACCEPT/FLAG decision |
| `project_story.png/.svg` | Diagram | Architecture knowledge | 5-stage project narrative for presentations |

---

### `data_pipeline/`

| Figure | Type | Data Source | Purpose |
|--------|------|-------------|---------|
| `end_to_end_pipeline.png/.svg` | Diagram | Architecture knowledge | Full pipeline: Raw signal → Calibration → Conformal → Reliability |
| `example_gait_signal.png` | Data | `gait-in-parkinsons-disease-1.0.0/*.txt` | Real 16-channel VGRF recording (15-second excerpt) |
| `windowing_example.png` | Data | `gait-in-parkinsons-disease-1.0.0/*.txt` | Sliding window segmentation with 50% overlap annotation |

---

### `cohort_shift/`

| Figure | Type | Data Source | Purpose |
|--------|------|-------------|---------|
| `loco_protocol.png/.svg` | Diagram + Data | `outputs/loco/loco_results.csv` | Visual LOCO protocol with actual per-fold accuracy and AUC |
| `corruption_accuracy.png` | Chart | `outputs/corruption_shift/corruption_shift_results.csv` | Accuracy vs corruption severity by method |
| `corruption_nll.png` | Chart | `outputs/corruption_shift/corruption_shift_results.csv` | NLL vs corruption severity |
| `corruption_ece.png` | Chart | `outputs/corruption_shift/corruption_shift_results.csv` | ECE vs corruption severity |
| `corruption_uncertainty.png` | Chart | `outputs/corruption_shift/corruption_shift_results.csv` | Mean uncertainty vs corruption severity |

---

### `performance/`

| Figure | Type | Data Source | Purpose |
|--------|------|-------------|---------|
| `loco_accuracy.png` | Bar chart | `outputs/loco/loco_results.csv` | Per-fold accuracy across 3 LOCO configurations |
| `loco_balanced_accuracy.png` | Bar chart | `outputs/loco/loco_results.csv` | Per-fold balanced accuracy |
| `loco_auc.png` | Bar chart | `outputs/loco/loco_results.csv` | Per-fold ROC-AUC |
| `loco_nll.png` | Bar chart | `outputs/loco/loco_results.csv` | Per-fold NLL |
| `loco_brier.png` | Bar chart | `outputs/loco/loco_results.csv` | Per-fold Brier score |
| `multiseed_performance.png` | Bar chart (CI) | `outputs/multiseed/multiseed_summary.csv` | Mean ± 95% CI over 5 seeds (42–46) |
| `multiseed_uncertainty.png` | Bar chart (CI) | `outputs/multiseed/multiseed_summary.csv` | Uncertainty on correct vs incorrect predictions |

---

### `calibration/`

| Figure | Type | Data Source | Purpose |
|--------|------|-------------|---------|
| `calibration_metrics.png` | Bar chart | `outputs/calibration/calibration_results.csv` | NLL / Brier / ECE: Softmax vs Temperature Scaling |
| `reliability_diagram.png` | Calibration curve | `outputs/calibration/calibration_subject_predictions.csv` | Predicted probability vs observed accuracy (reliability diagram) |

---

### `uncertainty/`

| Figure | Type | Data Source | Purpose |
|--------|------|-------------|---------|
| `deep_ensemble_prediction.png` | Diagram + Data | `outputs/deep_ensemble/deep_ensemble_results.csv` | 5-model ensemble schematic with real performance numbers |
| `deep_ensemble_comparison.png` | Bar chart | `outputs/deep_ensemble/deep_ensemble_comparison.csv` | All methods compared: Acc / AUC / NLL / Brier / ECE |
| `silent_confidence.png` | Dual-axis line | `outputs/corruption_shift/corruption_shift_results.csv` | Accuracy stays high while uncertainty fails to rise |

---

### `conformal/`

| Figure | Type | Data Source | Purpose |
|--------|------|-------------|---------|
| `conformal_coverage.png` | Bar chart | `outputs/conformal/conformal_results.csv` | Empirical vs target coverage at each alpha level |
| `conformal_set_size.png` | Bar chart | `outputs/conformal/conformal_results.csv` | Average prediction set size per fold and alpha |
| `conformal_corruption.png` | Line chart | `outputs/conformal/conformal_corruption_coverage.csv` | Coverage under input corruption |
| `conformal_prediction_explained.png/.svg` | Diagram | — | Visual explanation of the 4 possible prediction sets |

---

### `reliability/`

| Figure | Type | Data Source | Purpose |
|--------|------|-------------|---------|
| `raw_vs_accepted_accuracy.png` | Bar chart | `outputs/reliability/reliability_results.csv` | Raw accuracy vs accepted-only accuracy per cohort |
| `coverage_by_cohort.png` | Bar chart | `outputs/reliability/reliability_results.csv` | % of subjects accepted by the reliability layer |
| `error_capture_by_cohort.png` | Bar chart | `outputs/reliability/reliability_results.csv` | % of errors flagged / abstained |
| `flag_reason_breakdown.png` | Bar chart | `outputs/reliability/reliability_subject_decisions.csv` | Which flag reasons triggered for each cohort |
| `reliability_decision_flow.png/.svg` | Flow diagram | — | Three-layer ACCEPT/FLAG decision logic |
| `subject_confidence_distribution.png` | Histogram | `outputs/reliability/reliability_subject_decisions.csv` | Confidence distribution: ACCEPT vs FLAG subjects |
| `subject_uncertainty_distribution.png` | Histogram | `outputs/reliability/reliability_subject_decisions.csv` | Uncertainty distribution: ACCEPT vs FLAG subjects |

---

### `inference/`

| Figure | Type | Data Source | Purpose |
|--------|------|-------------|---------|
| `live_inference_flow.png/.svg` | Flow diagram | — | Zero-retraining inference pipeline for an individual recording |

---

## Notes

- Figures that require raw `.txt` dataset files will be **skipped** if `gait-in-parkinsons-disease-1.0.0/` is not present.
- Figures requiring specific CSV columns will be **skipped** gracefully with a printed notice.
- The generator prints `OK` / `SKIP` for each figure and a final summary count.
