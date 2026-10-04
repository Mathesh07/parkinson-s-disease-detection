"""
Quick smoke-test for the LOCO module — no dataset required.
Run: python Gait/experiments/_test_loco_imports.py
"""
import sys
sys.path.insert(0, ".")

# --- Project imports ---
from Gait import config
from Gait.dataset import GaitWindowDataset, collate_gait_batch
from Gait.model import GaitCNNBiLSTM
from Gait.preprocessing import GaitNormalizer, build_manifest, load_raw_recording
from Gait.utils import calculate_metrics, get_device, set_seed
from evidential import EvidentialLoss, dirichlet_from_evidence
print("Project imports OK")

# --- LOCO module imports ---
from Gait.experiments.leave_one_cohort_out import (
    LOCO_EXPERIMENTS,
    EXPECTED_COHORTS,
    build_cohort_splits,
    fit_normalizer_on_train,
    compute_loco_metrics,
    _safe_nll,
    _safe_brier,
    _safe_auroc,
    aggregate_to_subject_level,
)
print("LOCO module imports OK")

# --- Experiment definition checks ---
assert len(LOCO_EXPERIMENTS) == 3
assert LOCO_EXPERIMENTS[0]["test_cohort"] == "Si"
assert LOCO_EXPERIMENTS[1]["test_cohort"] == "Ju"
assert LOCO_EXPERIMENTS[2]["test_cohort"] == "Ga"
assert EXPECTED_COHORTS == {"Ga", "Ju", "Si"}
print("LOCO experiment definitions OK")

# --- Metric helper tests ---
import numpy as np

y_true = np.array([0, 0, 1, 1])
y_prob = np.array([0.1, 0.3, 0.7, 0.9])

nll   = _safe_nll(y_true, y_prob)
brier = _safe_brier(y_true, y_prob)
auroc = _safe_auroc(y_true, y_prob)

print(f"NLL={nll:.4f}  Brier={brier:.4f}  AUROC={auroc:.4f}")
assert nll > 0
assert 0 <= brier <= 1
assert auroc is not None and 0.5 <= auroc <= 1.0

# Guard: AUROC must return None when only one class present
auroc_none = _safe_auroc(np.array([1, 1, 1]), np.array([0.8, 0.9, 0.7]))
assert auroc_none is None, f"Expected None, got {auroc_none}"
print("_safe_auroc None guard OK")

# --- compute_loco_metrics with synthetic subject DataFrame ---
import pandas as pd

mock_sub = pd.DataFrame({
    "true_label":      [0, 0, 1, 1],
    "predicted_label": [0, 1, 1, 0],
    "P_PD":            [0.2, 0.6, 0.8, 0.4],
    "uncertainty":     [0.1, 0.4, 0.15, 0.45],
    "is_correct":      [1,   0,   1,    0],
})
metrics = compute_loco_metrics(mock_sub)

required_keys = [
    "accuracy", "balanced_accuracy", "sensitivity", "specificity",
    "f1", "roc_auc", "nll", "brier",
    "mean_uncertainty_correct", "mean_uncertainty_incorrect",
    "median_uncertainty_correct", "median_uncertainty_incorrect",
    "uncertainty_error_auroc",
]
for key in required_keys:
    assert key in metrics, f"Missing metric key: {key}"

bal_acc = metrics["balanced_accuracy"]
print(f"compute_loco_metrics OK — balanced_acc={bal_acc:.3f}")

# --- build_cohort_splits with synthetic manifest ---
manifest = pd.DataFrame({
    "subject_id": [
        "GaCo01", "GaCo02", "GaPt01",
        "JuCo01", "JuCo02", "JuPt01",
        "SiCo01", "SiCo02", "SiPt01",
    ],
    "study":    ["Ga", "Ga", "Ga", "Ju", "Ju", "Ju", "Si", "Si", "Si"],
    "label":    [0, 0, 1, 0, 0, 1, 0, 0, 1],
    "filepath": ["x.txt"] * 9,
})
train_df, val_df, test_df = build_cohort_splits(
    manifest, train_cohorts=["Ga", "Ju"], test_cohort="Si", seed=42
)
train_subjs = set(train_df["subject_id"])
test_subjs  = set(test_df["subject_id"])
assert not (train_subjs & test_subjs), "Subject leakage detected!"
assert all(s.startswith("Si") for s in test_subjs), "Wrong subjects in test set!"
print(f"build_cohort_splits OK — train={len(train_subjs)} subjs, test={len(test_subjs)} subjs")

# --- aggregate_to_subject_level with synthetic window data ---
window_mock = pd.DataFrame({
    "experiment":     [1, 1, 1, 1],
    "train_cohorts":  ["Ga+Ju"] * 4,
    "test_cohort":    ["Si"] * 4,
    "subject_id":     ["SiCo01", "SiCo01", "SiPt01", "SiPt01"],
    "study":          ["Si"] * 4,
    "true_label":     [0, 0, 1, 1],
    "e_HC":           [10.0, 9.0, 1.0, 1.5],
    "e_PD":           [1.0,  1.2, 9.0, 8.5],
    "alpha_HC":       [11.0, 10.0, 2.0, 2.5],
    "alpha_PD":       [2.0,  2.2,  10.0, 9.5],
    "P_HC":           [0.85, 0.82, 0.17, 0.21],
    "P_PD":           [0.15, 0.18, 0.83, 0.79],
    "uncertainty":    [0.15, 0.18, 0.17, 0.19],
    "predicted_label":[0, 0, 1, 1],
    "is_correct":     [1, 1, 1, 1],
})
sub_out = aggregate_to_subject_level(window_mock)
assert len(sub_out) == 2, f"Expected 2 subjects, got {len(sub_out)}"
assert "predicted_label" in sub_out.columns
assert "is_correct" in sub_out.columns
print(f"aggregate_to_subject_level OK — {len(sub_out)} subjects aggregated")

print()
print("=" * 40)
print("ALL SMOKE TESTS PASSED")
print("=" * 40)
