"""Cohort/Domain Shift Analysis on Learned Gait Representations.

Research Question:
    Are cohort/domain differences measurable in the learned 128-dimensional gait representation,
    and can these differences help explain variation in model reliability across cohorts?

Pairwise Domain Classification:
    1. Ga vs Ju
    2. Ga vs Si
    3. Ju vs Si

Target: Cohort/source-study identity (0 = Cohort A, 1 = Cohort B).
Features: 128-dimensional subject-level embeddings from Gait/outputs/features/gait_embeddings.pt.
Classifier: Logistic Regression (max_iter=2000, random_state=42).

Leakage Guards:
    - Subject-level split using GroupShuffleSplit(test_size=0.25, random_state=42).
    - StandardScaler fitted strictly on training subjects.
    - Zero subject overlap between train and test sets.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Path & Global Configurations
# ---------------------------------------------------------------------------
SEED = 42

_THIS_FILE = Path(__file__).resolve()
_EXPERIMENTS_DIR = _THIS_FILE.parent          # Gait/experiments/
_GAIT_DIR = _EXPERIMENTS_DIR.parent           # Gait/
_ROOT_DIR = _GAIT_DIR.parent                  # repo root

EMBEDDINGS_PATH = _GAIT_DIR / "outputs" / "features" / "gait_embeddings.pt"
OUTPUT_DIR = _GAIT_DIR / "outputs" / "cohort_shift"
RESULTS_CSV = OUTPUT_DIR / "cohort_shift_results.csv"
RESULTS_JSON = OUTPUT_DIR / "cohort_shift_results.json"


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_and_validate_embeddings(embeddings_path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    """
    Load subject-level embeddings and perform mandatory safety checks.

    Checks:
    1. File exists.
    2. 'subject_embeddings' key exists.
    3. Exactly 165 subjects loaded.
    4. Every embedding has shape [128].
    5. Cohorts contain only 'Ga', 'Ju', 'Si'.
    """
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found at: {embeddings_path}")

    data = torch.load(embeddings_path, map_location="cpu")
    if not isinstance(data, dict) or "subject_embeddings" not in data:
        raise KeyError(f"Expected dict with 'subject_embeddings' key in {embeddings_path}")

    subject_embeds = data["subject_embeddings"]
    num_subjects = len(subject_embeds)
    if num_subjects != 165:
        raise ValueError(f"Expected 165 subject embeddings, found {num_subjects}")

    subject_cohorts = {}
    valid_cohorts = {"Ga", "Ju", "Si"}

    for sub_id, tensor in subject_embeds.items():
        if getattr(tensor, "shape", None) != torch.Size([128]):
            raise ValueError(f"Subject {sub_id} embedding has invalid shape {getattr(tensor, 'shape', None)}, expected [128]")

        cohort = sub_id[:2]
        if cohort not in valid_cohorts:
            raise ValueError(f"Subject {sub_id} has invalid cohort prefix '{cohort}'. Expected one of {valid_cohorts}")

        subject_cohorts[sub_id] = cohort

    print(f"Loaded {num_subjects} subject embeddings across cohorts: {dict(pd.Series(list(subject_cohorts.values())).value_counts())}")
    return subject_embeds, subject_cohorts


def run_pairwise_experiment(
    cohort_a: str,
    cohort_b: str,
    subject_embeds: Dict[str, torch.Tensor],
    subject_cohorts: Dict[str, str],
    seed: int = SEED,
) -> Dict:
    """
    Run binary domain classification for Cohort A vs Cohort B.
    """
    # 1. Filter subjects for Cohort A and Cohort B
    subs_a = [s for s, c in subject_cohorts.items() if c == cohort_a]
    subs_b = [s for s, c in subject_cohorts.items() if c == cohort_b]

    pair_subjects = sorted(subs_a + subs_b)
    X = np.array([subject_embeds[s].numpy() for s in pair_subjects])
    y = np.array([0 if subject_cohorts[s] == cohort_a else 1 for s in pair_subjects])
    groups = np.array(pair_subjects)

    # 2. Subject-level train/test split using GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    train_subs, test_subs = groups[train_idx], groups[test_idx]

    # 3. Leakage Guard Assertions
    overlap = set(train_subs) & set(test_subs)
    assert len(overlap) == 0, f"LEAKAGE ERROR: Subjects overlap in train and test: {overlap}"

    # 4. Standardize features (Train-only fit)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Confirm scaler stats derived strictly from train set
    assert scaler.mean_.shape[0] == 128, "Scaler feature dimension mismatch."
    assert not np.isnan(scaler.mean_).any(), "NaN values found in scaler mean."

    # 5. Train Classifier
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X_train_scaled, y_train)

    # 6. Evaluate
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))

    if len(np.unique(y_test)) > 1:
        roc_auc = float(roc_auc_score(y_test, y_prob))
    else:
        roc_auc = None

    cm = confusion_matrix(y_test, y_pred).tolist()

    res = {
        "cohort_a": cohort_a,
        "cohort_b": cohort_b,
        "num_subjects": len(pair_subjects),
        "train_subjects": len(train_subs),
        "test_subjects": len(test_subs),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
    }

    # Print clear terminal report
    print(f"\n{'='*60}")
    print(f"{cohort_a} vs {cohort_b}")
    print(f"{'='*60}")
    print(f"Subjects:          {len(pair_subjects)}")
    print(f"Train subjects:    {len(train_subs)}")
    print(f"Test subjects:     {len(test_subs)}")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"ROC-AUC:           {roc_auc:.4f}" if roc_auc is not None else "ROC-AUC:           N/A")
    print("\nConfusion Matrix:")
    print(np.array(cm))

    return res


def main():
    set_seed(SEED)
    print("Initializing Cohort/Domain Shift Analysis...")

    # Load & Validate
    subject_embeds, subject_cohorts = load_and_validate_embeddings(EMBEDDINGS_PATH)

    # 3 Pairwise Experiments
    pairwise_pairs = [
        ("Ga", "Ju"),
        ("Ga", "Si"),
        ("Ju", "Si"),
    ]

    results = []
    for cohort_a, cohort_b in pairwise_pairs:
        res = run_pairwise_experiment(cohort_a, cohort_b, subject_embeds, subject_cohorts, seed=SEED)
        results.append(res)

    # Save Results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save CSV
    df_res = pd.DataFrame(results)
    df_res["confusion_matrix"] = df_res["confusion_matrix"].apply(json.dumps)
    df_res.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved CSV results -> {RESULTS_CSV}")

    # Save JSON
    json_output = {
        "experiment": "Cohort/Domain Shift Classification",
        "description": "Binary domain classification on 128-D subject gait embeddings using Logistic Regression.",
        "seed": SEED,
        "test_size": 0.25,
        "results": results,
    }
    with open(RESULTS_JSON, "w") as fh:
        json.dump(json_output, fh, indent=4)
    print(f"Saved JSON results -> {RESULTS_JSON}")


if __name__ == "__main__":
    main()
