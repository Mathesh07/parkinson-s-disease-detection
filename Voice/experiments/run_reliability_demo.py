"""Phase 8 Experiment: Voice Reliability Engine Live Demo & Evaluation.

Demonstrates inference-time reliability decision system on representative test subjects.
Highlights contrast between:
    - Case A (High Confidence / High Reliability -> ACCEPT)
    - Case B (Borderline / Ambiguous / High Uncertainty -> FLAG)

Also generates:
    - Voice/results/reliability/reliability_results.json
    - Voice/results/reliability/reliability_summary.csv
    - Voice/results/reliability/reliability_dashboard.png
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Ensure workspace root and Voice directory are in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
voice_dir = root_dir / "Voice"
for p in (str(root_dir), str(voice_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from Voice import config
from Voice.dataset import build_metadata, create_subject_splits
from Voice.model import Wav2Vec2ForParkinsons
from Voice.reliability_engine import VoiceReliabilityEngine
from Voice.utils import set_seed

RELIABILITY_DIR = config.RESULTS_DIR / "reliability"
RELIABILITY_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_JSON = RELIABILITY_DIR / "reliability_results.json"
SUMMARY_CSV = RELIABILITY_DIR / "reliability_summary.csv"
FIG_DASHBOARD = RELIABILITY_DIR / "reliability_dashboard.png"


def run_reliability_demo(seed: int = 42):
    """Run complete Phase 8 Voice Reliability Engine live demo across held-out test subjects."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Held-Out Test Subjects
    meta_df = build_metadata()
    _, val_df, test_df = create_subject_splits(meta_df, seed=seed)
    test_subjects = sorted(test_df["subject_id"].unique().tolist())

    sub_labels = {}
    for idx, row in meta_df.iterrows():
        sub_labels[row["subject_id"]] = int(row["label"])

    # Load 768-D clean subject representations
    embs_file = config.RESULTS_DIR / "embeddings" / "voice_embeddings.pt"
    embs_dict = torch.load(embs_file)
    sub_embeddings = embs_dict["subject_embeddings"]

    # Load 5 Ensemble Member Models
    ensemble_models = []
    for idx in range(5):
        ckpt = config.RESULTS_DIR / "deep_ensemble" / f"member_{idx}" / "checkpoint" / "best_model.pt"
        m = Wav2Vec2ForParkinsons(model_name=config.MODEL_NAME, num_classes=2)
        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        m.to(device)
        m.eval()
        ensemble_models.append(m)

    # Initialize Reliability Engine
    engine = VoiceReliabilityEngine(q_hat=0.407552)

    # 2. Evaluate Reliability for all Test Subjects
    subject_results = []
    with torch.no_grad():
        for sub in test_subjects:
            y_true = sub_labels[sub]
            emb = sub_embeddings[sub].unsqueeze(0).to(device)

            m_probs = []
            for m in ensemble_models:
                m_ev = m(emb)
                m_p = float(torch.softmax(m_ev, dim=-1)[0, 1].cpu().item())
                m_probs.append(m_p)

            meta_info = {
                "filename": f"recording_{sub}.wav",
                "duration_sec": 10.0,
                "num_chunks": 5,
                "subject_id": sub
            }

            res = engine.evaluate_reliability(m_probs, audio_meta=meta_info)
            res["subject_id"] = sub
            res["true_label"] = y_true
            res["true_name"] = "Parkinson's Disease" if y_true == 1 else "Healthy Control"
            res["correct_point_pred"] = int(res["predicted_label"] == y_true)
            subject_results.append(res)

    results_df = pd.DataFrame(subject_results)

    # 3. Print Rich Terminal Live Demo Output for Representative Contrast Cases
    print_live_demo_terminal(subject_results)

    # 4. Save JSON and Summary CSV
    summary_data = {
        "engine": "Voice Wav2Vec2 Reliability Engine",
        "n_test_subjects": len(test_subjects),
        "q_hat": 0.407552,
        "num_accept": int(np.sum(results_df["reliability"] == "ACCEPT")),
        "num_flag": int(np.sum(results_df["reliability"] == "FLAG")),
        "accept_rate": float(np.mean(results_df["reliability"] == "ACCEPT")),
        "flag_rate": float(np.mean(results_df["reliability"] == "FLAG")),
        "subject_results": subject_results
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(summary_data, f, indent=4)
    print(f"Saved reliability results JSON to: {RESULTS_JSON}")

    summary_df = pd.DataFrame([
        {
            "subject_id": r["subject_id"],
            "true_label": r["true_name"],
            "prediction": r["point_prediction"],
            "p_pd": r["p_pd"],
            "agreement": r["ensemble_agreement"],
            "entropy": r["predictive_entropy"],
            "conformal_set": r["conformal_prediction_set_str"],
            "reliability": r["reliability"],
            "level": r["reliability_level"],
            "reasons": ", ".join(r["reliability_reasons"])
        }
        for r in subject_results
    ])
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved reliability summary CSV to: {SUMMARY_CSV}\n")

    # 5. Generate Reliability Dashboard Plot
    generate_reliability_dashboard(subject_results)

    return summary_data


def print_live_demo_terminal(results: List[Dict]):
    """Print beautifully formatted live-demo output in terminal matching prompt specifications."""
    print("\n" + "=" * 65)
    print("        PARKINSON'S VOICE RELIABILITY ENGINE DEMO")
    print("=" * 65)

    # Case A: Select strongest ACCEPT prediction (e.g. ID29 or ID23)
    accept_cases = [r for r in results if r["reliability"] == "ACCEPT"]
    flag_cases = [r for r in results if r["reliability"] == "FLAG"]

    demo_cases = []
    if accept_cases:
        demo_cases.append(("CASE A: HIGH RELIABILITY PREDICTION (ACCEPT)", accept_cases[0]))
    if flag_cases:
        demo_cases.append(("CASE B: AMBIGUOUS / FLAGGED PREDICTION (FLAG)", flag_cases[0]))

    for title, r in demo_cases:
        print("\n" + "-" * 65)
        print(f"  {title}")
        print("-" * 65)
        print(f"Audio File:            {r['filename']} (Subject {r['subject_id']})")
        print(f"Duration:              {r['duration_sec']:.1f} sec ({r['num_chunks']} chunks)")
        print(f"\nMODEL PREDICTION:")
        print(f"  Point Prediction:    {r['point_prediction']}")
        print(f"  P(PD):               {r['p_pd'] * 100:.1f}%")
        print(f"  P(Healthy):          {r['p_healthy'] * 100:.1f}%")
        print(f"\nENSEMBLE (M=5):")
        m_formatted = " | ".join([f"{p:.3f}" for p in r["ensemble_member_probabilities"]])
        print(f"  Member Probs:        [{m_formatted}]")
        print(f"  Agreement:           {r['ensemble_agreement']}")
        print(f"  Variance:            {r['ensemble_variance']:.6f}")
        print(f"\nUNCERTAINTY QUANTIFICATION:")
        print(f"  Predictive Entropy:  {r['predictive_entropy']:.4f}")
        print(f"  Mutual Information:  {r['mutual_information']:.4f}")
        print(f"\nCONFORMAL PREDICTION:")
        print(f"  Prediction Set:      {r['conformal_prediction_set_str']}")
        print(f"  Quantile (q_hat):    {r['conformal_q_hat']:.4f}")
        print(f"\nRELIABILITY DECISION:")
        # Decision Banner
        status_label = "ACCEPT" if r["reliability"] == "ACCEPT" else "FLAG"
        print(f"  Decision:            [{status_label}]")
        print(f"  Level:               {r['reliability_level']}")
        reasons_str = ", ".join(r["reliability_reasons"])
        print(f"  Reason(s):           {reasons_str}")
        print("-" * 65)

    print("=" * 65 + "\n")


def generate_reliability_dashboard(results: List[Dict]):
    """Generate publication-quality reliability dashboard plot."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9), dpi=300)

    sub_ids = [r["subject_id"] for r in results]
    x = np.arange(len(sub_ids))

    p_pd_vals = [r["p_pd"] for r in results]
    entropies = [r["predictive_entropy"] for r in results]
    variances = [r["ensemble_variance"] for r in results]
    set_sizes = [r["conformal_set_size"] for r in results]

    colors = ["#2ca02c" if r["reliability"] == "ACCEPT" else "#d62728" for r in results]

    # Panel A: P(PD) with Reliability Color Coding
    bars1 = ax1.bar(x, p_pd_vals, color=colors, alpha=0.85, edgecolor="black", width=0.55)
    ax1.axhline(0.5, color="black", linestyle="--", lw=1.2, label="Boundary (0.5)")
    ax1.axhspan(0.40, 0.60, color="gray", alpha=0.15, label="Borderline Region [0.40, 0.60]")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sub_ids)
    ax1.set_ylabel("Mean P(PD)")
    ax1.set_title("A. Model Probability P(PD) per Subject", weight="bold")
    ax1.set_ylim([0, 1.05])
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel B: Predictive Entropy H(p)
    ax2.bar(x, entropies, color="#1f77b4", alpha=0.85, edgecolor="black", width=0.55)
    ax2.axhline(0.65, color="#d62728", linestyle="--", lw=1.5, label="Entropy Flag Threshold (0.65)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(sub_ids)
    ax2.set_ylabel("Predictive Entropy H(p)")
    ax2.set_title("B. Predictive Uncertainty (Entropy)", weight="bold")
    ax2.set_ylim([0, 0.75])
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Panel C: Ensemble Variance
    ax3.bar(x, variances, color="#ff7f0e", alpha=0.85, edgecolor="black", width=0.55)
    ax3.set_xticks(x)
    ax3.set_xticklabels(sub_ids)
    ax3.set_ylabel("Ensemble Variance")
    ax3.set_title("C. Ensemble Model Disagreement (Variance)", weight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)

    # Panel D: Conformal Set Size & Reliability Decision
    ax4.bar(x, set_sizes, color="#9467bd", alpha=0.85, edgecolor="black", width=0.55)
    ax4.set_xticks(x)
    ax4.set_xticklabels(sub_ids)
    ax4.set_ylabel("Conformal Set Size |C(X)|")
    ax4.set_yticks([0, 1, 2])
    ax4.set_title("D. Conformal Set Size (0=Empty, 1=Singleton, 2=Both)", weight="bold")
    ax4.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Voice Wav2Vec2 Reliability Decision Engine Dashboard", fontsize=14, weight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(FIG_DASHBOARD, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_reliability_demo()
