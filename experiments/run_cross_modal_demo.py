"""Phase 9 Experiment: Cross-Modal Reliability Fusion Live Demo & Evaluation.

Demonstrates decision-level fusion of Gait and Voice reliability engine outputs across:
    - CASE 1: Strong Cross-Modal Agreement (CONSISTENT -> ACCEPT)
    - CASE 2: Direct Modality Conflict (MODALITY_CONFLICT -> FLAG)
    - CASE 3: Partial Support / One Modality Uncertain (PARTIAL_SUPPORT -> FLAG)

Important Note on Paired Subjects:
    Gait (PhysioNet Ga/Ju/Si) and Voice (MDVR-KCL) originate from distinct clinical cohorts.
    No true paired subjects exist across datasets. Demonstrations use synthetic pairings
    of existing modality-level outputs strictly for evaluating fusion logic.

Outputs:
    - results/cross_modal/cross_modal_config.json
    - results/cross_modal/cross_modal_results.json
    - results/cross_modal/cross_modal_summary.csv
    - results/cross_modal/cross_modal_dashboard.png
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

# Ensure workspace root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from cross_modal_reliability import CrossModalReliabilityEngine

CROSS_MODAL_DIR = root_dir / "results" / "cross_modal"
CROSS_MODAL_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_JSON = CROSS_MODAL_DIR / "cross_modal_config.json"
RESULTS_JSON = CROSS_MODAL_DIR / "cross_modal_results.json"
SUMMARY_CSV = CROSS_MODAL_DIR / "cross_modal_summary.csv"
FIG_DASHBOARD = CROSS_MODAL_DIR / "cross_modal_dashboard.png"


def load_gait_results() -> List[Dict]:
    """Load existing Gait reliability results."""
    gait_json = root_dir / "Gait" / "outputs" / "reliability" / "reliability_results.json"
    if gait_json.exists():
        with open(gait_json, "r") as f:
            return json.load(f)
    return []


def load_voice_results() -> Dict:
    """Load existing Voice reliability results."""
    voice_json = root_dir / "Voice" / "results" / "reliability" / "reliability_results.json"
    if voice_json.exists():
        with open(voice_json, "r") as f:
            return json.load(f)
    return {}


def run_cross_modal_demo():
    """Run Phase 9 Cross-Modal Reliability Fusion live demo."""
    engine = CrossModalReliabilityEngine()

    voice_res = load_voice_results()
    voice_subs = voice_res.get("subject_results", [])

    # 1. Define Representative Demonstration Scenarios (Synthetic Pairings)
    scenarios = [
        {
            "scenario": "CASE 1: STRONG CROSS-MODAL AGREEMENT (PD)",
            "gait": {
                "prediction": "Parkinson's Disease", "predicted_label": 1,
                "p_pd": 0.89, "p_healthy": 0.11, "uncertainty": 0.35,
                "conformal_prediction_set": ["Parkinson's Disease"],
                "reliability": "ACCEPT", "reliability_level": "HIGH"
            },
            "voice": {
                "prediction": "Parkinson's Disease", "predicted_label": 1,
                "p_pd": 0.79, "p_healthy": 0.21, "uncertainty": 0.40,
                "conformal_prediction_set": ["Parkinson's Disease"],
                "reliability": "ACCEPT", "reliability_level": "HIGH"
            }
        },
        {
            "scenario": "CASE 2: DIRECT MODALITY CONFLICT (PD vs Healthy)",
            "gait": {
                "prediction": "Parkinson's Disease", "predicted_label": 1,
                "p_pd": 0.85, "p_healthy": 0.15, "uncertainty": 0.32,
                "conformal_prediction_set": ["Parkinson's Disease"],
                "reliability": "ACCEPT", "reliability_level": "HIGH"
            },
            "voice": {
                "prediction": "Healthy Control", "predicted_label": 0,
                "p_pd": 0.35, "p_healthy": 0.65, "uncertainty": 0.45,
                "conformal_prediction_set": ["Healthy Control"],
                "reliability": "ACCEPT", "reliability_level": "HIGH"
            }
        },
        {
            "scenario": "CASE 3: AGREEMENT BUT VOICE UNCERTAIN (FLAGGED)",
            "gait": {
                "prediction": "Parkinson's Disease", "predicted_label": 1,
                "p_pd": 0.88, "p_healthy": 0.12, "uncertainty": 0.30,
                "conformal_prediction_set": ["Parkinson's Disease"],
                "reliability": "ACCEPT", "reliability_level": "HIGH"
            },
            "voice": {
                "prediction": "Parkinson's Disease", "predicted_label": 1,
                "p_pd": 0.569, "p_healthy": 0.431, "uncertainty": 0.6835,
                "conformal_prediction_set": [],
                "reliability": "FLAG", "reliability_level": "LOW"
            }
        },
        {
            "scenario": "CASE 4: STRONG CROSS-MODAL AGREEMENT (Healthy)",
            "gait": {
                "prediction": "Healthy Control", "predicted_label": 0,
                "p_pd": 0.15, "p_healthy": 0.85, "uncertainty": 0.25,
                "conformal_prediction_set": ["Healthy Control"],
                "reliability": "ACCEPT", "reliability_level": "HIGH"
            },
            "voice": {
                "prediction": "Healthy Control", "predicted_label": 0,
                "p_pd": 0.18, "p_healthy": 0.82, "uncertainty": 0.28,
                "conformal_prediction_set": ["Healthy Control"],
                "reliability": "ACCEPT", "reliability_level": "HIGH"
            }
        }
    ]

    fusion_results = []
    for sc in scenarios:
        fused = engine.fuse_modalities(sc["gait"], sc["voice"])
        fused["scenario_title"] = sc["scenario"]
        fusion_results.append(fused)

    # 2. Terminal Live Demo Output
    print_cross_modal_terminal_demo(fusion_results)

    # 3. Save JSON & CSV Artifacts
    config_data = {
        "experiment": "Cross-Modal Reliability Fusion (Gait + Voice)",
        "paired_subjects_note": "Gait (PhysioNet Ga/Ju/Si) and Voice (MDVR-KCL) cohorts are distinct. Synthetic pairings are used strictly for demonstrating decision-level fusion logic.",
        "scenarios_evaluated": len(scenarios)
    }
    with open(CONFIG_JSON, "w") as f:
        json.dump(config_data, f, indent=4)

    results_data = {
        "config": config_data,
        "demonstration_results": fusion_results
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(results_data, f, indent=4)
    print(f"Saved cross-modal results JSON to: {RESULTS_JSON}")

    summary_rows = [
        {
            "scenario": r["scenario_title"],
            "gait_pred": r["gait"]["prediction"],
            "gait_p_pd": r["gait"]["p_pd"],
            "gait_rel": r["gait"]["reliability"],
            "voice_pred": r["voice"]["prediction"],
            "voice_p_pd": r["voice"]["p_pd"],
            "voice_rel": r["voice"]["reliability"],
            "pred_agree": r["prediction_agreement"],
            "conformal_status": r["conformal_agreement_status"],
            "cross_modal_status": r["cross_modal_status"],
            "final_decision": r["final_decision"],
            "reason": r["reason"]
        }
        for r in fusion_results
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved cross-modal summary CSV to: {SUMMARY_CSV}\n")

    # 4. Generate Visual Dashboard
    generate_cross_modal_dashboard(fusion_results)

    return results_data


def print_cross_modal_terminal_demo(results: List[Dict]):
    """Print terminal live-demo output in exact requested format."""
    print("\n" + "=" * 65)
    print("        PARKINSON'S CROSS-MODAL RELIABILITY FUSION")
    print("=" * 65)

    for r in results:
        print("\n" + "-" * 65)
        print(f"  {r['scenario_title']}")
        print("-" * 65)
        
        g = r["gait"]
        v = r["voice"]

        print("GAIT MODALITY:")
        print(f"  Prediction:       {g['prediction']}")
        print(f"  P(PD):            {g['p_pd']*100:.1f}%")
        print(f"  Conformal Set:    {g['conformal_set']}")
        print(f"  Reliability:      {g['reliability']} ({g['reliability_level']})")

        print("\nVOICE MODALITY:")
        print(f"  Prediction:       {v['prediction']}")
        print(f"  P(PD):            {v['p_pd']*100:.1f}%")
        print(f"  Conformal Set:    {v['conformal_set']}")
        print(f"  Reliability:      {v['reliability']} ({v['reliability_level']})")

        print("\nCROSS-MODAL EVIDENCE:")
        print(f"  Prediction Agree: {'YES' if r['prediction_agreement'] else 'NO'}")
        print(f"  Conformal Status: {r['conformal_agreement_status']}")
        print(f"  Final Status:     {r['cross_modal_status']}")
        print(f"  Final Reliability:{r['cross_modal_reliability']}")
        print(f"  Final Decision:   [{r['final_decision']}]")
        print(f"  Reason:           {r['reason']}")
        print("-" * 65)

    print("=" * 65 + "\n")


def generate_cross_modal_dashboard(results: List[Dict]):
    """Generate publication-quality cross-modal fusion dashboard figure."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 9), dpi=300)

    scenarios = [f"Case {i+1}" for i in range(len(results))]
    x = np.arange(len(scenarios))
    width = 0.35

    gait_pds = [r["gait"]["p_pd"] for r in results]
    voice_pds = [r["voice"]["p_pd"] for r in results]

    # Panel A: Gait vs Voice P(PD)
    ax1.bar(x - width/2, gait_pds, width, label="Gait P(PD)", color="#1f77b4", edgecolor="black")
    ax1.bar(x + width/2, voice_pds, width, label="Voice P(PD)", color="#ff7f0e", edgecolor="black")
    ax1.axhline(0.5, color="black", linestyle="--", lw=1.2, label="Boundary (0.5)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios)
    ax1.set_ylabel("P(PD)")
    ax1.set_title("A. Modality Probability Comparison", weight="bold")
    ax1.set_ylim([0, 1.05])
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel B: Conformal Set Sizes
    gait_cs = [r["gait"]["conformal_set_size"] for r in results]
    voice_cs = [r["voice"]["conformal_set_size"] for r in results]
    ax2.bar(x - width/2, gait_cs, width, label="Gait Set Size", color="#2ca02c", edgecolor="black")
    ax2.bar(x + width/2, voice_cs, width, label="Voice Set Size", color="#9467bd", edgecolor="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios)
    ax2.set_ylabel("Conformal Set Size |C(X)|")
    ax2.set_yticks([0, 1, 2])
    ax2.set_title("B. Conformal Set Size Alignment", weight="bold")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Panel C: Cross-Modal Reliability Decision Status
    dec_colors = ["#2ca02c" if r["final_decision"] == "ACCEPT" else "#d62728" for r in results]
    dec_vals = [1 if r["final_decision"] == "ACCEPT" else 0.5 for r in results]
    ax3.bar(x, dec_vals, color=dec_colors, edgecolor="black", width=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(scenarios)
    ax3.set_yticks([0.5, 1.0])
    ax3.set_yticklabels(["FLAG", "ACCEPT"])
    ax3.set_title("C. Final Cross-Modal Decision (ACCEPT vs FLAG)", weight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)

    # Panel D: Cross-Modal Status Breakdown Text Box
    ax4.axis("off")
    table_data = [
        [r["scenario_title"].split(":")[0], r["cross_modal_status"], r["final_decision"], r["reason"]]
        for r in results
    ]
    col_labels = ["Scenario", "Status", "Decision", "Reason"]
    table = ax4.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.1, 1.6)
    ax4.set_title("D. Decision Policy Summary Matrix", weight="bold")

    plt.suptitle("Parkinson's Disease Decision-Level Cross-Modal Reliability Fusion", fontsize=14, weight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(FIG_DASHBOARD, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_cross_modal_demo()
