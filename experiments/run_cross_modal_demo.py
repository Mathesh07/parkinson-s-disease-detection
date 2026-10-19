"""Phase 9 Experiment: Tri-Modal Cross-Modal Reliability Fusion Live Demo & Evaluation.

Demonstrates decision-level fusion across Gait, Voice, and Handwriting reliability outputs:
    - CASE 1: Strong Tri-Modal Agreement (CONSISTENT -> ACCEPT)
    - CASE 2: Modality Conflict (2 vs 1: Gait & Voice PD vs Handwriting Healthy -> FLAG)
    - CASE 3: Partial Support / Handwriting Modality Uncertain (PARTIAL_SUPPORT -> FLAG)
    - CASE 4: Strong Tri-Modal Agreement for Healthy Control (CONSISTENT -> ACCEPT)

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


def run_cross_modal_demo():
    """Run Tri-Modal Cross-Modal Reliability Fusion live demo."""
    engine = CrossModalReliabilityEngine()

    # 1. Define Representative Demonstration Scenarios across all 3 Modalities
    scenarios = [
        {
            "scenario": "CASE 1: STRONG TRI-MODAL AGREEMENT (PD)",
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
            },
            "handwriting": {
                "prediction": "Parkinson's Disease", "predicted_label": 1,
                "p_pd": 0.865, "p_healthy": 0.135, "uncertainty": 0.315,
                "conformal_prediction_set": ["Parkinson's Disease"],
                "reliability": "ACCEPT", "reliability_level": "HIGH"
            }
        },
        {
            "scenario": "CASE 2: DIRECT MODALITY CONFLICT (2 vs 1)",
            "gait": {
                "prediction": "Parkinson's Disease", "predicted_label": 1,
                "p_pd": 0.85, "p_healthy": 0.15, "uncertainty": 0.32,
                "conformal_prediction_set": ["Parkinson's Disease"],
                "reliability": "ACCEPT", "reliability_level": "HIGH"
            },
            "voice": {
                "prediction": "Parkinson's Disease", "predicted_label": 1,
                "p_pd": 0.80, "p_healthy": 0.20, "uncertainty": 0.39,
                "conformal_prediction_set": ["Parkinson's Disease"],
                "reliability": "ACCEPT", "reliability_level": "HIGH"
            },
            "handwriting": {
                "prediction": "Healthy Control", "predicted_label": 0,
                "p_pd": 0.22, "p_healthy": 0.78, "uncertainty": 0.34,
                "conformal_prediction_set": ["Healthy Control"],
                "reliability": "ACCEPT", "reliability_level": "HIGH"
            }
        },
        {
            "scenario": "CASE 3: AGREEMENT BUT HANDWRITING UNCERTAIN",
            "gait": {
                "prediction": "Parkinson's Disease", "predicted_label": 1,
                "p_pd": 0.88, "p_healthy": 0.12, "uncertainty": 0.30,
                "conformal_prediction_set": ["Parkinson's Disease"],
                "reliability": "ACCEPT", "reliability_level": "HIGH"
            },
            "voice": {
                "prediction": "Parkinson's Disease", "predicted_label": 1,
                "p_pd": 0.81, "p_healthy": 0.19, "uncertainty": 0.38,
                "conformal_prediction_set": ["Parkinson's Disease"],
                "reliability": "ACCEPT", "reliability_level": "HIGH"
            },
            "handwriting": {
                "prediction": "Parkinson's Disease", "predicted_label": 1,
                "p_pd": 0.545, "p_healthy": 0.455, "uncertainty": 0.582,
                "conformal_prediction_set": [],
                "reliability": "FLAG", "reliability_level": "MEDIUM"
            }
        },
        {
            "scenario": "CASE 4: STRONG TRI-MODAL AGREEMENT (Healthy)",
            "gait": {
                "prediction": "Healthy Control", "predicted_label": 0,
                "p_pd": 0.11, "p_healthy": 0.89, "uncertainty": 0.28,
                "conformal_prediction_set": ["Healthy Control"],
                "reliability": "ACCEPT", "reliability_level": "HIGH"
            },
            "voice": {
                "prediction": "Healthy Control", "predicted_label": 0,
                "p_pd": 0.14, "p_healthy": 0.86, "uncertainty": 0.33,
                "conformal_prediction_set": ["Healthy Control"],
                "reliability": "ACCEPT", "reliability_level": "HIGH"
            },
            "handwriting": {
                "prediction": "Healthy Control", "predicted_label": 0,
                "p_pd": 0.135, "p_healthy": 0.865, "uncertainty": 0.29,
                "conformal_prediction_set": ["Healthy Control"],
                "reliability": "ACCEPT", "reliability_level": "HIGH"
            }
        }
    ]

    fusion_results = []
    for sc in scenarios:
        fused = engine.fuse_modalities(sc["gait"], sc["voice"], sc["handwriting"])
        fused["scenario_title"] = sc["scenario"]
        fusion_results.append(fused)

    # 2. Terminal Live Demo Output
    print_cross_modal_terminal_demo(fusion_results)

    # 3. Save JSON & CSV Artifacts
    config_data = {
        "experiment": "Tri-Modal Reliability Fusion (Gait + Voice + Handwriting)",
        "modalities": ["Gait", "Voice", "Handwriting"],
        "paired_subjects_note": "Gait (PhysioNet Ga/Ju/Si), Voice (MDVR-KCL), and Handwriting (NewHandPD) cohorts are distinct. Synthetic pairings evaluate decision-level tri-modal fusion logic.",
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
            "fused_prediction": r["fused_prediction"],
            "fused_p_pd": r["fused_p_pd"],
            "gait_pred": r["gait"]["prediction"],
            "gait_p_pd": r["gait"]["p_pd"],
            "voice_pred": r["voice"]["prediction"],
            "voice_p_pd": r["voice"]["p_pd"],
            "hw_pred": r["handwriting"]["prediction"],
            "hw_p_pd": r["handwriting"]["p_pd"],
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
    """Print terminal live-demo output in structured format."""
    print("\n" + "=" * 70)
    print("        TRI-MODAL PARKINSON'S CROSS-MODAL RELIABILITY FUSION")
    print("=" * 70)

    for r in results:
        print("\n" + "-" * 70)
        print(f"  {r['scenario_title']}")
        print("-" * 70)
        
        g = r["gait"]
        v = r["voice"]
        h = r["handwriting"]

        print(f"GAIT:        Pred={g['prediction']:<20} P(PD)={g['p_pd']*100:>5.1f}% | Set={str(g['conformal_set']):<25} | [{g['reliability']}]")
        print(f"VOICE:       Pred={v['prediction']:<20} P(PD)={v['p_pd']*100:>5.1f}% | Set={str(v['conformal_set']):<25} | [{v['reliability']}]")
        print(f"HANDWRITING: Pred={h['prediction']:<20} P(PD)={h['p_pd']*100:>5.1f}% | Set={str(h['conformal_set']):<25} | [{h['reliability']}]")

        print("\nTRI-MODAL SYNTHESIS & FINAL PREDICTION:")
        print(f"  Final Prediction:  {r['fused_prediction']} (Fused P(PD) = {r['fused_p_pd']*100:.1f}%)")
        print(f"  Consensus Status:  {'3/3 UNANIMOUS' if r['prediction_agreement'] else '2/3 MAJORITY (CONFLICT)'}")
        print(f"  Conformal Status:  {r['conformal_agreement_status']}")
        print(f"  System Decision:   [{r['final_decision']}] (Reliability: {r['cross_modal_reliability']})")
        print(f"  Policy Reason:     {r['reason']}")
        print("-" * 70)

    print("=" * 70 + "\n")


def generate_cross_modal_dashboard(results: List[Dict]):
    """Generate publication-quality tri-modal fusion dashboard figure."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    scenarios = [f"Case {i+1}" for i in range(len(results))]
    x = np.arange(len(scenarios))
    width = 0.25

    gait_pds = [r["gait"]["p_pd"] for r in results]
    voice_pds = [r["voice"]["p_pd"] for r in results]
    hw_pds = [r["handwriting"]["p_pd"] for r in results]

    # Panel A: Gait vs Voice vs Handwriting P(PD)
    ax1.bar(x - width, gait_pds, width, label="Gait P(PD)", color="#1f77b4", edgecolor="black")
    ax1.bar(x, voice_pds, width, label="Voice P(PD)", color="#ff7f0e", edgecolor="black")
    ax1.bar(x + width, hw_pds, width, label="Handwriting P(PD)", color="#2ca02c", edgecolor="black")
    ax1.axhline(0.5, color="black", linestyle="--", lw=1.2, label="Boundary (0.5)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios)
    ax1.set_ylabel("P(PD)")
    ax1.set_title("A. Tri-Modal Probability Comparison", weight="bold")
    ax1.set_ylim([0, 1.05])
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel B: Conformal Set Sizes
    gait_cs = [r["gait"]["conformal_set_size"] for r in results]
    voice_cs = [r["voice"]["conformal_set_size"] for r in results]
    hw_cs = [r["handwriting"]["conformal_set_size"] for r in results]
    ax2.bar(x - width, gait_cs, width, label="Gait Set Size", color="#1f77b4", edgecolor="black")
    ax2.bar(x, voice_cs, width, label="Voice Set Size", color="#ff7f0e", edgecolor="black")
    ax2.bar(x + width, hw_cs, width, label="Handwriting Set Size", color="#2ca02c", edgecolor="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios)
    ax2.set_ylabel("Conformal Set Size |C(X)|")
    ax2.set_yticks([0, 1, 2])
    ax2.set_title("B. Conformal Set Size Alignment Across 3 Modalities", weight="bold")
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
        [
            r["scenario_title"].split(":")[0],
            r["fused_prediction"].replace("Parkinson's Disease", "PD").replace("Healthy Control", "HC"),
            r["cross_modal_status"],
            r["final_decision"],
            r["reason"]
        ]
        for r in results
    ]
    col_labels = ["Scenario", "Prediction", "Status", "Decision", "Reason"]
    table = ax4.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.1, 1.6)
    ax4.set_title("D. Decision Policy & Tri-Modal Prediction Matrix", weight="bold")

    plt.suptitle("Parkinson's Disease Tri-Modal Decision-Level Cross-Modal Reliability Fusion", fontsize=14, weight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(FIG_DASHBOARD, bbox_inches="tight")
    plt.close()
    print(f"Generated and saved tri-modal dashboard to: {FIG_DASHBOARD}")


if __name__ == "__main__":
    run_cross_modal_demo()
