"""CLI tool for Tri-Modal Parkinson's Disease classification and decision reliability evaluation."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from multimodal_pipeline import MultimodalPredictor


def main():
    parser = argparse.ArgumentParser(
        description="Run unified tri-modal Parkinson's detection and reliability assessment across Gait, Voice, and Handwriting."
    )
    parser.add_argument("--gait", type=str, default=None, help="Path to gait VGRF time-series (.txt / .csv)")
    parser.add_argument("--voice", type=str, default=None, help="Path to voice audio recording (.wav)")
    parser.add_argument("--handwriting", type=str, default=None, help="Path to handwriting drawing image (.png / .jpg)")
    parser.add_argument("--json", action="store_true", help="Print raw JSON decision structure")

    args = parser.parse_args()

    if not args.gait and not args.voice and not args.handwriting:
        parser.error("At least one biomarker input (--gait, --voice, or --handwriting) must be specified.")

    predictor = MultimodalPredictor()
    result = predictor.predict(
        gait=args.gait,
        voice=args.voice,
        handwriting=args.handwriting
    )

    if args.json:
        print(json.dumps(result, indent=2))
        return

    # Formatted CLI Presentation
    print("\n" + "=" * 70)
    print(" 🧠 TRI-MODAL PARKINSON'S RELIABILITY & EVIDENCE REPORT")
    print("=" * 70)
    print(f" Active Modalities:   {', '.join(result['active_modalities'])} ({result['active_modality_count']}/3)")
    print(f" Final Prediction:    {result['final_prediction']} (Class {result['final_predicted_label']})")
    print(f" Consensus P(PD):     {result['consensus_probability_pd'] * 100:.1f}%")
    print(f" Decision Case:       {result['decision_case']}")
    print("-" * 70)
    rel_color = "ACCEPT" if result["final_reliability"] == "ACCEPT" else "FLAG"
    print(f" System Decision:     [{rel_color}] (Level: {result['final_reliability_level']})")
    print(f" Actionable Guidance: {result['clinical_action']}")
    print(f" Clinical Summary:    {result['clinical_summary']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
