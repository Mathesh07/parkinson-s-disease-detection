"""CLI inference script for Handwriting-Based Parkinson's Disease classification and reliability evaluation."""

import argparse
import json
import sys
from pathlib import Path

# Add project root and Handwriting directory to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
for p in (str(current_dir), str(project_root)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from Handwriting import config
    from Handwriting.reliability_engine import HandwritingReliabilityEngine
except ImportError:
    import config
    from reliability_engine import HandwritingReliabilityEngine


def run_prediction(
    image_path: str,
    checkpoint: str = None,
    device: str = None,
    output_json: bool = False
):
    """Run evidential prediction and reliability quantification on a handwriting drawing image."""
    image_file = Path(image_path)
    if not image_file.exists():
        print(f"Error: Handwriting image file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    ckpt_path = Path(checkpoint) if checkpoint else (config.OUTPUT_DIR / "best_vit_evidential.pth")

    engine = HandwritingReliabilityEngine(
        checkpoint_path=ckpt_path if ckpt_path.exists() else None
    )

    try:
        result = engine.predict_image(image_path=image_file, device=device)
    except Exception as e:
        # Fallback if torch or weights unavailable
        print(f"Notice: Live model inference failed ({e}). Running evidential reliability check with fallback.", file=sys.stderr)
        result = engine.evaluate_reliability(
            p_pd=0.5,
            uncertainty=1.0,
            evidence=(0.0, 0.0),
            image_meta={"filename": image_file.name, "path": str(image_file), "note": "Uncertain / Unfitted Fallback"}
        )

    if output_json:
        print(json.dumps(result, indent=2))
        return result

    # Human-readable formatted summary
    print("\n" + "=" * 65)
    print(" ✍️  PARKINSON'S HANDWRITING RELIABILITY REPORT")
    print("=" * 65)
    print(f" Image File:        {image_file.name}")
    print(f" Final Prediction:  {result['prediction']} (Class {result['predicted_label']})")
    print(f" P(Parkinson's):    {result['p_pd'] * 100:.2f}%")
    print(f" P(Healthy):        {result['p_healthy'] * 100:.2f}%")
    print(f" Dirichlet Vacuity: {result['uncertainty']:.4f}")
    if result.get("evidence"):
        print(f" Evidence [HC, PD]: [{result['evidence'][0]:.3f}, {result['evidence'][1]:.3f}]")
    print(f" Conformal Set:     {{{', '.join(result['conformal_prediction_set'])}}}")
    print("-" * 65)
    rel_status = result["reliability"]
    level = result["reliability_level"]
    print(f" Reliability:       [{rel_status}] ({level})")
    print(f" Decision Reasons:  {', '.join(result['reliability_reasons'])}")
    print("=" * 65 + "\n")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Predict Parkinson's Disease and evaluate evidential reliability from a handwriting drawing image."
    )
    parser.add_argument("image_path", type=str, help="Path to spiral or meander drawing image (.png, .jpg)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to ViT evidential weights (.pth)")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cuda or cpu)")
    parser.add_argument("--json", action="store_true", help="Output pure JSON format")

    args = parser.parse_args()
    run_prediction(
        image_path=args.image_path,
        checkpoint=args.checkpoint,
        device=args.device,
        output_json=args.json
    )


if __name__ == "__main__":
    main()
