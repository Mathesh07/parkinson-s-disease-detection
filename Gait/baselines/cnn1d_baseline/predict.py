"""Inference CLI script for Gait 1D CNN Baseline."""

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

try:
    from . import config
    from .model import Gait1DCNN
except ImportError:
    import config
    from model import Gait1DCNN

if str(config.GAIT_DIR) not in sys.path:
    sys.path.insert(0, str(config.GAIT_DIR))

from dataset import segment_recording_into_windows
from preprocessing import GaitNormalizer, load_raw_recording


def predict_recording(filepath: str, checkpoint_path: Path = config.BEST_MODEL_PATH) -> Dict:
    """Run inference on a single raw Gait recording (.txt)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalizer = GaitNormalizer.load(config.NORMALIZATION_STATS_PATH)
    raw_data = load_raw_recording(filepath)
    norm_data = normalizer.transform(raw_data)
    windows = segment_recording_into_windows(norm_data, config.WINDOW_SIZE, config.STEP_SIZE)

    model = Gait1DCNN().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    window_tensors = torch.tensor(np.stack(windows), dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(window_tensors)
        probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()

    mean_prob = float(np.mean(probs))
    pred_label = int(mean_prob >= 0.5)

    p_clamp = np.clip(mean_prob, 1e-6, 1.0 - 1e-6)
    entropy = float(-(p_clamp * np.log2(p_clamp) + (1.0 - p_clamp) * np.log2(1.0 - p_clamp)))

    return {
        "filepath": filepath,
        "num_windows": len(windows),
        "pd_probability": mean_prob,
        "hc_probability": 1.0 - mean_prob,
        "predicted_label": pred_label,
        "predicted_class": config.ID2LABEL[pred_label],
        "predictive_entropy": entropy
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gait 1D CNN inference on a recording.")
    parser.add_argument("--filepath", type=str, required=True, help="Path to raw VGRF .txt file")
    args = parser.parse_args()

    result = predict_recording(args.filepath)
    print("\n--- Prediction Result ---")
    for k, v in result.items():
        print(f"  {k}: {v}")
