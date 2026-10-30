"""Inference CLI script for Voice 1D CNN Raw Waveform Baseline."""

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Voice.baselines.cnn1d_baseline import config
from Voice.baselines.cnn1d_baseline.dataset import VoiceWaveformNormalizer
from Voice.baselines.cnn1d_baseline.model import Voice1DCNN
from Voice.dataset import load_and_resample_audio


def predict_voice(audio_path: str, checkpoint_path: Path = config.BEST_MODEL_PATH) -> Dict:
    """Run inference on a single audio recording (.wav)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio = load_and_resample_audio(audio_path, target_sr=config.SAMPLE_RATE)
    normalizer = VoiceWaveformNormalizer.load(config.NORMALIZATION_STATS_PATH)

    total_samples = len(audio)
    if total_samples < config.CHUNK_SAMPLES:
        padded = np.zeros(config.CHUNK_SAMPLES, dtype=np.float32)
        padded[:total_samples] = audio
        chunks = [padded]
    else:
        chunks = []
        start = 0
        while start + config.CHUNK_SAMPLES <= total_samples:
            chunks.append(audio[start : start + config.CHUNK_SAMPLES])
            start += config.STEP_SAMPLES

    model = Voice1DCNN().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    chunk_tensors = [normalizer.transform(torch.tensor(ch, dtype=torch.float32).unsqueeze(0)) for ch in chunks]
    batch_tensors = torch.stack(chunk_tensors).to(device)

    with torch.no_grad():
        logits = model(batch_tensors)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    mean_prob_pd = float(np.mean(probs[:, 1]))
    pred_label = int(mean_prob_pd >= 0.5)

    p_clamp = np.clip(mean_prob_pd, 1e-6, 1.0 - 1e-6)
    entropy = float(-(p_clamp * np.log2(p_clamp) + (1.0 - p_clamp) * np.log2(1.0 - p_clamp)))

    return {
        "audio_path": audio_path,
        "num_chunks": len(chunks),
        "pd_probability": mean_prob_pd,
        "hc_probability": 1.0 - mean_prob_pd,
        "predicted_label": pred_label,
        "predicted_class": config.ID2LABEL[pred_label],
        "predictive_entropy": entropy
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Voice 1D CNN inference on a WAV file.")
    parser.add_argument("--audio", type=str, required=True, help="Path to raw audio WAV file")
    args = parser.parse_args()

    res = predict_voice(args.audio)
    print("\n--- Voice 1D CNN Prediction ---")
    for k, v in res.items():
        print(f"  {k}: {v}")
