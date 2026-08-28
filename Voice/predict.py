"""Inference script for Voice-Based Parkinson's Disease classification on arbitrary audio."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import Wav2Vec2Processor

# Support running as a standalone script or as a module
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
for p in (str(current_dir), str(parent_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from Voice import config
    from Voice.dataset import chunk_audio, load_and_resample_audio
    from Voice.model import Wav2Vec2ForParkinsons
    from Voice.utils import get_device
except ImportError:
    import config
    from dataset import chunk_audio, load_and_resample_audio
    from model import Wav2Vec2ForParkinsons
    from utils import get_device


def predict_audio(
    audio_path: str,
    checkpoint_dir: Path = config.CHECKPOINT_DIR,
    device: torch.device = None
):
    """Run inference on a single audio recording and return probabilities."""
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    weights_file = checkpoint_dir / "best_model.pt"
    if not weights_file.exists():
        raise FileNotFoundError(f"Trained checkpoint not found at: {weights_file}. Please train first.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load processor and model
    processor = Wav2Vec2Processor.from_pretrained(str(checkpoint_dir))
    model = Wav2Vec2ForParkinsons(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        freeze_feature_encoder=config.FREEZE_FEATURE_ENCODER
    )
    model.load_state_dict(torch.load(weights_file, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Load and resample audio
    audio = load_and_resample_audio(str(audio_file), target_sr=config.SAMPLE_RATE)
    duration_sec = len(audio) / float(config.SAMPLE_RATE)

    # Slice into chunks
    chunks = chunk_audio(audio, chunk_samples=config.CHUNK_SAMPLES, step_samples=config.STEP_SAMPLES)

    pd_probs = []
    use_cuda = device.type == "cuda"

    with torch.no_grad():
        for chunk in chunks:
            processed = processor(chunk, sampling_rate=config.SAMPLE_RATE, return_tensors="pt")
            input_values = processed.input_values.to(device)

            if use_cuda:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(input_values)
            else:
                logits = model(input_values)

            prob = torch.softmax(logits.float(), dim=-1)
            pd_probs.append(prob[0, 1].item())

    mean_pd_prob = float(np.mean(pd_probs))
    mean_hc_prob = 1.0 - mean_pd_prob
    pred_label = 1 if mean_pd_prob >= 0.5 else 0
    pred_name = "Parkinson's Disease" if pred_label == 1 else "Healthy Control"
    confidence = mean_pd_prob if pred_label == 1 else mean_hc_prob

    print("\n" + "=" * 45)
    print("WAV2VEC2 PARKINSON'S VOICE PREDICTION")
    print("=" * 45)
    print(f"Audio:          {audio_file.name}")
    print(f"Duration:       {duration_sec:.2f}s ({len(chunks)} chunks)")
    print(f"Prediction:     {pred_name}")
    print(f"PD Probability: {mean_pd_prob:.4f}")
    print(f"HC Probability: {mean_hc_prob:.4f}")
    print(f"Confidence:     {confidence * 100:.2f}%")
    print("=" * 45 + "\n")

    return {
        "filename": audio_file.name,
        "prediction": pred_name,
        "predicted_label": pred_label,
        "pd_probability": mean_pd_prob,
        "hc_probability": mean_hc_prob,
        "confidence": confidence,
        "num_chunks": len(chunks),
        "duration_sec": duration_sec
    }


def main():
    parser = argparse.ArgumentParser(description="Classify Parkinson's Disease from voice recording using Wav2Vec2.")
    parser.add_argument("--audio", type=str, required=True, help="Path to input .wav audio file")
    parser.add_argument("--checkpoint", type=str, default=str(config.CHECKPOINT_DIR), help="Path to checkpoint folder")
    args = parser.parse_args()

    predict_audio(args.audio, Path(args.checkpoint))


if __name__ == "__main__":
    main()
