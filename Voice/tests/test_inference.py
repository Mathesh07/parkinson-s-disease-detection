"""Tests for Voice prediction and inference on arbitrary audio recordings."""

import sys
import wave
from pathlib import Path
import numpy as np
import pytest
import torch

# Ensure workspace root and Voice directory are in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
voice_dir = root_dir / "Voice"
for p in (str(root_dir), str(voice_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from Voice import config
from Voice.predict import predict_audio


def create_synthetic_wav(filepath: Path, duration_sec: float = 12.0, sr: int = 16000):
    """Generate a clean synthetic WAV audio file for testing."""
    num_samples = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    signal = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(signal.tobytes())


def test_predict_audio_raises_file_not_found():
    """Verify predict_audio raises FileNotFoundError when input audio file does not exist."""
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        predict_audio("non_existent_audio_file.wav")


def test_predict_audio_inference_pipeline(tmp_path, monkeypatch):
    """
    Test predict_audio pipeline:
      - Loads audio and chunks into 10s segments
      - Runs model forward pass
      - Computes mean PD probability, prediction label, and confidence
      - Performs zero model parameter updates (inference mode)
    """
    audio_path = tmp_path / "sample_audio.wav"
    create_synthetic_wav(audio_path, duration_sec=12.0, sr=16000)

    checkpoint_dir = config.CHECKPOINT_DIR
    weights_path = checkpoint_dir / "best_model.pt"

    # If trained weight binary is absent, mock weights and processor loading
    if not weights_path.exists():
        from Voice.model import Wav2Vec2ForParkinsons
        from transformers import Wav2Vec2Processor

        # Save dummy model state_dict and processor to temporary checkpoint dir
        mock_ckpt = tmp_path / "mock_checkpoint"
        mock_ckpt.mkdir()

        model = Wav2Vec2ForParkinsons(model_name=config.MODEL_NAME, num_classes=2)
        torch.save(model.state_dict(), mock_ckpt / "best_model.pt")

        processor = Wav2Vec2Processor.from_pretrained(config.MODEL_NAME)
        processor.save_pretrained(mock_ckpt)

        checkpoint_dir = mock_ckpt

    result = predict_audio(
        audio_path=str(audio_path),
        checkpoint_dir=checkpoint_dir,
        device=torch.device("cpu")
    )

    assert isinstance(result, dict)
    assert result["filename"] == "sample_audio.wav"
    assert result["predicted_label"] in (0, 1)
    assert result["prediction"] in ("Parkinson's Disease", "Healthy Control")
    assert 0.0 <= result["pd_probability"] <= 1.0
    assert 0.0 <= result["hc_probability"] <= 1.0
    assert np.isclose(result["pd_probability"] + result["hc_probability"], 1.0)
    assert result["num_chunks"] >= 1
    assert result["duration_sec"] == 12.0
