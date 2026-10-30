"""Inference / Predictor interface for Voice Base-Paper 2D CNN Baseline.
"""

from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch

try:
    from . import config
    from .dataset import chunk_audio
    from .model import VoiceBasePaper2DCNN
    from .preprocessing import (
        VoiceSTFTNormalizer,
        compute_stft_spectrogram,
        load_and_resample_audio,
    )
except ImportError:
    import config
    from dataset import chunk_audio
    from model import VoiceBasePaper2DCNN
    from preprocessing import (
        VoiceSTFTNormalizer,
        compute_stft_spectrogram,
        load_and_resample_audio,
    )


class VoiceBasePaperPredictor:
    """Convenient predictor for Voice Base-Paper 2D CNN."""

    def __init__(
        self,
        checkpoint_path: Path = config.BEST_MODEL_PATH,
        normalizer_path: Path = config.NORMALIZATION_STATS_PATH
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.normalizer = VoiceSTFTNormalizer.load(normalizer_path)

        self.model = VoiceBasePaper2DCNN().to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

    def predict_audio(self, audio_input: Union[str, Path, np.ndarray]) -> Dict:
        """Predict Parkinson's Disease status from an audio recording or waveform."""
        if isinstance(audio_input, (str, Path)):
            audio = load_and_resample_audio(audio_input, config.SAMPLE_RATE)
        else:
            audio = np.asarray(audio_input, dtype=np.float32)

        chunks = chunk_audio(audio, config.CHUNK_SAMPLES, config.STEP_SAMPLES)
        chunk_probs = []

        with torch.no_grad():
            for c in chunks:
                spec = compute_stft_spectrogram(c)
                norm_spec = self.normalizer.transform(spec).unsqueeze(0).to(self.device)
                logits = self.model(norm_spec)
                prob = torch.softmax(logits, dim=-1)[0, 1].item()
                chunk_probs.append(float(prob))

        mean_p = float(np.mean(chunk_probs))
        pred_label = int(mean_p >= 0.5)

        return {
            "pd_probability": mean_p,
            "hc_probability": 1.0 - mean_p,
            "predicted_label": pred_label,
            "predicted_name": config.ID2LABEL[pred_label],
            "num_chunks": len(chunk_probs),
            "chunk_probabilities": chunk_probs
        }
