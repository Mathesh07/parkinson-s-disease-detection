"""Inference / Prediction script for Gait Base-Paper 2D CNN Baseline.
"""

from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch

try:
    from . import config
    from .dataset import segment_recording_into_windows
    from .model import GaitBasePaper2DCNN
    from .preprocessing import (
        GaitSTFTNormalizer,
        compute_stft_tensor,
        load_raw_recording,
    )
except ImportError:
    import config
    from dataset import segment_recording_into_windows
    from model import GaitBasePaper2DCNN
    from preprocessing import (
        GaitSTFTNormalizer,
        compute_stft_tensor,
        load_raw_recording,
    )


class GaitBasePaperPredictor:
    """Convenient predictor interface for Gait Base-Paper 2D CNN."""

    def __init__(
        self,
        checkpoint_path: Path = config.BEST_MODEL_PATH,
        normalizer_path: Path = config.NORMALIZATION_STATS_PATH
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.normalizer = GaitSTFTNormalizer.load(normalizer_path)

        self.model = GaitBasePaper2DCNN(
            in_channels=config.NUM_CHANNELS,
            conv_channels=config.CONV_CHANNELS,
            dense_units=config.DENSE_UNITS,
            dropout_rate=config.DROPOUT_RATE
        ).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

    def predict_recording(self, recording_input: Union[str, Path, np.ndarray]) -> Dict:
        """
        Predict Parkinson's Disease status for a continuous gait trial.

        Args:
            recording_input: Filepath to .txt recording or (N, 16) array.

        Returns:
            Dict containing subject-level and window-level prediction details.
        """
        if isinstance(recording_input, (str, Path)):
            raw_data = load_raw_recording(recording_input)
        else:
            raw_data = np.asarray(recording_input, dtype=np.float32)

        windows = segment_recording_into_windows(raw_data)
        window_probs = []

        with torch.no_grad():
            for win in windows:
                stft_tensor = compute_stft_tensor(win)
                norm_tensor = self.normalizer.transform(stft_tensor).unsqueeze(0).to(self.device)
                logit = self.model(norm_tensor)
                prob = float(torch.sigmoid(logit).item())
                window_probs.append(prob)

        mean_prob = float(np.mean(window_probs))
        pred_label = int(mean_prob >= 0.5)

        return {
            "pd_probability": mean_prob,
            "hc_probability": 1.0 - mean_prob,
            "predicted_label": pred_label,
            "predicted_name": config.ID2LABEL[pred_label],
            "num_windows": len(window_probs),
            "window_probabilities": window_probs
        }
