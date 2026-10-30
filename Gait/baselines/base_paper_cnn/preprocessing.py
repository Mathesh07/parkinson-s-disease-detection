"""Preprocessing and STFT representation module for Gait Base-Paper-Style 2D CNN.

Converts 16-channel raw VGRF continuous force signals into time-frequency
spectrogram tensors suitable for 2D Convolutional Neural Networks, inspired by
Vásquez-Correa et al. (2019).
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

try:
    from . import config
except ImportError:
    import config


def parse_filename(filename: str) -> Dict[str, Union[str, int]]:
    """Parse PhysioNet Gait filename convention."""
    basename = Path(filename).name
    match = re.match(r"^([A-Za-z]{2})([A-Za-z]{2})(\d{2})_(\d{2})\.txt$", basename)
    if not match:
        match = re.match(r"^([A-Za-z]{2})([A-Za-z]{2})(\d+)_(\d+)\.txt$", basename)
        if not match:
            raise ValueError(f"Unrecognized gait filename: {basename}")

    study = match.group(1).capitalize()
    group = match.group(2).capitalize()
    subj_num = match.group(3)
    walk_id = match.group(4)
    subject_id = f"{study}{group}{subj_num}"

    label = config.LABEL_MAP.get(group, None)
    if label is None:
        raise ValueError(f"Unknown group code '{group}' in {basename}")

    return {
        "filename": basename,
        "study": study,
        "group": group,
        "subject_id": subject_id,
        "walk_id": walk_id,
        "label": label,
        "label_name": config.ID2LABEL[label]
    }


def load_raw_recording(
    filepath: Union[str, Path],
    num_sensor_channels: int = config.NUM_CHANNELS
) -> np.ndarray:
    """Load a raw .txt gait recording and extract 16 VGRF channels."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Recording file not found: {filepath}")

    try:
        df = pd.read_csv(
            filepath,
            sep=r"\s+",
            header=None,
            usecols=list(range(1, num_sensor_channels + 1)),
            dtype=np.float32,
            engine="c"
        )
        arr = df.values
    except Exception:
        data = []
        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) >= 17:
                    data.append([float(x) for x in parts[1:17]])
                elif len(parts) >= 16:
                    data.append([float(x) for x in parts[:16]])
        arr = np.array(data, dtype=np.float32)

    if np.isnan(arr).any() or np.isinf(arr).any():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    return arr


def build_manifest(data_dir: Union[str, Path] = config.DATA_DIR) -> pd.DataFrame:
    """Build structured metadata manifest for all gait recordings."""
    data_dir = Path(data_dir)
    txt_files = sorted([
        f for f in data_dir.glob("*.txt")
        if not f.name.endswith(("format.txt", "demographics.txt", "SHA256SUMS.txt"))
    ])

    records = []
    for f in txt_files:
        info = parse_filename(f.name)
        info["filepath"] = str(f.resolve())
        records.append(info)

    return pd.DataFrame(records)


def compute_stft_tensor(
    window: np.ndarray,
    n_fft: int = config.N_FFT,
    hop_length: int = config.HOP_LENGTH,
    win_length: int = config.WIN_LENGTH
) -> torch.Tensor:
    """
    Transform a 16-channel time-series window (500, 16) into an STFT log-magnitude spectrogram.

    Args:
        window: np.ndarray of shape (time_steps=500, channels=16).
        n_fft: FFT window size (default: 64).
        hop_length: STFT hop size (default: 16).
        win_length: STFT window length (default: 64).

    Returns:
        torch.Tensor of shape (16, freq_bins=33, time_frames=32), dtype float32.
    """
    # Transpose to (channels=16, time_steps=500)
    x = torch.from_numpy(window.T).float()  # (16, 500)
    hann = torch.hann_window(win_length)

    # Compute complex STFT across all 16 channels
    # center=True ensures boundary padding to yield 32 time frames
    stft = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=hann,
        center=True,
        return_complex=True
    )  # (16, freq_bins=33, time_frames=32)

    # Compute log-magnitude spectrogram: log(1 + |STFT|)
    magnitude = torch.abs(stft)
    log_spec = torch.log1p(magnitude)

    return log_spec


class GaitSTFTNormalizer:
    """
    Channel-wise and frequency-wise standard scaling fitted STRICTLY on training data.
    Ensures zero subject leakage into validation or testing.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        self.is_fit = False

    def fit(self, stft_tensors: List[torch.Tensor]) -> "GaitSTFTNormalizer":
        """
        Fit mean and standard deviation per channel and frequency bin across training samples.
        Each tensor has shape (16, freq_bins, time_frames).
        """
        if not stft_tensors:
            raise ValueError("Cannot fit on empty tensor list.")

        # Stack over batch dimension: (N, 16, freq_bins, time_frames)
        stacked = torch.stack(stft_tensors)
        # Compute mean and std over batch (0) and time (3) dimensions: shape (16, freq_bins, 1)
        self.mean = torch.mean(stacked, dim=(0, 3), keepdim=True)
        self.std = torch.std(stacked, dim=(0, 3), keepdim=True)
        self.std = torch.clamp(self.std, min=self.eps)
        self.is_fit = True
        return self

    def transform(self, stft_tensor: torch.Tensor) -> torch.Tensor:
        """Normalize a single tensor of shape (16, freq_bins, time_frames)."""
        if not self.is_fit:
            raise RuntimeError("Normalizer must be fit before transform.")
        return (stft_tensor - self.mean.squeeze(0)) / self.std.squeeze(0)

    def transform_batch(self, stft_batch: torch.Tensor) -> torch.Tensor:
        """Normalize a batch of shape (B, 16, freq_bins, time_frames)."""
        if not self.is_fit:
            raise RuntimeError("Normalizer must be fit before transform.")
        return (stft_batch - self.mean) / self.std

    def save(self, filepath: Union[str, Path] = config.NORMALIZATION_STATS_PATH) -> None:
        """Save fitted normalizer statistics to JSON."""
        if not self.is_fit:
            raise RuntimeError("Cannot save unfitted normalizer.")
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        stats = {
            "mean": self.mean.squeeze().tolist(),
            "std": self.std.squeeze().tolist(),
            "eps": self.eps
        }
        with open(filepath, "w") as f:
            json.dump(stats, f, indent=4)

    @classmethod
    def load(cls, filepath: Union[str, Path] = config.NORMALIZATION_STATS_PATH) -> "GaitSTFTNormalizer":
        """Load fitted normalizer statistics from JSON."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Stats file not found: {filepath}")
        with open(filepath, "r") as f:
            stats = json.load(f)
        norm = cls(eps=stats.get("eps", 1e-6))
        norm.mean = torch.tensor(stats["mean"]).unsqueeze(0).unsqueeze(-1)
        norm.std = torch.tensor(stats["std"]).unsqueeze(0).unsqueeze(-1)
        norm.is_fit = True
        return norm
