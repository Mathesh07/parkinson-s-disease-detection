"""Audio preprocessing, STFT computation, and speech transition extraction for Voice Base-Paper CNN.

Implements Short-Time Fourier Transform (STFT) time-frequency representations and
voiced/unvoiced phonation transition modeling (Onset and Offset) inspired by
Vásquez-Correa et al. (2019).
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import torch

try:
    from . import config
except ImportError:
    import config


def load_and_resample_audio(
    filepath: Union[str, Path],
    target_sr: int = config.SAMPLE_RATE
) -> np.ndarray:
    """Load WAV file, convert to mono float32, and resample to target_sr."""
    filepath = str(filepath)
    sr, audio = wavfile.read(filepath)

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != target_sr:
        gcd = math.gcd(target_sr, sr)
        up = target_sr // gcd
        down = sr // gcd
        audio = signal.resample_poly(audio, up, down).astype(np.float32)

    return audio


def compute_stft_spectrogram(
    audio_chunk: np.ndarray,
    n_fft: int = config.N_FFT,
    hop_length: int = config.HOP_LENGTH,
    win_length: int = config.WIN_LENGTH
) -> torch.Tensor:
    """
    Compute log-magnitude STFT spectrogram for a continuous audio chunk.

    Args:
        audio_chunk: 1D np.ndarray of shape (N,) float32.
        n_fft: FFT window size (default: 512).
        hop_length: Frame shift (default: 160 = 10 ms).
        win_length: Analysis window (default: 400 = 25 ms).

    Returns:
        torch.Tensor of shape (1, freq_bins=257, time_frames).
    """
    x = torch.from_numpy(audio_chunk).float()
    window = torch.hann_window(win_length)

    stft = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True
    )  # (freq_bins=257, time_frames)

    magnitude = torch.abs(stft)
    log_spec = torch.log1p(magnitude).unsqueeze(0)  # (1, 257, time_frames)
    return log_spec


def detect_speech_transitions(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    transition_ms: float = 80.0
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Detect voiced/unvoiced boundaries and extract Onset & Offset STFT transitions.

    - Onset: Boundary transitioning from unvoiced to voiced (40 ms unvoiced + 40 ms voiced).
    - Offset: Boundary transitioning from voiced to unvoiced (40 ms voiced + 40 ms unvoiced).

    Returns:
        (onset_spectrograms, offset_spectrograms) as lists of tensors of shape (1, 129, 33).
    """
    frame_len = int(sr * frame_ms / 1000.0)
    hop_len = int(sr * hop_ms / 1000.0)
    half_trans = int(sr * (transition_ms / 2.0) / 1000.0)  # 40 ms = 640 samples

    if len(audio) < frame_len:
        return [], []

    # Vectorized framing
    frames = np.lib.stride_tricks.sliding_window_view(audio, frame_len)[::hop_len]
    energies = np.mean(frames ** 2, axis=1)
    signs = np.sign(frames)
    zcrs = np.mean(np.abs(np.diff(signs, axis=1)), axis=1) / 2.0

    # Adaptive voicing thresholds
    e_thresh = float(np.percentile(energies, 40))
    z_thresh = float(np.percentile(zcrs, 60))

    is_voiced = (energies > e_thresh) & (zcrs < z_thresh)

    onset_specs = []
    offset_specs = []

    # Find transition points (max 50 per recording to maintain balance)
    on_transitions = np.where(~is_voiced[:-1] & is_voiced[1:])[0]
    off_transitions = np.where(is_voiced[:-1] & ~is_voiced[1:])[0]

    for idx in on_transitions[:50]:
        sample_idx = (idx + 1) * hop_len
        if sample_idx >= half_trans and (sample_idx + half_trans) <= len(audio):
            segment = audio[sample_idx - half_trans : sample_idx + half_trans]
            spec = compute_stft_spectrogram(
                segment,
                n_fft=config.TRANSITION_N_FFT,
                hop_length=config.TRANSITION_HOP_LENGTH,
                win_length=config.TRANSITION_WIN_LENGTH
            )
            onset_specs.append(spec)

    for idx in off_transitions[:50]:
        sample_idx = (idx + 1) * hop_len
        if sample_idx >= half_trans and (sample_idx + half_trans) <= len(audio):
            segment = audio[sample_idx - half_trans : sample_idx + half_trans]
            spec = compute_stft_spectrogram(
                segment,
                n_fft=config.TRANSITION_N_FFT,
                hop_length=config.TRANSITION_HOP_LENGTH,
                win_length=config.TRANSITION_WIN_LENGTH
            )
            offset_specs.append(spec)

    return onset_specs, offset_specs


class VoiceSTFTNormalizer:
    """Frequency-wise standard scaling fitted STRICTLY on training subjects."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        self.is_fit = False

    def fit(self, spectrogram_list: List[torch.Tensor]) -> "VoiceSTFTNormalizer":
        """Fit frequency-wise mean and std across training spectrograms."""
        if not spectrogram_list:
            raise ValueError("Cannot fit normalizer on empty list.")

        # Compute running mean and std across frequency bins
        stacked = torch.stack(spectrogram_list)  # (N, 1, freq_bins, time_frames)
        self.mean = torch.mean(stacked, dim=(0, 3), keepdim=True)  # (1, 1, freq_bins, 1)
        self.std = torch.std(stacked, dim=(0, 3), keepdim=True)
        self.std = torch.clamp(self.std, min=self.eps)
        self.is_fit = True
        return self

    def transform(self, spec: torch.Tensor) -> torch.Tensor:
        """Normalize a single spectrogram (1, freq_bins, time_frames)."""
        if not self.is_fit:
            raise RuntimeError("Normalizer must be fit before transform.")
        return (spec - self.mean.squeeze(0)) / self.std.squeeze(0)

    def save(self, filepath: Union[str, Path] = config.NORMALIZATION_STATS_PATH) -> None:
        """Save fitted statistics to JSON."""
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
    def load(cls, filepath: Union[str, Path] = config.NORMALIZATION_STATS_PATH) -> "VoiceSTFTNormalizer":
        """Load fitted statistics from JSON."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Stats file not found: {filepath}")
        with open(filepath, "r") as f:
            stats = json.load(f)
        norm = cls(eps=stats.get("eps", 1e-6))
        norm.mean = torch.tensor(stats["mean"]).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        norm.std = torch.tensor(stats["std"]).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        norm.is_fit = True
        return norm
