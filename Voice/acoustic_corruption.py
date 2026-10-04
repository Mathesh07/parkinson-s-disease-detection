"""Acoustic Corruption and Robustness Module for Voice Wav2Vec2 Pipeline.

Implements Additive Gaussian Noise corruption at exact SNR levels (Clean, 20dB, 10dB, 5dB, 0dB)
applied directly to audio waveforms prior to feature extraction and model inference.

Leakage Guards:
    - Pure inference mode only under torch.no_grad().
    - Zero model retraining, fine-tuning, or weight modification.
    - Zero threshold tuning or calibration using corrupted test data.
    - Test set labels used strictly for retrospective performance metrics.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch

# Ensure workspace root and Voice directory are in sys.path
root_dir = Path(__file__).resolve().parent.parent
voice_dir = root_dir / "Voice"
for p in (str(root_dir), str(voice_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from Voice import config


def add_additive_gaussian_noise(
    audio: np.ndarray,
    snr_db: Optional[float],
    seed: int = 42
) -> Tuple[np.ndarray, float]:
    """
    Add Additive Gaussian Noise to 1D audio waveform signal to achieve exact target SNR (dB).
    
    Args:
        audio: 1D float32 audio waveform numpy array.
        snr_db: Target Signal-to-Noise Ratio in dB. None or inf returns original clean audio.
        seed: Random seed for noise generation reproducibility.
        
    Returns:
        Tuple of (corrupted_audio, achieved_snr_db).
    """
    audio = np.asarray(audio, dtype=np.float32)
    
    if snr_db is None or np.isinf(snr_db):
        return audio.copy(), float("inf")

    rng = np.random.RandomState(seed)
    signal_power = float(np.mean(audio ** 2))
    if signal_power <= 1e-12:
        signal_power = 1e-6

    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise_std = np.sqrt(noise_power)
    noise = rng.normal(loc=0.0, scale=noise_std, size=audio.shape).astype(np.float32)

    corrupted = audio + noise
    
    # Audio safety check: document clipping to [-1.0, 1.0] if range exceeds unit amplitude
    corrupted_clipped = np.clip(corrupted, -1.0, 1.0).astype(np.float32)

    # Verify no NaN or Inf values
    assert not np.isnan(corrupted_clipped).any(), "NaN detected in corrupted audio waveform!"
    assert not np.isinf(corrupted_clipped).any(), "Inf detected in corrupted audio waveform!"

    actual_noise = corrupted_clipped - audio
    actual_noise_power = float(np.mean(actual_noise ** 2))
    achieved_snr = float(10.0 * np.log10(signal_power / max(1e-12, actual_noise_power)))

    return corrupted_clipped, achieved_snr


def compute_spearman_correlation(snr_numeric: List[float], uncertainty_values: List[float]) -> Tuple[float, float]:
    """
    Compute Spearman rank correlation between numeric SNR levels and uncertainty values.
    
    Note: SNR level 100 is used to represent Clean (inf dB) for rank calculation.
    """
    snrs = np.array(snr_numeric, dtype=float)
    uncs = np.array(uncertainty_values, dtype=float)

    if len(snrs) < 3 or np.std(uncs) <= 1e-12:
        return 0.0, 1.0

    corr, p_val = stats.spearmanr(snrs, uncs)
    if np.isnan(corr):
        corr = 0.0
        p_val = 1.0

    return float(corr), float(p_val)
