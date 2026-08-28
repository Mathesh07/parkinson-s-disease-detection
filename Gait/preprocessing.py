"""Preprocessing module for Gait VGRF recordings with zero-leakage normalization."""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Support running as a standalone script or as a module
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
for p in (str(current_dir), str(parent_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from Gait import config
except ImportError:
    import config


def parse_filename(filename: str) -> Dict[str, Union[str, int]]:
    """
    Parse standard PhysioNet Gait filename convention.
    Examples:
        'GaCo01_01.txt' -> study='Ga', group='Co', subj_num='01', subject_id='GaCo01', walk_id='01', label=0
        'JuPt11_03.txt' -> study='Ju', group='Pt', subj_num='11', subject_id='JuPt11', walk_id='03', label=1
    """
    basename = Path(filename).name
    # Match pattern: 2-letter study (Ga/Ju/Si) + 2-letter group (Co/Pt) + 2-digit subj + '_' + 2-digit walk + '.txt'
    match = re.match(r"^([A-Za-z]{2})([A-Za-z]{2})(\d{2})_(\d{2})\.txt$", basename)
    if not match:
        # Fallback regex for variations
        match = re.match(r"^([A-Za-z]{2})([A-Za-z]{2})(\d+)_(\d+)\.txt$", basename)
        if not match:
            raise ValueError(f"Unrecognized gait filename pattern: {basename}")

    study = match.group(1).capitalize()
    group = match.group(2).capitalize()
    subj_num = match.group(3)
    walk_id = match.group(4)
    subject_id = f"{study}{group}{subj_num}"

    label = config.LABEL_MAP.get(group, None)
    if label is None:
        raise ValueError(f"Unknown group code '{group}' in filename {basename}")

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
    """
    Load a raw .txt gait recording and extract the 16 VGRF sensor channels.
    
    19 raw columns:
        Col 0: Time
        Col 1-8: Left foot sensors (L1..L8)
        Col 9-16: Right foot sensors (R1..R8)
        Col 17: Total Left VGRF (ignored)
        Col 18: Total Right VGRF (ignored)
        
    Returns:
        np.ndarray of shape (time_steps, 16) with dtype float32.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Recording file not found: {filepath}")

    try:
        # Fast C-parser loading columns 1..16 (0-indexed)
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
        # Robust fallback
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

    # Clean NaNs and Infs safely
    if np.isnan(arr).any() or np.isinf(arr).any():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure shape is (N, 16)
    if arr.ndim != 2 or arr.shape[1] != num_sensor_channels:
        raise ValueError(f"Expected array shape (N, {num_sensor_channels}), got {arr.shape} for {filepath.name}")

    return arr


class GaitNormalizer:
    """
    Channel-wise Z-Score Normalizer fit strictly on training subject data to prevent leakage.
    Formula: x_norm = (x - mean) / (std + eps)
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.is_fit = False

    def fit(self, training_recordings: List[np.ndarray]) -> "GaitNormalizer":
        """Fit channel-wise mean and standard deviation from training recordings."""
        if not training_recordings:
            raise ValueError("Cannot fit normalizer on empty list of recordings.")

        # Concatenate along time dimension: (total_time_steps, 16)
        all_data = np.concatenate(training_recordings, axis=0)
        self.mean = np.mean(all_data, axis=0).astype(np.float32)
        self.std = np.std(all_data, axis=0).astype(np.float32)
        self.std[self.std < self.eps] = 1.0  # Prevent zero division
        self.is_fit = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply fitted normalization to a recording or window array."""
        if not self.is_fit:
            raise RuntimeError("GaitNormalizer must be fit before calling transform.")
        return (data - self.mean) / (self.std + self.eps)

    def fit_transform(self, training_recordings: List[np.ndarray]) -> List[np.ndarray]:
        """Fit on training data and transform it."""
        self.fit(training_recordings)
        return [self.transform(r) for r in training_recordings]

    def save(self, filepath: Union[str, Path] = config.NORMALIZATION_STATS_PATH) -> None:
        """Save fitted normalization parameters to JSON."""
        if not self.is_fit:
            raise RuntimeError("Cannot save unfitted GaitNormalizer.")
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        stats = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "eps": self.eps,
            "num_channels": int(len(self.mean))
        }
        with open(filepath, "w") as f:
            json.dump(stats, f, indent=4)

    @classmethod
    def load(cls, filepath: Union[str, Path] = config.NORMALIZATION_STATS_PATH) -> "GaitNormalizer":
        """Load fitted normalization parameters from JSON."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Normalization stats file not found at: {filepath}")
        with open(filepath, "r") as f:
            stats = json.load(f)
        normalizer = cls(eps=stats.get("eps", 1e-6))
        normalizer.mean = np.array(stats["mean"], dtype=np.float32)
        normalizer.std = np.array(stats["std"], dtype=np.float32)
        normalizer.is_fit = True
        return normalizer


def build_manifest(data_dir: Union[str, Path] = config.DATA_DIR) -> pd.DataFrame:
    """Inspect all gait recordings in data directory and build structured metadata manifest."""
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

    df = pd.DataFrame(records)
    return df
