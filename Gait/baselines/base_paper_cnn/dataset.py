"""PyTorch Dataset, subject-level zero-leakage splits, and STFT windowing for Gait Base-Paper-Style CNN.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from . import config
    from .preprocessing import (
        GaitSTFTNormalizer,
        build_manifest,
        compute_stft_tensor,
        load_raw_recording,
    )
except ImportError:
    import config
    from preprocessing import (
        GaitSTFTNormalizer,
        build_manifest,
        compute_stft_tensor,
        load_raw_recording,
    )

# In-memory cache for raw recordings to eliminate redundant disk I/O
_RECORDING_CACHE: Dict[str, np.ndarray] = {}


def get_cached_recording(filepath: str) -> np.ndarray:
    """Load recording from cache or read from disk once."""
    if filepath not in _RECORDING_CACHE:
        _RECORDING_CACHE[filepath] = load_raw_recording(filepath)
    return _RECORDING_CACHE[filepath]


def create_subject_splits(
    manifest_df: pd.DataFrame,
    train_ratio: float = config.TRAIN_SPLIT_RATIO,
    val_ratio: float = config.VAL_SPLIT_RATIO,
    test_ratio: float = config.TEST_SPLIT_RATIO,
    seed: int = config.RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform a strict, stratified subject-level split to eliminate patient data leakage.
    Guarantees that test subjects are 100% identical to the existing Gait project model.
    """
    rng = np.random.RandomState(seed)

    hc_subjects = sorted(manifest_df[manifest_df["label"] == 0]["subject_id"].unique())
    pd_subjects = sorted(manifest_df[manifest_df["label"] == 1]["subject_id"].unique())

    hc_shuffled = list(rng.permutation(hc_subjects))
    pd_shuffled = list(rng.permutation(pd_subjects))

    def compute_counts(total: int) -> Tuple[int, int, int]:
        n_train = int(round(total * train_ratio))
        n_val = int(round(total * val_ratio))
        n_test = total - n_train - n_val
        if n_val == 0:
            n_val = 1
            n_train -= 1
        if n_test == 0:
            n_test = 1
            n_train -= 1
        return n_train, n_val, n_test

    hc_n_train, hc_n_val, hc_n_test = compute_counts(len(hc_shuffled))
    pd_n_train, pd_n_val, pd_n_test = compute_counts(len(pd_shuffled))

    train_hc = hc_shuffled[:hc_n_train]
    val_hc = hc_shuffled[hc_n_train : hc_n_train + hc_n_val]
    test_hc = hc_shuffled[hc_n_train + hc_n_val:]

    train_pd = pd_shuffled[:pd_n_train]
    val_pd = pd_shuffled[pd_n_train : pd_n_train + pd_n_val]
    test_pd = pd_shuffled[pd_n_train + pd_n_val:]

    train_subjects = set(train_hc + train_pd)
    val_subjects = set(val_hc + val_pd)
    test_subjects = set(test_hc + test_pd)

    # Rigorous Zero-Leakage Asserts
    assert len(train_subjects & val_subjects) == 0, "Subject leakage between Train and Val sets!"
    assert len(train_subjects & test_subjects) == 0, "Subject leakage between Train and Test sets!"
    assert len(val_subjects & test_subjects) == 0, "Subject leakage between Val and Test sets!"

    train_df = manifest_df[manifest_df["subject_id"].isin(train_subjects)].copy().reset_index(drop=True)
    val_df = manifest_df[manifest_df["subject_id"].isin(val_subjects)].copy().reset_index(drop=True)
    test_df = manifest_df[manifest_df["subject_id"].isin(test_subjects)].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def segment_recording_into_windows(
    data: np.ndarray,
    window_size: int = config.WINDOW_SIZE,
    step_size: int = config.STEP_SIZE
) -> List[np.ndarray]:
    """Segment a continuous recording (time_steps, 16) into overlapping fixed-size windows."""
    total_time_steps = len(data)
    if total_time_steps < window_size:
        padded = np.zeros((window_size, data.shape[1]), dtype=np.float32)
        padded[:total_time_steps, :] = data
        return [padded]

    windows = []
    start = 0
    while start + window_size <= total_time_steps:
        windows.append(data[start : start + window_size])
        start += step_size

    if start < total_time_steps and (total_time_steps - start) >= (window_size // 2):
        windows.append(data[-window_size:])

    if not windows:
        padded = np.zeros((window_size, data.shape[1]), dtype=np.float32)
        padded[:total_time_steps, :] = data
        windows.append(padded)

    return windows


class GaitSTFTDataset(Dataset):
    """
    PyTorch Dataset providing STFT time-frequency spectrograms with subject/recording metadata.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        normalizer: Optional[GaitSTFTNormalizer] = None,
        window_size: int = config.WINDOW_SIZE,
        step_size: int = config.STEP_SIZE
    ):
        self.manifest_df = manifest_df.reset_index(drop=True)
        self.normalizer = normalizer
        self.window_size = window_size
        self.step_size = step_size
        self.samples: List[Dict] = []
        self._build_dataset()

    def _build_dataset(self):
        for _, row in self.manifest_df.iterrows():
            raw_data = get_cached_recording(row["filepath"])
            windows = segment_recording_into_windows(raw_data, self.window_size, self.step_size)
            num_windows = len(windows)

            for win_idx, win in enumerate(windows):
                if win_idx == 0:
                    win_type = "onset"
                elif win_idx == num_windows - 1:
                    win_type = "offset"
                else:
                    win_type = "steady_state"

                stft_tensor = compute_stft_tensor(win)

                if self.normalizer is not None:
                    stft_tensor = self.normalizer.transform(stft_tensor)

                self.samples.append({
                    "features": stft_tensor,
                    "label": row["label"],
                    "label_name": row["label_name"],
                    "subject_id": row["subject_id"],
                    "recording_id": row["filename"],
                    "study": row["study"],
                    "window_idx": win_idx,
                    "num_windows": num_windows,
                    "window_type": win_type
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        item = self.samples[idx]
        return {
            "features": item["features"].clone().detach(),
            "label": torch.tensor(item["label"], dtype=torch.float32),
            "subject_id": item["subject_id"],
            "recording_id": item["recording_id"],
            "study": item["study"],
            "window_idx": item["window_idx"],
            "num_windows": item["num_windows"],
            "window_type": item["window_type"]
        }


def collate_gait_stft_batch(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """Collate STFT spectrogram batches."""
    features = torch.stack([item["features"] for item in batch])  # (B, 16, 33, 32)
    labels = torch.stack([item["label"] for item in batch])        # (B,)

    return {
        "features": features,
        "labels": labels,
        "subject_ids": [item["subject_id"] for item in batch],
        "recording_ids": [item["recording_id"] for item in batch],
        "studies": [item["study"] for item in batch],
        "window_indices": [item["window_idx"] for item in batch],
        "window_types": [item["window_type"] for item in batch]
    }


def create_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    seed: int = config.RANDOM_SEED
) -> Tuple[DataLoader, DataLoader, DataLoader, GaitSTFTNormalizer, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build splits, fit normalizer, and return DataLoaders."""
    manifest_df = build_manifest()
    train_df, val_df, test_df = create_subject_splits(manifest_df, seed=seed)

    # 1. Build unnormalized train dataset
    train_dataset = GaitSTFTDataset(train_df, normalizer=None)

    # 2. Fit normalizer on train samples directly
    normalizer = GaitSTFTNormalizer()
    normalizer.fit([s["features"] for s in train_dataset.samples])
    normalizer.save(config.NORMALIZATION_STATS_PATH)

    # Apply fitted normalization to training samples in-place
    for s in train_dataset.samples:
        s["features"] = normalizer.transform(s["features"])
    train_dataset.normalizer = normalizer

    # 3. Build val and test datasets with normalizer
    val_dataset = GaitSTFTDataset(val_df, normalizer=normalizer)
    test_dataset = GaitSTFTDataset(test_df, normalizer=normalizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_gait_stft_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_gait_stft_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_gait_stft_batch)

    return train_loader, val_loader, test_loader, normalizer, train_df, val_df, test_df
