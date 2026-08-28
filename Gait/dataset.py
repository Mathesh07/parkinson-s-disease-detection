"""PyTorch Dataset, subject-level split, and windowing pipeline for Gait VGRF signals."""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Support running as a standalone script or as a module
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
for p in (str(current_dir), str(parent_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from Gait import config
    from Gait.preprocessing import (
        GaitNormalizer,
        build_manifest,
        load_raw_recording,
    )
except ImportError:
    import config
    from preprocessing import (
        GaitNormalizer,
        build_manifest,
        load_raw_recording,
    )


def create_subject_splits(
    manifest_df: pd.DataFrame,
    train_ratio: float = config.TRAIN_SPLIT_RATIO,
    val_ratio: float = config.VAL_SPLIT_RATIO,
    test_ratio: float = config.TEST_SPLIT_RATIO,
    seed: int = config.RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform a strict, stratified subject-level split to eliminate patient data leakage.
    
    All recordings belonging to a subject are placed exclusively in one split (Train, Val, or Test).
    """
    rng = np.random.RandomState(seed)

    # Group subjects by class label
    hc_subjects = sorted(manifest_df[manifest_df["label"] == 0]["subject_id"].unique())
    pd_subjects = sorted(manifest_df[manifest_df["label"] == 1]["subject_id"].unique())

    # Shuffle subjects deterministically
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

    # Rigorous Zero-Leakage Checks
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
    """
    Segment a continuous 2D recording (time_steps, 16) into overlapping fixed-size windows (window_size, 16).
    """
    total_time_steps = len(data)
    if total_time_steps < window_size:
        # Pad short recordings to minimum window size
        padded = np.zeros((window_size, data.shape[1]), dtype=np.float32)
        padded[:total_time_steps, :] = data
        return [padded]

    windows = []
    start = 0
    while start + window_size <= total_time_steps:
        windows.append(data[start : start + window_size])
        start += step_size

    # Ensure tail is captured if sufficient samples remain
    if start < total_time_steps and (total_time_steps - start) >= (window_size // 2):
        windows.append(data[-window_size:])

    if not windows:
        padded = np.zeros((window_size, data.shape[1]), dtype=np.float32)
        padded[:total_time_steps, :] = data
        windows.append(padded)

    return windows


class GaitWindowDataset(Dataset):
    """
    PyTorch Dataset yielding normalized fixed-length windows with patient and recording metadata.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        normalizer: Optional[GaitNormalizer] = None,
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
            raw_data = load_raw_recording(row["filepath"])
            
            # Apply zero-leakage normalization if normalizer is provided
            norm_data = self.normalizer.transform(raw_data) if self.normalizer is not None else raw_data
            
            windows = segment_recording_into_windows(norm_data, self.window_size, self.step_size)
            num_windows = len(windows)

            for win_idx, win in enumerate(windows):
                self.samples.append({
                    "window": win,                      # np.ndarray shape (500, 16)
                    "label": row["label"],               # int (0 or 1)
                    "label_name": row["label_name"],     # str
                    "subject_id": row["subject_id"],     # str (e.g. 'GaCo01')
                    "recording_id": row["filename"],     # str (e.g. 'GaCo01_01.txt')
                    "study": row["study"],               # str
                    "window_idx": win_idx,               # int
                    "num_windows": num_windows           # int
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        item = self.samples[idx]
        return {
            "features": torch.tensor(item["window"], dtype=torch.float32),  # (500, 16)
            "label": torch.tensor(item["label"], dtype=torch.float32),       # float scalar for BCE
            "subject_id": item["subject_id"],
            "recording_id": item["recording_id"],
            "study": item["study"],
            "window_idx": item["window_idx"],
            "num_windows": item["num_windows"]
        }


def collate_gait_batch(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """Custom batch collator for Gait DataLoader."""
    features = torch.stack([item["features"] for item in batch])  # (B, 500, 16)
    labels = torch.stack([item["label"] for item in batch])        # (B,)

    return {
        "features": features,
        "labels": labels,
        "subject_ids": [item["subject_id"] for item in batch],
        "recording_ids": [item["recording_id"] for item in batch],
        "studies": [item["study"] for item in batch],
        "window_indices": [item["window_idx"] for item in batch]
    }


def create_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    window_size: int = config.WINDOW_SIZE,
    step_size: int = config.STEP_SIZE,
    seed: int = config.RANDOM_SEED
) -> Tuple[DataLoader, DataLoader, DataLoader, GaitNormalizer, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build train, val, test splits, fit normalizer on train subjects, and return PyTorch DataLoaders.
    """
    manifest_df = build_manifest()
    train_df, val_df, test_df = create_subject_splits(manifest_df, seed=seed)

    # 1. Fit normalizer STRICTLY on training recordings
    print("Fitting Gait Normalizer on training subjects...")
    train_raw_recordings = [load_raw_recording(fp) for fp in train_df["filepath"]]
    normalizer = GaitNormalizer()
    normalizer.fit(train_raw_recordings)
    normalizer.save(config.NORMALIZATION_STATS_PATH)
    print(f"Saved normalization parameters to: {config.NORMALIZATION_STATS_PATH}")

    # 2. Build window datasets
    train_dataset = GaitWindowDataset(train_df, normalizer=normalizer, window_size=window_size, step_size=step_size)
    val_dataset = GaitWindowDataset(val_df, normalizer=normalizer, window_size=window_size, step_size=step_size)
    test_dataset = GaitWindowDataset(test_df, normalizer=normalizer, window_size=window_size, step_size=step_size)

    # 3. Build DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_gait_batch,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_gait_batch,
        drop_last=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_gait_batch,
        drop_last=False
    )

    return train_loader, val_loader, test_loader, normalizer, train_df, val_df, test_df
