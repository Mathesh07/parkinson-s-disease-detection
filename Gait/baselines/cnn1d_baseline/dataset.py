"""Dataset pipeline, subject-level split, and PyTorch DataLoaders for Gait 1D CNN Baseline."""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from . import config
except ImportError:
    import config

# Ensure ROOT_DIR is in sys.path
if str(config.ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(config.ROOT_DIR))

from Gait.dataset import create_subject_splits, segment_recording_into_windows
from Gait.preprocessing import GaitNormalizer, build_manifest, load_raw_recording


class Gait1DCNNWindowDataset(Dataset):
    """PyTorch Dataset yielding normalized (500, 16) windows with full metadata."""

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
            norm_data = self.normalizer.transform(raw_data) if self.normalizer is not None else raw_data
            windows = segment_recording_into_windows(norm_data, self.window_size, self.step_size)
            num_windows = len(windows)

            for win_idx, win in enumerate(windows):
                self.samples.append({
                    "window": win,                      # shape (500, 16)
                    "label": row["label"],               # 0 or 1
                    "label_name": row["label_name"],
                    "subject_id": row["subject_id"],
                    "recording_id": row["filename"],
                    "study": row["study"],
                    "window_idx": win_idx,
                    "num_windows": num_windows
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        item = self.samples[idx]
        return {
            "features": torch.tensor(item["window"], dtype=torch.float32),  # (500, 16)
            "label": torch.tensor(item["label"], dtype=torch.float32),       # float for BCE
            "subject_id": item["subject_id"],
            "recording_id": item["recording_id"],
            "study": item["study"],
            "window_idx": item["window_idx"],
            "num_windows": item["num_windows"]
        }


def collate_gait_1d_batch(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """Batch collator for Gait 1D CNN DataLoader."""
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


def create_gait_1d_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    seed: int = config.RANDOM_SEED
) -> Tuple[DataLoader, DataLoader, DataLoader, GaitNormalizer, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build zero-leakage splits, fit normalizer strictly on train, and return DataLoaders."""
    manifest_df = build_manifest(config.DATA_DIR)
    train_df, val_df, test_df = create_subject_splits(manifest_df, seed=seed)

    print("Fitting normalizer strictly on training subjects...")
    train_raw_recordings = [load_raw_recording(fp) for fp in train_df["filepath"]]
    normalizer = GaitNormalizer()
    normalizer.fit(train_raw_recordings)
    normalizer.save(config.NORMALIZATION_STATS_PATH)
    print(f"Saved normalizer to: {config.NORMALIZATION_STATS_PATH}")

    train_ds = Gait1DCNNWindowDataset(train_df, normalizer=normalizer)
    val_ds = Gait1DCNNWindowDataset(val_df, normalizer=normalizer)
    test_ds = Gait1DCNNWindowDataset(test_df, normalizer=normalizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_gait_1d_batch, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_gait_1d_batch)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_gait_1d_batch)

    return train_loader, val_loader, test_loader, normalizer, train_df, val_df, test_df
