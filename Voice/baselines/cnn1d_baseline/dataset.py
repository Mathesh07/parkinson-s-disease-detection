"""Dataset loader and windowing pipeline for Voice 1D CNN Raw Waveform Baseline."""

import json
import re
import sys
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from . import config
except ImportError:
    import config

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Voice.dataset import build_metadata, create_subject_splits, load_and_resample_audio


class VoiceWaveformNormalizer:
    """Mean and variance normalization fitted strictly on training waveform chunks."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean: float = 0.0
        self.std: float = 1.0
        self.is_fit: bool = False

    def fit(self, chunks: List[torch.Tensor]) -> "VoiceWaveformNormalizer":
        """Fit global scalar mean and standard deviation across training audio chunks."""
        if not chunks:
            raise ValueError("Cannot fit normalizer on empty list.")
        stacked = torch.stack(chunks)  # (N, 1, 48000)
        self.mean = float(torch.mean(stacked))
        self.std = float(torch.std(stacked))
        if self.std < self.eps:
            self.std = 1.0
        self.is_fit = True
        return self

    def transform(self, chunk: torch.Tensor) -> torch.Tensor:
        """Standardize a single chunk tensor."""
        if not self.is_fit:
            return chunk
        return (chunk - self.mean) / self.std

    def save(self, filepath: Union[str, Path] = config.NORMALIZATION_STATS_PATH):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump({"mean": self.mean, "std": self.std, "eps": self.eps}, f, indent=4)

    @classmethod
    def load(cls, filepath: Union[str, Path] = config.NORMALIZATION_STATS_PATH) -> "VoiceWaveformNormalizer":
        filepath = Path(filepath)
        with open(filepath, "r") as f:
            stats = json.load(f)
        norm = cls(eps=stats.get("eps", 1e-6))
        norm.mean = stats["mean"]
        norm.std = stats["std"]
        norm.is_fit = True
        return norm


class VoiceRawWaveformChunkDataset(Dataset):
    """PyTorch Dataset yielding 3.0s raw audio waveform chunks (1, 48000)."""

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        normalizer: Optional[VoiceWaveformNormalizer] = None,
        chunk_samples: int = config.CHUNK_SAMPLES,
        step_samples: int = config.STEP_SAMPLES
    ):
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.normalizer = normalizer
        self.chunk_samples = chunk_samples
        self.step_samples = step_samples
        self.samples: List[Dict] = []
        self._build_dataset()

    def _build_dataset(self):
        for _, row in self.metadata_df.iterrows():
            fp = row.get("filepath", "")
            if not fp or not Path(fp).exists():
                continue

            audio = load_and_resample_audio(fp, target_sr=config.SAMPLE_RATE)
            total_samples = len(audio)

            if total_samples < self.chunk_samples:
                padded = np.zeros(self.chunk_samples, dtype=np.float32)
                padded[:total_samples] = audio
                chunks = [padded]
            else:
                chunks = []
                start = 0
                while start + self.chunk_samples <= total_samples:
                    chunks.append(audio[start : start + self.chunk_samples])
                    start += self.step_samples
                if not chunks:
                    padded = np.zeros(self.chunk_samples, dtype=np.float32)
                    padded[:total_samples] = audio
                    chunks.append(padded)

            num_chunks = len(chunks)
            for chunk_idx, ch in enumerate(chunks):
                tensor_ch = torch.tensor(ch, dtype=torch.float32).unsqueeze(0)  # (1, 48000)
                if self.normalizer is not None:
                    tensor_ch = self.normalizer.transform(tensor_ch)

                self.samples.append({
                    "features": tensor_ch,
                    "label": int(row["label"]),
                    "label_name": row["label_name"],
                    "subject_id": row["subject_id"],
                    "recording_id": row["filename"],
                    "task": row["task"],
                    "chunk_idx": chunk_idx,
                    "num_chunks": num_chunks
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def collate_voice_1d_batch(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """Batch collator for Voice 1D CNN DataLoader."""
    features = torch.stack([item["features"] for item in batch])  # (B, 1, 48000)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    return {
        "features": features,
        "labels": labels,
        "subject_ids": [item["subject_id"] for item in batch],
        "recording_ids": [item["recording_id"] for item in batch],
        "tasks": [item["task"] for item in batch],
        "chunk_indices": [item["chunk_idx"] for item in batch]
    }


def create_voice_1d_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    seed: int = config.RANDOM_SEED
) -> Tuple[DataLoader, DataLoader, DataLoader, VoiceWaveformNormalizer, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create zero-leakage subject-level splits and DataLoaders."""
    meta_df = build_metadata(config.DATA_DIR)
    train_df, val_df, test_df = create_subject_splits(meta_df, seed=seed)

    print("Fitting waveform normalizer on training subjects...")
    unnorm_train_ds = VoiceRawWaveformChunkDataset(train_df, normalizer=None)
    normalizer = VoiceWaveformNormalizer()
    normalizer.fit([s["features"] for s in unnorm_train_ds.samples])
    normalizer.save(config.NORMALIZATION_STATS_PATH)
    print(f"Saved waveform normalization stats to: {config.NORMALIZATION_STATS_PATH}")

    train_ds = VoiceRawWaveformChunkDataset(train_df, normalizer=normalizer)
    val_ds = VoiceRawWaveformChunkDataset(val_df, normalizer=normalizer)
    test_ds = VoiceRawWaveformChunkDataset(test_df, normalizer=normalizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_voice_1d_batch, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_voice_1d_batch)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_voice_1d_batch)

    return train_loader, val_loader, test_loader, normalizer, train_df, val_df, test_df
