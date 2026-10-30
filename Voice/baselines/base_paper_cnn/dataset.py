"""PyTorch Dataset and subject-level zero-leakage splits for Voice Base-Paper CNN.
"""

import math
import os
import re
import sys
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from . import config
    from .preprocessing import (
        VoiceSTFTNormalizer,
        compute_stft_spectrogram,
        detect_speech_transitions,
        load_and_resample_audio,
    )
except ImportError:
    import config
    from preprocessing import (
        VoiceSTFTNormalizer,
        compute_stft_spectrogram,
        detect_speech_transitions,
        load_and_resample_audio,
    )


def extract_subject_id(filename: str) -> str:
    """Extract standard subject ID from audio filename (e.g., 'ID04_pd_2_0_1.wav' -> 'ID04')."""
    match = re.search(r"(ID\d+)", filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return Path(filename).stem.upper()


def build_voice_metadata(data_dir: Path = config.DATA_DIR) -> pd.DataFrame:
    """
    Build metadata DataFrame from the MDVR-KCL audio dataset folder.
    Falls back to metadata CSV if raw folder is currently being unzipped.
    """
    records = []
    tasks = ["ReadText", "SpontaneousDialogue"]
    labels = ["HC", "PD"]

    if data_dir.exists():
        for task in tasks:
            for label_name in labels:
                folder = data_dir / task / label_name
                if not folder.exists():
                    continue
                for wav_file in sorted(folder.glob("*.wav")):
                    sub_id = extract_subject_id(wav_file.name)
                    try:
                        with wave.open(str(wav_file), "rb") as wf:
                            nframes = wf.getnframes()
                            framerate = wf.getframerate()
                            duration = nframes / float(framerate) if framerate > 0 else 0.0
                    except Exception:
                        duration = 10.0

                    records.append({
                        "filepath": str(wav_file.resolve()),
                        "filename": wav_file.name,
                        "label": config.LABEL_MAP[label_name],
                        "label_name": label_name,
                        "task": task,
                        "subject_id": sub_id,
                        "duration": duration
                    })

    if not records:
        # Check if the existing repository's metadata CSV exists
        meta_csv = config.VOICE_DIR / "results" / "embeddings" / "voice_embeddings_metadata.csv"
        if meta_csv.exists():
            meta_df = pd.read_csv(meta_csv)
            for _, r in meta_df.iterrows():
                fname = r.get("recording_id", r.get("filename", ""))
                task = r.get("task", "ReadText")
                label_name = r.get("label_name", "HC")
                # Construct expected filepath in DATA_DIR
                potential_fp = data_dir / task / label_name / fname
                records.append({
                    "filepath": str(potential_fp.resolve()),
                    "filename": fname,
                    "label": int(r["label"]),
                    "label_name": label_name,
                    "task": task,
                    "subject_id": r["subject_id"],
                    "duration": float(r.get("duration", 10.0))
                })

    df = pd.DataFrame(records)
    return df


def create_subject_splits(
    df: pd.DataFrame,
    train_ratio: float = config.TRAIN_SPLIT_RATIO,
    val_ratio: float = config.VAL_SPLIT_RATIO,
    test_ratio: float = config.TEST_SPLIT_RATIO,
    seed: int = config.RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split Voice dataset at the subject level to prevent patient leakage.
    Produces identical test subjects to the proposed Wav2Vec2 model.
    """
    rng = np.random.RandomState(seed)

    hc_subjects = sorted(df[df["label_name"] == "HC"]["subject_id"].unique())
    pd_subjects = sorted(df[df["label_name"] == "PD"]["subject_id"].unique())

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

    hc_n_train, hc_n_val, hc_n_test = compute_counts(len(hc_subjects))
    pd_n_train, pd_n_val, pd_n_test = compute_counts(len(pd_subjects))

    train_hc = hc_shuffled[:hc_n_train]
    val_hc = hc_shuffled[hc_n_train : hc_n_train + hc_n_val]
    test_hc = hc_shuffled[hc_n_train + hc_n_val:]

    train_pd = pd_shuffled[:pd_n_train]
    val_pd = pd_shuffled[pd_n_train : pd_n_train + pd_n_val]
    test_pd = pd_shuffled[pd_n_train + pd_n_val:]

    train_subs = set(train_hc + train_pd)
    val_subs = set(val_hc + val_pd)
    test_subs = set(test_hc + test_pd)

    assert len(train_subs & val_subs) == 0, "Voice train/val leakage detected!"
    assert len(train_subs & test_subs) == 0, "Voice train/test leakage detected!"
    assert len(val_subs & test_subs) == 0, "Voice val/test leakage detected!"

    train_df = df[df["subject_id"].isin(train_subs)].copy().reset_index(drop=True)
    val_df = df[df["subject_id"].isin(val_subs)].copy().reset_index(drop=True)
    test_df = df[df["subject_id"].isin(test_subs)].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def chunk_audio(
    audio: np.ndarray,
    chunk_samples: int = config.CHUNK_SAMPLES,
    step_samples: int = config.STEP_SAMPLES
) -> List[np.ndarray]:
    """Slice audio waveform into overlapping chunks."""
    total_samples = len(audio)
    if total_samples <= chunk_samples:
        padded = np.zeros(chunk_samples, dtype=np.float32)
        padded[:total_samples] = audio
        return [padded]

    chunks = []
    start = 0
    while start + chunk_samples <= total_samples:
        chunks.append(audio[start : start + chunk_samples])
        start += step_samples

    if start < total_samples and (total_samples - start) >= (config.SAMPLE_RATE * 1):
        chunks.append(audio[-chunk_samples:])

    if not chunks:
        padded = np.zeros(chunk_samples, dtype=np.float32)
        padded[:total_samples] = audio
        chunks.append(padded)

    return chunks


class VoiceSTFTChunkDataset(Dataset):
    """PyTorch Dataset yielding 2D STFT spectrograms for continuous speech chunks."""

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        normalizer: Optional[VoiceSTFTNormalizer] = None,
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

            audio = load_and_resample_audio(fp, config.SAMPLE_RATE)
            chunks = chunk_audio(audio, self.chunk_samples, self.step_samples)

            for c_idx, c_data in enumerate(chunks):
                # (1, 257, 301)
                stft_spec = compute_stft_spectrogram(
                    c_data,
                    n_fft=config.N_FFT,
                    hop_length=config.HOP_LENGTH,
                    win_length=config.WIN_LENGTH
                )

                if self.normalizer is not None:
                    stft_spec = self.normalizer.transform(stft_spec)

                self.samples.append({
                    "features": stft_spec,
                    "label": row["label"],
                    "label_name": row["label_name"],
                    "subject_id": row["subject_id"],
                    "recording_id": row["filename"],
                    "task": row["task"],
                    "chunk_idx": c_idx,
                    "num_chunks": len(chunks)
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        item = self.samples[idx]
        return {
            "features": item["features"].clone().detach(),
            "label": torch.tensor(item["label"], dtype=torch.long),
            "subject_id": item["subject_id"],
            "recording_id": item["recording_id"],
            "task": item["task"],
            "chunk_idx": item["chunk_idx"],
            "num_chunks": item["num_chunks"]
        }


def collate_voice_stft_batch(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """Collate Voice STFT batches."""
    features = torch.stack([item["features"] for item in batch])  # (B, 1, 257, 301)
    labels = torch.stack([item["label"] for item in batch])        # (B,)

    return {
        "features": features,
        "labels": labels,
        "subject_ids": [item["subject_id"] for item in batch],
        "recording_ids": [item["recording_id"] for item in batch],
        "tasks": [item["task"] for item in batch],
        "chunk_indices": [item["chunk_idx"] for item in batch]
    }
