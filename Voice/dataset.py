"""Dataset discovery, subject-level split, and PyTorch Dataset for Wav2Vec2."""

import math
import os
import re
import sys
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import torch
from torch.utils.data import Dataset

# Support running as a standalone script or as a module
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
for p in (str(current_dir), str(parent_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from Voice import config
except ImportError:
    import config


def extract_subject_id(filename: str) -> str:
    """Extract standard subject ID from audio filename."""
    match = re.match(r"(ID\d+)", filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return Path(filename).stem.upper()


def build_metadata(data_dir: Path = config.DATA_DIR) -> pd.DataFrame:
    """Inspect the dataset folder and build a structured metadata DataFrame."""
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
                            channels = wf.getnchannels()
                            framerate = wf.getframerate()
                            nframes = wf.getnframes()
                            duration = nframes / float(framerate) if framerate > 0 else 0.0
                    except Exception:
                        sr, data = wavfile.read(str(wav_file))
                        channels = 1 if data.ndim == 1 else data.shape[1]
                        framerate = sr
                        duration = len(data) / float(sr)

                    records.append({
                        "filepath": str(wav_file.resolve()),
                        "filename": wav_file.name,
                        "label": config.LABEL_MAP[label_name],
                        "label_name": label_name,
                        "task": task,
                        "subject_id": sub_id,
                        "duration": duration,
                        "sample_rate": framerate,
                        "channels": channels
                    })

    if not records:
        # Fallback to pre-extracted embeddings metadata if raw audio folder is missing
        meta_csv = config.RESULTS_DIR / "embeddings" / "voice_embeddings_metadata.csv"
        if meta_csv.exists():
            meta_df = pd.read_csv(meta_csv)
            if "filepath" not in meta_df.columns:
                meta_df["filepath"] = ""
            if "filename" not in meta_df.columns and "recording_id" in meta_df.columns:
                meta_df["filename"] = meta_df["recording_id"]
            return meta_df

    df = pd.DataFrame(records)
    return df



def create_subject_splits(
    df: pd.DataFrame,
    train_ratio: float = config.TRAIN_SPLIT_RATIO,
    val_ratio: float = config.VAL_SPLIT_RATIO,
    test_ratio: float = config.TEST_SPLIT_RATIO,
    seed: int = config.RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset at subject level to prevent patient data leakage."""
    rng = np.random.RandomState(seed)

    # Separate HC and PD unique subject IDs
    hc_subjects = sorted(df[df["label_name"] == "HC"]["subject_id"].unique())
    pd_subjects = sorted(df[df["label_name"] == "PD"]["subject_id"].unique())

    hc_subjects_shuffled = list(rng.permutation(hc_subjects))
    pd_subjects_shuffled = list(rng.permutation(pd_subjects))

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

    train_hc = hc_subjects_shuffled[:hc_n_train]
    val_hc = hc_subjects_shuffled[hc_n_train : hc_n_train + hc_n_val]
    test_hc = hc_subjects_shuffled[hc_n_train + hc_n_val:]

    train_pd = pd_subjects_shuffled[:pd_n_train]
    val_pd = pd_subjects_shuffled[pd_n_train : pd_n_train + pd_n_val]
    test_pd = pd_subjects_shuffled[pd_n_train + pd_n_val:]

    train_subjects = set(train_hc + train_pd)
    val_subjects = set(val_hc + val_pd)
    test_subjects = set(test_hc + test_pd)

    # Strictly enforce zero data leakage across subject splits
    assert len(train_subjects & val_subjects) == 0, "Leakage detected between train and val subjects!"
    assert len(train_subjects & test_subjects) == 0, "Leakage detected between train and test subjects!"
    assert len(val_subjects & test_subjects) == 0, "Leakage detected between val and test subjects!"

    train_df = df[df["subject_id"].isin(train_subjects)].copy().reset_index(drop=True)
    val_df = df[df["subject_id"].isin(val_subjects)].copy().reset_index(drop=True)
    test_df = df[df["subject_id"].isin(test_subjects)].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def load_and_resample_audio(filepath: str, target_sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """Load WAV audio, convert to mono float32, and polyphase resample to target_sr."""
    sr, audio = wavfile.read(filepath)

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Polyphase resampling (fast & high fidelity)
    if sr != target_sr:
        gcd = math.gcd(target_sr, sr)
        up = target_sr // gcd
        down = sr // gcd
        audio = signal.resample_poly(audio, up, down).astype(np.float32)

    return audio


def chunk_audio(
    audio: np.ndarray,
    chunk_samples: int = config.CHUNK_SAMPLES,
    step_samples: int = config.STEP_SAMPLES
) -> List[np.ndarray]:
    """Slice audio into overlapping fixed-length chunks."""
    total_samples = len(audio)
    if total_samples <= chunk_samples:
        # Pad short audio to minimum chunk size
        padded = np.zeros(chunk_samples, dtype=np.float32)
        padded[:total_samples] = audio
        return [padded]

    chunks = []
    start = 0
    while start + chunk_samples <= total_samples:
        chunks.append(audio[start : start + chunk_samples])
        start += step_samples

    # Handle remaining samples at the end of recording
    if start < total_samples and (total_samples - start) >= (config.SAMPLE_RATE * 2):
        chunks.append(audio[-chunk_samples:])

    if not chunks:
        padded = np.zeros(chunk_samples, dtype=np.float32)
        padded[:total_samples] = audio
        chunks.append(padded)

    return chunks


class VoiceChunkDataset(Dataset):
    """PyTorch Dataset that generates audio chunks with full metadata tracking."""

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        processor=None,
        target_sr: int = config.SAMPLE_RATE,
        chunk_samples: int = config.CHUNK_SAMPLES,
        step_samples: int = config.STEP_SAMPLES
    ):
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.processor = processor
        self.target_sr = target_sr
        self.chunk_samples = chunk_samples
        self.step_samples = step_samples

        # Index all chunks across all recordings
        self.chunk_items = []
        self._build_chunk_index()

    def _build_chunk_index(self):
        embeddings_dict = None
        embs_file = config.RESULTS_DIR / "embeddings" / "voice_embeddings.pt"
        if embs_file.exists():
            try:
                embeddings_dict = torch.load(embs_file, weights_only=True)
            except Exception:
                embeddings_dict = torch.load(embs_file)

        for rec_idx, row in self.metadata_df.iterrows():
            filepath = row.get("filepath", "")
            if filepath and Path(filepath).exists():
                audio = load_and_resample_audio(filepath, self.target_sr)
                chunks = chunk_audio(audio, self.chunk_samples, self.step_samples)
                num_chunks = len(chunks)
                for chunk_idx, chunk_data in enumerate(chunks):
                    self.chunk_items.append({
                        "audio_chunk": chunk_data,
                        "label": row["label"],
                        "label_name": row["label_name"],
                        "task": row["task"],
                        "subject_id": row["subject_id"],
                        "recording_id": row["filename"],
                        "filepath": filepath,
                        "chunk_idx": chunk_idx,
                        "num_chunks": num_chunks
                    })
            elif embeddings_dict is not None:
                rec_key = row.get("recording_key", f"{row['task']}_{row['filename']}")
                rec_embs = embeddings_dict.get("recording_embeddings", {})
                if rec_key in rec_embs:
                    emb = rec_embs[rec_key]
                    if isinstance(emb, torch.Tensor):
                        emb = emb.numpy()
                else:
                    sub_embs = embeddings_dict.get("subject_embeddings", {})
                    emb = sub_embs[row["subject_id"]]
                    if isinstance(emb, torch.Tensor):
                        emb = emb.numpy()

                self.chunk_items.append({
                    "audio_chunk": emb,
                    "label": row["label"],
                    "label_name": row["label_name"],
                    "task": row["task"],
                    "subject_id": row["subject_id"],
                    "recording_id": row.get("recording_id", row.get("filename", "")),
                    "filepath": "",
                    "chunk_idx": 0,
                    "num_chunks": 1
                })

    def __len__(self) -> int:
        return len(self.chunk_items)

    def __getitem__(self, idx: int) -> Dict:
        item = self.chunk_items[idx]
        audio_chunk = item["audio_chunk"]

        if self.processor is not None:
            # Process via Wav2Vec2Processor
            processed = self.processor(
                audio_chunk,
                sampling_rate=self.target_sr,
                return_tensors="pt"
            )
            input_values = processed.input_values.squeeze(0)
        else:
            input_values = torch.tensor(audio_chunk, dtype=torch.float32)

        return {
            "input_values": input_values,
            "label": torch.tensor(item["label"], dtype=torch.long),
            "recording_id": item["recording_id"],
            "subject_id": item["subject_id"],
            "task": item["task"],
            "label_name": item["label_name"],
            "chunk_idx": item["chunk_idx"],
            "num_chunks": item["num_chunks"]
        }


def collate_audio_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function to batch dynamic audio chunks."""
    input_values = [item["input_values"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])

    # Determine max sequence length in batch
    max_len = max(x.shape[0] for x in input_values)
    padded_inputs = torch.zeros(len(input_values), max_len, dtype=torch.float32)
    attention_mask = torch.zeros(len(input_values), max_len, dtype=torch.long)

    for i, x in enumerate(input_values):
        padded_inputs[i, :x.shape[0]] = x
        attention_mask[i, :x.shape[0]] = 1

    return {
        "input_values": padded_inputs,
        "attention_mask": attention_mask,
        "labels": labels,
        "recording_ids": [item["recording_id"] for item in batch],
        "subject_ids": [item["subject_id"] for item in batch],
        "tasks": [item["task"] for item in batch],
        "chunk_indices": [item["chunk_idx"] for item in batch]
    }


if __name__ == "__main__":
    # Test discovery and splitting
    print("Inspecting KCL Dataset...")
    meta_df = build_metadata()
    print(f"Total recordings: {len(meta_df)}")
    print(f"Total subjects: {meta_df['subject_id'].nunique()}")

    train_df, val_df, test_df = create_subject_splits(meta_df)
    print("\n--- Subject Split Statistics ---")
    print(f"Train subjects ({train_df['subject_id'].nunique()}): {sorted(train_df['subject_id'].unique().tolist())}")
    print(f"  HC: {train_df[train_df['label_name']=='HC']['subject_id'].nunique()} | PD: {train_df[train_df['label_name']=='PD']['subject_id'].nunique()}")
    print(f"Val subjects ({val_df['subject_id'].nunique()}): {sorted(val_df['subject_id'].unique().tolist())}")
    print(f"  HC: {val_df[val_df['label_name']=='HC']['subject_id'].nunique()} | PD: {val_df[val_df['label_name']=='PD']['subject_id'].nunique()}")
    print(f"Test subjects ({test_df['subject_id'].nunique()}): {sorted(test_df['subject_id'].unique().tolist())}")
    print(f"  HC: {test_df[test_df['label_name']=='HC']['subject_id'].nunique()} | PD: {test_df[test_df['label_name']=='PD']['subject_id'].nunique()}")
