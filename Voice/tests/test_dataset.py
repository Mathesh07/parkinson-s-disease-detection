"""Tests for Voice dataset parsing, audio loading, resampling, and chunking."""

import sys
import wave
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

# Ensure workspace root and Voice directory are in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
voice_dir = root_dir / "Voice"
for p in (str(root_dir), str(voice_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from Voice import config
from Voice.dataset import (
    VoiceChunkDataset,
    build_metadata,
    chunk_audio,
    collate_audio_batch,
    extract_subject_id,
    load_and_resample_audio,
)


def test_subject_id_extraction():
    """Verify regex extraction of subject IDs from audio filenames."""
    assert extract_subject_id("ID04_pd_2_0_1.wav") == "ID04"
    assert extract_subject_id("ID10_hc_0_0_0.wav") == "ID10"
    assert extract_subject_id("id29_PD_3_1_2.WAV") == "ID29"
    assert extract_subject_id("custom_subject_audio.wav") == "CUSTOM_SUBJECT_AUDIO"


def test_label_mappings():
    """Verify dataset label mapping constants."""
    assert config.LABEL_MAP["HC"] == 0
    assert config.LABEL_MAP["PD"] == 1
    assert config.ID2LABEL[0] == "Healthy Control (HC)"
    assert config.ID2LABEL[1] == "Parkinson's Disease (PD)"


def test_audio_resampling_and_loading(tmp_path):
    """Test loading synthetic WAV audio, channel downmixing, and polyphase resampling to 16kHz."""
    wav_path = tmp_path / "test_sample.wav"
    sr_orig = 44100
    duration_sec = 2.5
    num_samples = int(sr_orig * duration_sec)

    # Generate synthetic 44.1kHz stereo audio
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    sig_l = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    sig_r = (0.5 * np.sin(2 * np.pi * 880 * t) * 32767).astype(np.int16)
    stereo = np.column_stack((sig_l, sig_r))

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr_orig)
        wf.writeframes(stereo.tobytes())

    # Load and resample to target_sr (16000 Hz)
    audio = load_and_resample_audio(str(wav_path), target_sr=16000)

    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert audio.ndim == 1  # Downmixed to mono
    expected_samples = int(16000 * duration_sec)
    assert abs(len(audio) - expected_samples) <= 10  # Resampling rounding margin
    assert np.all(audio >= -1.0) and np.all(audio <= 1.0)


def test_audio_chunking_dimensions():
    """Verify chunking hyperparameters: 10s chunks (160k samples), 2s overlap (32k samples)."""
    target_sr = 16000
    chunk_samples = 10 * target_sr      # 160,000
    overlap_samples = 2 * target_sr    # 32,000
    step_samples = chunk_samples - overlap_samples  # 128,000

    # 1. Audio shorter than chunk length (5 seconds) -> Padded to 10s chunk
    short_audio = np.random.randn(5 * target_sr).astype(np.float32)
    chunks_short = chunk_audio(short_audio, chunk_samples, step_samples)
    assert len(chunks_short) == 1
    assert len(chunks_short[0]) == chunk_samples
    assert np.all(chunks_short[0][5 * target_sr:] == 0.0)  # Padded zeros

    # 2. Audio long enough for multiple chunks (25 seconds = 400,000 samples)
    long_audio = np.random.randn(25 * target_sr).astype(np.float32)
    chunks_long = chunk_audio(long_audio, chunk_samples, step_samples)
    # Starts: 0, 128000, 256000, and tail handling at end
    assert len(chunks_long) >= 3
    for chunk in chunks_long:
        assert len(chunk) == chunk_samples


def test_chunk_dataset_and_collate_batch(tmp_path):
    """Test VoiceChunkDataset instantiation and collate_audio_batch."""
    # Create 2 synthetic WAV files
    wav1 = tmp_path / "ID01_hc.wav"
    wav2 = tmp_path / "ID02_pd.wav"

    sr = 16000
    for w in (wav1, wav2):
        sig = (np.random.randn(sr * 12) * 1000).astype(np.int16)  # 12s audio
        with wave.open(str(w), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(sig.tobytes())

    meta_df = pd.DataFrame([
        {
            "filepath": str(wav1),
            "filename": wav1.name,
            "label": 0,
            "label_name": "HC",
            "task": "ReadText",
            "subject_id": "ID01",
            "duration": 12.0,
            "sample_rate": 16000,
            "channels": 1,
        },
        {
            "filepath": str(wav2),
            "filename": wav2.name,
            "label": 1,
            "label_name": "PD",
            "task": "SpontaneousDialogue",
            "subject_id": "ID02",
            "duration": 12.0,
            "sample_rate": 16000,
            "channels": 1,
        },
    ])

    dataset = VoiceChunkDataset(meta_df, processor=None, target_sr=16000)
    assert len(dataset) > 0

    item = dataset[0]
    assert "input_values" in item
    assert "label" in item
    assert "recording_id" in item
    assert "subject_id" in item
    assert item["input_values"].shape[0] == 160000

    # Test batch collate
    batch = [dataset[0], dataset[1]]
    collated = collate_audio_batch(batch)
    assert collated["input_values"].shape == (2, 160000)
    assert collated["attention_mask"].shape == (2, 160000)
    assert collated["labels"].shape == (2,)
    assert torch.equal(collated["attention_mask"], torch.ones(2, 160000, dtype=torch.long))
