"""Feature extraction module to extract 128-dimensional fixed-length gait embeddings for multimodal fusion."""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

# Support running as a standalone script or as a module
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
for p in (str(current_dir), str(parent_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from Gait import config
    from Gait.dataset import segment_recording_into_windows
    from Gait.model import GaitCNNBiLSTM
    from Gait.preprocessing import (
        GaitNormalizer,
        build_manifest,
        load_raw_recording,
        parse_filename,
    )
    from Gait.utils import get_device, set_seed
except ImportError:
    import config
    from dataset import segment_recording_into_windows
    from model import GaitCNNBiLSTM
    from preprocessing import (
        GaitNormalizer,
        build_manifest,
        load_raw_recording,
        parse_filename,
    )
    from utils import get_device, set_seed


def extract_gait_features(
    recording: Union[str, Path, np.ndarray],
    model: Optional[GaitCNNBiLSTM] = None,
    normalizer: Optional[GaitNormalizer] = None,
    checkpoint_path: Path = config.BEST_MODEL_PATH,
    normalization_path: Path = config.NORMALIZATION_STATS_PATH,
    device: Optional[torch.device] = None,
    window_size: int = config.WINDOW_SIZE,
    step_size: int = config.STEP_SIZE
) -> np.ndarray:
    """
    Extract a single 128-dimensional fixed-length gait feature vector from an input recording.
    
    Pipeline:
        1. Loads gait recording (or uses provided raw array).
        2. Applies zero-leakage training normalization.
        3. Segments into 5.0-second overlapping windows.
        4. Passes windows through CNN-BiLSTM feature extractor.
        5. Extracts 128-dimensional dense embedding (pre-classification layer).
        6. Aggregates window embeddings (mean pooling) into one fixed 128-d vector.
        
    Returns:
        np.ndarray of shape (128,) with dtype float32.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Normalizer if not passed
    if normalizer is None:
        normalizer = GaitNormalizer.load(normalization_path)

    # Load Model if not passed
    if model is None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        model = GaitCNNBiLSTM()
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()

    # 1. Load Raw Data
    if isinstance(recording, (str, Path)):
        raw_data = load_raw_recording(recording)
    elif isinstance(recording, np.ndarray):
        raw_data = recording
    else:
        raise TypeError(f"Unsupported recording type: {type(recording)}")

    # 2. Normalize
    norm_data = normalizer.transform(raw_data)

    # 3. Windowing
    windows = segment_recording_into_windows(norm_data, window_size=window_size, step_size=step_size)

    # 4. Batch Tensor Creation
    win_tensor = torch.tensor(np.stack(windows), dtype=torch.float32).to(device)  # (N_windows, 500, 16)

    # 5. Extract 128-dim Embedding per window
    use_cuda = device.type == "cuda"
    with torch.no_grad():
        if use_cuda:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                _, window_embs = model(win_tensor, return_embedding=True)
        else:
            _, window_embs = model(win_tensor, return_embedding=True)

    # 6. Global Temporal Mean Pooling across all windows in recording
    recording_embedding = window_embs.mean(dim=0).cpu().numpy().astype(np.float32)  # Shape: (128,)

    assert recording_embedding.shape == (config.EMBEDDING_DIM,), f"Expected shape ({config.EMBEDDING_DIM},), got {recording_embedding.shape}"
    return recording_embedding


def extract_all_dataset_embeddings(
    checkpoint_path: Path = config.BEST_MODEL_PATH,
    normalization_path: Path = config.NORMALIZATION_STATS_PATH,
    output_dir: Path = config.FEATURES_DIR
):
    """
    Extract and save 128-dim embeddings for all recordings and subjects in the Gait dataset.
    """
    print("=" * 60)
    print("EXTRACTING 128-DIMENSIONAL GAIT EMBEDDINGS")
    print("=" * 60)

    set_seed(config.RANDOM_SEED)
    device = get_device()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Model & Normalizer
    print(f"Loading normalizer from: {normalization_path}")
    normalizer = GaitNormalizer.load(normalization_path)

    print(f"Loading model checkpoint from: {checkpoint_path}")
    model = GaitCNNBiLSTM().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    # 2. Build Dataset Manifest
    manifest_df = build_manifest()
    print(f"Found {len(manifest_df)} recordings across {manifest_df['subject_id'].nunique()} subjects.")

    recording_embeddings = {}
    metadata_records = []

    print("Extracting recording embeddings...")
    for idx, row in manifest_df.iterrows():
        emb = extract_gait_features(
            recording=row["filepath"],
            model=model,
            normalizer=normalizer,
            device=device
        )
        rec_key = row["filename"]
        recording_embeddings[rec_key] = torch.tensor(emb, dtype=torch.float32)

        metadata_records.append({
            "filename": row["filename"],
            "subject_id": row["subject_id"],
            "study": row["study"],
            "walk_id": row["walk_id"],
            "label": row["label"],
            "label_name": row["label_name"],
            "embedding_dim": len(emb)
        })

    meta_df = pd.DataFrame(metadata_records)

    # 3. Aggregate Recording Embeddings to Subject-Level Embeddings
    subject_embeddings = {}
    for sub_id, group in meta_df.groupby("subject_id"):
        filenames = group["filename"].tolist()
        sub_emb = torch.stack([recording_embeddings[f] for f in filenames]).mean(dim=0)
        subject_embeddings[sub_id] = sub_emb

    # 4. Save to PT, NPY, and CSV Formats
    pt_path = output_dir / "gait_embeddings.pt"
    npy_path = output_dir / "gait_embeddings.npy"
    csv_path = output_dir / "gait_embeddings_metadata.csv"

    # PyTorch dictionary format
    torch.save({
        "recording_embeddings": recording_embeddings,
        "subject_embeddings": subject_embeddings,
        "embedding_dim": config.EMBEDDING_DIM
    }, pt_path)

    # NumPy dictionary format
    np.save(npy_path, {
        "recording_embeddings": {k: v.numpy() for k, v in recording_embeddings.items()},
        "subject_embeddings": {k: v.numpy() for k, v in subject_embeddings.items()},
        "embedding_dim": config.EMBEDDING_DIM
    })

    # Metadata CSV
    meta_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 60)
    print("GAIT EMBEDDINGS EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Total Recordings Processed: {len(recording_embeddings)}")
    print(f"Total Subjects Processed:   {len(subject_embeddings)}")
    print(f"Embedding Dimension:        {config.EMBEDDING_DIM}")
    print(f"Saved PyTorch Embeddings:   {pt_path}")
    print(f"Saved NumPy Embeddings:     {npy_path}")
    print(f"Saved Metadata CSV:         {csv_path}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Extract 128-dim Gait representations for multimodal fusion.")
    parser.add_argument("--checkpoint", type=str, default=str(config.BEST_MODEL_PATH), help="Path to best model checkpoint")
    parser.add_argument("--normalization", type=str, default=str(config.NORMALIZATION_STATS_PATH), help="Path to normalization stats")
    parser.add_argument("--output", type=str, default=str(config.FEATURES_DIR), help="Output directory for embeddings")
    args = parser.parse_args()

    extract_all_dataset_embeddings(Path(args.checkpoint), Path(args.normalization), Path(args.output))


if __name__ == "__main__":
    main()
