"""Extract 768-dimensional Wav2Vec2 pooled embeddings for downstream multimodal fusion."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import Wav2Vec2Processor

# Support running as a standalone script or as a module
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
for p in (str(current_dir), str(parent_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from Voice import config
    from Voice.dataset import build_metadata, chunk_audio, load_and_resample_audio
    from Voice.model import Wav2Vec2ForParkinsons
    from Voice.utils import get_device, set_seed
except ImportError:
    import config
    from dataset import build_metadata, chunk_audio, load_and_resample_audio
    from model import Wav2Vec2ForParkinsons
    from utils import get_device, set_seed


def extract_embeddings(
    checkpoint_dir: Path = config.CHECKPOINT_DIR,
    output_dir: Path = config.EMBEDDINGS_DIR
):
    """Extract and save pre-classification 768-dim embeddings for all recordings and subjects."""
    print("=" * 50)
    print("EXTRACTING WAV2VEC2 VOICE EMBEDDINGS")
    print("=" * 50)

    set_seed(config.RANDOM_SEED)
    device = get_device()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load trained model & processor
    weights_path = checkpoint_dir / "best_model.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Trained checkpoint not found at: {weights_path}")

    print(f"Loading checkpoint from: {checkpoint_dir}")
    processor = Wav2Vec2Processor.from_pretrained(str(checkpoint_dir))
    model = Wav2Vec2ForParkinsons(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        freeze_feature_encoder=config.FREEZE_FEATURE_ENCODER
    )
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # 2. Build metadata
    df = build_metadata()
    print(f"Found {len(df)} audio recordings from {df['subject_id'].nunique()} subjects.")

    recording_embeddings = {}
    metadata_records = []
    use_cuda = device.type == "cuda"

    print("Extracting chunk representations and pooling to recording representations...")
    with torch.no_grad():
        for idx, row in df.iterrows():
            filepath = row["filepath"]
            audio = load_and_resample_audio(filepath, target_sr=config.SAMPLE_RATE)
            chunks = chunk_audio(audio, chunk_samples=config.CHUNK_SAMPLES, step_samples=config.STEP_SAMPLES)

            chunk_embs = []
            for chunk in chunks:
                processed = processor(chunk, sampling_rate=config.SAMPLE_RATE, return_tensors="pt")
                input_values = processed.input_values.to(device)

                if use_cuda:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        _, emb = model(input_values, return_embedding=True)
                else:
                    _, emb = model(input_values, return_embedding=True)

                chunk_embs.append(emb.cpu().squeeze(0))

            # Mean pool chunk embeddings to recording-level representation
            recording_emb = torch.stack(chunk_embs).mean(dim=0)  # Shape: [768]
            recording_key = f"{row['task']}_{row['filename']}"
            recording_embeddings[recording_key] = recording_emb

            metadata_records.append({
                "recording_key": recording_key,
                "recording_id": row["filename"],
                "subject_id": row["subject_id"],
                "task": row["task"],
                "label": row["label"],
                "label_name": row["label_name"],
                "duration": row["duration"],
                "num_chunks": len(chunks),
                "embedding_dim": recording_emb.shape[0]
            })

    meta_df = pd.DataFrame(metadata_records)

    # 3. Aggregate to Subject-level representations
    subject_embeddings = {}
    for sub_id, group in meta_df.groupby("subject_id"):
        keys = group["recording_key"].tolist()
        sub_emb = torch.stack([recording_embeddings[k] for k in keys]).mean(dim=0)
        subject_embeddings[sub_id] = sub_emb

    # 4. Save Embeddings and Metadata
    emb_dict = {
        "recording_embeddings": recording_embeddings,
        "subject_embeddings": subject_embeddings
    }

    pt_output_path = output_dir / "voice_embeddings.pt"
    csv_output_path = output_dir / "voice_embeddings_metadata.csv"

    torch.save(emb_dict, pt_output_path)
    meta_df.to_csv(csv_output_path, index=False)

    print("\n" + "=" * 50)
    print("EMBEDDING EXTRACTION SUMMARY")
    print("=" * 50)
    print(f"Total Recordings Extracted: {len(recording_embeddings)}")
    print(f"Total Subjects Extracted:   {len(subject_embeddings)}")
    print(f"Embedding Dimension:        768")
    print(f"Saved PyTorch Embeddings:   {pt_output_path}")
    print(f"Saved Metadata CSV:         {csv_output_path}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Extract Wav2Vec2 representations for multimodal fusion.")
    parser.add_argument("--checkpoint", type=str, default=str(config.CHECKPOINT_DIR), help="Path to checkpoint folder")
    parser.add_argument("--output", type=str, default=str(config.EMBEDDINGS_DIR), help="Output directory for embeddings")
    args = parser.parse_args()

    extract_embeddings(Path(args.checkpoint), Path(args.output))


if __name__ == "__main__":
    main()
