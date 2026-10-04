"""Script to train 5 independent Deep Ensemble model members for Voice Wav2Vec2.

Each member is trained with a distinct random seed (42, 43, 44, 45, 46) for model initialization,
while enforcing identical subject-level data partitioning (split_seed=42).

Checkpoints and metrics are saved under Voice/results/deep_ensemble/member_X/.
"""

import json
import sys
from pathlib import Path

# Support running as a standalone script or as a module
current_dir = Path(__file__).resolve().parent
voice_dir = current_dir.parent
root_dir = voice_dir.parent
for p in (str(root_dir), str(voice_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from Voice import config
from Voice.dataset import build_metadata, create_subject_splits
from Voice.train import train

ENSEMBLE_DIR = config.RESULTS_DIR / "deep_ensemble"
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]
SPLIT_SEED = 42


def main():
    print("=" * 65)
    print("VOICE PHASE 4: TRAINING DEEP ENSEMBLE (M=5 MEMBERS)")
    print(f"Seeds: {SEEDS} | Data Split Seed: {SPLIT_SEED}")
    print("=" * 65)

    # 1. Verify and save ensemble configuration
    metadata_df = build_metadata()
    train_df, val_df, test_df = create_subject_splits(metadata_df, seed=SPLIT_SEED)

    train_subs = sorted(train_df["subject_id"].unique().tolist())
    val_subs = sorted(val_df["subject_id"].unique().tolist())
    test_subs = sorted(test_df["subject_id"].unique().tolist())

    assert len(set(train_subs) & set(test_subs)) == 0, "Leakage in train/test subjects!"
    assert len(set(val_subs) & set(test_subs)) == 0, "Leakage in val/test subjects!"

    config_json = {
        "num_members": len(SEEDS),
        "seeds": SEEDS,
        "split_seed": SPLIT_SEED,
        "dataset": "MDVR-KCL",
        "train_subject_ids": train_subs,
        "validation_subject_ids": val_subs,
        "test_subject_ids": test_subs,
        "model_architecture": f"{config.MODEL_NAME} + EvidentialHead",
        "preprocessing_config": {
            "sample_rate": config.SAMPLE_RATE,
            "chunk_seconds": config.CHUNK_SECONDS,
            "overlap_seconds": config.OVERLAP_SECONDS
        },
        "training_config": {
            "batch_size": config.BATCH_SIZE,
            "gradient_accumulation_steps": config.GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": config.LEARNING_RATE,
            "weight_decay": config.WEIGHT_DECAY,
            "epochs": config.EPOCHS,
            "patience": config.PATIENCE,
            "freeze_feature_encoder": config.FREEZE_FEATURE_ENCODER
        }
    }

    config_path = ENSEMBLE_DIR / "deep_ensemble_config.json"
    with open(config_path, "w") as f:
        json.dump(config_json, f, indent=4)
    print(f"Saved ensemble master configuration to: {config_path}\n")

    # 2. Train each member sequentially
    for idx, seed in enumerate(SEEDS):
        member_dir = ENSEMBLE_DIR / f"member_{idx}"
        checkpoint_dir = member_dir / "checkpoint"
        
        print("\n" + "#" * 65)
        print(f"TRAINING ENSEMBLE MEMBER {idx} / {len(SEEDS)-1} (SEED {seed})")
        print("#" * 65 + "\n")

        try:
            train(
                seed=seed,
                split_seed=SPLIT_SEED,
                save_dir=checkpoint_dir
            )
            print(f"\nSuccessfully trained ensemble member {idx} (seed {seed})!")
        except Exception as e:
            print(f"\nCRITICAL FAILURE: Training member {idx} (seed {seed}) failed with error: {e}")
            raise e

    print("\n" + "=" * 65)
    print("ALL 5 ENSEMBLE MEMBERS TRAINED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
