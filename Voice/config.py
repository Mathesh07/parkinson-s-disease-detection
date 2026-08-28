"""Configuration parameters for Voice-Based Parkinson's Detection using Wav2Vec2."""

from pathlib import Path

# Paths
VOICE_DIR = Path(__file__).resolve().parent
ROOT_DIR = VOICE_DIR.parent
DATA_DIR = VOICE_DIR / "26-29_09_2017_KCL"
RESULTS_DIR = VOICE_DIR / "results"
CHECKPOINT_DIR = RESULTS_DIR / "best_wav2vec2_model"
EMBEDDINGS_DIR = RESULTS_DIR / "embeddings"

# Ensure output directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# Pretrained Wav2Vec2 Checkpoint
MODEL_NAME = "facebook/wav2vec2-base"

# Audio Preprocessing Hyperparameters
SAMPLE_RATE = 16000
CHUNK_SECONDS = 10
OVERLAP_SECONDS = 2
CHUNK_SAMPLES = CHUNK_SECONDS * SAMPLE_RATE       # 160,000 samples
OVERLAP_SAMPLES = OVERLAP_SECONDS * SAMPLE_RATE   # 32,000 samples
STEP_SAMPLES = CHUNK_SAMPLES - OVERLAP_SAMPLES    # 128,000 samples

# Dataset Labels & Split
LABEL_MAP = {"HC": 0, "PD": 1}
ID2LABEL = {0: "Healthy Control (HC)", 1: "Parkinson's Disease (PD)"}
NUM_CLASSES = 2

TRAIN_SPLIT_RATIO = 0.70
VAL_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15
RANDOM_SEED = 42

# Training Hyperparameters
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
EPOCHS = 15
PATIENCE = 5
MAX_GRAD_NORM = 1.0
FREEZE_FEATURE_ENCODER = True

# Output Artifact Files
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = RESULTS_DIR / "roc_curve.png"
TRAINING_CURVES_PATH = RESULTS_DIR / "training_curves.png"
TEST_PREDICTIONS_PATH = RESULTS_DIR / "test_predictions.csv"
SUBJECT_PREDICTIONS_PATH = RESULTS_DIR / "subject_predictions.csv"
METRICS_PATH = RESULTS_DIR / "metrics.json"
TRAINING_HISTORY_PATH = RESULTS_DIR / "training_history.json"
EMBEDDINGS_OUTPUT_PATH = EMBEDDINGS_DIR / "voice_embeddings.pt"
EMBEDDINGS_METADATA_PATH = EMBEDDINGS_DIR / "voice_embeddings_metadata.csv"
