"""Configuration for Voice 1D CNN Baseline on Raw Waveforms.

Conventional 1D CNN operating directly on raw 16 kHz audio waveforms (without Wav2Vec2 or STFT).
"""

from pathlib import Path

# Paths
BASELINE_DIR = Path(__file__).resolve().parent
VOICE_DIR = BASELINE_DIR.parent.parent
ROOT_DIR = VOICE_DIR.parent

DATA_DIR = VOICE_DIR / "26-29_09_2017_KCL"
OUTPUT_DIR = BASELINE_DIR / "results"
CHECKPOINT_DIR = BASELINE_DIR / "checkpoints"
METRICS_DIR = OUTPUT_DIR / "metrics"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Ensure output directories exist
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Audio Preprocessing Hyperparameters
SAMPLE_RATE = 16000            # Standard 16 kHz resampled audio
CHUNK_DURATION = 3.0          # Seconds per acoustic chunk
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)  # 48,000 samples
CHUNK_STEP_RATIO = 0.5        # 50% overlap
STEP_SAMPLES = int(CHUNK_SAMPLES * (1.0 - CHUNK_STEP_RATIO))  # 24,000 samples

# Dataset Labels & Split
LABEL_MAP = {"HC": 0, "PD": 1}
ID2LABEL = {0: "Healthy Control (HC)", 1: "Parkinson's Disease (PD)"}
NUM_CLASSES = 2

TRAIN_SPLIT_RATIO = 0.70
VAL_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15
RANDOM_SEED = 42

# 1D CNN Architecture Parameters for Raw Waveform
CONV_CHANNELS = [32, 64, 128, 256]
DENSE_DIM = 128
DROPOUT = 0.3

# Training Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 30
PATIENCE = 7
MAX_GRAD_NORM = 1.0

# Output Artifact Files
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_voice_1d_cnn.pt"
MODEL_CONFIG_PATH = CHECKPOINT_DIR / "model_config.json"
NORMALIZATION_STATS_PATH = CHECKPOINT_DIR / "normalization_stats.json"

LOSS_CURVE_PATH = PLOTS_DIR / "loss_curves.png"
ACCURACY_CURVE_PATH = PLOTS_DIR / "accuracy_curves.png"
CONFUSION_MATRIX_PATH = PLOTS_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = PLOTS_DIR / "roc_curve.png"
COMPARISON_BAR_PATH = PLOTS_DIR / "voice_models_comparison.png"

METRICS_JSON_PATH = METRICS_DIR / "test_metrics.json"
SUBJECT_PREDICTIONS_PATH = METRICS_DIR / "subject_predictions.csv"
RECORDING_PREDICTIONS_PATH = METRICS_DIR / "recording_predictions.csv"
CHUNK_PREDICTIONS_PATH = METRICS_DIR / "chunk_predictions.csv"
