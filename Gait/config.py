"""Configuration parameters for Gait-Based Parkinson's Detection using 1D CNN + BiLSTM."""

from pathlib import Path

# Base Paths
GAIT_DIR = Path(__file__).resolve().parent
ROOT_DIR = GAIT_DIR.parent
DATA_DIR = GAIT_DIR / "gait-in-parkinsons-disease-1.0.0"
OUTPUT_DIR = GAIT_DIR / "outputs"
CHECKPOINT_DIR = GAIT_DIR / "checkpoints"
METRICS_DIR = OUTPUT_DIR / "metrics"
PLOTS_DIR = OUTPUT_DIR / "plots"
FEATURES_DIR = OUTPUT_DIR / "features"

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# Signal & Windowing Parameters
SAMPLING_RATE = 100            # Hz (100 samples per second)
WINDOW_DURATION = 5.0          # Seconds per window
WINDOW_SIZE = int(SAMPLING_RATE * WINDOW_DURATION)  # 500 time steps
OVERLAP_RATIO = 0.5            # 50% overlap
STEP_SIZE = int(WINDOW_SIZE * (1.0 - OVERLAP_RATIO))  # 250 time steps
NUM_CHANNELS = 16              # 8 Left foot sensors + 8 Right foot sensors

# Sensor Channel Names
SENSOR_NAMES = [
    "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8",
    "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"
]

# Dataset Labels & Split
LABEL_MAP = {"Co": 0, "Pt": 1}
ID2LABEL = {0: "Healthy Control (HC)", 1: "Parkinson's Disease (PD)"}
NUM_CLASSES = 1               # Binary classification (BCEWithLogitsLoss)

TRAIN_SPLIT_RATIO = 0.70
VAL_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15
RANDOM_SEED = 42

# Model Architecture Parameters
CNN_CHANNELS = [64, 128]
CNN_KERNEL_SIZES = [5, 5]
CNN_POOL_SIZES = [2, 2]
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_BIDIRECTIONAL = True
EMBEDDING_DIM = 128           # Output feature vector dimension
DROPOUT = 0.3

# Training Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50
PATIENCE = 10
MIN_DELTA = 1e-4
MAX_GRAD_NORM = 1.0

# Output Artifact Files
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_gait_model.pt"
MODEL_CONFIG_PATH = CHECKPOINT_DIR / "model_config.json"
NORMALIZATION_STATS_PATH = CHECKPOINT_DIR / "normalization_stats.json"

LOSS_CURVE_PATH = PLOTS_DIR / "loss_curves.png"
ACCURACY_CURVE_PATH = PLOTS_DIR / "accuracy_curves.png"
CONFUSION_MATRIX_PATH = PLOTS_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = PLOTS_DIR / "roc_curve.png"

TEST_METRICS_PATH = METRICS_DIR / "test_metrics.json"
WINDOW_PREDICTIONS_PATH = METRICS_DIR / "window_predictions.csv"
RECORDING_PREDICTIONS_PATH = METRICS_DIR / "recording_predictions.csv"
SUBJECT_PREDICTIONS_PATH = METRICS_DIR / "subject_predictions.csv"

EMBEDDINGS_PT_PATH = FEATURES_DIR / "gait_embeddings.pt"
EMBEDDINGS_NPY_PATH = FEATURES_DIR / "gait_embeddings.npy"
EMBEDDINGS_METADATA_PATH = FEATURES_DIR / "gait_embeddings_metadata.csv"
