"""Configuration for Gait 1D CNN Baseline.

Conventional 1D CNN baseline for comparing against the proposed 1D CNN + BiLSTM model.
"""

from pathlib import Path

# Paths
BASELINE_DIR = Path(__file__).resolve().parent
GAIT_DIR = BASELINE_DIR.parent.parent
ROOT_DIR = GAIT_DIR.parent

DATA_DIR = GAIT_DIR / "gait-in-parkinsons-disease-1.0.0"
OUTPUT_DIR = BASELINE_DIR / "results"
CHECKPOINT_DIR = BASELINE_DIR / "checkpoints"
METRICS_DIR = OUTPUT_DIR / "metrics"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Ensure directories exist
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Signal & Windowing Parameters (Identical to proposed model)
SAMPLING_RATE = 100            # Hz (100 samples per second)
WINDOW_DURATION = 5.0          # Seconds per window
WINDOW_SIZE = int(SAMPLING_RATE * WINDOW_DURATION)  # 500 time steps
OVERLAP_RATIO = 0.5            # 50% overlap
STEP_SIZE = int(WINDOW_SIZE * (1.0 - OVERLAP_RATIO))  # 250 time steps
NUM_CHANNELS = 16              # 16 VGRF channels (8 Left + 8 Right)

# Sensor Channel Names
SENSOR_NAMES = [
    "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8",
    "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"
]

# Dataset Labels & Split
LABEL_MAP = {"Co": 0, "Pt": 1}
ID2LABEL = {0: "Healthy Control (HC)", 1: "Parkinson's Disease (PD)"}
NUM_CLASSES = 1                # Binary logit output

TRAIN_SPLIT_RATIO = 0.70
VAL_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15
RANDOM_SEED = 42

# 1D CNN Architecture Parameters (Pure Conv1D without LSTM/evidential heads)
CNN_CHANNELS = [64, 128, 256]
CNN_KERNEL_SIZE = 5
POOL_SIZE = 2
DENSE_DIM = 128
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
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_gait_1d_cnn.pt"
MODEL_CONFIG_PATH = CHECKPOINT_DIR / "model_config.json"
NORMALIZATION_STATS_PATH = CHECKPOINT_DIR / "normalization_stats.json"

LOSS_CURVE_PATH = PLOTS_DIR / "loss_curves.png"
ACCURACY_CURVE_PATH = PLOTS_DIR / "accuracy_curves.png"
CONFUSION_MATRIX_PATH = PLOTS_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = PLOTS_DIR / "roc_curve.png"
PR_CURVE_PATH = PLOTS_DIR / "precision_recall_curve.png"
ABLATION_BAR_PATH = PLOTS_DIR / "ablation_comparison.png"

TEST_METRICS_PATH = METRICS_DIR / "test_metrics.json"
WINDOW_PREDICTIONS_PATH = METRICS_DIR / "window_predictions.csv"
RECORDING_PREDICTIONS_PATH = METRICS_DIR / "recording_predictions.csv"
SUBJECT_PREDICTIONS_PATH = METRICS_DIR / "subject_predictions.csv"
LOCO_METRICS_PATH = METRICS_DIR / "leave_one_cohort_out_metrics.json"
ABLATION_METRICS_PATH = METRICS_DIR / "ablation_study_metrics.json"
