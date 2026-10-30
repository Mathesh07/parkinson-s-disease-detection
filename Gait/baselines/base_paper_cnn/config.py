"""Configuration for Gait Base-Paper-Style 2D CNN Baseline.

Reference:
    "Multimodal assessment of Parkinson's disease: a deep learning approach"
    by J. C. Vásquez-Correa et al., IEEE JBHI, 2019.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
GAIT_DIR = BASE_DIR.parent.parent
ROOT_DIR = GAIT_DIR.parent

DATA_DIR = GAIT_DIR / "gait-in-parkinsons-disease-1.0.0"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"

# Ensure isolated baseline directories exist
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Signal & Windowing Parameters (matched to existing Gait pipeline)
SAMPLING_RATE = 100            # Hz
WINDOW_DURATION = 5.0          # Seconds per temporal window
WINDOW_SIZE = int(SAMPLING_RATE * WINDOW_DURATION)  # 500 time steps
OVERLAP_RATIO = 0.5            # 50% overlap
STEP_SIZE = int(WINDOW_SIZE * (1.0 - OVERLAP_RATIO))  # 250 time steps
NUM_CHANNELS = 16              # 8 Left + 8 Right VGRF sensor channels

SENSOR_NAMES = [
    "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8",
    "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"
]

# STFT Time-Frequency Parameters
# With 100 Hz sampling rate, n_fft=64 provides 33 frequency bins (0 to 50 Hz, 1.56 Hz/bin)
# win_length=64 (0.64s window), hop_length=16 (0.16s hop) yields 32 time frames per 5s window.
N_FFT = 64
HOP_LENGTH = 16
WIN_LENGTH = 64
FREQ_BINS = N_FFT // 2 + 1     # 33
TIME_FRAMES = 32               # (500 - 64) // 16 + 1 = 28 or centered = 32

# Dataset Labels & Subject Split (strictly matched to existing Gait model)
LABEL_MAP = {"Co": 0, "Pt": 1}
ID2LABEL = {0: "Healthy Control (HC)", 1: "Parkinson's Disease (PD)"}
NUM_CLASSES = 1                # Binary logit output for BCEWithLogitsLoss

TRAIN_SPLIT_RATIO = 0.70
VAL_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15
RANDOM_SEED = 42

# 2D CNN Architecture Hyperparameters (inspired by Vásquez-Correa et al.)
CONV_CHANNELS = [32, 64, 128]
CONV_KERNEL_SIZE = (3, 3)
POOL_KERNEL_SIZE = (2, 2)
DENSE_UNITS = 128
DROPOUT_RATE = 0.3

# Training Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 40
PATIENCE = 8
MIN_DELTA = 1e-4
MAX_GRAD_NORM = 1.0

# Output Artifact Paths inside baseline directory
BEST_MODEL_PATH = CHECKPOINTS_DIR / "best_gait_base_paper_cnn.pt"
MODEL_CONFIG_PATH = CHECKPOINTS_DIR / "model_config.json"
NORMALIZATION_STATS_PATH = CHECKPOINTS_DIR / "normalization_stats.json"

LOSS_CURVE_PATH = PLOTS_DIR / "loss_curves.png"
ACCURACY_CURVE_PATH = PLOTS_DIR / "accuracy_curves.png"
CONFUSION_MATRIX_PATH = PLOTS_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = PLOTS_DIR / "roc_curve.png"
COMPARISON_BAR_PATH = PLOTS_DIR / "baseline_vs_proposed_comparison.png"

METRICS_JSON_PATH = METRICS_DIR / "test_metrics.json"
SUBJECT_PREDICTIONS_PATH = METRICS_DIR / "subject_predictions.csv"
RECORDING_PREDICTIONS_PATH = METRICS_DIR / "recording_predictions.csv"
WINDOW_PREDICTIONS_PATH = METRICS_DIR / "window_predictions.csv"
TRANSITION_METRICS_PATH = METRICS_DIR / "transition_analysis_metrics.json"
LOCO_METRICS_PATH = METRICS_DIR / "loco_comparison_metrics.json"
