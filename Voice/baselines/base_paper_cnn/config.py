"""Configuration for Voice Base-Paper-Style 2D CNN Baseline.

Reference:
    "Multimodal assessment of Parkinson's disease: a deep learning approach"
    by J. C. Vásquez-Correa et al., IEEE JBHI, 2019.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
VOICE_DIR = BASE_DIR.parent.parent
ROOT_DIR = VOICE_DIR.parent

DATA_DIR = VOICE_DIR / "26-29_09_2017_KCL"
RESULTS_DIR = BASE_DIR / "results"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"

# Ensure directories exist
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Audio Preprocessing & Sampling
SAMPLE_RATE = 16000            # Standard 16 kHz resampled audio
CHUNK_DURATION = 3.0          # Seconds per acoustic chunk
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)  # 48,000 samples
CHUNK_STEP_RATIO = 0.5        # 50% overlap
STEP_SAMPLES = int(CHUNK_SAMPLES * (1.0 - CHUNK_STEP_RATIO))  # 24,000 samples

# STFT Time-Frequency Parameters for Continuous Speech
# 25 ms window with 10 ms shift, standard for acoustic modeling
N_FFT = 512
WIN_LENGTH = 400              # 25 ms at 16 kHz
HOP_LENGTH = 160              # 10 ms at 16 kHz
FREQ_BINS = N_FFT // 2 + 1    # 257 frequency bins (0 to 8000 Hz)
TIME_FRAMES = 301             # Spectrogram time frames for 3.0s

# Speech Transition Parameters (Onset & Offset Phonation Modeling)
# 80 ms boundary segments (40 ms unvoiced + 40 ms voiced for onset; vice versa for offset)
TRANSITION_DURATION = 0.080   # 80 ms
TRANSITION_SAMPLES = int(SAMPLE_RATE * TRANSITION_DURATION)  # 1280 samples
TRANSITION_N_FFT = 256
TRANSITION_WIN_LENGTH = 160   # 10 ms
TRANSITION_HOP_LENGTH = 40    # 2.5 ms
TRANSITION_FREQ_BINS = TRANSITION_N_FFT // 2 + 1  # 129
TRANSITION_TIME_FRAMES = 33

# Dataset Labels & Split
LABEL_MAP = {"HC": 0, "PD": 1}
ID2LABEL = {0: "Healthy Control (HC)", 1: "Parkinson's Disease (PD)"}
NUM_CLASSES = 2

TRAIN_SPLIT_RATIO = 0.70
VAL_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15
RANDOM_SEED = 42

# 2D CNN Architecture Hyperparameters
CONV_CHANNELS = [32, 64, 128]
DENSE_UNITS = 128
DROPOUT_RATE = 0.3

# Training Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 35
PATIENCE = 7
MAX_GRAD_NORM = 1.0

# Checkpoints & Artifact Paths inside baseline directory
BEST_MODEL_PATH = CHECKPOINTS_DIR / "best_voice_base_paper_cnn.pt"
ONSET_MODEL_PATH = CHECKPOINTS_DIR / "best_voice_onset_cnn.pt"
OFFSET_MODEL_PATH = CHECKPOINTS_DIR / "best_voice_offset_cnn.pt"
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
CHUNK_PREDICTIONS_PATH = METRICS_DIR / "chunk_predictions.csv"
TRANSITION_METRICS_PATH = METRICS_DIR / "transition_analysis_metrics.json"
