from pathlib import Path

# Project root is the workspace folder that contains the NewHandPD directory.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "NewHandPD"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILES = ["NewSpiral.csv", "NewMeander.csv"]

# Static handwriting image folders.
IMAGE_FOLDERS = {
    "spiral": {
        "healthy": DATA_ROOT / "HealthySpiral",
        "parkinson": DATA_ROOT / "PatientSpiral",
    },
    "meander": {
        "healthy": DATA_ROOT / "HealthyMeander",
        "parkinson": DATA_ROOT / "PatientMeander",
    },
}

# Raw dataset labels from the CSV files.
RAW_CLASS_LABELS = {1: "Healthy", 2: "Parkinson"}

# Model-friendly binary labels (0: Healthy, 1: Parkinson's Disease).
MODEL_LABELS = {0: "Healthy", 1: "Parkinson"}

# Global reproducibility seed
RANDOM_SEED = 42

# Training & Hyperparameter Configuration
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 25
LEARNING_RATE_HEAD = 3e-4
LEARNING_RATE_BACKBONE = 1e-5
WEIGHT_DECAY = 1e-4
PATIENCE = 3
MIN_DELTA = 1e-4

# Hugging Face ViT model
VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"

# Checkpoint & Plot Output Paths
VIT_CHECKPOINT = OUTPUT_DIR / "best_vit_model.pth"
CNN_CHECKPOINT = OUTPUT_DIR / "best_cnn_model.pth"
SPLIT_MANIFEST_PATH = OUTPUT_DIR / "patient_split_manifest.csv"
METRICS_OUTPUT_PATH = OUTPUT_DIR / "evaluation_results.json"
CURVES_PLOT_PATH = OUTPUT_DIR / "training_curves.png"
EXPLAINABILITY_PLOT_PATH = OUTPUT_DIR / "vit_explainability.png"

