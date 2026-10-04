"""Interactive Gait Reliability Demonstration Pipeline (Zero-Retraining Inference).

Research Objective:
    Provide an intuitive CLI and programmatic inference tool that evaluates an individual gait recording
    end-to-end using frozen, pre-trained model checkpoints, precomputed zero-leakage normalization
    statistics, and precomputed calibration / conformal / reliability thresholds.

Pipeline:
    Raw 16-channel VGRF gait signal (.txt)
    -> load_raw_recording() (extract columns 1..16, shape (T, 16))
    -> 5.0-second windows (500 samples at 100 Hz) with 50% overlap (250 samples)
    -> Channel-wise GaitNormalizer using saved training statistics (normalization_stats.json)
    -> Pretrained GaitCNNBiLSTM checkpoint (ensemble_exp{fold}_member0_seed42.pt)
    -> Window-level forward pass & predictions
    -> Recording-level mean aggregation
    -> Temperature scaling using precomputed fold T
    -> P(Healthy), P(PD)
    -> Confidence = max(P(Healthy), P(PD))
    -> Uncertainty = 1.0 - Confidence
    -> Conformal prediction set using precomputed q_hat
    -> Reliability decision (ACCEPT / FLAG) using precomputed tau_conf & tau_unc
    -> Human-readable report & structured programmatic return

Usage:
    python -m Gait.demo --input Gait/gait-in-parkinsons-disease-1.0.0/GaPt03_01.txt --fold 3
    python -m Gait.demo --input Gait/gait-in-parkinsons-disease-1.0.0/JuCo01_01.txt --fold 2
    python -m Gait.demo --input Gait/gait-in-parkinsons-disease-1.0.0/SiPt01_01.txt --fold 1 --plot
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Path bootstrapping
_THIS_FILE = Path(__file__).resolve()
_GAIT_DIR = _THIS_FILE.parent
_ROOT_DIR = _GAIT_DIR.parent

for _p in (str(_GAIT_DIR), str(_ROOT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Gait import config
from Gait.dataset import segment_recording_into_windows
from Gait.experiments.conformal_prediction import construct_prediction_set, format_set_string
from Gait.experiments.corruption_shift import apply_corruption
from Gait.experiments.reliability_analysis import evaluate_decision_layer
from Gait.model import GaitCNNBiLSTM
from Gait.preprocessing import GaitNormalizer, load_raw_recording
from Gait.utils import get_device

# Precomputed fold artifacts and calibration thresholds (frozen from offline experiments)
PRECOMPUTED_FOLD_ASSETS: Dict[int, Dict] = {
    1: {
        "train_cohorts": "Ga+Ju",
        "test_cohort": "Si",
        "checkpoint_name": "ensemble_exp1_member0_seed42.pt",
        "temperature": 1.7542187822198632,
        "tau_conf": 0.55,
        "tau_unc": 0.4044305468308314,
        "q_hat": 0.9192679752302372,
        "alpha": 0.10,
    },
    2: {
        "train_cohorts": "Ga+Si",
        "test_cohort": "Ju",
        "checkpoint_name": "ensemble_exp2_member0_seed42.pt",
        "temperature": 1.5752042429080755,
        "tau_conf": 0.55,
        "tau_unc": 0.4500894249330588,
        "q_hat": 0.8147560958710676,
        "alpha": 0.10,
    },
    3: {
        "train_cohorts": "Ju+Si",
        "test_cohort": "Ga",
        "checkpoint_name": "ensemble_exp3_member0_seed42.pt",
        "temperature": 9.733706469812919,
        "tau_conf": 0.7487072792971063,
        "tau_unc": 0.20760254608081974,
        "q_hat": 0.764675178376716,
        "alpha": 0.10,
    },
}


def find_default_recording() -> Path:
    """Find a representative Parkinson's Disease gait recording."""
    data_dir = config.DATA_DIR
    candidates = sorted(list(data_dir.glob("*Pt*.txt")))
    if not candidates:
        candidates = sorted(list(data_dir.glob("*.txt")))
    if not candidates:
        raise FileNotFoundError(f"No gait recording files (.txt) found in {data_dir}")
    return candidates[0]


def load_fold_assets(
    fold: int = 1,
    device: torch.device = torch.device("cpu"),
) -> Tuple[GaitCNNBiLSTM, GaitNormalizer, float, Dict[str, float], str]:
    """
    Load pre-trained model checkpoint, normalizer, and precomputed calibration/reliability thresholds.
    Zero training, fitting, or test label access is performed.
    """
    if fold not in PRECOMPUTED_FOLD_ASSETS:
        raise ValueError(f"Invalid fold: {fold}. Expected one of {list(PRECOMPUTED_FOLD_ASSETS.keys())}.")

    meta = PRECOMPUTED_FOLD_ASSETS[fold]
    ckpt_name = meta["checkpoint_name"]
    ckpt_path = config.CHECKPOINT_DIR / "deep_ensemble" / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Required model checkpoint not found at: {ckpt_path}")

    # 1. Load Normalizer (precomputed channel statistics)
    norm_path = config.NORMALIZATION_STATS_PATH
    if not norm_path.exists():
        raise FileNotFoundError(f"Normalization stats file not found at: {norm_path}")
    normalizer = GaitNormalizer.load(norm_path)

    # 2. Load Model Checkpoint (GaitCNNBiLSTM)
    model = GaitCNNBiLSTM()
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. Resolve Precomputed Calibration & Reliability Thresholds
    learned_t = float(meta["temperature"])
    thresholds = {
        "tau_conf": float(meta["tau_conf"]),
        "tau_unc": float(meta["tau_unc"]),
        "q_hat": float(meta["q_hat"]),
        "alpha": float(meta["alpha"]),
    }

    # Dynamically verify / sync from CSV if present
    conf_csv = config.OUTPUT_DIR / "conformal" / "conformal_results.csv"
    if conf_csv.exists():
        try:
            cdf = pd.read_csv(conf_csv)
            row = cdf[(cdf["experiment"] == fold) & np.isclose(cdf["alpha"], 0.10)]
            if len(row) > 0:
                learned_t = float(row["learned_temperature"].iloc[0])
                thresholds["q_hat"] = float(row["q_hat"].iloc[0])
        except Exception:
            pass

    rel_csv = config.OUTPUT_DIR / "reliability" / "reliability_results.csv"
    if rel_csv.exists():
        try:
            rdf = pd.read_csv(rel_csv)
            row = rdf[rdf["experiment"] == fold]
            if len(row) > 0:
                thresholds["tau_conf"] = float(row["tau_conf"].iloc[0])
                thresholds["tau_unc"] = float(row["tau_unc"].iloc[0])
                thresholds["q_hat"] = float(row["q_hat"].iloc[0])
        except Exception:
            pass

    return model, normalizer, learned_t, thresholds, ckpt_name


@torch.no_grad()
def evaluate_single_recording(
    model: GaitCNNBiLSTM,
    raw_signal: np.ndarray,
    normalizer: GaitNormalizer,
    learned_t: float,
    thresholds: Dict[str, float],
    corruption_type: Optional[str] = None,
    severity: int = 0,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    """
    Evaluate one recording with windowing, normalization, optional corruption, and reliability assessment.
    Uses strictly precomputed parameters without training or label access.
    """
    # 1. Window segmentation: 500 steps (5.0s at 100 Hz), 250 step shift (50% overlap)
    windows = segment_recording_into_windows(
        raw_signal,
        window_size=config.WINDOW_SIZE,
        step_size=config.STEP_SIZE,
    )
    if len(windows) == 0:
        raise ValueError("Recording is shorter than window duration (5.0s).")

    # 2. Normalize using saved training statistics
    norm_windows = normalizer.transform(windows)  # (N, 500, 16)

    # 3. Optional signal corruption (for demonstration / stress testing)
    if corruption_type and severity > 0:
        norm_windows = apply_corruption(norm_windows, corruption_type, severity).cpu().numpy()

    # 4. Model inference
    model.eval()
    x_tensor = torch.tensor(norm_windows, dtype=torch.float32, device=device)
    evidence, emb = model(x_tensor, return_embedding=True)
    _, raw_logits = model.evidential_head(emb, return_raw_logits=True)

    # Scale logits by precomputed validation temperature
    calib_probs = torch.softmax(raw_logits / learned_t, dim=1).cpu().numpy()

    # 5. Window -> Recording mean aggregation
    mean_p_hc = float(np.mean(calib_probs[:, 0]))
    mean_p_pd = float(np.mean(calib_probs[:, 1]))
    p_sum = mean_p_hc + mean_p_pd
    mean_p_hc /= p_sum
    mean_p_pd /= p_sum

    confidence = max(mean_p_hc, mean_p_pd)
    uncertainty = 1.0 - confidence
    pred_label = 1 if mean_p_pd >= 0.5 else 0

    # 6. Conformal & Reliability decision using precomputed thresholds
    subject_df = pd.DataFrame([{
        "subject_id": "INFERENCE_SAMPLE",
        "predicted_label": pred_label,
        "true_label": -1,  # Unseen test instance - never accessed
        "P_HC": mean_p_hc,
        "P_PD": mean_p_pd,
        "confidence": confidence,
        "uncertainty": uncertainty,
    }])

    decision_df = evaluate_decision_layer(subject_df, thresholds)
    rec = decision_df.iloc[0].to_dict()

    flag_reasons_str = rec.get("flag_reasons", "NONE")
    flag_reasons_list = [r for r in flag_reasons_str.split(";") if r and r != "NONE"]

    return {
        "prediction": "Parkinson's Disease" if pred_label == 1 else "Healthy",
        "predicted_class": "Parkinson's Disease" if pred_label == 1 else "Healthy Control",
        "predicted_label": pred_label,
        "probability_pd": mean_p_pd,
        "probability_hc": mean_p_hc,
        "confidence": confidence,
        "uncertainty": uncertainty,
        "uncertainty_level": rec.get("uncertainty_level", "LOW" if uncertainty <= thresholds["tau_unc"] else "HIGH"),
        "confidence_level": rec.get("confidence_level", "HIGH" if confidence >= thresholds["tau_conf"] else "LOW"),
        "conformal_set": rec.get("conformal_set", "{Healthy, PD}"),
        "set_size": rec.get("set_size", 1),
        "reliability_status": rec.get("reliability_status", "ACCEPT"),
        "flag_reasons": flag_reasons_list if flag_reasons_list else ["None (Model is Reliable)"],
        "flag_reasons_raw": flag_reasons_str,
        "num_windows": len(windows),
    }


def generate_demo_plot(
    rec_path: Path,
    model: GaitCNNBiLSTM,
    raw_signal: np.ndarray,
    normalizer: GaitNormalizer,
    learned_t: float,
    thresholds: Dict[str, float],
    corruption_type: str = "gaussian_noise",
    device: torch.device = torch.device("cpu"),
) -> Path:
    """Generate visual demonstration of confidence, uncertainty, and reliability status across severities."""
    out_plot_dir = config.OUTPUT_DIR / "demo"
    out_plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_plot_dir / f"demo_{rec_path.stem}_{corruption_type}.png"

    severities = [0, 1, 2, 3, 4]
    sweep_results = []
    for sev in severities:
        sev_eval = evaluate_single_recording(
            model=model,
            raw_signal=raw_signal,
            normalizer=normalizer,
            learned_t=learned_t,
            thresholds=thresholds,
            corruption_type=corruption_type,
            severity=sev,
            device=device,
        )
        sev_eval["severity"] = sev
        sweep_results.append(sev_eval)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    sevs = [r["severity"] for r in sweep_results]
    confs = [r["confidence"] for r in sweep_results]
    uncs = [r["uncertainty"] for r in sweep_results]
    statuses = [r["reliability_status"] for r in sweep_results]

    # Left plot: Confidence & Uncertainty
    ax1.plot(sevs, confs, "o-", color="#2b6cb0", linewidth=2.5, label="Prediction Confidence")
    ax1.plot(sevs, uncs, "s--", color="#c53030", linewidth=2.5, label="Uncertainty (1 - Conf)")
    ax1.axhline(thresholds["tau_conf"], color="#2b6cb0", linestyle=":", alpha=0.7, label=f"Confidence Threshold ({thresholds['tau_conf']:.2f})")
    ax1.axhline(thresholds["tau_unc"], color="#c53030", linestyle=":", alpha=0.7, label=f"Uncertainty Threshold ({thresholds['tau_unc']:.2f})")
    ax1.set_xlabel("Corruption Severity Level")
    ax1.set_ylabel("Value")
    ax1.set_xticks(sevs)
    ax1.set_xticklabels(["0 (Clean)", "1", "2", "3", "4 (Severe)"])
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("Prediction Confidence vs Uncertainty", fontweight="bold")
    ax1.legend(loc="best")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Right plot: Reliability Status
    colors = ["#38a169" if s == "ACCEPT" else "#e53e3e" for s in statuses]
    ax2.bar(sevs, [1] * len(sevs), color=colors, alpha=0.85, width=0.5)
    for i, s in enumerate(statuses):
        ax2.text(sevs[i], 0.5, s, ha="center", va="center", color="white", fontweight="bold", fontsize=11)

    ax2.set_xlabel("Corruption Severity Level")
    ax2.set_yticks([])
    ax2.set_xticks(sevs)
    ax2.set_xticklabels(["0 (Clean)", "1", "2", "3", "4 (Severe)"])
    ax2.set_title("Reliability Layer Status (ACCEPT / FLAG)", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(f"Gait Inference: {rec_path.name} under {corruption_type.replace('_', ' ').title()}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nSaved demonstration visual plot -> {plot_path}")
    return plot_path


def run_demo(
    input_path: Optional[str] = None,
    fold: int = 1,
    device_name: Optional[str] = None,
    plot: bool = False,
    corruption_type: Optional[str] = None,
    severity: int = 0,
) -> Dict:
    """
    Execute true zero-retraining inference on an individual gait recording.
    Loads frozen checkpoint, normalizer, and precomputed thresholds without any training or fitting.
    """
    device = torch.device(device_name) if device_name else get_device()

    rec_path = Path(input_path) if input_path else find_default_recording()
    if not rec_path.exists():
        raise FileNotFoundError(f"Recording path does not exist: {rec_path}")

    # 1. Load Precomputed Assets (Zero Retraining)
    model, normalizer, learned_t, thresholds, ckpt_name = load_fold_assets(fold=fold, device=device)

    # 2. Load Raw Signal (Columns 1..16, shape (T, 16))
    raw_signal = load_raw_recording(rec_path)

    # 3. Execute Inference Pipeline
    eval_result = evaluate_single_recording(
        model=model,
        raw_signal=raw_signal,
        normalizer=normalizer,
        learned_t=learned_t,
        thresholds=thresholds,
        corruption_type=corruption_type,
        severity=severity,
        device=device,
    )

    result = {
        "input_file": rec_path.name,
        "input_path": str(rec_path),
        "fold": fold,
        "checkpoint": ckpt_name,
        "learned_temperature": learned_t,
        "tau_conf": thresholds["tau_conf"],
        "tau_unc": thresholds["tau_unc"],
        "q_hat": thresholds["q_hat"],
        **eval_result,
    }

    # 4. Clean, Human-Readable Console Output
    print("=" * 60)
    print("GAIT PARKINSON'S INFERENCE")
    print("=" * 60)
    print(f"Input: {rec_path.name}")
    print(f"LOCO Fold: {fold}")
    print(f"Checkpoint: {ckpt_name}\n")
    print(f"Windows processed: {eval_result['num_windows']}\n")
    print(f"Prediction: {eval_result['prediction']}")
    print(f"P(PD): {eval_result['probability_pd'] * 100:.2f}%")
    print(f"P(Healthy): {eval_result['probability_hc'] * 100:.2f}%\n")
    print(f"Confidence: {eval_result['confidence'] * 100:.2f}%")
    print(f"Uncertainty: {eval_result['uncertainty'] * 100:.2f}%\n")
    print(f"Conformal set: {eval_result['conformal_set']}\n")
    print(f"Reliability: {eval_result['reliability_status']}\n")
    print("Flag reasons:")
    for r in eval_result["flag_reasons"]:
        print(f"- {r}")
    print("=" * 60)

    # Optional plot generation if requested
    if plot:
        generate_demo_plot(
            rec_path=rec_path,
            model=model,
            raw_signal=raw_signal,
            normalizer=normalizer,
            learned_t=learned_t,
            thresholds=thresholds,
            corruption_type=corruption_type or "gaussian_noise",
            device=device,
        )

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-Retraining Gait Parkinson's Individual Inference Demo")
    parser.add_argument("--input", type=str, default=None, help="Path to raw gait .txt recording")
    parser.add_argument("--fold", type=int, default=1, choices=[1, 2, 3], help="LOCO Fold to use (1, 2, or 3)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g. cuda, cpu)")
    parser.add_argument("--plot", action="store_true", help="Generate demonstration figure")
    parser.add_argument("--corruption", type=str, default=None, help="Corruption type to simulate (for plot/stress testing)")
    parser.add_argument("--severity", type=int, default=0, help="Corruption severity level (0-4)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_demo(
        input_path=args.input,
        fold=args.fold,
        device_name=args.device,
        plot=args.plot,
        corruption_type=args.corruption,
        severity=args.severity,
    )
