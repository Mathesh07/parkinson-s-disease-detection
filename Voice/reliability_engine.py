"""Voice Reliability Engine.

Integrates Wav2Vec2 Deep Ensemble (M=5) inference, multi-source uncertainty quantification,
and Split Conformal Prediction into a unified, deterministic reliability decision system.

Mission:
    Evaluates whether a model prediction is sufficiently supported by uncertainty
    and conformal evidence before accepting the decision.

Decision Policy:
    - ACCEPT (HIGH): Conformal set is singleton, strong ensemble agreement (>= 4/5),
      low/moderate predictive entropy (< 0.65), non-borderline probability.
    - FLAG (CAUTION / UNRELIABLE): Triggered when conformal set is ambiguous or empty,
      ensemble disagreement is high (< 4/5), entropy is high (>= 0.65), or probability
      is near boundary [0.40, 0.60].

Disclaimer:
    This engine evaluates model prediction reliability only.
    It does NOT provide medical diagnosis or clinical patient assessment.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch

from Voice import config
from Voice.conformal_prediction import construct_prediction_set, format_prediction_set_str
from Voice.deep_ensemble import compute_ensemble_statistics


# Default Conformal Quantile calibrated on Validation subjects (n=5, alpha=0.10)
DEFAULT_CONFORMAL_Q_HAT = 0.407552


class VoiceReliabilityEngine:
    """Inference-time decision engine for Voice Parkinson's classification reliability."""

    def __init__(
        self,
        q_hat: float = DEFAULT_CONFORMAL_Q_HAT,
        entropy_threshold: float = 0.65,
        min_agreement_score: float = 0.80,
        borderline_margin: float = 0.10,
        class_names: List[str] = None
    ):
        """
        Initialize Voice Reliability Engine with pre-declared heuristic thresholds.
        
        Args:
            q_hat: Calibrated conformal quantile threshold (default 0.407552 from Phase 7).
            entropy_threshold: Maximum acceptable predictive entropy for ACCEPT (default 0.65).
            min_agreement_score: Minimum required ensemble agreement score for ACCEPT (default 0.80 = 4/5).
            borderline_margin: Margin around 0.5 decision boundary considered borderline (default 0.10 -> [0.40, 0.60]).
            class_names: Binary class names list (default ["Healthy Control", "Parkinson's Disease"]).
        """
        self.q_hat = float(q_hat)
        self.entropy_threshold = float(entropy_threshold)
        self.min_agreement_score = float(min_agreement_score)
        self.borderline_margin = float(borderline_margin)
        self.class_names = class_names or ["Healthy Control", "Parkinson's Disease"]

    def evaluate_reliability(
        self,
        member_probabilities: List[float],
        audio_meta: Optional[Dict] = None,
        q_hat_override: Optional[float] = None
    ) -> Dict:
        """
        Evaluate full reliability decision for an audio recording given ensemble member probabilities.
        
        Args:
            member_probabilities: List of M=5 probabilities P_m(PD) for ensemble members.
            audio_meta: Optional metadata dict (filename, duration, num_chunks, etc.).
            q_hat_override: Optional override for conformal q_hat threshold.
            
        Returns:
            JSON-serializable result dictionary with prediction, uncertainty, conformal set,
            reliability decision (ACCEPT/FLAG), level (HIGH/MEDIUM/LOW), and explanation reasons.
        """
        q_hat = float(q_hat_override) if q_hat_override is not None else self.q_hat
        member_probs = [float(p) for p in member_probabilities]
        M = len(member_probs)

        # 1. Compute Ensemble Distribution and Uncertainty Statistics
        ens_stats = compute_ensemble_statistics(member_probs)
        mean_p_pd = ens_stats["ensemble_mean_probability"]
        mean_p_hc = 1.0 - mean_p_pd

        pred_label = 1 if mean_p_pd >= 0.5 else 0
        pred_name = self.class_names[pred_label]

        # 2. Conformal Prediction Set Construction
        pred_set = construct_prediction_set(mean_p_hc, mean_p_pd, q_hat, self.class_names)
        set_size = len(pred_set)
        set_str = format_prediction_set_str(pred_set)

        # 3. Reliability Signal Evaluation & Flag Reason Detection
        flag_reasons = []

        # Signal A: Conformal Set Status
        if set_size == 2:
            flag_reasons.append("AMBIGUOUS_CONFORMAL_SET")
        elif set_size == 0:
            flag_reasons.append("EMPTY_CONFORMAL_SET")

        # Signal B: Ensemble Agreement
        agreement_score = ens_stats["prediction_agreement"]
        if agreement_score < self.min_agreement_score:
            flag_reasons.append("HIGH_ENSEMBLE_DISAGREEMENT")

        # Signal C: Predictive Entropy
        pred_entropy = ens_stats["predictive_entropy"]
        if pred_entropy >= self.entropy_threshold:
            flag_reasons.append("HIGH_UNCERTAINTY")

        # Signal D: Borderline Decision Boundary Proximity
        lower_bound = 0.5 - self.borderline_margin
        upper_bound = 0.5 + self.borderline_margin
        if lower_bound <= mean_p_pd <= upper_bound:
            flag_reasons.append("BORDERLINE_PROBABILITY")

        # 4. Final Reliability Decision & Categorical Level
        if len(flag_reasons) == 0:
            reliability = "ACCEPT"
            reliability_level = "HIGH"
            reliability_reasons = ["CONSISTENT_SINGLETON_PREDICTION"]
        else:
            reliability = "FLAG"
            reliability_reasons = flag_reasons
            if len(flag_reasons) == 1:
                reliability_level = "MEDIUM"
            else:
                reliability_level = "LOW"

        # 5. Assemble Structured Result Dict
        audio_meta = audio_meta or {}
        result = {
            "filename": audio_meta.get("filename", "unknown_audio.wav"),
            "duration_sec": audio_meta.get("duration_sec", 0.0),
            "num_chunks": audio_meta.get("num_chunks", 1),
            "point_prediction": pred_name,
            "predicted_label": pred_label,
            "p_pd": mean_p_pd,
            "p_healthy": mean_p_hc,
            "ensemble_member_probabilities": member_probs,
            "ensemble_num_pd": ens_stats["num_members_pd"],
            "ensemble_num_hc": ens_stats["num_members_hc"],
            "ensemble_agreement": ens_stats["member_agreement"],
            "ensemble_agreement_score": agreement_score,
            "ensemble_variance": ens_stats["ensemble_variance"],
            "predictive_entropy": pred_entropy,
            "expected_entropy": ens_stats["expected_entropy"],
            "mutual_information": ens_stats["mutual_information"],
            "conformal_prediction_set": sorted(list(pred_set)),
            "conformal_prediction_set_str": set_str,
            "conformal_set_size": set_size,
            "conformal_q_hat": q_hat,
            "reliability": reliability,
            "reliability_level": reliability_level,
            "reliability_reasons": reliability_reasons
        }
        return result


def evaluate_subject_reliability(
    member_probabilities: List[float],
    audio_meta: Optional[Dict] = None,
    q_hat: float = DEFAULT_CONFORMAL_Q_HAT
) -> Dict:
    """Convenience functional interface for VoiceReliabilityEngine."""
    engine = VoiceReliabilityEngine(q_hat=q_hat)
    return engine.evaluate_reliability(member_probabilities, audio_meta=audio_meta)
