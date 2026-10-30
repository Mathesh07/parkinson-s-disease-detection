"""Handwriting Reliability Engine for Parkinson's Disease Detection.

Integrates Vision Transformer (ViT) Evidential Deep Learning, Dirichlet uncertainty estimation,
and Split Conformal Prediction into a unified, deterministic reliability decision system.

Mission:
    Evaluates whether a handwriting drawing prediction (Spiral/Meander) is sufficiently supported
    by low Dirichlet uncertainty (vacuity) and conformal evidence before accepting the decision.

Decision Policy:
    - ACCEPT (HIGH): Conformal set is singleton, vacuity uncertainty is low (< 0.45),
      and probability is clearly separated from decision boundary.
    - FLAG (MEDIUM / LOW): Triggered when conformal set is ambiguous or empty,
      predictive uncertainty is elevated (>= 0.45), or probability is near boundary [0.40, 0.60].

Disclaimer:
    Evaluates machine learning model prediction reliability only.
    Does NOT provide medical diagnosis or clinical patient assessment.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    from PIL import Image
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


DEFAULT_HANDWRITING_CONFORMAL_Q_HAT = 0.450


class HandwritingReliabilityEngine:
    """Inference-time decision engine for Handwriting Parkinson's classification reliability."""

    def __init__(
        self,
        q_hat: float = DEFAULT_HANDWRITING_CONFORMAL_Q_HAT,
        uncertainty_threshold: float = 0.45,
        borderline_margin: float = 0.10,
        class_names: Optional[List[str]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize Handwriting Reliability Engine with pre-declared heuristic thresholds.
        
        Args:
            q_hat: Calibrated conformal quantile threshold (default 0.450).
            uncertainty_threshold: Maximum acceptable Dirichlet uncertainty for ACCEPT (default 0.45).
            borderline_margin: Margin around 0.5 decision boundary considered borderline (default 0.10 -> [0.40, 0.60]).
            class_names: Binary class names list (default ["Healthy Control", "Parkinson's Disease"]).
            checkpoint_path: Optional path to evidential ViT checkpoint.
        """
        self.q_hat = float(q_hat)
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.borderline_margin = float(borderline_margin)
        self.class_names = class_names or ["Healthy Control", "Parkinson's Disease"]
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._model = None

    def construct_prediction_set(
        self,
        p_hc: float,
        p_pd: float,
        q_hat: Optional[float] = None
    ) -> List[str]:
        """Construct split conformal prediction set using softmax/Dirichlet probabilities."""
        q = float(q_hat if q_hat is not None else self.q_hat)
        prediction_set = []
        # Conformal non-conformity score: s_k = 1 - p_k <= q  <=>  p_k >= 1 - q
        threshold = 1.0 - q
        if p_hc >= threshold:
            prediction_set.append(self.class_names[0])
        if p_pd >= threshold:
            prediction_set.append(self.class_names[1])
        return sorted(prediction_set)

    def evaluate_reliability(
        self,
        p_pd: float,
        uncertainty: float,
        evidence: Optional[Tuple[float, float]] = None,
        image_meta: Optional[Dict] = None,
        q_hat_override: Optional[float] = None
    ) -> Dict:
        """
        Evaluate full reliability decision for handwriting drawing given probability and uncertainty.
        
        Args:
            p_pd: Calibrated probability P(PD) in [0, 1].
            uncertainty: Evidential Dirichlet uncertainty (vacuity u = K/S) in [0, 1].
            evidence: Optional tuple (e_HC, e_PD) >= 0.
            image_meta: Optional metadata dict (filename, task, patient_id).
            q_hat_override: Optional override for conformal quantile.
            
        Returns:
            JSON-serializable result dictionary with prediction, uncertainty, conformal set,
            reliability decision (ACCEPT/FLAG), level (HIGH/MEDIUM/LOW), and explanation reasons.
        """
        p_pd = float(np.clip(p_pd, 0.0, 1.0))
        p_hc = float(1.0 - p_pd)
        uncertainty = float(np.clip(uncertainty, 0.0, 1.0))
        q_hat = float(q_hat_override if q_hat_override is not None else self.q_hat)

        pred_label = 1 if p_pd >= 0.5 else 0
        pred_name = self.class_names[pred_label]

        # 1. Conformal Prediction Set Construction
        pred_set = self.construct_prediction_set(p_hc, p_pd, q_hat)
        set_size = len(pred_set)

        # 2. Reliability Flag Checks
        flag_reasons = []

        # Signal A: Conformal Set Status
        if set_size == 2:
            flag_reasons.append("AMBIGUOUS_CONFORMAL_SET")
        elif set_size == 0:
            flag_reasons.append("EMPTY_CONFORMAL_SET")

        # Signal B: Evidential Dirichlet Uncertainty (Vacuity)
        if uncertainty >= self.uncertainty_threshold:
            flag_reasons.append("HIGH_EVIDENTIAL_UNCERTAINTY")

        # Signal C: Borderline Decision Boundary Proximity
        lower_bound = 0.5 - self.borderline_margin
        upper_bound = 0.5 + self.borderline_margin
        if lower_bound <= p_pd <= upper_bound:
            flag_reasons.append("BORDERLINE_PROBABILITY")

        # 3. Final Reliability Decision & Categorical Level
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

        # 4. Assemble Result Dict
        image_meta = image_meta or {}
        return {
            "modality": "Handwriting",
            "prediction": pred_name,
            "predicted_label": int(pred_label),
            "p_pd": float(round(p_pd, 4)),
            "p_healthy": float(round(p_hc, 4)),
            "uncertainty": float(round(uncertainty, 4)),
            "evidence": [float(round(e, 4)) for e in evidence] if evidence is not None else None,
            "conformal_prediction_set": pred_set,
            "conformal_set_size": int(set_size),
            "conformal_q_hat": float(round(q_hat, 4)),
            "reliability": reliability,
            "reliability_level": reliability_level,
            "reliability_reasons": reliability_reasons,
            "flag_reasons": flag_reasons,
            "image_meta": image_meta
        }

    def predict_image(
        self,
        image_path: Union[str, Path],
        device: Optional[str] = None
    ) -> Dict:
        """
        Run end-to-end inference & evidential reliability evaluation on a handwriting image file.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch and torchvision are required for live image inference.")

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Handwriting image file not found: {image_path}")

        # Lazy load model
        if self._model is None:
            from Handwriting.model import ViTBinaryClassifier
            from Handwriting.config import OUTPUT_DIR

            ckpt = self.checkpoint_path or (OUTPUT_DIR / "best_vit_evidential.pth")
            dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._dev = torch.device(dev)

            model = ViTBinaryClassifier(freeze_backbone=False)
            if ckpt.exists():
                model.load_state_dict(torch.load(ckpt, map_location=self._dev, weights_only=True))
            model.to(self._dev)
            model.eval()
            self._model = model

        # Preprocess Image
        img = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensor_img = transform(img).unsqueeze(0).to(self._dev)

        with torch.no_grad():
            evidence = self._model(tensor_img)  # (1, 2)
            alpha = evidence + 1.0
            S = torch.sum(alpha, dim=-1, keepdim=True)
            probs = alpha / S
            u = 2.0 / S

        p_pd = float(probs[0, 1].item())
        uncertainty = float(u[0, 0].item())
        ev_tuple = (float(evidence[0, 0].item()), float(evidence[0, 1].item()))

        return self.evaluate_reliability(
            p_pd=p_pd,
            uncertainty=uncertainty,
            evidence=ev_tuple,
            image_meta={"filename": image_path.name, "path": str(image_path)}
        )
