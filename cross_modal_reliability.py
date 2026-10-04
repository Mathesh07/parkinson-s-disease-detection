"""Decision-Level Cross-Modal Reliability Fusion Engine for Parkinson's Disease Detection.

Integrates independent reliability outputs from the Gait and Voice pipelines into a single,
evidence-consistent decision framework.

Mission:
    Answers: "Do Gait and Voice provide consistent and reliable evidence, or do the modalities disagree?"
    
Policy:
    - CASE A (STRONG AGREEMENT): Predictions agree, both modalities RELIABLE, conformal sets concur.
      => Status: CONSISTENT | Level: HIGH | Decision: ACCEPT | Reason: CROSS_MODAL_AGREEMENT
    - CASE B (AGREEMENT BUT ONE UNCERTAIN): Predictions agree, but one modality is CAUTION or UNRELIABLE.
      => Status: PARTIAL_SUPPORT | Level: MEDIUM | Decision: FLAG | Reason: ONE_MODALITY_UNCERTAIN
    - CASE C (DIRECT CONFLICT): Gait prediction != Voice prediction.
      => Status: MODALITY_CONFLICT | Level: LOW | Decision: FLAG | Reason: GAIT_VOICE_CONFLICT
    - CASE D (BOTH UNCERTAIN): Both modalities UNRELIABLE.
      => Status: INSUFFICIENT_EVIDENCE | Level: LOW | Decision: FLAG | Reason: BOTH_MODALITIES_UNRELIABLE
    - CASE E (SINGLE MODALITY): Only Gait or Voice available.
      => Status: SINGLE_MODALITY | Decision: Inherited | Reason: SINGLE_MODALITY_EVIDENCE

Disclaimer:
    Evaluates evidence consistency across machine learning predictions.
    Does NOT provide medical diagnosis or clinical patient assessment.
"""

import json
from typing import Dict, List, Optional, Set, Tuple, Union


class CrossModalReliabilityEngine:
    """Decision-level fusion engine evaluating evidence consistency across Gait and Voice modalities."""

    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names or ["Healthy Control", "Parkinson's Disease"]

    def _normalize_modality_input(self, mod_dict: Optional[Dict], mod_name: str) -> Optional[Dict]:
        """Normalize Gait or Voice modality output dictionary into standard schema."""
        if not mod_dict:
            return None

        # Standardize prediction string and binary label
        pred_label = mod_dict.get("predicted_label", None)
        if pred_label is None:
            point_pred = str(mod_dict.get("point_prediction", mod_dict.get("prediction", "")))
            if "PD" in point_pred or "Parkinson" in point_pred:
                pred_label = 1
            else:
                pred_label = 0
        
        pred_name = self.class_names[pred_label]

        p_pd = float(mod_dict.get("p_pd", mod_dict.get("P_PD", mod_dict.get("pd_probability", 0.5))))
        p_hc = float(mod_dict.get("p_healthy", mod_dict.get("P_HC", 1.0 - p_pd)))

        # Conformal set extraction
        conf_set = mod_dict.get("conformal_prediction_set", mod_dict.get("conformal_set", []))
        if isinstance(conf_set, str):
            if "Healthy" in conf_set and "PD" in conf_set:
                conf_set = ["Healthy Control", "Parkinson's Disease"]
            elif "PD" in conf_set or "Parkinson" in conf_set:
                conf_set = ["Parkinson's Disease"]
            elif "Healthy" in conf_set:
                conf_set = ["Healthy Control"]
            else:
                conf_set = []

        conf_set_list = sorted(list(conf_set))
        conf_set_size = len(conf_set_list)

        # Modality reliability decision & level
        reliability = str(mod_dict.get("reliability", mod_dict.get("reliability_status", "FLAG"))).upper()
        level = str(mod_dict.get("reliability_level", mod_dict.get("confidence_level", "MEDIUM"))).upper()

        # Uncertainty metric
        uncertainty = float(mod_dict.get("predictive_entropy", mod_dict.get("uncertainty", 0.0)))

        return {
            "modality": mod_name,
            "prediction": pred_name,
            "predicted_label": int(pred_label),
            "p_pd": p_pd,
            "p_healthy": p_hc,
            "uncertainty": uncertainty,
            "conformal_set": conf_set_list,
            "conformal_set_size": conf_set_size,
            "reliability": reliability,
            "reliability_level": level,
            "reasons": mod_dict.get("reliability_reasons", mod_dict.get("flag_reasons", []))
        }

    def _determine_conformal_status(self, gait: Optional[Dict], voice: Optional[Dict]) -> str:
        """Determine conformal prediction set agreement status."""
        if not gait or not voice:
            return "SINGLE_MODALITY_CONFORMAL"

        g_set = set(gait["conformal_set"])
        v_set = set(voice["conformal_set"])

        if len(g_set) == 0 or len(v_set) == 0:
            return "UNCERTAIN_CONFORMAL_EVIDENCE"

        if g_set == v_set:
            if len(g_set) == 1:
                return "CONFORMAL_AGREEMENT"
            else:
                return "AMBIGUOUS_CONFORMAL_AGREEMENT"

        intersection = g_set & v_set
        if len(intersection) > 0:
            return "PARTIAL_CONFORMAL_AGREEMENT"

        return "CONFORMAL_CONFLICT"

    def fuse_modalities(
        self,
        gait_output: Optional[Dict] = None,
        voice_output: Optional[Dict] = None
    ) -> Dict:
        """
        Perform decision-level fusion of Gait and Voice reliability engine outputs.
        
        Args:
            gait_output: Structured dictionary from Gait reliability engine or saved results.
            voice_output: Structured dictionary from Voice reliability engine or saved results.
            
        Returns:
            JSON-serializable dictionary with fusion status, final decision, and evidence summary.
        """
        gait = self._normalize_modality_input(gait_output, "Gait")
        voice = self._normalize_modality_input(voice_output, "Voice")

        # CASE E: Only one modality available
        if gait is not None and voice is None:
            return {
                "gait": gait,
                "voice": None,
                "prediction_agreement": None,
                "conformal_agreement_status": "SINGLE_MODALITY_CONFORMAL",
                "cross_modal_status": "SINGLE_MODALITY",
                "cross_modal_reliability": gait["reliability_level"],
                "final_decision": gait["reliability"],
                "reason": "SINGLE_MODALITY_EVIDENCE",
                "evidence_summary": [
                    f"Gait prediction: {gait['prediction']} (P(PD) = {gait['p_pd']*100:.1f}%).",
                    "Voice modality data unavailable.",
                    "Cross-modal agreement evaluation skipped."
                ]
            }

        if voice is not None and gait is None:
            return {
                "gait": None,
                "voice": voice,
                "prediction_agreement": None,
                "conformal_agreement_status": "SINGLE_MODALITY_CONFORMAL",
                "cross_modal_status": "SINGLE_MODALITY",
                "cross_modal_reliability": voice["reliability_level"],
                "final_decision": voice["reliability"],
                "reason": "SINGLE_MODALITY_EVIDENCE",
                "evidence_summary": [
                    f"Voice prediction: {voice['prediction']} (P(PD) = {voice['p_pd']*100:.1f}%).",
                    "Gait modality data unavailable.",
                    "Cross-modal agreement evaluation skipped."
                ]
            }

        if gait is None and voice is None:
            raise ValueError("At least one modality (Gait or Voice) output must be provided for fusion.")

        # Both modalities available
        pred_agree = (gait["predicted_label"] == voice["predicted_label"])
        conformal_status = self._determine_conformal_status(gait, voice)

        gait_reliable = (gait["reliability"] == "ACCEPT" and gait["reliability_level"] == "HIGH")
        voice_reliable = (voice["reliability"] == "ACCEPT" and voice["reliability_level"] == "HIGH")

        gait_unreliable = (gait["reliability"] == "FLAG" and gait["reliability_level"] in ["LOW", "UNRELIABLE"])
        voice_unreliable = (voice["reliability"] == "FLAG" and voice["reliability_level"] in ["LOW", "UNRELIABLE"])

        summary = []

        # Decision Policy Logic
        if not pred_agree:
            # CASE C: Direct Modality Conflict
            status = "MODALITY_CONFLICT"
            reliability_level = "LOW"
            final_decision = "FLAG"
            reason = "GAIT_VOICE_CONFLICT"
            summary.append(f"Gait predicts {gait['prediction']} (P={gait['p_pd']*100:.1f}%), whereas Voice predicts {voice['prediction']} (P={voice['p_pd']*100:.1f}%).")
            summary.append("Direct prediction disagreement detected between modalities.")
            summary.append("Prediction flagged due to cross-modal evidence conflict.")

        elif gait_unreliable and voice_unreliable:
            # CASE D: Both Modalities Uncertain
            status = "INSUFFICIENT_EVIDENCE"
            reliability_level = "LOW"
            final_decision = "FLAG"
            reason = "BOTH_MODALITIES_UNRELIABLE"
            summary.append(f"Both Gait and Voice predict {gait['prediction']}, but both modalities report UNRELIABLE predictions.")
            summary.append("Excessive uncertainty across both modalities.")

        elif gait_reliable and voice_reliable and conformal_status == "CONFORMAL_AGREEMENT":
            # CASE A: Strong Agreement
            status = "CONSISTENT"
            reliability_level = "HIGH"
            final_decision = "ACCEPT"
            reason = "CROSS_MODAL_AGREEMENT"
            summary.append(f"Gait and Voice predictions concur on {gait['prediction']}.")
            summary.append("Both modalities report HIGH reliability.")
            summary.append(f"Conformal prediction sets concur on {gait['conformal_set']}.")

        else:
            # CASE B: Agreement but One Modality Uncertain / Partial Support
            status = "PARTIAL_SUPPORT"
            reliability_level = "MEDIUM"
            final_decision = "FLAG"
            reason = "ONE_MODALITY_UNCERTAIN"
            summary.append(f"Gait and Voice predictions concur on {gait['prediction']}.")
            if not gait_reliable:
                summary.append(f"Gait reliability is {gait['reliability_level']} (Flagged).")
            if not voice_reliable:
                summary.append(f"Voice reliability is {voice['reliability_level']} (Flagged).")
            summary.append(f"Conformal status: {conformal_status}.")

        return {
            "gait": gait,
            "voice": voice,
            "prediction_agreement": pred_agree,
            "conformal_agreement_status": conformal_status,
            "cross_modal_status": status,
            "cross_modal_reliability": reliability_level,
            "final_decision": final_decision,
            "reason": reason,
            "evidence_summary": summary
        }


def fuse_cross_modal_reliability(
    gait_output: Optional[Dict] = None,
    voice_output: Optional[Dict] = None
) -> Dict:
    """Convenience function for CrossModalReliabilityEngine.fuse_modalities."""
    engine = CrossModalReliabilityEngine()
    return engine.fuse_modalities(gait_output, voice_output)
