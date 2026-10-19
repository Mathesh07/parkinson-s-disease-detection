"""Decision-Level Tri-Modal Cross-Modal Reliability Fusion Engine for Parkinson's Disease Detection.

Integrates independent reliability outputs from Gait, Voice, and Handwriting pipelines into a single,
evidence-consistent decision framework.

Mission:
    Evaluates evidence consistency across machine learning predictions from:
    1. Gait (VGRF time-series / 1D-CNN + BiLSTM / Deep Ensemble M=5)
    2. Voice (16 kHz WAV audio / Wav2Vec2 / Deep Ensemble M=5)
    3. Handwriting (Drawing strokes / Vision Transformer / Evidential Dirichlet Learning)

Policy:
    - CASE A (STRONG AGREEMENT / TRI-MODAL CONSENSUS):
      All available modalities agree, all report HIGH reliability (ACCEPT), and conformal prediction sets concur.
      => Status: CONSISTENT | Level: HIGH | Decision: ACCEPT | Reason: TRI_MODAL_AGREEMENT (or CROSS_MODAL_AGREEMENT)
    
    - CASE B (AGREEMENT BUT ONE OR MORE UNCERTAIN):
      Modalities agree on class, but one or more report CAUTION / UNRELIABLE or ambiguous conformal sets.
      => Status: PARTIAL_SUPPORT | Level: MEDIUM | Decision: FLAG | Reason: ONE_OR_MORE_MODALITIES_UNCERTAIN
    
    - CASE C (DIRECT CONFLICT / DISAGREEMENT):
      Disagreement exists among modalities (e.g. 2 vs 1 vote, or pairwise conflict).
      => Status: MODALITY_CONFLICT | Level: LOW | Decision: FLAG | Reason: MODALITY_DISAGREEMENT
    
    - CASE D (ALL UNCERTAIN):
      All available modalities report excessive uncertainty / UNRELIABLE status.
      => Status: INSUFFICIENT_EVIDENCE | Level: LOW | Decision: FLAG | Reason: ALL_MODALITIES_UNRELIABLE
    
    - CASE E (SINGLE MODALITY):
      Only one modality available.
      => Status: SINGLE_MODALITY | Decision: Inherited | Reason: SINGLE_MODALITY_EVIDENCE

Disclaimer:
    Evaluates evidence consistency and machine learning prediction reliability only.
    Does NOT provide medical diagnosis or clinical patient assessment.
"""

import json
from typing import Dict, List, Optional, Set, Tuple, Union


class CrossModalReliabilityEngine:
    """Decision-level fusion engine evaluating evidence consistency across Gait, Voice, and Handwriting modalities."""

    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names or ["Healthy Control", "Parkinson's Disease"]

    def _normalize_modality_input(self, mod_dict: Optional[Dict], mod_name: str) -> Optional[Dict]:
        """Normalize Gait, Voice, or Handwriting modality output dictionary into standard schema."""
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
        
        pred_label = int(pred_label)
        pred_name = self.class_names[pred_label]

        p_pd = float(mod_dict.get("p_pd", mod_dict.get("P_PD", mod_dict.get("pd_probability", 0.5))))
        p_hc = float(mod_dict.get("p_healthy", mod_dict.get("P_HC", 1.0 - p_pd)))

        # Uncertainty metric (entropy or Dirichlet vacuity)
        uncertainty = float(mod_dict.get("uncertainty", mod_dict.get("predictive_entropy", 0.0)))

        # Conformal set extraction
        conf_set = mod_dict.get("conformal_prediction_set", mod_dict.get("conformal_set", None))
        if conf_set is None:
            # Construct default conformal set based on calibrated probabilities if omitted
            if uncertainty < 0.45 and (p_pd >= 0.70 or p_hc >= 0.70):
                conf_set = [pred_name]
            elif uncertainty >= 0.60:
                conf_set = ["Healthy Control", "Parkinson's Disease"]
            else:
                conf_set = [pred_name]
        elif isinstance(conf_set, str):
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

        reasons = mod_dict.get("reliability_reasons", mod_dict.get("flag_reasons", []))

        return {
            "modality": mod_name,
            "prediction": pred_name,
            "predicted_label": int(pred_label),
            "p_pd": float(round(p_pd, 4)),
            "p_healthy": float(round(p_hc, 4)),
            "uncertainty": float(round(uncertainty, 4)),
            "conformal_set": conf_set_list,
            "conformal_set_size": int(conf_set_size),
            "reliability": reliability,
            "reliability_level": level,
            "reasons": reasons
        }

    def _determine_conformal_status(self, active_modalities: List[Dict]) -> str:
        """Determine conformal prediction set agreement status across 2 or 3 active modalities."""
        if len(active_modalities) <= 1:
            return "SINGLE_MODALITY_CONFORMAL"

        sets = [set(m["conformal_set"]) for m in active_modalities]

        # Any empty conformal set -> uncertain
        if any(len(s) == 0 for s in sets):
            return "UNCERTAIN_CONFORMAL_EVIDENCE"

        # Check if all sets are identical
        first_set = sets[0]
        if all(s == first_set for s in sets):
            if len(first_set) == 1:
                return "CONFORMAL_AGREEMENT"
            else:
                return "AMBIGUOUS_CONFORMAL_AGREEMENT"

        # Check intersection
        common = set.intersection(*sets)
        if len(common) > 0:
            return "PARTIAL_CONFORMAL_AGREEMENT"

        return "CONFORMAL_CONFLICT"

    def _fuse_probabilities(self, active_modalities: List[Dict]) -> Tuple[float, float]:
        """Compute uncertainty-weighted fused probability across active modalities."""
        weights = []
        p_pd_values = []
        for m in active_modalities:
            u = max(float(m.get("uncertainty", 0.3)), 0.01)
            # Uncertainty-weighted fusion: lower uncertainty -> higher weight
            w = 1.0 / (u + 0.05)
            weights.append(w)
            p_pd_values.append(float(m.get("p_pd", 0.5)))

        total_weight = sum(weights)
        if total_weight <= 0:
            fused_p_pd = float(sum(p_pd_values) / len(p_pd_values))
        else:
            fused_p_pd = float(sum(w * p for w, p in zip(weights, p_pd_values)) / total_weight)

        fused_p_pd = max(0.0, min(1.0, fused_p_pd))
        fused_p_hc = 1.0 - fused_p_pd
        return round(fused_p_pd, 4), round(fused_p_hc, 4)

    def fuse_modalities(
        self,
        gait_output: Optional[Dict] = None,
        voice_output: Optional[Dict] = None,
        handwriting_output: Optional[Dict] = None,
        **kwargs
    ) -> Dict:
        """
        Perform decision-level fusion of Gait, Voice, and Handwriting reliability outputs.
        
        Args:
            gait_output: Structured dictionary from Gait reliability engine or saved results.
            voice_output: Structured dictionary from Voice reliability engine or saved results.
            handwriting_output: Structured dictionary from Handwriting reliability engine or saved results.
            
        Returns:
            JSON-serializable dictionary with fusion status, final prediction, decision, and evidence summary.
        """
        gait = self._normalize_modality_input(gait_output, "Gait")
        voice = self._normalize_modality_input(voice_output, "Voice")
        handwriting = self._normalize_modality_input(handwriting_output, "Handwriting")

        active_mods = {}
        if gait is not None:
            active_mods["gait"] = gait
        if voice is not None:
            active_mods["voice"] = voice
        if handwriting is not None:
            active_mods["handwriting"] = handwriting

        num_active = len(active_mods)
        if num_active == 0:
            raise ValueError("At least one modality (Gait, Voice, or Handwriting) output must be provided for fusion.")

        active_list = list(active_mods.values())

        # =====================================================================
        # CASE E: SINGLE MODALITY FALLBACK (Only 1 active modality)
        # =====================================================================
        if num_active == 1:
            mod_key, mod = list(active_mods.items())[0]
            summary = [
                f"{mod['modality']} prediction: {mod['prediction']} (P(PD) = {mod['p_pd']*100:.1f}%, uncertainty = {mod['uncertainty']:.3f}).",
                f"Other modalities unavailable.",
                "Cross-modal agreement evaluation skipped."
            ]
            return {
                "gait": gait,
                "voice": voice,
                "handwriting": handwriting,
                "active_modalities": [mod["modality"]],
                "num_modalities": 1,
                "prediction_agreement": None,
                "majority_agreement": True,
                "fused_prediction": mod["prediction"],
                "predicted_label": mod["predicted_label"],
                "fused_p_pd": mod["p_pd"],
                "fused_p_healthy": mod["p_healthy"],
                "conformal_agreement_status": "SINGLE_MODALITY_CONFORMAL",
                "cross_modal_status": "SINGLE_MODALITY",
                "cross_modal_reliability": mod["reliability_level"],
                "final_decision": mod["reliability"],
                "reason": "SINGLE_MODALITY_EVIDENCE",
                "evidence_summary": summary
            }

        # Calculate Fused Probability across active modalities
        fused_p_pd, fused_p_hc = self._fuse_probabilities(active_list)
        conformal_status = self._determine_conformal_status(active_list)

        pred_labels = [m["predicted_label"] for m in active_list]
        pd_votes = sum(1 for l in pred_labels if l == 1)
        hc_votes = num_active - pd_votes

        # Majority Vote Prediction
        if pd_votes > hc_votes:
            fused_label = 1
            fused_prediction = self.class_names[1]
        elif hc_votes > pd_votes:
            fused_label = 0
            fused_prediction = self.class_names[0]
        else:
            # Tie break by fused probability
            fused_label = 1 if fused_p_pd >= 0.5 else 0
            fused_prediction = self.class_names[fused_label]

        unanimous = (pd_votes == num_active or hc_votes == num_active)
        all_reliable = all(m["reliability"] == "ACCEPT" and m["reliability_level"] == "HIGH" for m in active_list)
        all_unreliable = all(m["reliability"] == "FLAG" and m["reliability_level"] in ["LOW", "UNRELIABLE"] for m in active_list)

        summary = []
        for m in active_list:
            summary.append(
                f"{m['modality']} predicts {m['prediction']} "
                f"(P(PD) = {m['p_pd']*100:.1f}%, uncertainty = {m['uncertainty']:.3f}, {m['reliability']}/{m['reliability_level']})."
            )

        # =====================================================================
        # TWO MODALITIES AVAILABLE (e.g. Gait + Voice, Gait + Handwriting, Voice + Handwriting)
        # =====================================================================
        if num_active == 2:
            pred_agree = (active_list[0]["predicted_label"] == active_list[1]["predicted_label"])
            mod_names = [m["modality"] for m in active_list]

            if not pred_agree:
                # Case C: Direct Modality Conflict
                status = "MODALITY_CONFLICT"
                reliability_level = "LOW"
                final_decision = "FLAG"
                if "Gait" in mod_names and "Voice" in mod_names:
                    reason = "GAIT_VOICE_CONFLICT"
                else:
                    reason = f"{mod_names[0].upper()}_{mod_names[1].upper()}_CONFLICT"
                summary.append(f"Direct prediction disagreement detected between {mod_names[0]} and {mod_names[1]}.")
                summary.append("Prediction flagged due to cross-modal evidence conflict.")

            elif all_unreliable:
                # Case D: Both Modalities Uncertain
                status = "INSUFFICIENT_EVIDENCE"
                reliability_level = "LOW"
                final_decision = "FLAG"
                reason = "BOTH_MODALITIES_UNRELIABLE"
                summary.append(f"Both modalities predict {fused_prediction}, but both report UNRELIABLE predictions.")
                summary.append("Excessive uncertainty across both modalities.")

            elif all_reliable and conformal_status == "CONFORMAL_AGREEMENT":
                # Case A: Strong Agreement
                status = "CONSISTENT"
                reliability_level = "HIGH"
                final_decision = "ACCEPT"
                reason = "CROSS_MODAL_AGREEMENT"
                summary.append(f"{mod_names[0]} and {mod_names[1]} predictions concur on {fused_prediction}.")
                summary.append("Both modalities report HIGH reliability with matching conformal sets.")

            else:
                # Case B: Agreement but One Modality Uncertain
                status = "PARTIAL_SUPPORT"
                reliability_level = "MEDIUM"
                final_decision = "FLAG"
                reason = "ONE_MODALITY_UNCERTAIN"
                summary.append(f"Modalities concur on {fused_prediction}, but one or more report caution/uncertainty.")
                summary.append(f"Conformal agreement status: {conformal_status}.")

            summary.append(f"Fused Probability: P(PD) = {fused_p_pd*100:.1f}%, P(HC) = {fused_p_hc*100:.1f}%.")

            return {
                "gait": gait,
                "voice": voice,
                "handwriting": handwriting,
                "active_modalities": mod_names,
                "num_modalities": 2,
                "prediction_agreement": bool(pred_agree),
                "majority_agreement": True,
                "fused_prediction": fused_prediction,
                "predicted_label": int(fused_label),
                "fused_p_pd": fused_p_pd,
                "fused_p_healthy": fused_p_hc,
                "conformal_agreement_status": conformal_status,
                "cross_modal_status": status,
                "cross_modal_reliability": reliability_level,
                "final_decision": final_decision,
                "reason": reason,
                "evidence_summary": summary
            }

        # =====================================================================
        # TRI-MODAL FUSION (All 3 Modalities: Gait + Voice + Handwriting)
        # =====================================================================
        mod_names = ["Gait", "Voice", "Handwriting"]

        if unanimous:
            summary.append(f"Tri-modal consensus: All 3 modalities agree on {fused_prediction} ({pd_votes}/3 PD, {hc_votes}/3 Healthy).")
            
            if all_reliable and conformal_status == "CONFORMAL_AGREEMENT":
                # CASE A: Strong Tri-Modal Agreement
                status = "CONSISTENT"
                reliability_level = "HIGH"
                final_decision = "ACCEPT"
                reason = "TRI_MODAL_AGREEMENT"
                summary.append("All 3 modalities report HIGH reliability with matching conformal prediction sets.")
                summary.append(f"Tri-modal fused decision ACCEPTED for {fused_prediction}.")
            elif all_unreliable:
                # CASE D: All Modalities Uncertain
                status = "INSUFFICIENT_EVIDENCE"
                reliability_level = "LOW"
                final_decision = "FLAG"
                reason = "ALL_MODALITIES_UNRELIABLE"
                summary.append("All 3 modalities predict the same class but report excessive uncertainty.")
                summary.append("Flagged due to insufficient evidence across the entire tri-modal pipeline.")
            else:
                # CASE B: Unanimous but 1 or 2 modalities uncertain
                status = "PARTIAL_SUPPORT"
                reliability_level = "MEDIUM"
                final_decision = "FLAG"
                reason = "ONE_OR_MORE_MODALITIES_UNCERTAIN"
                summary.append(f"All 3 modalities predict {fused_prediction}, but one or more report caution/uncertainty.")
                summary.append(f"Conformal status: {conformal_status}. Flagged for clinical verification.")
        else:
            # CASE C: 2 vs 1 Conflict / Disagreement
            # Identify dissenting modality
            dissenting = [m for m in active_list if m["predicted_label"] != fused_label]
            dissenting_names = ", ".join(m["modality"] for m in dissenting)
            agreeing_names = ", ".join(m["modality"] for m in active_list if m["predicted_label"] == fused_label)

            status = "MODALITY_CONFLICT"
            reliability_level = "LOW"
            final_decision = "FLAG"
            reason = "MODALITY_DISAGREEMENT_2_VS_1"

            summary.append(f"Modality disagreement detected: {agreeing_names} predict {fused_prediction}, whereas {dissenting_names} dissents.")
            summary.append(f"Majority vote points to {fused_prediction} ({max(pd_votes, hc_votes)}/3 modalities), but conflict requires manual clinical review.")
            summary.append("Automated acceptance rejected due to cross-modal conflict.")

        summary.append(f"Fused Tri-Modal Probability: P(PD) = {fused_p_pd*100:.1f}%, P(HC) = {fused_p_hc*100:.1f}%.")

        return {
            "gait": gait,
            "voice": voice,
            "handwriting": handwriting,
            "active_modalities": mod_names,
            "num_modalities": 3,
            "prediction_agreement": bool(unanimous),
            "majority_agreement": True,
            "fused_prediction": fused_prediction,
            "predicted_label": int(fused_label),
            "fused_p_pd": fused_p_pd,
            "fused_p_healthy": fused_p_hc,
            "conformal_agreement_status": conformal_status,
            "cross_modal_status": status,
            "cross_modal_reliability": reliability_level,
            "final_decision": final_decision,
            "reason": reason,
            "evidence_summary": summary
        }


def fuse_cross_modal_reliability(
    gait_output: Optional[Dict] = None,
    voice_output: Optional[Dict] = None,
    handwriting_output: Optional[Dict] = None,
    **kwargs
) -> Dict:
    """Convenience function for CrossModalReliabilityEngine.fuse_modalities."""
    engine = CrossModalReliabilityEngine()
    return engine.fuse_modalities(
        gait_output=gait_output,
        voice_output=voice_output,
        handwriting_output=handwriting_output,
        **kwargs
    )
