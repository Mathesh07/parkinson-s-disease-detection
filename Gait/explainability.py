"""Explainability and Feature Attribution for Gait-Based Parkinson's Disease Detection.

This module implements two complementary explainability methods for continuous VGRF time-series:
1. Integrated Gradients (IG):
   - Path-integral feature attribution (Sundararajan et al., 2017) across the 16 VGRF channels and 500 time steps.
   - Computes exact per-sensor importance (L1-L8 Left foot vs R1-R8 Right foot) and bilateral asymmetry indices.
2. 1D Grad-CAM:
   - Gradient-weighted Class Activation Mapping hooked to the final 1D CNN feature layer.
   - Identifies temporal gait-cycle phases (heel strike, mid-stance, toe-off) that triggered the prediction.
3. Evidential Decision Explanations:
   - Couples attribution heatmaps with evidential belief & epistemic vacuity metrics.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
for p in (str(current_dir), str(parent_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from Gait import config
    from Gait.model import GaitCNNBiLSTM
except ImportError:
    import config
    from model import GaitCNNBiLSTM

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F




def compute_integrated_gradients(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_class: int = 1,
    baseline: Optional[torch.Tensor] = None,
    steps: int = 50
) -> torch.Tensor:
    """Compute Integrated Gradients attribution for 1D VGRF time-series.
    
    Formula:
        IG_i(x) = (x_i - x'_i) * integral_{alpha=0..1} (dF(x' + alpha*(x - x')) / dx_i) d_alpha
        
    Args:
        model: Trained GaitCNNBiLSTM neural network in eval mode.
        x: Input tensor of shape (batch_size, time_steps, 16).
        target_class: Class index for attribution (0: HC, 1: PD).
        baseline: Baseline reference tensor (default: zeros of same shape).
        steps: Number of Riemann interpolation steps along path.
        
    Returns:
        attributions: Tensor of shape (batch_size, time_steps, 16) with signed attribution scores.
    """
    # Put model into training mode with dropout/batchnorm frozen to allow RNN backward with cuDNN
    was_training = model.training
    model.train()
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            m.eval()

    device = x.device
    if baseline is None:
        baseline = torch.zeros_like(x, device=device)

    # Pre-allocate path steps: (steps + 1, batch_size, time_steps, 16)
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=device)
    delta = x - baseline

    # Compute gradients along the path
    accumulated_grads = torch.zeros_like(x, device=device)

    try:
        for alpha in alphas:
            interpolated_x = (baseline + alpha * delta).clone().detach().requires_grad_(True)
            # Forward pass
            evidence = model(interpolated_x)
            # If model outputs 2D evidence (B, 2), select target class score
            if evidence.dim() == 2 and evidence.shape[1] >= 2:
                target_score = evidence[:, target_class].sum()
            else:
                target_score = evidence.squeeze().sum()

            model.zero_grad()
            grads = torch.autograd.grad(target_score, interpolated_x, create_graph=False)[0]
            accumulated_grads += grads
    finally:
        model.train(was_training)

    # Trapezoidal approximation average: (accumulated / (steps + 1)) * (x - baseline)
    avg_grads = accumulated_grads / (steps + 1)
    attributions = delta * avg_grads
    return attributions



class GradCAM1D:
    """1D Grad-CAM implementation hooked to the final Conv1d feature layer."""

    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.model.eval()
        if target_layer is None:
            # Default to conv_block2[0] (Conv1d layer of CNN block 2)
            self.target_layer = self.model.cnn_block2[0]
        else:
            self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, x: torch.Tensor, target_class: int = 1) -> torch.Tensor:
        """Generate 1D Grad-CAM activation map upsampled to input sequence length.
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, 16).
            target_class: Target class index.
            
        Returns:
            cam: 1D normalized activation map of shape (batch_size, time_steps) in [0, 1].
        """
        was_training = self.model.training
        self.model.train()
        for m in self.model.modules():
            if isinstance(m, (torch.nn.Dropout, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                m.eval()

        self.model.zero_grad()
        try:
            evidence = self.model(x)

            if evidence.dim() == 2 and evidence.shape[1] >= 2:
                score = evidence[:, target_class].sum()
            else:
                score = evidence.squeeze().sum()

            score.backward(retain_graph=True)
        finally:
            self.model.train(was_training)


        # Activations shape: (B, Channels, T_conv) e.g., (1, 128, 125)
        # Gradients shape:   (B, Channels, T_conv)
        alpha = torch.mean(self.gradients, dim=2, keepdim=True)  # (B, Channels, 1)
        cam = torch.sum(alpha * self.activations, dim=1, keepdim=True)  # (B, 1, T_conv)
        cam = F.relu(cam)

        # Upsample back to temporal input length (e.g., 500)
        target_len = x.shape[1]
        cam_upsampled = F.interpolate(cam, size=target_len, mode="linear", align_corners=False)
        cam_upsampled = cam_upsampled.squeeze(1)  # (B, time_steps)

        # Min-max normalization per sample
        b_min = cam_upsampled.min(dim=-1, keepdim=True)[0]
        b_max = cam_upsampled.max(dim=-1, keepdim=True)[0]
        norm_cam = (cam_upsampled - b_min) / (b_max - b_min + 1e-8)
        return norm_cam


def summarize_sensor_attribution(attributions: torch.Tensor) -> Dict[str, Union[float, Dict[str, float]]]:
    """Compute per-sensor importance scores and bilateral gait asymmetry index.
    
    Args:
        attributions: Tensor of shape (time_steps, 16) or (1, time_steps, 16).
        
    Returns:
        Dictionary with per-sensor scores, left vs right sum, and Asymmetry Index.
    """
    if attributions.dim() == 3:
        attr_2d = attributions[0].detach().cpu().numpy()
    else:
        attr_2d = attributions.detach().cpu().numpy()

    # Aggregate absolute feature contribution per channel
    channel_importance = np.mean(np.abs(attr_2d), axis=0)  # (16,)

    sensor_dict = {
        name: float(score)
        for name, score in zip(config.SENSOR_NAMES, channel_importance)
    }

    left_sum = float(np.sum(channel_importance[:8]))
    right_sum = float(np.sum(channel_importance[8:]))
    asymmetry_index = float(abs(left_sum - right_sum) / (left_sum + right_sum + 1e-8))

    return {
        "sensor_scores": sensor_dict,
        "left_foot_sum": left_sum,
        "right_foot_sum": right_sum,
        "asymmetry_index": asymmetry_index,
        "dominant_sensor": max(sensor_dict, key=sensor_dict.get)
    }


def visualize_gait_explainability(
    x: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    sample_id: str = "Gait_Sample",
    true_label: Optional[int] = None,
    save_path: Optional[Path] = None
) -> Tuple[plt.Figure, Dict[str, float]]:
    """Generate and save comprehensive 4-panel Gait Explainability Dashboard."""
    model.eval()
    x = x.to(device)
    if x.dim() == 2:
        x = x.unsqueeze(0)

    # 1. Model Forward Pass & Evidential Inference
    with torch.no_grad():
        evidence = model(x)
        if evidence.dim() == 2 and evidence.shape[1] >= 2:
            e_hc = evidence[0, 0].item()
            e_pd = evidence[0, 1].item()
        else:
            e_hc, e_pd = 1.0, float(evidence.item())

        s = e_hc + e_pd + 2.0
        p_hc = (e_hc + 1.0) / s
        p_pd = (e_pd + 1.0) / s
        vacuity = 2.0 / s
        pred_label = 1 if p_pd >= p_hc else 0

    # 2. Compute Integrated Gradients
    ig_attr = compute_integrated_gradients(model, x, target_class=1, steps=40)
    summary = summarize_sensor_attribution(ig_attr)

    # 3. Compute 1D Grad-CAM
    gradcam_engine = GradCAM1D(model)
    cam_1d = gradcam_engine.generate(x, target_class=1)[0].detach().cpu().numpy()

    # Data arrays for plotting
    vgrf_raw = x[0].detach().cpu().numpy()  # (500, 16)
    ig_raw = ig_attr[0].detach().cpu().numpy()  # (500, 16)
    time_sec = np.linspace(0, config.WINDOW_DURATION, config.WINDOW_SIZE)

    # 4. Create Publication-Grade Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor("#FFFFFF")

    # Panel 1: Raw VGRF Signal (Left Foot vs Right Foot Total Force)
    left_total_force = np.sum(vgrf_raw[:, :8], axis=1)
    right_total_force = np.sum(vgrf_raw[:, 8:], axis=1)
    ax1 = axes[0, 0]
    ax1.plot(time_sec, left_total_force, color="#1E88E5", linewidth=2.0, label="Left Foot (L1-L8 Total)")
    ax1.plot(time_sec, right_total_force, color="#D81B60", linewidth=2.0, linestyle="--", label="Right Foot (R1-R8 Total)")
    ax1.set_title("VGRF Continuous Stance Signals (Left vs Right Foot)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Time (seconds)", fontsize=10)
    ax1.set_ylabel("Force (N)", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    # Panel 2: Sensor Attribution Bar Chart (L1..L8 vs R1..R8)
    ax2 = axes[0, 1]
    sensor_names = config.SENSOR_NAMES
    scores = [summary["sensor_scores"][s] for s in sensor_names]
    colors = ["#1E88E5" if s.startswith("L") else "#D81B60" for s in sensor_names]
    bars = ax2.bar(sensor_names, scores, color=colors, edgecolor="black", alpha=0.85)
    ax2.set_title(
        f"Sensor Feature Attribution (IG) | Bilateral Asymmetry Index = {summary['asymmetry_index']:.3f}",
        fontsize=12, fontweight="bold"
    )
    ax2.set_xlabel("VGRF Sensor Channel", fontsize=10)
    ax2.set_ylabel("Mean |Integrated Gradient|", fontsize=10)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
    for bar in bars:
        h = bar.get_height()
        if h > 0.001:
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.0005, f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    # Panel 3: Temporal Attribution Heatmap (Sensors x Time)
    ax3 = axes[1, 0]
    im = ax3.imshow(
        np.abs(ig_raw).T,
        aspect="auto",
        cmap="magma",
        extent=[0, config.WINDOW_DURATION, 16, 0]
    )
    ax3.set_yticks(np.arange(16) + 0.5)
    ax3.set_yticklabels(sensor_names, fontsize=8)
    ax3.set_title("Temporal Attribution Heatmap across 16 Sensors (IG)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Time (seconds)", fontsize=10)
    ax3.set_ylabel("Sensors", fontsize=10)
    plt.colorbar(im, ax=ax3, orientation="vertical", label="|Attribution|", pad=0.02)

    # Panel 4: 1D Grad-CAM Temporal Profile & Evidential Reliability Readout
    ax4 = axes[1, 1]
    ax4.plot(time_sec, cam_1d, color="#FF6F00", linewidth=2.5, label="1D Grad-CAM Activation")
    ax4.fill_between(time_sec, cam_1d, color="#FF6F00", alpha=0.25)
    ax4.set_title("1D Grad-CAM Temporal Activation Profile & Decision Confidence", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Time (seconds)", fontsize=10)
    ax4.set_ylabel("Activation Intensity [0, 1]", fontsize=10)
    ax4.grid(True, linestyle=":", alpha=0.6)

    # Annotation Box with Evidential & Confidence Metrics
    status_str = "Parkinson's Disease (PD)" if pred_label == 1 else "Healthy Control (HC)"
    status_color = "#C62828" if pred_label == 1 else "#2E7D32"
    info_text = (
        f"Prediction: {status_str}\n"
        f"Confidence P(PD): {p_pd*100:.1f}%\n"
        f"Evidence: e_PD={e_pd:.2f}, e_HC={e_hc:.2f}\n"
        f"Vacuity (Uncertainty u): {vacuity:.3f}\n"
        f"Dominant Sensor: {summary['dominant_sensor']}"
    )
    ax4.text(
        0.03, 0.95, info_text, transform=ax4.transAxes,
        fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", edgecolor=status_color, linewidth=2.0)
    )
    ax4.legend(loc="upper right", frameon=True)

    plt.suptitle(
        f"Gait Modality Explainability & Clinical Attribution Report — {sample_id}",
        fontsize=14, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    results = {
        "p_pd": p_pd,
        "p_hc": p_hc,
        "vacuity": vacuity,
        "asymmetry_index": summary["asymmetry_index"],
        "dominant_sensor": summary["dominant_sensor"]
    }
    return fig, results


if __name__ == "__main__":
    print("Testing Gait Explainability module...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GaitCNNBiLSTM().to(device)

    # Dummy continuous VGRF window: (1, 500, 16)
    dummy_x = torch.randn(1, config.WINDOW_SIZE, config.NUM_CHANNELS, device=device)

    # Run Integrated Gradients
    ig = compute_integrated_gradients(model, dummy_x, target_class=1, steps=10)
    print("Integrated Gradients shape:", ig.shape)
    assert ig.shape == dummy_x.shape, "IG shape mismatch!"

    # Run Grad-CAM 1D
    gradcam = GradCAM1D(model)
    cam = gradcam.generate(dummy_x, target_class=1)
    print("Grad-CAM 1D shape:", cam.shape)
    assert cam.shape == (1, config.WINDOW_SIZE), "GradCAM shape mismatch!"

    # Run visualization
    out_file = config.PLOTS_DIR / "test_gait_explainability.png"
    fig, metrics = visualize_gait_explainability(dummy_x, model, device, sample_id="Test_Run", save_path=out_file)
    plt.close(fig)
    print("Metrics:", metrics)
    print("Gait Explainability successfully verified!")
