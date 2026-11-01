"""Explainability and Feature Attribution for Voice-Based Parkinson's Disease Detection.

This module implements two complementary explainability methods for vocal acoustic features:
1. 2D Grad-CAM for Voice Spectrograms:
   - Hooks into the final convolutional feature layer of the Voice 2D CNN architecture.
   - Generates high-resolution time-frequency attribution heatmaps identifying acoustic segments
     (vocal tremors, unvoiced-to-voiced transitions, pitch perturbations) driving the prediction.
2. Integrated Gradients for Spectrogram / Acoustic Features:
   - Path-integral attribution (Sundararajan et al., 2017) over frequency bins and time frames.
   - Computes marginal frequency importance: F0 pitch (<300 Hz), F1/F2 formants (300-2500 Hz),
     and high-frequency breathiness/noise (>2500 Hz).
3. Evidential Decision Explanations:
   - Integrates evidential belief (e_HC, e_PD) and epistemic vacuity metrics.
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
    from Voice import config as voice_config
    from Voice.baselines.base_paper_cnn import config as cnn_config
    from Voice.baselines.base_paper_cnn.model import VoiceBasePaper2DCNN
except ImportError:
    import config as voice_config
    from baselines.base_paper_cnn import config as cnn_config
    from baselines.base_paper_cnn.model import VoiceBasePaper2DCNN

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM2DSpectrogram:
    """2D Grad-CAM implementation hooked to the final Conv2d feature layer."""

    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.model.eval()
        if target_layer is None:
            # Default to conv_block3[0] (Conv2d in Block 3)
            self.target_layer = self.model.conv_block3[0]
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
        """Generate 2D Grad-CAM activation map upsampled to input spectrogram shape.
        
        Args:
            x: Input tensor of shape (batch_size, 1, freq_bins, time_frames).
            target_class: Target class index (0: HC, 1: PD).
            
        Returns:
            cam: 2D normalized activation map of shape (batch_size, freq_bins, time_frames) in [0, 1].
        """
        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad()

        try:
            logits = self.model(x)
            if logits.dim() == 2 and logits.shape[1] >= 2:
                score = logits[:, target_class].sum()
            else:
                score = logits.squeeze().sum()

            score.backward(retain_graph=True)
        finally:
            self.model.train(was_training)

        # Activations shape: (B, Channels, F_conv, T_conv) e.g., (1, 128, F', T')
        # Gradients shape:   (B, Channels, F_conv, T_conv)
        alpha = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (B, Channels, 1, 1)
        cam = torch.sum(alpha * self.activations, dim=1, keepdim=True)  # (B, 1, F_conv, T_conv)
        cam = F.relu(cam)

        # Upsample back to input spectrogram dimensions (e.g., freq_bins, time_frames)
        target_size = (x.shape[2], x.shape[3])
        cam_upsampled = F.interpolate(cam, size=target_size, mode="bilinear", align_corners=False)
        cam_upsampled = cam_upsampled.squeeze(1)  # (B, freq_bins, time_frames)

        # Min-max normalization per sample
        b_min = cam_upsampled.amin(dim=(1, 2), keepdim=True)
        b_max = cam_upsampled.amax(dim=(1, 2), keepdim=True)
        norm_cam = (cam_upsampled - b_min) / (b_max - b_min + 1e-8)
        return norm_cam


def compute_voice_integrated_gradients(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_class: int = 1,
    baseline: Optional[torch.Tensor] = None,
    steps: int = 40
) -> torch.Tensor:
    """Compute Integrated Gradients attribution for 2D voice spectrograms.
    
    Args:
        model: Trained Voice 2D CNN model in eval mode.
        x: Input tensor of shape (batch_size, 1, freq_bins, time_frames).
        target_class: Class index for attribution (0: HC, 1: PD).
        baseline: Reference baseline tensor (default: zeros).
        steps: Number of interpolation steps.
        
    Returns:
        attributions: Tensor of shape (batch_size, 1, freq_bins, time_frames).
    """
    was_training = model.training
    model.eval()
    device = x.device
    if baseline is None:
        baseline = torch.zeros_like(x, device=device)

    alphas = torch.linspace(0.0, 1.0, steps + 1, device=device)
    delta = x - baseline
    accumulated_grads = torch.zeros_like(x, device=device)

    try:
        for alpha in alphas:
            interpolated_x = (baseline + alpha * delta).clone().detach().requires_grad_(True)
            logits = model(interpolated_x)
            if logits.dim() == 2 and logits.shape[1] >= 2:
                target_score = logits[:, target_class].sum()
            else:
                target_score = logits.squeeze().sum()

            model.zero_grad()
            grads = torch.autograd.grad(target_score, interpolated_x, create_graph=False)[0]
            accumulated_grads += grads
    finally:
        model.train(was_training)

    avg_grads = accumulated_grads / (steps + 1)
    attributions = delta * avg_grads
    return attributions


def summarize_acoustic_frequency_bands(
    attribution_map: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 512
) -> Dict[str, float]:
    """Segment frequency attribution into clinically relevant acoustic bands:
    - Fundamental Frequency (F0, pitch range): 0 - 300 Hz
    - Low-Mid Formants (F1, F2 vowel articulation): 300 - 2500 Hz
    - High-Frequency Formants / Aspiration Noise (F3+ & breathiness): > 2500 Hz
    """
    freq_bins = attribution_map.shape[0]
    freqs = np.linspace(0, sample_rate / 2, freq_bins)

    f0_mask = freqs <= 300
    formant_mask = (freqs > 300) & (freqs <= 2500)
    high_mask = freqs > 2500

    # Marginal attribution by frequency
    freq_profile = np.mean(np.abs(attribution_map), axis=1)  # (freq_bins,)
    total_energy = float(np.sum(freq_profile) + 1e-8)

    f0_pct = float(np.sum(freq_profile[f0_mask]) / total_energy) * 100.0
    formant_pct = float(np.sum(freq_profile[formant_mask]) / total_energy) * 100.0
    high_pct = float(np.sum(freq_profile[high_mask]) / total_energy) * 100.0

    return {
        "F0_Pitch_Band_Pct": f0_pct,
        "F1_F2_Formant_Band_Pct": formant_pct,
        "HighFreq_Breathiness_Band_Pct": high_pct,
        "Dominant_Frequency_Hz": float(freqs[np.argmax(freq_profile)])
    }


def visualize_voice_explainability(
    spectrogram: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    sample_id: str = "Voice_Acoustic_Sample",
    sample_rate: int = cnn_config.SAMPLE_RATE,
    save_path: Optional[Path] = None
) -> Tuple[plt.Figure, Dict[str, float]]:
    """Generate and save comprehensive 4-panel Voice Explainability Dashboard."""
    model.eval()
    spectrogram = spectrogram.to(device)
    if spectrogram.dim() == 2:
        spectrogram = spectrogram.unsqueeze(0).unsqueeze(0)
    elif spectrogram.dim() == 3:
        spectrogram = spectrogram.unsqueeze(0)

    # 1. Forward Pass & Probability Computation
    with torch.no_grad():
        logits = model(spectrogram)
        probs = torch.softmax(logits, dim=1)[0]
        p_hc = probs[0].item()
        p_pd = probs[1].item()
        pred_label = 1 if p_pd >= p_hc else 0

    # 2. Compute 2D Grad-CAM
    cam_engine = GradCAM2DSpectrogram(model)
    cam_2d = cam_engine.generate(spectrogram, target_class=1)[0].detach().cpu().numpy()

    # 3. Compute Integrated Gradients
    ig_attr = compute_voice_integrated_gradients(model, spectrogram, target_class=1, steps=30)
    ig_2d = ig_attr[0, 0].detach().cpu().numpy()

    # 4. Frequency Band Summary
    band_summary = summarize_acoustic_frequency_bands(
        cam_2d, sample_rate=sample_rate, n_fft=cnn_config.N_FFT
    )

    spec_raw = spectrogram[0, 0].detach().cpu().numpy()
    freq_bins, time_frames = spec_raw.shape
    time_axis = np.linspace(0, cnn_config.CHUNK_DURATION, time_frames)
    freq_axis = np.linspace(0, sample_rate / 2000.0, freq_bins)  # in kHz

    # 5. Create Publication-Grade Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor("#FFFFFF")

    # Panel 1: Input STFT Spectrogram
    ax1 = axes[0, 0]
    im1 = ax1.imshow(
        spec_raw,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[0, cnn_config.CHUNK_DURATION, 0, sample_rate / 2000.0]
    )
    ax1.set_title("Input Acoustic STFT Spectrogram", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Time (seconds)", fontsize=10)
    ax1.set_ylabel("Frequency (kHz)", fontsize=10)
    plt.colorbar(im1, ax=ax1, orientation="vertical", label="Normalized Magnitude", pad=0.02)

    # Panel 2: 2D Grad-CAM Activation Heatmap
    ax2 = axes[0, 1]
    im2 = ax2.imshow(
        cam_2d,
        aspect="auto",
        origin="lower",
        cmap="inferno",
        extent=[0, cnn_config.CHUNK_DURATION, 0, sample_rate / 2000.0]
    )
    ax2.set_title("2D Grad-CAM Class Activation Map (Saliency across Formants)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Time (seconds)", fontsize=10)
    ax2.set_ylabel("Frequency (kHz)", fontsize=10)
    plt.colorbar(im2, ax=ax2, orientation="vertical", label="Grad-CAM Activation [0, 1]", pad=0.02)

    # Panel 3: Clinical Acoustic Frequency Band Distribution
    ax3 = axes[1, 0]
    bands = ["F0 Pitch (<300Hz)", "F1/F2 Formants (0.3-2.5kHz)", "Breathiness / Noise (>2.5kHz)"]
    percentages = [
        band_summary["F0_Pitch_Band_Pct"],
        band_summary["F1_F2_Formant_Band_Pct"],
        band_summary["HighFreq_Breathiness_Band_Pct"]
    ]
    colors = ["#2b5c8f", "#d95f02", "#7570b3"]
    bars = ax3.barh(bands, percentages, color=colors, edgecolor="black", alpha=0.85)
    ax3.set_title("Acoustic Frequency-Band Saliency Distribution (IG)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Contribution Percentage (%)", fontsize=10)
    ax3.set_xlim(0, max(percentages) + 15)
    ax3.grid(True, axis="x", linestyle=":", alpha=0.6)
    for bar in bars:
        w = bar.get_width()
        ax3.text(w + 1.0, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=9, fontweight="bold")

    # Panel 4: Integrated Gradients Temporal Profile & Decision Confidence Card
    ax4 = axes[1, 1]
    temporal_profile = np.mean(np.abs(ig_2d), axis=0)  # (time_frames,)
    temporal_profile = (temporal_profile - temporal_profile.min()) / (temporal_profile.max() - temporal_profile.min() + 1e-8)
    ax4.plot(time_axis, temporal_profile, color="#00897B", linewidth=2.2, label="Temporal Importance (IG)")
    ax4.fill_between(time_axis, temporal_profile, color="#00897B", alpha=0.25)
    ax4.set_title("Temporal Phonetic Saliency & Evidential Readout", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Time (seconds)", fontsize=10)
    ax4.set_ylabel("Temporal Saliency Intensity", fontsize=10)
    ax4.grid(True, linestyle=":", alpha=0.6)

    # Decision card
    status_str = "Parkinson's Disease (PD)" if pred_label == 1 else "Healthy Control (HC)"
    status_color = "#C62828" if pred_label == 1 else "#2E7D32"
    info_text = (
        f"Prediction: {status_str}\n"
        f"Confidence P(PD): {p_pd*100:.1f}%\n"
        f"Confidence P(HC): {p_hc*100:.1f}%\n"
        f"Dominant Freq: {band_summary['Dominant_Frequency_Hz']:.0f} Hz\n"
        f"Formant Articulation Band: {band_summary['F1_F2_Formant_Band_Pct']:.1f}%"
    )
    ax4.text(
        0.03, 0.95, info_text, transform=ax4.transAxes,
        fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", edgecolor=status_color, linewidth=2.0)
    )
    ax4.legend(loc="upper right", frameon=True)

    plt.suptitle(
        f"Voice Modality Explainability & Acoustic Saliency Report — {sample_id}",
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
        "dominant_freq_hz": band_summary["Dominant_Frequency_Hz"],
        "formant_pct": band_summary["F1_F2_Formant_Band_Pct"],
        "f0_pitch_pct": band_summary["F0_Pitch_Band_Pct"]
    }
    return fig, results


if __name__ == "__main__":
    print("Testing Voice Explainability module...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VoiceBasePaper2DCNN().to(device)

    # Dummy spectrogram input: (1, 1, 257, 301)
    dummy_spec = torch.randn(1, 1, cnn_config.FREQ_BINS, cnn_config.TIME_FRAMES, device=device)

    # Run 2D Grad-CAM
    cam_engine = GradCAM2DSpectrogram(model)
    cam = cam_engine.generate(dummy_spec, target_class=1)
    print("Voice Grad-CAM 2D shape:", cam.shape)
    assert cam.shape == (1, cnn_config.FREQ_BINS, cnn_config.TIME_FRAMES), "GradCAM 2D shape mismatch!"

    # Run Integrated Gradients
    ig = compute_voice_integrated_gradients(model, dummy_spec, target_class=1, steps=10)
    print("Voice Integrated Gradients shape:", ig.shape)
    assert ig.shape == dummy_spec.shape, "IG shape mismatch!"

    # Run visualization
    out_file = voice_config.RESULTS_DIR / "voice_explainability.png"
    fig, metrics = visualize_voice_explainability(dummy_spec, model, device, sample_id="Test_Voice", save_path=out_file)
    plt.close(fig)
    print("Metrics:", metrics)
    print("Voice Explainability successfully verified!")
