"""Explainability and Feature Attribution for Handwriting-Based Parkinson's Disease Detection.

This module implements complementary explainability methods for handwriting drawings (spirals, meanders):
1. Attention Rollout for Vision Transformer (ViT):
   - Extracts multi-head self-attention matrices across all transformer encoder layers.
   - Combines heads, models residual skip connections, and propagates attention from [CLS] to spatial patches.
   - Interpolates the 14x14 patch attention to 224x224 heatmap highlighting high-impact drawing stroke regions.
2. 2D Grad-CAM for Handwriting CNNBaseline:
   - Hooks into the final Conv2d stage of the 4-stage convolutional baseline.
   - Computes gradient-weighted feature map activations localized on kinematic stroke abnormalities.
3. Evidential Decision Metrics:
   - Integrates evidential belief (e_HC, e_PD) and Dirichlet vacuity into the explainability report.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
for p in (str(current_dir), str(parent_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from Handwriting.config import (
        EXPLAINABILITY_PLOT_PATH, IMAGE_SIZE, MODEL_LABELS, VIT_CHECKPOINT
    )
    from Handwriting.dataset import (
        build_manifest, create_patient_level_split, create_transforms
    )
    from Handwriting.model import ViTBinaryClassifier, CNNBaseline
except ImportError:
    from config import (
        EXPLAINABILITY_PLOT_PATH, IMAGE_SIZE, MODEL_LABELS, VIT_CHECKPOINT
    )
    from dataset import (
        build_manifest, create_patient_level_split, create_transforms
    )
    from model import ViTBinaryClassifier, CNNBaseline


def compute_attention_rollout(attentions, head_fusion: str = "mean") -> torch.Tensor:
    """Compute ViT Attention Rollout across transformer encoder layers.
    
    Args:
        attentions: Tuple of attention matrices from each layer of shape (batch_size, num_heads, N, N)
        head_fusion: Strategy to aggregate head attentions ('mean', 'max', 'min')
    Returns:
        2D tensor heatmap array of shape (grid_size, grid_size)
    """
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for layer_attn in attentions:
            # layer_attn shape: (1, num_heads, N, N)
            if head_fusion == "mean":
                attn_fused = layer_attn[0].mean(dim=0)
            elif head_fusion == "max":
                attn_fused = layer_attn[0].max(dim=0)[0]
            else:
                attn_fused = layer_attn[0].mean(dim=0)

            # Add identity matrix to model residual connections
            I = torch.eye(attn_fused.size(-1)).to(attn_fused.device)
            a = (attn_fused + I) / 2.0
            a = a / a.sum(dim=-1, keepdim=True)

            result = torch.matmul(a, result)

    # Extract attention from [CLS] token (index 0) to all spatial patches (indices 1..)
    cls_attn = result[0, 1:]
    grid_size = int(np.sqrt(cls_attn.size(0)))  # 14x14 for patch16 on 224x224 image
    heatmap = cls_attn.reshape(grid_size, grid_size)

    # Normalize heatmap between 0 and 1
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap


class GradCAMHandwritingCNN:
    """2D Grad-CAM implementation for Handwriting CNNBaseline classifier."""

    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.model.eval()
        if target_layer is None:
            # Locate the last Conv2d layer in the model features
            conv_layers = [m for m in self.model.modules() if isinstance(m, torch.nn.Conv2d)]
            self.target_layer = conv_layers[-1] if conv_layers else self.model.features[-4]
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
        """Generate 2D Grad-CAM activation map upsampled to input image dimensions (B, H, W).
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            target_class: Target class (0: HC, 1: PD)
        Returns:
            cam: 2D normalized activation map of shape (B, H, W) in [0, 1]
        """
        was_training = self.model.training
        self.model.eval()
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

        alpha = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(alpha * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        target_size = (x.shape[2], x.shape[3])
        cam_upsampled = F.interpolate(cam, size=target_size, mode="bilinear", align_corners=False)
        cam_np = cam_upsampled.squeeze(1)

        # Min-max normalize per sample
        for i in range(cam_np.shape[0]):
            c_min = cam_np[i].min()
            c_max = cam_np[i].max()
            cam_np[i] = (cam_np[i] - c_min) / (c_max - c_min + 1e-8)

        return cam_np


def visualize_handwriting_explainability(
    image_or_tensor: Union[str, Path, Image.Image, torch.Tensor],
    model: torch.nn.Module,
    device: torch.device,
    sample_id: str = "Handwriting_Sample",
    target_class: int = 1,
    save_path: Optional[Union[str, Path]] = None
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """Generate comprehensive 3-panel explainability figure for Handwriting (ViT or CNN).
    
    Args:
        image_or_tensor: Image path, PIL Image, or (1, 3, H, W) torch.Tensor
        model: Trained ViT or CNN model
        device: Computation device (cpu or cuda)
        sample_id: Identifier for figure title
        target_class: Class to explain (default 1: Parkinson's)
        save_path: Optional path to save resulting figure PNG
        
    Returns:
        (fig, metrics): Matplotlib Figure and dictionary of clinical explainability metrics.
    """
    model.eval()
    model = model.to(device)

    # 1. Prepare image tensor and display RGB image
    if isinstance(image_or_tensor, (str, Path)):
        raw_img = Image.open(str(image_or_tensor)).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        transform = create_transforms(image_size=IMAGE_SIZE, train=False)
        img_tensor = transform(raw_img).unsqueeze(0).to(device)
        display_img = np.array(raw_img)
    elif isinstance(image_or_tensor, Image.Image):
        raw_img = image_or_tensor.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        transform = create_transforms(image_size=IMAGE_SIZE, train=False)
        img_tensor = transform(raw_img).unsqueeze(0).to(device)
        display_img = np.array(raw_img)
    elif isinstance(image_or_tensor, torch.Tensor):
        img_tensor = image_or_tensor.to(device)
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        # Denormalize tensor for display
        np_img = img_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
        np_img = (np_img - np_img.min()) / (np_img.max() - np_img.min() + 1e-8)
        display_img = (np_img * 255).astype(np.uint8)
    else:
        raise TypeError(f"Unsupported image input type: {type(image_or_tensor)}")

    # 2. Forward pass & compute attribution
    is_vit = hasattr(model, "backbone")
    if is_vit:
        method_name = "ViT Attention Rollout"
        with torch.no_grad():
            outputs = model(img_tensor)
            if isinstance(outputs, tuple):
                evidence, attentions = outputs
            else:
                evidence = outputs
                attentions = None

        if attentions is not None:
            heatmap_raw = compute_attention_rollout(attentions, head_fusion="mean")
            heatmap_tensor = heatmap_raw.unsqueeze(0).unsqueeze(0)
            heatmap_resized = F.interpolate(
                heatmap_tensor, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
            )
            heatmap_np = heatmap_resized.squeeze().cpu().numpy()
        else:
            # Fallback if attentions were not captured
            heatmap_np = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32) * 0.5
    else:
        method_name = "CNN 2D Grad-CAM"
        gradcam = GradCAMHandwritingCNN(model)
        cam = gradcam.generate(img_tensor, target_class=target_class)
        heatmap_np = cam[0].detach().cpu().numpy()
        with torch.no_grad():
            evidence = model(img_tensor)

    # 3. Evidential metrics
    if evidence.dim() == 2 and evidence.shape[1] >= 2:
        e_hc = float(evidence[0, 0].item())
        e_pd = float(evidence[0, 1].item())
    else:
        e_hc, e_pd = 1.0, float(evidence.squeeze().item())

    s = e_hc + e_pd + 2.0
    p_hc = (e_hc + 1.0) / s
    p_pd = (e_pd + 1.0) / s
    vacuity = 2.0 / s
    pred_label = 1 if p_pd >= p_hc else 0
    pred_name = MODEL_LABELS[pred_label] if pred_label < len(MODEL_LABELS) else "Unknown"

    # 4. Generate 3-Panel Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#FFFFFF")

    # Panel 1: Original Image
    axes[0].imshow(display_img)
    axes[0].set_title(f"Input Drawing: {sample_id}", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Saliency Heatmap
    im1 = axes[1].imshow(heatmap_np, cmap="inferno")
    axes[1].set_title(f"Attribution ({method_name})", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Blended Overlay
    axes[2].imshow(display_img)
    axes[2].imshow(heatmap_np, cmap="inferno", alpha=0.55)
    axes[2].set_title(
        f"Prediction: {pred_name} (P={p_pd*100:.1f}%, u={vacuity:.2f})",
        fontsize=11,
        fontweight="bold"
    )
    axes[2].axis("off")

    plt.suptitle(
        f"Handwriting Modality Explainability & Kinematic Saliency — {sample_id}",
        fontsize=13,
        fontweight="bold",
        y=0.98
    )
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    metrics = {
        "sample_id": sample_id,
        "method": method_name,
        "prediction": pred_name,
        "p_pd": round(p_pd, 4),
        "p_hc": round(p_hc, 4),
        "vacuity": round(vacuity, 4),
        "mean_saliency": round(float(np.mean(heatmap_np)), 4),
        "max_saliency": round(float(np.max(heatmap_np)), 4)
    }

    return fig, metrics


def visualize_vit_explainability(
    image_path: str,
    model: torch.nn.Module,
    device: torch.device,
    save_path: Path = EXPLAINABILITY_PLOT_PATH
) -> None:
    """Backwards-compatible wrapper for ViT explainability visualization."""
    fig, _ = visualize_handwriting_explainability(
        image_or_tensor=image_path,
        model=model,
        device=device,
        save_path=save_path
    )
    plt.close(fig)
    print(f"Saved explainability visualization to {save_path}")


def generate_explainability_reports() -> None:
    """Load trained ViT model and generate explainability maps for test set samples."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not VIT_CHECKPOINT.exists():
        raise FileNotFoundError(f"ViT checkpoint not found at {VIT_CHECKPOINT}")

    print("\n==========================================")
    print(" Generating ViT Explainability Maps")
    print("==========================================")

    model = ViTBinaryClassifier(freeze_backbone=False, output_attentions=True).to(device)
    checkpoint = torch.load(VIT_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    manifest, _ = build_manifest()
    manifest = create_patient_level_split(manifest)
    test_rows = manifest[(manifest["split"] == "test") & (manifest["image_exists"])]

    sample_row = test_rows[test_rows["target"] == 1].iloc[0]
    sample_image_path = sample_row["image_path"]
    print(f"Selected test image for explainability: {sample_image_path}")

    visualize_vit_explainability(sample_image_path, model, device)


if __name__ == "__main__":
    generate_explainability_reports()
