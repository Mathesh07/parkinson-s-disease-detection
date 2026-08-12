"""Explainability visualization for Vision Transformer (ViT) classifier.

This module implements Attention Rollout for ViT using PyTorch, PIL, and Matplotlib:
1. Extracts attention weights from all transformer layers.
2. Performs matrix multiplication across layers considering residual connections (Attention Rollout).
3. Extracts attention from the [CLS] token to input spatial image patches.
4. Reshapes 1D patch attention weights back to 2D spatial grid (14x14 patches).
5. Overlays heatmap onto the original handwriting image to highlight drawing strokes that influenced the Parkinson's prediction.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import (EXPLAINABILITY_PLOT_PATH, IMAGE_SIZE, MODEL_LABELS,
                    VIT_CHECKPOINT)
from dataset import (build_manifest, create_patient_level_split,
                     create_transforms)
from model import ViTBinaryClassifier


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


def visualize_vit_explainability(
    image_path: str,
    model: torch.nn.Module,
    device: torch.device,
    save_path: Path = EXPLAINABILITY_PLOT_PATH
) -> None:
    """Generate and save Attention Rollout visualization for a single handwriting image."""
    raw_img = Image.open(image_path).convert("RGB")
    raw_img_resized = raw_img.resize((IMAGE_SIZE, IMAGE_SIZE))

    transform = create_transforms(image_size=IMAGE_SIZE, train=False)
    img_tensor = transform(raw_img).unsqueeze(0).to(device)

    # Forward pass with attention extraction
    model.eval()
    with torch.no_grad():
        logits, attentions = model(img_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_label = torch.argmax(logits, dim=1).item()

    # Compute attention rollout heatmap
    heatmap = compute_attention_rollout(attentions, head_fusion="mean")

    # Resize heatmap to 224x224 using bilinear interpolation
    heatmap_tensor = heatmap.unsqueeze(0).unsqueeze(0)  # (1, 1, 14, 14)
    heatmap_resized = F.interpolate(heatmap_tensor, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    heatmap_np = heatmap_resized.squeeze().cpu().numpy()

    # Plot using Matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original Image
    axes[0].imshow(raw_img_resized)
    axes[0].set_title("Input Handwriting Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Attention Map
    im1 = axes[1].imshow(heatmap_np, cmap="jet")
    axes[1].set_title("ViT Attention Rollout Map", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Blended Overlay
    axes[2].imshow(raw_img_resized)
    axes[2].imshow(heatmap_np, cmap="jet", alpha=0.5)
    axes[2].set_title(f"Prediction: {MODEL_LABELS[pred_label]} ({probs[pred_label]*100:.1f}%)", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    plt.suptitle("Vision Transformer (ViT) Explainability — Handwriting Region Importance", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved explainability visualization to {save_path}")


def generate_explainability_reports() -> None:
    """Load trained ViT model and generate explainability maps for test set samples."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not VIT_CHECKPOINT.exists():
        raise FileNotFoundError(f"ViT checkpoint not found at {VIT_CHECKPOINT}")

    print("\n==========================================")
    print(" Generating ViT Explainability Maps")
    print("==========================================")

    # Instantiate model with output_attentions=True
    model = ViTBinaryClassifier(freeze_backbone=False, output_attentions=True).to(device)
    checkpoint = torch.load(VIT_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load test sample
    manifest, _ = build_manifest()
    manifest = create_patient_level_split(manifest)
    test_rows = manifest[(manifest["split"] == "test") & (manifest["image_exists"])]

    # Pick sample Parkinson's drawing
    sample_row = test_rows[test_rows["target"] == 1].iloc[0]
    sample_image_path = sample_row["image_path"]
    print(f"Selected test image for explainability: {sample_image_path}")

    visualize_vit_explainability(sample_image_path, model, device)


if __name__ == "__main__":
    generate_explainability_reports()
