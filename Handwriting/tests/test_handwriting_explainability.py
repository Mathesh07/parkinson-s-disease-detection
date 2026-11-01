"""Unit tests for Handwriting Explainability (ViT Attention Rollout & CNN 2D Grad-CAM)."""

import pytest
import sys
from pathlib import Path
from PIL import Image

# Ensure paths
root_dir = Path(__file__).resolve().parent.parent.parent
hw_dir = root_dir / "Handwriting"
for p in (str(root_dir), str(hw_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import torch
    from Handwriting.model import CNNBaseline
    from Handwriting.explainability import (
        compute_attention_rollout,
        GradCAMHandwritingCNN,
        visualize_handwriting_explainability,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.fixture
def cnn_model():
    model = CNNBaseline(num_classes=2)
    model.eval()
    return model


@pytest.fixture
def sample_drawing_tensor():
    # Shape: (1, 3, 224, 224)
    return torch.randn(1, 3, 224, 224)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required for explainability tests")
def test_compute_attention_rollout():
    """Verify ViT Attention Rollout aggregates layers into normalized 14x14 heatmap."""
    num_layers = 4
    num_heads = 4
    num_tokens = 197  # 1 CLS token + 14x14 spatial tokens
    dummy_attentions = tuple(
        torch.softmax(torch.randn(1, num_heads, num_tokens, num_tokens), dim=-1)
        for _ in range(num_layers)
    )

    heatmap = compute_attention_rollout(dummy_attentions, head_fusion="mean")
    assert heatmap.shape == (14, 14)
    assert not torch.isnan(heatmap).any()
    assert heatmap.min().item() >= -1e-6
    assert heatmap.max().item() <= 1.0 + 1e-6


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required for explainability tests")
def test_gradcam_handwriting_cnn_shape_and_bounds(cnn_model, sample_drawing_tensor):
    """Verify 2D Grad-CAM on CNN baseline outputs activation map matching (1, 224, 224)."""
    gradcam = GradCAMHandwritingCNN(cnn_model)
    cam = gradcam.generate(sample_drawing_tensor, target_class=1)

    assert cam.shape == (1, 224, 224)
    assert not torch.isnan(cam).any()
    assert cam.min().item() >= -1e-6
    assert cam.max().item() <= 1.0 + 1e-6


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required for explainability tests")
def test_visualize_handwriting_explainability_cnn(cnn_model, sample_drawing_tensor, tmp_path):
    """Verify end-to-end handwriting explainability dashboard generation with CNN model."""
    out_file = tmp_path / "test_hw_explainability.png"
    device = torch.device("cpu")

    fig, metrics = visualize_handwriting_explainability(
        image_or_tensor=sample_drawing_tensor,
        model=cnn_model,
        device=device,
        sample_id="TestHandwritingSample",
        save_path=out_file
    )

    assert out_file.exists()
    assert "p_pd" in metrics
    assert "vacuity" in metrics
    assert "mean_saliency" in metrics
    assert metrics["method"] == "CNN 2D Grad-CAM"
