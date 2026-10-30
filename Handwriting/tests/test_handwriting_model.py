"""Unit tests for Handwriting Model architectures and Evidential Head."""

import pytest
import sys
from pathlib import Path

# Ensure paths
root_dir = Path(__file__).resolve().parent.parent.parent
hw_dir = root_dir / "Handwriting"
for p in (str(root_dir), str(hw_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import torch
    import torch.nn as nn
    from Handwriting.model import EvidentialHead, CNNBaseline
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required for model tests")
def test_evidential_head_non_negative_evidence():
    """Verify evidential head produces strictly non-negative evidence."""
    batch_size = 4
    feature_dim = 128
    head = EvidentialHead(in_features=feature_dim, num_classes=2)
    
    dummy_features = torch.randn(batch_size, feature_dim)
    evidence = head(dummy_features)
    
    assert evidence.shape == (batch_size, 2)
    assert (evidence >= 0.0).all(), "Evidence values must be >= 0 (Softplus guarantee)"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required for model tests")
def test_evidential_head_return_raw_logits():
    """Verify evidential head can optionally return raw unconstrained logits."""
    head = EvidentialHead(in_features=64, num_classes=2)
    dummy_features = torch.randn(2, 64)
    
    evidence, raw_logits = head(dummy_features, return_raw_logits=True)
    assert evidence.shape == (2, 2)
    assert raw_logits.shape == (2, 2)
    assert torch.allclose(evidence, torch.nn.functional.softplus(raw_logits))


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required for model tests")
def test_cnn_baseline_forward_pass():
    """Verify CNN baseline runs forward pass and yields (B, 2) evidential output."""
    model = CNNBaseline(num_classes=2)
    model.eval()
    
    # 2 sample 224x224 RGB images
    dummy_images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        evidence = model(dummy_images)
        
    assert evidence.shape == (2, 2)
    assert (evidence >= 0.0).all()
