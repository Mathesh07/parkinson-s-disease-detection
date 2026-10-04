"""Tests for Evidential Deep Learning (EDL), Dirichlet strength, and vacuity uncertainty."""

import sys
from pathlib import Path
import numpy as np
import pytest
import torch

# Ensure workspace root and Voice directory are in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
voice_dir = root_dir / "Voice"
for p in (str(root_dir), str(voice_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from evidential import EvidentialLoss, dirichlet_from_evidence, kl_divergence_dirichlet


def test_dirichlet_from_evidence_identities():
    """Verify Subjective Logic identities: alpha = e + 1, S = sum(alpha), P = alpha / S, u = K / S."""
    # Synthetic evidence tensor for 3 samples (K=2 classes)
    evidence = torch.tensor([
        [0.0, 0.0],     # Zero evidence -> maximum uncertainty (u = 1.0)
        [5.0, 15.0],    # High evidence for class 1
        [10.0, 0.0],    # High evidence for class 0
    ], dtype=torch.float32)

    num_classes = 2
    alpha, S, probs, u = dirichlet_from_evidence(evidence, num_classes=num_classes)

    # 1. Alpha >= 1.0
    assert (alpha >= 1.0).all()
    assert torch.allclose(alpha, evidence + 1.0)

    # 2. Total Dirichlet strength S >= K
    assert (S >= num_classes).all()
    assert torch.allclose(S, torch.sum(alpha, dim=-1, keepdim=True))

    # 3. Expected Class Probabilities sum to 1.0
    assert torch.allclose(probs.sum(dim=-1), torch.ones(3))

    # 4. Predictive Vacuity Uncertainty 0 < u <= 1.0
    assert (u > 0.0).all() and (u <= 1.0).all()
    assert torch.allclose(u.squeeze(-1), num_classes / S.squeeze(-1))

    # 5. Subjective Logic Identity: sum(e_k / S) + u = 1.0
    belief = torch.sum(evidence / S, dim=-1, keepdim=True)
    assert torch.allclose(belief + u, torch.ones_like(u))

    # Zero evidence sample check
    assert u[0].item() == 1.0  # Max uncertainty when evidence is zero
    assert torch.allclose(probs[0], torch.tensor([0.5, 0.5]))


def test_kl_divergence_dirichlet_bounds():
    """Verify KL divergence between Dir(alpha) and uniform Dir(1) is >= 0 and non-NaN."""
    # Dir(1, 1) should have zero KL divergence from prior Dir(1, 1)
    alpha_prior = torch.ones(2, 2)
    kl_prior = kl_divergence_dirichlet(alpha_prior, num_classes=2)
    assert torch.allclose(kl_prior, torch.zeros(2), atol=1e-5)

    # Concentrated Dirichlet distribution should have positive KL divergence
    alpha_conc = torch.tensor([[10.0, 2.0], [1.0, 20.0]])
    kl_conc = kl_divergence_dirichlet(alpha_conc, num_classes=2)
    assert (kl_conc > 0.0).all()
    assert not torch.isnan(kl_conc).any()
    assert not torch.isinf(kl_conc).any()


def test_evidential_loss_forward_pass():
    """Test EvidentialLoss computation, ACE loss decomposition, and KL annealing."""
    loss_fn = EvidentialLoss(num_classes=2, annealing_epochs=5)

    evidence = torch.tensor([
        [2.0, 8.0],
        [6.0, 1.0],
    ], dtype=torch.float32)
    target = torch.tensor([1, 0], dtype=torch.long)

    # Epoch 1 (lambda_t = 1/5 = 0.2)
    total_loss, loss_dict = loss_fn(evidence, target, epoch=1)

    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.dim() == 0
    assert total_loss.item() > 0.0
    assert not torch.isnan(total_loss)

    assert "loss" in loss_dict
    assert "ace_loss" in loss_dict
    assert "kl_loss" in loss_dict
    assert loss_dict["lambda_t"] == 0.2

    # Check that total_loss = ace_loss + lambda_t * kl_loss
    expected_total = loss_dict["ace_loss"] + 0.2 * loss_dict["kl_loss"]
    assert np.isclose(loss_dict["loss"], expected_total, atol=1e-4)
