"""Mathematical Unit Tests for Step 3: Evidential Deep Learning, Dirichlet & Uncertainty."""

import math
import sys
from pathlib import Path
import torch

# Ensure workspace is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evidential import dirichlet_from_evidence, kl_divergence_dirichlet, EvidentialLoss

print("=" * 70)
print("RUNNING MATHEMATICAL UNIT TESTS FOR STEP 3 (EDL / TMC)")
print("=" * 70)

# -------------------------------------------------------------
# TEST 1: Zero-Evidence Edge Case [0, 0]
# -------------------------------------------------------------
print("\n[TEST 1] Zero-Evidence Case: evidence = [0.0, 0.0]")
e_zero = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
alpha, S, probs, u = dirichlet_from_evidence(e_zero, num_classes=2)

print(f"  alpha:       {alpha.tolist()[0]} (Expected: [1.0, 1.0])")
print(f"  S:           {S.item():.4f} (Expected: 2.0000)")
print(f"  probability: {probs.tolist()[0]} (Expected: [0.5, 0.5])")
print(f"  uncertainty: {u.item():.4f} (Expected: 1.0000)")

assert torch.allclose(alpha, torch.tensor([[1.0, 1.0]])), "Test 1 Failed: alpha != [1, 1]"
assert math.isclose(S.item(), 2.0, abs_tol=1e-5), "Test 1 Failed: S != 2.0"
assert torch.allclose(probs, torch.tensor([[0.5, 0.5]])), "Test 1 Failed: probs != [0.5, 0.5]"
assert math.isclose(u.item(), 1.0, abs_tol=1e-5), "Test 1 Failed: u != 1.0"
print("  --> [PASS] Zero-Evidence test passed exactly!")

# -------------------------------------------------------------
# TEST 2: Asymmetric Evidence Case [1, 9]
# -------------------------------------------------------------
print("\n[TEST 2] Asymmetric Evidence Case: evidence = [1.0, 9.0]")
e_asym = torch.tensor([[1.0, 9.0]], dtype=torch.float32)
alpha, S, probs, u = dirichlet_from_evidence(e_asym, num_classes=2)

expected_p_hc = 2.0 / 12.0
expected_p_pd = 10.0 / 12.0
expected_u = 2.0 / 12.0

print(f"  alpha:       {alpha.tolist()[0]} (Expected: [2.0, 10.0])")
print(f"  S:           {S.item():.4f} (Expected: 12.0000)")
print(f"  probability: [P_HC={probs[0,0].item():.4f}, P_PD={probs[0,1].item():.4f}] (Expected: [{expected_p_hc:.4f}, {expected_p_pd:.4f}])")
print(f"  uncertainty: {u.item():.4f} (Expected: {expected_u:.4f})")

assert torch.allclose(alpha, torch.tensor([[2.0, 10.0]])), "Test 2 Failed: alpha != [2, 10]"
assert math.isclose(S.item(), 12.0, abs_tol=1e-5), "Test 2 Failed: S != 12.0"
assert torch.allclose(probs, torch.tensor([[expected_p_hc, expected_p_pd]])), "Test 2 Failed: probs mismatch"
assert math.isclose(u.item(), expected_u, abs_tol=1e-5), "Test 2 Failed: u mismatch"
print("  --> [PASS] Asymmetric Evidence test passed exactly!")

# -------------------------------------------------------------
# TEST 3: Evidence Scaling with Constant Ratio
# -------------------------------------------------------------
print("\n[TEST 3] Uncertainty Reduction with Total Evidence Scaling (Ratio 1:9 maintained)")
e_low = torch.tensor([[1.0, 9.0]], dtype=torch.float32)
e_high = torch.tensor([[10.0, 90.0]], dtype=torch.float32)

_, _, p_low, u_low = dirichlet_from_evidence(e_low, num_classes=2)
_, _, p_high, u_high = dirichlet_from_evidence(e_high, num_classes=2)

print(f"  Low Evidence [1, 9]:     P_PD={p_low[0,1].item():.4f}, uncertainty={u_low.item():.4f}")
print(f"  High Evidence [10, 90]:  P_PD={p_high[0,1].item():.4f}, uncertainty={u_high.item():.4f}")

# Ratio of evidence is 1:9 in both, but uncertainty must strictly decrease
assert u_high.item() < u_low.item(), f"Test 3 Failed: u_high ({u_high.item()}) not < u_low ({u_low.item()})"
assert math.isclose(u_high.item(), 2.0 / 102.0, abs_tol=1e-5), "Test 3 Failed: u_high value mismatch"
print(f"  --> [PASS] Total evidence scaling test passed: u dropped from {u_low.item():.4f} to {u_high.item():.4f}!")

# -------------------------------------------------------------
# TEST 4: Subjective Logic Identity (b_0 + b_1 + u == 1.0)
# -------------------------------------------------------------
print("\n[TEST 4] Subjective Logic Identity: b_HC + b_PD + u == 1.0")
e_test = torch.tensor([[3.5, 7.2], [0.0, 5.0], [12.0, 0.0]], dtype=torch.float32)
alpha, S, _, u = dirichlet_from_evidence(e_test, num_classes=2)
belief = e_test / S
identity_sum = torch.sum(belief, dim=-1, keepdim=True) + u

print(f"  Identity sum for 3 test batches: {identity_sum.squeeze().tolist()} (All must be 1.0)")
assert torch.allclose(identity_sum, torch.ones_like(identity_sum)), "Test 4 Failed: Belief + Uncertainty != 1"
print("  --> [PASS] Subjective Logic identity satisfied across all batches!")

# -------------------------------------------------------------
# TEST 5: Evidential Loss & Backprop Verification
# -------------------------------------------------------------
print("\n[TEST 5] Evidential Loss Computation & Gradient Backpropagation")
criterion = EvidentialLoss(num_classes=2, annealing_epochs=10)

evidence_param = torch.tensor([[0.5, 2.0], [3.0, 0.2]], requires_grad=True)
targets = torch.tensor([1, 0], dtype=torch.long)  # Batch 0 is PD (1), Batch 1 is HC (0)

loss, loss_dict = criterion(evidence_param, targets, epoch=5)
loss.backward()

print(f"  Total Loss (Epoch 5): {loss.item():.4f}")
print(f"  ACE Component:        {loss_dict['ace_loss']:.4f}")
print(f"  KL Component:         {loss_dict['kl_loss']:.4f}")
print(f"  Annealing Factor:     {loss_dict['lambda_t']:.4f}")
print(f"  Gradients on evidence: shape {evidence_param.grad.shape}, min={evidence_param.grad.min().item():.4f}, max={evidence_param.grad.max().item():.4f}")

assert not torch.isnan(loss), "Test 5 Failed: Loss is NaN"
assert not torch.isnan(evidence_param.grad).any(), "Test 5 Failed: Gradients are NaN"
assert evidence_param.grad is not None, "Test 5 Failed: Gradients not computed"
print("  --> [PASS] Evidential loss computed cleanly with valid gradients!")

print("\n" + "=" * 70)
print("ALL MATHEMATICAL TESTS PASSED SUCCESSFULLY!")
print("=" * 70)
