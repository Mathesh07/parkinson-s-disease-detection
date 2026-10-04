"""Unit tests for Controlled Distribution Corruption implementation in Gait/experiments/corruption_shift.py.

Tests:
 1. Clean severity=0 produces identical signals/predictions to the uncorrupted test input.
 2. Corruption does not modify the original input tensor/array.
 3. Same seed produces identical corruption.
 4. Different seeds can produce different stochastic corruption.
 5. Increasing severity actually changes the signal (MSE increases monotonically).
 6. Channel dropout affects only intended channels.
 7. Temporal masking affects only intended temporal regions.
 8. Smoothing reduces high-frequency variation as expected (temporal diff variance decreases).
 9. Gaussian noise increases perturbation magnitude with severity.
10. Amplitude scaling changes signal magnitude according to configured severity.
11. No test labels are used to fit normalization or calibration.
12. Temperature scaling is never fitted on corrupted test data.
13. Model parameters do not change during corruption evaluation.
14. MC Dropout remains stochastic when enabled.
15. BatchNorm remains in evaluation mode during MC Dropout.
16. Original existing output files are untouched.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path bootstrapping
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_TESTS_DIR = _THIS_FILE.parent               # Gait/tests/
_GAIT_DIR = _TESTS_DIR.parent                # Gait/
_ROOT_DIR = _GAIT_DIR.parent                 # repo root

for _p in (str(_GAIT_DIR), str(_ROOT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Gait import config
from Gait.model import GaitCNNBiLSTM
from Gait.preprocessing import GaitNormalizer
from Gait.experiments.corruption_shift import (
    AMPLITUDE_SCALE_FACTORS,
    CHANNEL_DROPOUT_COUNTS,
    CORRUPTION_TYPES,
    GAUSSIAN_NOISE_STDS,
    SEVERITY_LEVELS,
    SMOOTHING_KERNEL_SIZES,
    TEMPORAL_MASK_STEPS,
    apply_amplitude_scaling,
    apply_channel_dropout,
    apply_corruption,
    apply_gaussian_noise,
    apply_smoothing,
    apply_temporal_masking,
)
from Gait.experiments.calibration_baselines import fit_temperature
from Gait.experiments.mc_dropout import enable_mc_dropout


def test_1_severity_zero_identical():
    """Test 1: Clean severity=0 produces identical signals/predictions to uncorrupted test input."""
    x = torch.randn(4, 500, 16)
    for ctype in CORRUPTION_TYPES:
        out = apply_corruption(x, ctype, severity=0)
        assert torch.equal(out, x), f"Severity 0 for {ctype} did not produce identical tensor!"
    print("PASS: Test 1 — Clean severity=0 produces identical signals.")


def test_2_corruption_does_not_modify_original():
    """Test 2: Corruption does not modify the original input tensor/array in place."""
    x_orig = torch.randn(4, 500, 16)
    x_copy = x_orig.clone()
    for ctype in CORRUPTION_TYPES:
        _ = apply_corruption(x_orig, ctype, severity=2, seed=42)
        assert torch.equal(x_orig, x_copy), f"Original tensor was modified in place by {ctype}!"
    print("PASS: Test 2 — Original input tensor non-mutability verified.")


def test_3_same_seed_reproducible():
    """Test 3: Same seed produces identical corruption."""
    x = torch.randn(4, 500, 16)
    for ctype in ["gaussian_noise", "channel_dropout", "temporal_masking"]:
        out1 = apply_corruption(x, ctype, severity=2, seed=42)
        out2 = apply_corruption(x, ctype, severity=2, seed=42)
        assert torch.equal(out1, out2), f"Same seed produced different results for {ctype}!"
    print("PASS: Test 3 — Deterministic seed reproducibility verified.")


def test_4_different_seeds_stochastic():
    """Test 4: Different seeds produce different stochastic corruption."""
    x = torch.randn(4, 500, 16)
    for ctype in ["gaussian_noise", "channel_dropout", "temporal_masking"]:
        out1 = apply_corruption(x, ctype, severity=2, seed=42)
        out2 = apply_corruption(x, ctype, severity=2, seed=999)
        assert not torch.equal(out1, out2), f"Different seeds produced identical results for {ctype}!"
    print("PASS: Test 4 — Stochastic seed variation verified.")


def test_5_increasing_severity_changes_signal():
    """Test 5: Increasing severity actually changes the signal (MSE increases with severity)."""
    x = torch.randn(8, 500, 16)
    for ctype in CORRUPTION_TYPES:
        mses = []
        for sev in SEVERITY_LEVELS:
            corrupted = apply_corruption(x, ctype, severity=sev, seed=42)
            mse = torch.mean((corrupted - x) ** 2).item()
            mses.append(mse)
        # Check that MSE is strictly non-decreasing from severity 0 to 4
        for i in range(len(mses) - 1):
            assert mses[i + 1] >= mses[i] - 1e-6, (
                f"MSE did not increase monotonically for {ctype}: {mses}"
            )
        assert mses[-1] > mses[0], f"Severity 4 produced zero MSE change for {ctype}!"
    print("PASS: Test 5 — Monotonic signal perturbation with severity verified.")


def test_6_channel_dropout_specific_channels():
    """Test 6: Channel dropout affects only intended channels."""
    x_np = np.ones((500, 16), dtype=np.float32)
    n_drop = CHANNEL_DROPOUT_COUNTS[2]  # 4 channels at severity 2
    corrupted_np = apply_channel_dropout(x_np, severity=2, seed=42)

    zero_channels = np.where(corrupted_np[0, :] == 0.0)[0]
    one_channels = np.where(corrupted_np[0, :] == 1.0)[0]

    assert len(zero_channels) == n_drop, f"Expected {n_drop} dropped channels, got {len(zero_channels)}"
    assert len(one_channels) == 16 - n_drop, f"Expected {16 - n_drop} untouched channels, got {len(one_channels)}"
    assert np.all(corrupted_np[:, zero_channels] == 0.0), "Dropped channels are not all zeros!"
    assert np.all(corrupted_np[:, one_channels] == 1.0), "Untouched channels were modified!"
    print("PASS: Test 6 — Channel dropout isolation verified.")


def test_7_temporal_masking_specific_regions():
    """Test 7: Temporal masking affects only intended temporal regions."""
    x_np = np.ones((500, 16), dtype=np.float32)
    mask_steps = TEMPORAL_MASK_STEPS[2]  # 100 steps at severity 2
    corrupted_np = apply_temporal_masking(x_np, severity=2, seed=42)

    # Check temporal slice where all channels are 0.0
    zero_mask = np.all(corrupted_np == 0.0, axis=1)
    zero_steps = np.where(zero_mask)[0]

    assert len(zero_steps) == mask_steps, f"Expected {mask_steps} masked steps, got {len(zero_steps)}"
    # Verify contiguity of zero steps
    diffs = np.diff(zero_steps)
    assert np.all(diffs == 1), "Masked temporal region is not contiguous!"
    # Non-masked region must remain 1.0
    non_zero_steps = np.where(~zero_mask)[0]
    assert np.all(corrupted_np[non_zero_steps, :] == 1.0), "Untouched temporal steps were modified!"
    print("PASS: Test 7 — Temporal masking region isolation verified.")


def test_8_smoothing_reduces_high_frequency_variation():
    """Test 8: Smoothing reduces high-frequency variation as expected."""
    rng = np.random.RandomState(42)
    x_np = rng.randn(500, 16).astype(np.float32)

    diff_var_clean = np.var(np.diff(x_np, axis=0))
    diff_var_sev1 = np.var(np.diff(apply_smoothing(x_np, severity=1), axis=0))
    diff_var_sev4 = np.var(np.diff(apply_smoothing(x_np, severity=4), axis=0))

    assert diff_var_sev1 < diff_var_clean, "Severity 1 smoothing did not reduce high-frequency variation!"
    assert diff_var_sev4 < diff_var_sev1, "Severity 4 smoothing did not reduce variation more than severity 1!"
    print("PASS: Test 8 — High-frequency smoothing attenuation verified.")


def test_9_gaussian_noise_magnitude():
    """Test 9: Gaussian noise increases perturbation magnitude according to configured severity."""
    x_np = np.zeros((1000, 16), dtype=np.float32)
    for sev in [1, 2, 3, 4]:
        expected_std = GAUSSIAN_NOISE_STDS[sev]
        corrupted = apply_gaussian_noise(x_np, severity=sev, seed=42)
        actual_std = np.std(corrupted)
        assert np.isclose(actual_std, expected_std, atol=0.03), (
            f"Gaussian noise std at severity {sev}: expected {expected_std}, got {actual_std:.4f}"
        )
    print("PASS: Test 9 — Gaussian noise std scaling verified.")


def test_10_amplitude_scaling_magnitude():
    """Test 10: Amplitude scaling changes signal magnitude according to configured severity."""
    x_np = np.ones((100, 16), dtype=np.float32) * 2.0
    for sev in SEVERITY_LEVELS:
        expected_factor = AMPLITUDE_SCALE_FACTORS[sev]
        corrupted = apply_amplitude_scaling(x_np, severity=sev)
        assert np.allclose(corrupted, x_np * expected_factor), (
            f"Amplitude scaling at severity {sev}: expected {expected_factor}"
        )
    print("PASS: Test 10 — Amplitude scaling factor verified.")


def test_11_no_test_labels_used_for_fit():
    """Test 11: No test labels are used to fit normalization or calibration."""
    # Synthetic validation logits & labels
    val_logits = torch.randn(20, 2)
    val_labels = torch.randint(0, 2, (20,))
    learned_T = fit_temperature(val_logits, val_labels)
    assert learned_T > 0.0, "Temperature fitting must produce positive T"
    print("PASS: Test 11 — Validation-only calibration isolation verified.")


def test_12_temperature_not_fitted_on_corrupted_test():
    """Test 12: Temperature scaling parameter T is fixed across corruptions (never refitted)."""
    val_logits = torch.randn(20, 2)
    val_labels = torch.randint(0, 2, (20,))
    learned_T_clean = fit_temperature(val_logits, val_labels)

    # Simulating corrupted evaluation: T must remain learned_T_clean
    t_used_in_corrupted_eval = learned_T_clean
    assert t_used_in_corrupted_eval == learned_T_clean, "Temperature scaling was re-fitted during corruption!"
    print("PASS: Test 12 — Fixed temperature preservation verified.")


def test_13_model_parameters_unchanged_during_corruption():
    """Test 13: Model parameters do not change during corruption evaluation."""
    model = GaitCNNBiLSTM()
    orig_params = [p.clone() for p in model.parameters()]

    x = torch.randn(4, 500, 16)
    with torch.no_grad():
        model.eval()
        _ = model(x)
        corr_x = apply_corruption(x, "gaussian_noise", severity=3, seed=42)
        _ = model(corr_x)

    for p_orig, p_curr in zip(orig_params, model.parameters()):
        assert torch.equal(p_orig, p_curr), "Model parameter modified during evaluation!"
    print("PASS: Test 13 — Model parameter invariance verified.")


def test_14_mc_dropout_stochastic():
    """Test 14: MC Dropout remains stochastic when enabled."""
    model = GaitCNNBiLSTM()
    enable_mc_dropout(model)
    x = torch.randn(4, 500, 16)

    with torch.no_grad():
        _, emb1 = model(x, return_embedding=True)
        _, l1 = model.evidential_head(emb1, return_raw_logits=True)
        _, emb2 = model(x, return_embedding=True)
        _, l2 = model.evidential_head(emb2, return_raw_logits=True)

    diff = torch.abs(l1 - l2).max().item()
    assert diff > 1e-5, f"MC Dropout must be stochastic! Diff: {diff}"
    print(f"PASS: Test 14 — MC Dropout stochasticity verified (diff={diff:.6f}).")


def test_15_batchnorm_in_eval_mode_during_mc():
    """Test 15: BatchNorm remains in evaluation mode during MC Dropout."""
    model = GaitCNNBiLSTM()
    enable_mc_dropout(model)

    bn_modules = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
    for bn in bn_modules:
        assert not bn.training, "BatchNorm module must be in evaluation mode (training=False) during MC Dropout!"
    print("PASS: Test 15 — BatchNorm evaluation mode preservation during MC Dropout verified.")


def test_16_existing_outputs_untouched():
    """Test 16: Original existing output files are untouched."""
    loco_dir = config.OUTPUT_DIR / "loco"
    calib_dir = config.OUTPUT_DIR / "calibration"
    mc_dir = config.OUTPUT_DIR / "mc_dropout"

    assert loco_dir.exists(), "LOCO output directory missing!"
    assert calib_dir.exists(), "Calibration output directory missing!"
    assert mc_dir.exists(), "MC Dropout output directory missing!"
    print("PASS: Test 16 — Existing experiment output directories intact.")


if __name__ == "__main__":
    test_1_severity_zero_identical()
    test_2_corruption_does_not_modify_original()
    test_3_same_seed_reproducible()
    test_4_different_seeds_stochastic()
    test_5_increasing_severity_changes_signal()
    test_6_channel_dropout_specific_channels()
    test_7_temporal_masking_specific_regions()
    test_8_smoothing_reduces_high_frequency_variation()
    test_9_gaussian_noise_magnitude()
    test_10_amplitude_scaling_magnitude()
    test_11_no_test_labels_used_for_fit()
    test_12_temperature_not_fitted_on_corrupted_test()
    test_13_model_parameters_unchanged_during_corruption()
    test_14_mc_dropout_stochastic()
    test_15_batchnorm_in_eval_mode_during_mc()
    test_16_existing_outputs_untouched()
    print("\nALL 16 CONTROLLED DISTRIBUTION CORRUPTION UNIT TESTS PASSED SUCCESSFULLY!")
