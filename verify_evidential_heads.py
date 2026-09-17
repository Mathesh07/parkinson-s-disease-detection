"""Verification test script for STEP 2: Evidential Head Implementation across all 3 Modalities."""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F

# Add workspace roots
base_dir = Path(__file__).resolve().parent
for sub in ["Gait", "Voice", "Handwriting"]:
    p = str(base_dir / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

print("=" * 70)
print("VERIFYING EVIDENTIAL HEAD IMPLEMENTATION ACROSS ALL 3 MODALITIES")
print("=" * 70)

# -------------------------------------------------------------
# 1. GAIT MODALITY (1D-CNN + BiLSTM)
# -------------------------------------------------------------
print("\n[1/3] Testing Gait Evidential Model (GaitCNNBiLSTM)...")
from Gait.model import GaitCNNBiLSTM
import Gait.config as gait_cfg

gait_model = GaitCNNBiLSTM()
gait_model.eval()

dummy_gait = torch.randn(4, gait_cfg.WINDOW_SIZE, gait_cfg.NUM_CHANNELS)  # (4, 500, 16)
with torch.no_grad():
    gait_evidence, gait_emb = gait_model(dummy_gait, return_embedding=True)

print(f"  * Input Shape:       {tuple(dummy_gait.shape)}")
print(f"  * Embedding Shape:   {tuple(gait_emb.shape)} (Expected: [4, 128])")
print(f"  * Evidence Shape:    {tuple(gait_evidence.shape)} (Expected: [4, 2])")
print(f"  * Min Evidence Value: {gait_evidence.min().item():.6f} (Must be >= 0)")
print(f"  * Sample Evidence:   e_HC={gait_evidence[0,0].item():.4f}, e_PD={gait_evidence[0,1].item():.4f}")
print(f"  * Evidence Sum:      {gait_evidence[0].sum().item():.4f} (Not forced to 1, since Softmax is not applied)")

assert gait_evidence.shape == (4, 2), "Gait evidence shape mismatch!"
assert (gait_evidence >= 0).all(), "Gait evidence contains negative values!"
assert not torch.allclose(gait_evidence.sum(dim=-1), torch.ones(4)), "Softmax should not be applied to evidence!"
print("  [PASS] Gait Evidential Head PASSES all criteria!")

# -------------------------------------------------------------
# 2. HANDWRITING MODALITY (Vision Transformer)
# -------------------------------------------------------------
print("\n[2/3] Testing Handwriting Evidential Model (ViTBinaryClassifier)...")
from Handwriting.model import ViTBinaryClassifier

vit_model = ViTBinaryClassifier()
vit_model.eval()

dummy_img = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    vit_evidence = vit_model(dummy_img)

print(f"  * Input Shape:       {tuple(dummy_img.shape)}")
print(f"  * Evidence Shape:    {tuple(vit_evidence.shape)} (Expected: [2, 2])")
print(f"  * Min Evidence Value: {vit_evidence.min().item():.6f} (Must be >= 0)")
print(f"  * Sample Evidence:   e_HC={vit_evidence[0,0].item():.4f}, e_PD={vit_evidence[0,1].item():.4f}")
print(f"  * Evidence Sum:      {vit_evidence[0].sum().item():.4f} (Not forced to 1)")

assert vit_evidence.shape == (2, 2), "ViT evidence shape mismatch!"
assert (vit_evidence >= 0).all(), "ViT evidence contains negative values!"
assert not torch.allclose(vit_evidence.sum(dim=-1), torch.ones(2)), "Softmax should not be applied to evidence!"
print("  [PASS] Handwriting ViT Evidential Head PASSES all criteria!")

# -------------------------------------------------------------
# 3. VOICE MODALITY (Wav2Vec2)
# -------------------------------------------------------------
print("\n[3/3] Testing Voice Evidential Model (Wav2Vec2ForParkinsons)...")
from Voice.model import Wav2Vec2ForParkinsons

voice_model = Wav2Vec2ForParkinsons()
voice_model.eval()

dummy_audio = torch.randn(2, 16000)  # 1s audio
with torch.no_grad():
    voice_evidence, voice_emb = voice_model(dummy_audio, return_embedding=True)

print(f"  * Input Shape:       {tuple(dummy_audio.shape)}")
print(f"  * Embedding Shape:   {tuple(voice_emb.shape)} (Expected: [2, 768])")
print(f"  * Evidence Shape:    {tuple(voice_evidence.shape)} (Expected: [2, 2])")
print(f"  * Min Evidence Value: {voice_evidence.min().item():.6f} (Must be >= 0)")
print(f"  * Sample Evidence:   e_HC={voice_evidence[0,0].item():.4f}, e_PD={voice_evidence[0,1].item():.4f}")
print(f"  * Evidence Sum:      {voice_evidence[0].sum().item():.4f} (Not forced to 1)")

assert voice_evidence.shape == (2, 2), "Voice evidence shape mismatch!"
assert (voice_evidence >= 0).all(), "Voice evidence contains negative values!"
assert not torch.allclose(voice_evidence.sum(dim=-1), torch.ones(2)), "Softmax should not be applied to evidence!"
print("  [PASS] Voice Wav2Vec2 Evidential Head PASSES all criteria!")

print("\n" + "=" * 70)
print("ALL 3 UNIMODAL EVIDENTIAL HEADS SUCCESSFULLY VERIFIED!")
print("=" * 70)
