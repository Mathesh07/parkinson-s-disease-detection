"""Unit tests verifying isolation of Voice baseline from existing project assets."""

import pytest

from Voice.baselines.base_paper_cnn import config


def test_voice_baseline_isolation():
    """Verify existing Voice checkpoints and outputs remain untouched."""
    exist_metrics = config.VOICE_DIR / "results" / "metrics.json"
    assert exist_metrics.exists(), "Existing Voice metrics.json must not be modified!"

    exist_model_dir = config.VOICE_DIR / "results" / "best_wav2vec2_model"
    assert exist_model_dir.exists(), "Existing Wav2Vec2 directory must remain intact!"

    assert config.BEST_MODEL_PATH != exist_model_dir / "best_model.pt"
    assert config.METRICS_JSON_PATH != exist_metrics
