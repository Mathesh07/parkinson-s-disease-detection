"""Unit test ensuring baseline does not modify existing Gait model checkpoints or metrics."""

from pathlib import Path
import pytest

from Gait.baselines.base_paper_cnn import config


def test_existing_artifacts_isolated():
    """Verify existing Gait output files remain intact."""
    existing_metrics = config.GAIT_DIR / "outputs" / "metrics" / "test_metrics.json"
    assert existing_metrics.exists(), "Existing test_metrics.json must remain untouched!"

    existing_model_config = config.GAIT_DIR / "checkpoints" / "model_config.json"
    assert existing_model_config.exists(), "Existing model_config.json must remain untouched!"

    # Ensure baseline paths do not equal existing paths
    assert config.BEST_MODEL_PATH != config.GAIT_DIR / "checkpoints" / "best_gait_model.pt"
    assert config.METRICS_JSON_PATH != existing_metrics
