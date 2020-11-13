from pathlib import Path
import pytest

from gretel_synthetics.config import config_from_model_dir, TensorFlowConfig

test_data_dir = Path(__file__).parent / "data"


@pytest.mark.parametrize("model_name,dp,expected_learning_rate",
                         [("non-dp-model", False, 0.01),
                          ("dp-model", True, 0.001)])
def test_load_legacy_config(model_name, dp, expected_learning_rate):
    legacy_model_dir = test_data_dir / '0.14.x' / model_name

    config = config_from_model_dir(legacy_model_dir)

    assert isinstance(config, TensorFlowConfig)
    assert 'dp_learning_rate' not in config.__dict__
    assert config.learning_rate == expected_learning_rate
    assert config.dp == dp
