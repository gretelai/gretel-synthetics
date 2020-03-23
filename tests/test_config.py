from unittest.mock import patch
from pathlib import Path
import uuid

from gretel_synthetics.config import LocalConfig


@patch('gretel_synthetics.config.Path.mkdir')
def test_local_config(mkdir):
    target = uuid.uuid4().hex
    test_data_dir = Path(__file__).parent
    test_data_file = test_data_dir / 'data' / 'smol.txt'
    lc = LocalConfig(checkpoint_dir=target, input_data=test_data_file.as_posix())

    mkdir.assert_called
    assert lc.epochs == 30
    assert lc.input_data == test_data_file.as_posix()
    assert lc.training_data == Path(target, 'training_data.txt').as_posix()
    assert lc.tokenizer_model == Path(target, 'm.model').as_posix()
