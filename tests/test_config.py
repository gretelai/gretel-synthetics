from unittest.mock import patch
import uuid
import os

from gretel_synthetics.config import LocalConfig


@patch('os.mkdir')
def test_local_config(mkdir):
    target = uuid.uuid4().hex
    lc = LocalConfig(checkpoint_dir=target, input_data='blah')
    mkdir.assert_called_with(target)
    assert lc.epochs == 30
    assert lc.input_data == 'blah'
    assert lc.training_data == os.path.join(target, 'training_data.txt')
    assert lc.tokenizer_model == os.path.join(target, 'm.model')