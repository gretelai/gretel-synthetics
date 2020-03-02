from unittest.mock import patch
import uuid
import os

from gretel_synthetics.config import LocalConfig


@patch('os.mkdir')
def test_local_config(mkdir):
    target = uuid.uuid4().hex
    lc = LocalConfig(checkpoint_dir=target, training_data='blah')
    mkdir.assert_called_with(target)
    assert lc.epochs == 30
    assert lc.training_data == 'blah'
    assert lc.char2idx == os.path.join(target, 'char2idx.p')