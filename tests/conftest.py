import pytest
from pathlib import Path
import shutil

from gretel_synthetics.config import TensorFlowConfig

test_data_dir = Path(__file__).parent


@pytest.fixture(scope="session")
def tf_config():
    target = test_data_dir / "ckpoint"
    input_data = test_data_dir / "data" / "smol.txt"
    if not target.exists():
        target.mkdir()
    config = TensorFlowConfig(
        checkpoint_dir=target.as_posix(),
        input_data_path=input_data.as_posix(),
        field_delimiter=",",
        predict_batch_size=1,
        overwrite=True,
    )
    yield config
    shutil.rmtree(target)
