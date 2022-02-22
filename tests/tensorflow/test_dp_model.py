from typing import Tuple

import numpy as np

from gretel_synthetics.tensorflow.dp_model import compute_dp_sgd_privacy, ORDERS


def test_compute_dp_sgd_privacy():
    out = compute_dp_sgd_privacy(
        n=2000,
        batch_size=128,
        noise_multiplier=0.01,
        epochs=50,
        delta=1 / 2000,
        orders=ORDERS,
    )
    assert np.isclose(out[0], 4060510)
    assert out[1] == 1.05
    assert len(out) == 2
    assert isinstance(out, Tuple)
