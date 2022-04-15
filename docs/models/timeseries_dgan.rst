Timeseries DGAN
===============

The Timeseries DGAN module contains a PyTorch implementation of the DoppelGANger
model, see https://arxiv.org/abs/1909.13403 for a detailed description of the
model.

.. code-block:: python

   import numpy as np
    from gretel_synthetics.timeseries_dgan.dgan import DGAN
    from gretel_synthetics.timeseries_dgan.config import DGANConfig

    attributes = np.random.rand(10000, 3)
    features = np.random.rand(10000, 20, 2)

    config = DGANConfig(
        max_sequence_len=20,
        sample_len=5,
        batch_size=1000,
        epochs=10
    )
    model = DGAN(config)

    model.train(attributes, features)

    synthetic_attributes, synthetic_features = model.generate(1000)

.. automodule:: gretel_synthetics.timeseries_dgan.config
    :members:

.. automodule:: gretel_synthetics.timeseries_dgan.dgan
    :special-members: __init__
    :members:
