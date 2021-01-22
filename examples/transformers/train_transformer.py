#!/usr/bin/env python
# coding: utf-8

import json
import os
from pathlib import Path

from gretel_synthetics.train import train
from gretel_synthetics.config_transformer import TransformerConfig

# Create a config that we can use for both training and generating data
config = TransformerConfig(
    epochs=5,
    model_name_or_path='gpt2',
    do_train=True,
    do_eval=True,
    block_size=200,
    save_steps=5000,
    overwrite=True,
    checkpoint_dir=(Path.cwd() / 'checkpoints').as_posix(),
    input_data_path="../examples/data/safecast.txt"
)

# Train the model
train(config)
