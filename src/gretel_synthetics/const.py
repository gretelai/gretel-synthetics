"""
Constants
"""
from enum import Enum

NEWLINE = "<n>"
MODEL_PARAMS = "model_params.json"
METRIC_LOSS = "loss"
METRIC_ACCURACY = "accuracy"
METRIC_VAL_LOSS = "val_loss"
METRIC_VAL_ACCURACY = "val_accuracy"
METRIC_EPSILON = "epsilon"
METRIC_DELTA = "delta"
MODEL_TYPE = "model_type"
TRAINING_DATA = "training_data.txt"
MODEL_PREFIX = "m"


class Data(Enum):
    train = 1
    validate = 2
