"""
Constants
"""
from enum import Enum

NEWLINE = "<n>"
MODEL_PARAMS = "model_params.json"
VAL_LOSS = "loss"
VAL_ACC = "accuracy"
MODEL_TYPE = "model_type"
TRAINING_DATA = "training_data.txt"
TESTING_DATA = "testing_data.txt"
MODEL_PREFIX = "m"


class Data(Enum):
    train = 1
    test = 2
