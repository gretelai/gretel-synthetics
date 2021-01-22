from dataclasses import dataclass
from pathlib import Path

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
)

import gretel_synthetics.const as const
from gretel_synthetics.config import BaseConfig, BaseGenerator
from gretel_synthetics.transformer.train import train

@dataclass
class TransformerConfig(BaseConfig):
    """
    Huggingface transformer config that contains all the main parameters for
    training or fine-tuning a model and generating data
    """

    MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
    MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

    # Training arguments
    epochs: int = 5
    do_eval: bool = True
    do_train: bool = True
    save_steps: int = 5000
    seed: int = 42

    # Model config
    cache_dir: str = None
    config_name: str = None
    model_name_or_path: str = None
    model_revision: str = "main"
    model_type: str = None
    use_auth_token: str = None

    # Tokenizer config
    preprocessing_num_workers: int = None
    tokenizer_name: str = None
    use_fast_tokenizer: bool = True

    # Dataset config
    block_size: int = None
    dataset_name: str = None
    input_data_path: str = None
    overwrite_cache: bool = False
    validation_split_percentage: int = 5

    def get_generator_class(self) -> BaseGenerator:
        return NotImplemented
        pass

    def __post_init__(self):
        if not self.checkpoint_dir:
            raise AttributeError(
                "Must provide checkpoint_dir and input_data_path params!"
            )
        if not self.dataset_name and not self.input_data_path:
            raise AttributeError(
                "Must provide a dataset_name (huggingface.co hub) or input_data_path!"
            )

        if not Path(self.checkpoint_dir).exists():
            Path(self.checkpoint_dir).resolve().mkdir()
        self.training_data_path = Path(
            self.checkpoint_dir, const.TRAINING_DATA
        ).as_posix()

        # assign the model type for serialization
        self.model_type = self.__class__.__name__

    def get_training_callable(self):
        return train

    def gpu_check(self):
        pass