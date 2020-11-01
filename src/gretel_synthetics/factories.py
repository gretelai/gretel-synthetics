"""
Create instances from serialized configs
"""
from pathlib import Path
import json

from gretel_synthetics.tensorflow.config import TensorFlowConfig
from gretel_synthetics.tokenizers.base import BaseTokenizerTrainer
from gretel_synthetics.tokenizers.sentencepiece import (
    SentencePieceTokenizer,
    SentencepieceTokenizerTrainer,
)
from gretel_synthetics.tokenizers.char import CharTokenizer, CharTokenizerTrainer
from gretel_synthetics.const import MODEL_PARAMS, MODEL_TYPE

CONFIG_MAP = {TensorFlowConfig.__name__: TensorFlowConfig}

TOK_MAP = {
    SentencepieceTokenizerTrainer.__name__: SentencePieceTokenizer,
    CharTokenizerTrainer.__name__: CharTokenizer,
}


def config_from_model_dir(model_dir: str):
    params_file = Path(model_dir) / MODEL_PARAMS
    params_dict = json.loads(open(params_file).read())
    model_type = params_dict.pop(MODEL_TYPE, None)
    if model_type is None:
        return TensorFlowConfig(**params_dict)
    cls = CONFIG_MAP[model_type]
    return cls(**params_dict)


def tokenizer_from_model_dir(model_dir: str):
    params_file = Path(model_dir) / BaseTokenizerTrainer.settings_fname
    params_dict = json.loads(open(params_file).read())
    tok_type = params_dict.pop(BaseTokenizerTrainer.tokenizer_type)
    tok_cls = TOK_MAP[tok_type]
    return tok_cls.load(model_dir)
