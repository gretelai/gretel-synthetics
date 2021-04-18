from copy import deepcopy

from gretel_synthetics.train import _create_default_tokenizer
from gretel_synthetics.tokenizers import (
    SentencePieceTokenizerTrainer,
    CharTokenizerTrainer
)


def test_default_tokenizer(tf_config):
    config = deepcopy(tf_config)

    config.vocab_size = 15
    tokenizer = _create_default_tokenizer(config)
    assert(isinstance(tokenizer, SentencePieceTokenizerTrainer))

    config.vocab_size = 0
    tokenizer = _create_default_tokenizer(config)
    assert(isinstance(tokenizer, CharTokenizerTrainer))
