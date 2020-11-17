"""
Train models for creating synthetic data.  This module is the primary entrypoint for creating
a model. It depends on having created a engine specifc configuration and optionally a tokenizer
to be used.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from gretel_synthetics.tokenizers import SentencePieceTokenizerTrainer, tokenizer_from_model_dir

if TYPE_CHECKING:
    from gretel_synthetics.config import BaseConfig
    from gretel_synthetics.tokenizers import BaseTokenizerTrainer, BaseTokenizer
else:
    BaseConfig = None
    BaseTokenizerTrainer = None
    BaseTokenizer = None


@dataclass
class TrainingParams:
    """A structure that is created and passed into the engine-specific training
    entrypoint. All engine-specific training entrypoints should expect to receive
    this object and process accordingly.
    """
    tokenizer_trainer: BaseTokenizerTrainer
    tokenizer: BaseTokenizer
    config: BaseConfig


def _create_default_tokenizer(store: BaseConfig) -> SentencePieceTokenizerTrainer:
    trainer = SentencePieceTokenizerTrainer(
        vocab_size=store.vocab_size,
        character_coverage=store.character_coverage,
        pretrain_sentence_count=store.pretrain_sentence_count,
        max_line_len=store.max_line_len,
        config=store,
    )
    return trainer


def train(store: BaseConfig, tokenizer_trainer: Optional[BaseTokenizerTrainer] = None):
    """Train a Synthetic Model.  This is a facade entrypoint that implements the engine
    specific training operation based on the provided configuration.

    Args:
        store: A subclass instance of ``BaseConfig.`` This config is reponsible for
            providing the actual training entrypoint for a specific training routine.

        tokenizer_trainer: An optional subclass instance of a ``BaseTokenizerTrainer``.  If provided
            this tokenizer will be used to pre-process and create an annotated dataset for training.
            If not provided a default tokenizer will be used.
    """
    if tokenizer_trainer is None:
        tokenizer_trainer = _create_default_tokenizer(store)
    tokenizer_trainer.create_annotated_training_data()
    tokenizer_trainer.train()
    tokenizer = tokenizer_from_model_dir(store.checkpoint_dir)
    params = TrainingParams(
        tokenizer_trainer=tokenizer_trainer,
        tokenizer=tokenizer,
        config=store
    )
    train_fn = store.get_training_callable()
    store.save_model_params()
    train_fn(params)


def train_rnn(store: BaseConfig):
    """
    Facade to support backwards compatibility for <= 0.14.x versions.
    """
    train(store)
