"""
Abstract model training module
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
    tokenizer_trainer: BaseTokenizerTrainer
    tokenizer: BaseTokenizer
    config: BaseConfig


def create_default_tokenizer(store: BaseConfig) -> SentencePieceTokenizerTrainer:
    trainer = SentencePieceTokenizerTrainer(
        vocab_size=store.vocab_size,
        character_coverage=store.character_coverage,
        pretrain_sentence_count=store.pretrain_sentence_count,
        max_line_len=store.max_line_len,
        config=store,
    )
    return trainer


def train(store: BaseConfig, tokenizer_trainer: Optional[BaseTokenizerTrainer] = None):
    """Train a Synthetic Model.
    """
    if tokenizer_trainer is None:
        tokenizer_trainer = create_default_tokenizer(store)
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
    FIXME: Facade pass through to maintain backwards compat. Just call into
    the new train interface
    """
    train(store)
