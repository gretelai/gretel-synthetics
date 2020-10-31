"""
Google Sentencepiece Tokenizer
"""
import logging
from pathlib import Path
import shutil
import json
from typing import List

import sentencepiece as spm

from gretel_synthetics.tokenizers.base import BaseTokenizerTrainer, BaseTokenizer
from gretel_synthetics.const import NEWLINE
from gretel_synthetics.base_config import BaseConfig

spm_logger = logging.getLogger("sentencepiece")
spm_logger.setLevel(logging.INFO)

logging.basicConfig(
    format="%(asctime)s : %(threadName)s : %(levelname)s : %(message)s",
    level=logging.INFO,
)

MODEL_PREFIX = "m"


class SentencepieceTokenizerTrainer(BaseTokenizerTrainer):

    vocab_size: int
    character_coverage: float
    pretrain_sentence_count: int
    max_line_line: int

    def __init__(
        self,
        *,
        character_coverage: float = 1.0,
        pretrain_sentence_count: int = 1000000,
        max_line_len: int = 2048,
        **kwargs
    ):
        self.character_coverage = character_coverage
        self.pretrain_sentence_count = pretrain_sentence_count
        self.max_line_line = max_line_len

        super().__init__(**kwargs)

    def _annotate_training_line(self, line: str):
        if self.config.field_delimiter is not None:
            line = line.strip().replace(
                self.config.field_delimiter, self.config.field_delimiter_token
            )
            line += f"{NEWLINE}\n"
        else:
            line = line.strip() + NEWLINE + "\n"

        return line

    def _train(self):
        logging.info("Training SentencePiece tokenizer")
        spm.SentencePieceTrainer.Train(
            input=self.config.training_data_path,
            model_prefix=MODEL_PREFIX,
            user_defined_symbols=[NEWLINE, self.config.field_delimiter_token],
            vocab_size=self.vocab_size,
            hard_vocab_limit=False,
            max_sentence_length=self.max_line_line,
            input_sentence_size=self.pretrain_sentence_count,
            shuffle_input_sentence=True,
            character_coverage=self.character_coverage
        )

        # The training automatically saves to disk,
        # so we have to now load it back in after we move
        # it to our checkpoint directory.
        self._save_model()

    def _save_model(self):
        for part in ("model", "vocab"):
            src = Path.cwd() / f"{MODEL_PREFIX}.{part}"
            dst = Path(self.config.checkpoint_dir) / f"{MODEL_PREFIX}.{part}"
            shutil.move(src.as_posix(), dst.as_posix())

    def _save_settings(self):
        with open(Path(self.config.checkpoint_dir) / self._settings_fname, "w") as fout:
            fout.write(
                json.dumps({
                    "vocab_size": self.vocab_size,
                    "character_coverage": self.character_coverage,
                    "pretrain_sentence_count": self.pretrain_sentence_count,
                    "max_line_len": self.max_line_line
                })
            )


def _log_sample_data(config: BaseConfig, sp: spm.SentencePieceProcessor):
    training_data = Path(config.training_data_path)
    if not training_data.is_file():
        logging.info("Training data not found for SP sampling")
        return

    with open(config.training_data_path) as fin:
        sample = fin.readline().strip()

    logging.info(f"Tokenizer model vocabulary size: {len(sp)} tokens")
    logging.info(
        "Mapping first line of training data\n\n{}\n ---- sample tokens mapped to pieces ---- > \n{}\n".format(
            repr(sample), ", ".join(sp.SampleEncodeAsPieces(sample, -1, 0.1))
        )
    )
    logging.info(
        "Mapping first line of training data\n\n{}\n ---- sample tokens mapped to int ---- > \n{}\n".format(
            repr(sample), ", ".join([str(idx) for idx in sp.EncodeAsIds(sample)])
        )
    )


class SentencePieceTokenizer(BaseTokenizer):

    _model: spm.SentencePieceProcessor

    @classmethod
    def load(cls, config: BaseConfig):
        sp = spm.SentencePieceProcessor()
        model_fname = f"{MODEL_PREFIX}.model"
        logging.info("Loading tokenizer from: %s", model_fname)
        model_path = Path(config.checkpoint_dir) / model_fname
        sp.Load(str(model_path))
        _log_sample_data(config, sp)
        return cls(sp)

    @property
    def total_vocab_size(self):
        return len(self._model)

    def encode_to_ids(self, data: str) -> List[int]:
        return self._model.EncodeAsIds(data)

    def decode_from_ids(self, ids: List[int]) -> str:
        return self._model.DecodeIds(ids)
