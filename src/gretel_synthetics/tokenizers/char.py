"""
Basic character encoder
"""
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import cloudpickle

from gretel_synthetics.tokenizers.base import BaseTokenizer, BaseTokenizerTrainer
from gretel_synthetics.base_config import BaseConfig


class CharTokenizerTrainer(BaseTokenizerTrainer):

    def _train(self):
        text = open(
            self.config.training_data_path, "rb"
        ).read().decode()
        vocab = sorted(set(text))[:self.vocab_size]
        char2idx = {u: i for i, u in enumerate(vocab)}
        idx2char = np.array(vocab)
        self._save_model(char2idx, idx2char)

    def _save_model(self, char2idx, idx2char):
        cloudpickle.dump(char2idx, open(Path(self.config.checkpoint_dir) / "char2idx.p", "wb"))
        cloudpickle.dump(idx2char, open(Path(self.config.checkpoint_dir) / "idx2char.p", "wb"))

    def _save_settings(self):
        with open(Path(self.config.checkpoint_dir) / self._settings_fname, "w") as fout:
            fout.write(
                json.dumps({
                    "vocab_size": self.vocab_size
                })
            )


class CharTokenizer(BaseTokenizer):

    _model: Dict[str, Any]

    @classmethod
    def load(cls, config: BaseConfig):
        model = {
            "char2idx": cloudpickle.load(
                open(Path(config.checkpoint_dir) / "char2idx.p", "rb")
            ),
            "idx2char": cloudpickle.load(
                open(Path(config.checkpoint_dir) / "idx2char.p", "rb")
            )
        }
        return cls(model)

    @property
    def total_vocab_size(self):
        return len(self._model["idx2char"])

    def encode_to_ids(self, data: str) -> List[int]:
        return [self._model["char2idx"][char] for char in data]

    def decode_from_ids(self, ids: List[int]) -> str:
        chars = [self._model["idx2char"][id] for id in ids]
        return "".join(chars)
