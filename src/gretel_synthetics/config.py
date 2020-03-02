import os
import logging
from abc import ABC, abstractmethod


logging.basicConfig(
    format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
    level=logging.INFO)


class BaseConfig(ABC):

    def __init__(self, *, max_chars=0, epochs=30, batch_size=64, buffer_size=10000, seq_length=100, embedding_dim=256,
                 rnn_units=256, dropout_rate=.2, rnn_initializer='glorot_uniform', dp=False, dp_learning_rate=0.015,
                 dp_noise_multiplier=1.1, dp_l2_norm_clip=1.0, dp_microbatches=256, gen_temp=1.0, gen_chars=0,
                 gen_lines=500):
        self.char2idx = None
        self.idx2char = None

        # Diff privacy configs
        self.dp = dp
        self.dp_learning_rate = dp_learning_rate
        self.dp_noise_multiplier = dp_noise_multiplier
        self.dp_l2_norm_clip = dp_l2_norm_clip
        self.dp_microbatches = dp_microbatches

        # Generative model configs
        self.max_chars = max_chars
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.epochs = epochs
        self.dropout_rate = dropout_rate
        self.rnn_initializer = rnn_initializer

        # Text generation settings
        self.gen_temp = gen_temp
        self.gen_chars = gen_chars
        self.gen_lines = gen_lines

        @abstractmethod
        def _gen_idxs(self):  # pragma: no cover
            pass


class LocalConfig(BaseConfig):

    def __init__(self, *, checkpoint_dir, training_data, **kwargs):
        self.checkpoint_dir = checkpoint_dir
        self.training_data = training_data
        super().__init__(**kwargs)

        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        self._set_idxs()

    def _set_idxs(self):
        self.char2idx = os.path.join(self.checkpoint_dir, 'char2idx.p')
        self.idx2char = os.path.join(self.checkpoint_dir, 'idx2char.p')
