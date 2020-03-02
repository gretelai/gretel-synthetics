#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_rnn.py
    Train a character-based RNN to generate text.
    Edit the default_config.yaml to get started.
    Sources:
        * https://www.tensorflow.org/tutorials/text/text_generation
        * http://karpathy.github.io/2015/05/21/rnn-effectiveness/
"""
import logging
import numpy as np
import os
import pickle

import tensorflow as tf
from smart_open import open

from gretel_synthetics.model import build_sequential_model, compute_epsilon
from gretel_synthetics.config import BaseConfig


logging.basicConfig(
    format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
    level=logging.INFO)


def create_vocab(store: BaseConfig, text):
    if store.max_chars > 0:
        text = text[:store.max_chars]
    vocab = sorted(set(text))
    return vocab


def read_training_data(path):  # pragma: no cover
    return open(path, 'rb').read().decode(encoding='utf-8')


def train_rnn(store: BaseConfig):
    text = read_training_data(store.training_data)
    vocab = create_vocab(store, text)
    logging.info(f'Length of training data: {len(text)} characters')
    logging.info(f'Training set contains: {len(vocab)} unique characters')

    dataset = create_dataset(store, text, vocab)

    logging.info("Initializing model")
    model = build_sequential_model(
        vocab_size=len(vocab),
        batch_size=store.batch_size,
        store=store
        )

    # Save checkpoints during training
    checkpoint_prefix = os.path.join(store.checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True
    )

    all_cbs = [checkpoint_callback]

    logging.info("Training model")
    model.fit(dataset, epochs=store.epochs, callbacks=all_cbs)
    logging.info(
        f"Wrote latest checkpoint to disk: {tf.train.latest_checkpoint(store.checkpoint_dir)}")

    if store.dp:
        logging.info(compute_epsilon(len(text), store))
    else:
        logging.info('Trained with non-private Adam optimizer')


def create_dataset(store: BaseConfig, text: str, vocab: list) -> tf.data.Dataset:
    """
    Before training, we need to map strings to a numerical representation.
    Create two lookup tables: one mapping characters to numbers,
    and another for numbers to characters.
    """
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    # text_as_int = np.array([char2idx[c] for c in text])
    tmp = []
    for c in text:
        try:
            tmp.append(char2idx[c])
        except KeyError:
            pass
    text_as_int = np.array(tmp)

    logging.info("Pickling vocabulary to disk")
    pickle.dump(char2idx, open(store.char2idx, "wb"))
    pickle.dump(idx2char, open(store.idx2char, "wb"))

    logging.info("Character->int mapping (showing 20 chars)")
    logging.info('{')
    for char, _ in zip(char2idx, range(20)):
        logging.info('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
    logging.info('  ...\n}')

    # Show how the first 13 characters from the text are mapped to integers
    logging.info(
        '{} ---- characters mapped to int ---- > {}'.format(
            repr(text[:13]), text_as_int[:13]))

    # Create training dataset
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(store.seq_length + 1, drop_remainder=True)
    dataset = sequences.map(split_input_target)
    dataset = dataset.shuffle(
        store.buffer_size).batch(
            store.batch_size, drop_remainder=True)
    return dataset


def split_input_target(chunk: str) -> (str, str):
    """
    For each sequence, duplicate and shift it to form the input and target text
    by using the map method to apply a simple function to each batch:

    Examples:
        split_input_target("So hot right now")
        Returns: ('So hot right now', 'o hot right now.')
    """
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text
