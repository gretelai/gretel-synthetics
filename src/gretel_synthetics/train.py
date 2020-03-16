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
import sentencepiece as spm
import tf_sentencepiece as tfs
from smart_open import open

from gretel_synthetics.model import build_sequential_model, compute_epsilon
from gretel_synthetics.config import BaseConfig


logging.basicConfig(
    format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
    level=logging.INFO)


def read_training_data(path):  # pragma: no cover
    return open(path, 'rb').read().decode(encoding='utf-8')


def train_rnn(store: BaseConfig):
    text = read_training_data(store.training_data)
    logging.info(f'Length of training data: {len(text)} characters')

    spm = train_tokenizer(store, text)
    dataset = create_dataset(store, text, spm)

    logging.info("Initializing model")
    model = build_sequential_model(
        vocab_size=len(spm),
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


def train_tokenizer(store: BaseConfig, text: str) -> spm.SentencePieceProcessor:
    vocab_size = 500
    character_coverage = 1.0

    logging.info("Training SentencePiece tokenizer")
    spm.SentencePieceTrainer.Train(
        f'--input={store.training_data} '
        f'--model_prefix=m '
        f'--vocab_size={vocab_size} '
        f'--character_coverage={character_coverage}')

    sp = spm.SentencePieceProcessor()
    sp.Load("m.model")

    # print sample output
    with open(store.training_data) as f:
        sample = f.readline()
    logging.info(f"Vocabulary size: {len(sp)} tokens")
    logging.info(
        '{} ---- sample tokens mapped to int ---- > {}'.format(
            repr(sample), ", ".join(sp.SampleEncodeAsPieces(sample, -1, 0.1))))
    return sp


def create_dataset(store: BaseConfig, text: str, sp: spm.SentencePieceProcessor) -> tf.data.Dataset:
    """
    Before training, we need to map strings to a numerical representation.
    Create two lookup tables: one mapping characters to numbers,
    and another for numbers to characters.
    """
    # Create training dataset
    char_dataset = tf.data.Dataset.from_tensor_slices(sp.EncodeAsIds(text))
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
