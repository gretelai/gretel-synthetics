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
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import sentencepiece as spm
from smart_open import open
import tensorflow as tf
from tqdm import tqdm

from gretel_synthetics.model import build_sequential_model, compute_epsilon
from gretel_synthetics.config import BaseConfig

spm_logger = logging.getLogger('sentencepiece')
spm_logger.setLevel(logging.INFO)

logging.basicConfig(
    format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
    level=logging.INFO)


class LossHistory(tf.keras.callbacks.Callback):
    """
    Callback class to compute loss and accuracy during model training
    """
    def __init__(self):
        self.losses = []
        self.accuracy = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('accuracy'))


def save_history_csv(history: LossHistory, save_dir: str):
    """
    Save model training history to CSV format
    """
    loss = [np.average(x) for x in history.losses]
    accuracy = [np.average(x) for x in history.accuracy]
    perplexity = [2**np.average(x) for x in history.losses]
    df = pd.DataFrame(zip(range(len(loss)), loss, accuracy, perplexity),
                      columns=['epoch', 'loss', 'accuracy', 'perplexity'])

    save_path = Path(save_dir) / 'model_history.csv'
    logging.info(f"Saving model history to {save_path.name}")
    df.to_csv(save_path.as_posix(), index=False)


def train_rnn(store: BaseConfig):
    """
    Fit synthetic data model on training data
    """
    text = annotate_training_data(store)
    sp = train_tokenizer(store)
    dataset = create_dataset(store, text, sp)
    logging.info("Initializing synthetic model")
    model = build_sequential_model(
        vocab_size=len(sp),
        batch_size=store.batch_size,
        store=store
        )

    # Save checkpoints during training
    checkpoint_prefix = (Path(store.checkpoint_dir) / "synthetic").as_posix()
    if store.save_all_checkpoints:
        checkpoint_prefix = checkpoint_prefix + "-{epoch}"

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True,
        monitor='accuracy'
    )
    history_callback = LossHistory()

    model.fit(dataset, epochs=store.epochs, callbacks=[checkpoint_callback, history_callback])
    save_history_csv(history_callback, store.checkpoint_dir)
    store.save_model_params()
    logging.info(f"Saving model to {tf.train.latest_checkpoint(store.checkpoint_dir)}")

    if store.dp:
        logging.info(compute_epsilon(len(text), store))


def annotate_training_data(store: BaseConfig):
    """
    Prepare training data for tokenization with SentencePiece model.
    Including: use reserved tokens <n> to indicate end of sentences in training data.
    """
    # required for SentencePiece to tokenize newline characters
    logging.info(f"Loading training data from {store.input_data_path}")
    training_text = []
    with open(store.input_data_path, 'r', encoding='utf-8', errors='replace') as infile:
        for line in infile:
            if store.max_lines and len(training_text) >= store.max_lines:
                break
            line = line.strip().replace(",", "<c>")
            training_text.append(line)

    logging.info(f"Storing annotations to {Path(store.training_data).name}")
    labeled_text = ''
    with open(store.training_data, 'w') as f:
        for sample in training_text:
            chunk = f"{sample}<n>\n"
            f.write(chunk)
            labeled_text += chunk
    logging.info(f"Dataset size: {len(training_text)} lines, {len(labeled_text)} characters")
    return labeled_text


def move_tokenizer_model(store: BaseConfig):
    """
    Move SentencePiece tokenizer to model storage directory
    """
    for model in ['model', 'vocab']:
        src = Path.cwd() / f'{store.tokenizer_prefix}.{model}'
        dst = Path(store.checkpoint_dir) / f'{store.tokenizer_prefix}.{model}'
        shutil.move(src.as_posix(), dst.as_posix())


def train_tokenizer(store: BaseConfig) -> spm.SentencePieceProcessor:
    """
    Trains SentencePiece tokenizer on training data
    """
    logging.info("Training SentencePiece tokenizer")
    spm.SentencePieceTrainer.Train(
        f'--input={store.training_data} '
        f'--model_prefix={store.tokenizer_prefix} '
        f'--user_defined_symbols=<n>,<c> '
        f'--vocab_size={store.vocab_size} '
        f'--hard_vocab_limit=false '
        f'--character_coverage={store.character_coverage}')
    move_tokenizer_model(store)

    sp = spm.SentencePieceProcessor()
    logging.info(f"Loading tokenizer from: {Path(store.tokenizer_model).name}")
    sp.Load(store.tokenizer_model)

    # print sample output
    with open(store.training_data) as f:
        sample = f.readline().strip()
    logging.info(f"Tokenizer model vocabulary size: {len(sp)} tokens")
    logging.info(
        'Mapping first line of training data\n\n{}\n ---- sample tokens mapped to pieces ---- > \n{}\n'.format(
            repr(sample), ", ".join(sp.SampleEncodeAsPieces(sample, -1, 0.1))))
    logging.info(
        'Mapping first line of training data\n\n{}\n ---- sample tokens mapped to int ---- > \n{}\n'.format(
            repr(sample), ", ".join([str(idx) for idx in sp.EncodeAsIds(sample)])))
    logging.info(f"Saving SentencePiece model to {store.tokenizer_prefix}.model and {store.tokenizer_prefix}.vocab")
    return sp


def create_dataset(store: BaseConfig, text: str, sp: spm.SentencePieceProcessor) -> tf.data.Dataset:
    """
    Before training, we need to map strings to a numerical representation.
    Create two lookup tables: one mapping characters to numbers,
    and another for numbers to characters.
    """
    logging.info("Tokenizing training data")
    ids = []
    for line in tqdm(text.split("\n")):
        ids.extend(sp.EncodeAsIds(line))

    logging.info("Creating and shuffling tensorflow dataset")
    char_dataset = tf.data.Dataset.from_tensor_slices(ids)
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
