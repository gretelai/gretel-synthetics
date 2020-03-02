#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""text_generator.py
    Edit the default_config.yaml to get started.

    Sources:
        * https://www.tensorflow.org/tutorials/text/text_generation
        * http://karpathy.github.io/2015/05/21/rnn-effectiveness/
"""
import logging
import numpy as np
import pickle
import tensorflow as tf
from collections import namedtuple

from gretel_synthetics.config import BaseConfig
from gretel_synthetics.model import build_sequential_model

pred_string = namedtuple('pred_string', ['data'])
gen_text = namedtuple('gen_text', ['valid', 'text', 'explain'])


logging.basicConfig(
    format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
    level=logging.INFO)


def prepare_model(char2idx, batch_size, store):  # pragma: no cover
    model = build_sequential_model(
                vocab_size=len(char2idx),
                batch_size=1,
                store=store)

    load_dir = store.checkpoint_dir

    model.load_weights(
        tf.train.latest_checkpoint(
            load_dir)).expect_partial()

    model.build(tf.TensorShape([1, None]))
    model.summary()

    return model


def gen_text_factory(text, valid, explain):
    return dict(
        gen_text(valid, text, explain)._asdict()
    )


def generate_text(store: BaseConfig, start_string="\n", line_validator=None):
    logging.info(
        f"Latest checkpoint: {tf.train.latest_checkpoint(store.checkpoint_dir)}")  # noqa

    # Restore the latest model
    char2idx = pickle.load(open(store.char2idx, "rb"))
    idx2char = pickle.load(open(store.idx2char, "rb"))

    model = prepare_model(char2idx, 1, store)

    lines_generated = 0

    while True:
        rec = predict_chars(model, char2idx, idx2char, start_string, store).data
        try:
            if not line_validator:
                yield gen_text_factory(rec, None, None)
            else:
                line_validator(rec)
                yield gen_text_factory(rec, True, None)
        except Exception as err:
            # logging.warning(f'Line failed validation: {rec} errored with {str(err)}')
            yield gen_text_factory(rec, False, str(err))
        finally:
            lines_generated += 1

        if lines_generated >= store.gen_lines:
            break


def predict_chars(model: tf.keras.Sequential,
                  char2idx: dict, idx2char: np.array, start_string: str,
                  store: BaseConfig) -> str:
    """
    Evaluation step (generating text using the learned model).

    Args:
        model: tf.keras.Sequential model
        char2idx: character to index mapping
        idx2char: index to character mapping
        start_string: string to bootstrap model
        store: our config object
    Returns:
        Yields line of text per iteration
    """

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store each line
    text_generated = ""

    # Here batch size == 1
    model.reset_states()

    while True:
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to
        # predict the word returned by the model
        predictions = predictions / store.gen_temp
        predicted_id = tf.random.categorical(
            predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        if idx2char[predicted_id] == "\n":
            return pred_string(text_generated)
        elif store.gen_chars > 0 and len(text_generated) >= store.gen_chars:
            return pred_string(text_generated)
        else:
            text_generated += idx2char[predicted_id]
