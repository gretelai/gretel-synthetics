#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""text_generator.py
    Edit the default_config.yaml to get started.

    Sources:
        * https://www.tensorflow.org/tutorials/text/text_generation
        * http://karpathy.github.io/2015/05/21/rnn-effectiveness/
"""
import logging
import sentencepiece as spm
import tensorflow as tf
from collections import namedtuple

from gretel_synthetics.config import BaseConfig
from gretel_synthetics.model import build_sequential_model

pred_string = namedtuple('pred_string', ['data'])
gen_text = namedtuple('gen_text', ['valid', 'text', 'explain'])


logging.basicConfig(
    format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
    level=logging.INFO)


def load_tokenizer(store: BaseConfig) -> spm.SentencePieceProcessor:
    logging.info("Loading SentencePiece tokenizer")
    sp = spm.SentencePieceProcessor()
    sp.Load(store.tokenizer_model)
    return sp


def prepare_model(sp: spm, batch_size: int, store: BaseConfig) -> tf.keras.Sequential:
    model = build_sequential_model(
                vocab_size=len(sp),
                batch_size=batch_size,
                store=store)

    load_dir = store.checkpoint_dir

    model.load_weights(
        tf.train.latest_checkpoint(
            load_dir)).expect_partial()

    model.build(tf.TensorShape([1, None]))
    model.summary()

    return model


def gen_text_factory(text: str, valid, explain) -> dict:
    return dict(
        gen_text(valid, text, explain)._asdict()
    )


def generate_text(store: BaseConfig, start_string="<n>", line_validator=None):
    logging.info(
        f"Latest checkpoint: {tf.train.latest_checkpoint(store.checkpoint_dir)}")  # noqa

    # Restore the latest SentencePiece model
    sp = load_tokenizer(store)

    # Load the RNN
    model = prepare_model(sp, 1, store)

    lines_generated = 0

    while True:
        rec = predict_chars(model, sp, start_string, store).data
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
                  sp: spm.SentencePieceProcessor,
                  start_string: str,
                  store: BaseConfig) -> str:
    """
    Evaluation step (generating text using the learned model).

    Args:
        model: tf.keras.Sequential model
        sp: SentencePiece tokenizer
        start_string: string to bootstrap model
        store: our config object
    Returns:
        Yields line of text per iteration
    """

    # Converting our start string to numbers (vectorizing)
    input_eval = sp.EncodeAsIds(start_string)
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store each line
    sentence_ids = []

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
        sentence_ids.append(int(predicted_id))

        decoded = sp.DecodeIds(sentence_ids)
        decoded = decoded.replace('<c>', ',')

        if '<n>' in decoded:
            return pred_string(decoded.replace('<n>', ''))
        elif 0 < store.gen_chars <= len(decoded):
            return pred_string(decoded)
