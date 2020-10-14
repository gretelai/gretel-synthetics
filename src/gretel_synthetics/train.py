"""
Train a machine learning model, based on automatically annotated input data
to generate synthetic records.

In order to use this module you must first create a config and then pass that config
to the ``train_rnn`` function.
"""
import io
from contextlib import redirect_stdout
import logging
from pathlib import Path
import shutil
from typing import Tuple, Optional

import pandas as pd
import sentencepiece as spm
from smart_open import open
import tensorflow as tf
from tqdm import tqdm

from gretel_synthetics.model import build_sequential_model, compute_epsilon
from gretel_synthetics.config import BaseConfig, VAL_ACC, VAL_LOSS
from gretel_synthetics.generator import _load_model, NEWLINE


spm_logger = logging.getLogger("sentencepiece")
spm_logger.setLevel(logging.INFO)

logging.basicConfig(
    format="%(asctime)s : %(threadName)s : %(levelname)s : %(message)s",
    level=logging.INFO,
)


class _ModelHistory(tf.keras.callbacks.Callback):
    """
    Callback class to compute loss and accuracy during model training
    """

    def __init__(self, total_token_count: int, config: BaseConfig):
        self.total_token_count = total_token_count
        self.config = config
        self.losses = []
        self.accuracy = []
        self.epsilons = []
        self.deltas = []
        self.best = []

    def on_epoch_end(self, epoch, logs: dict = None):
        self.losses.append(logs.get(VAL_LOSS))
        self.accuracy.append(logs.get(VAL_ACC))
        # Account for tf-privacy library writing to stdout
        with redirect_stdout(io.StringIO()):
            eps, _ = compute_epsilon(self.total_token_count, self.config, epoch)
            self.epsilons.append(eps)

        # NOTE: this is just a list of the same value, but
        # is simpler for creating the history csv
        self.deltas.append(1 / float(self.total_token_count))

        # NOTE: When we do the final history saving, one of these items
        # will flip to 1 to denote the actual best model that is saved
        # This just seeds the column for now
        self.best.append(0)


def _save_history_csv(
    history: _ModelHistory,
    save_dir: str,
    dp: bool,
    best_col: str,
    best_val: Optional[float] = None,
):
    """
    Save model training history to CSV format
    """
    perplexity = [2 ** x for x in history.losses]
    df = pd.DataFrame(
        zip(
            range(len(history.losses)),
            history.losses,
            history.accuracy,
            perplexity,
            history.epsilons,
            history.deltas,
            history.best,
        ),
        columns=["epoch", VAL_LOSS, VAL_ACC, "perplexity", "epsilon", "delta", "best"],
    )

    if not dp:
        df.drop(["epsilon", "delta"], axis=1, inplace=True)

    # Grab that last idx in case we need to use it in lieu of finding
    # a better one
    last_idx = df.iloc[[-1]].index.values.astype(int)[0]

    # Here we want to find the row that contains the value "best_val" within
    # the specified row by "best_col". We are looking for the first occurance
    # of "best_val" in either the "loss" or "accuracy" column. This row will
    # then be marked as best and represents the final model weights that were
    # stored.
    if best_val is None:
        best_idx = last_idx
    else:
        try:
            best_idx = df[df[best_col] == best_val].index.values.astype(int)[0]
        except IndexError:
            best_idx = last_idx

    df.at[best_idx, "best"] = 1

    save_path = Path(save_dir) / "model_history.csv"
    logging.info(f"Saving model history to {save_path.name}")
    df.to_csv(save_path.as_posix(), index=False)


def train_rnn(store: BaseConfig):
    """
    Fit synthetic data model on training data.

    This will annotate the training data and create a new file that
    will be used to actually train on. The updated training data, model,
    checkkpoints, etc will all be saved in the location specified
    by your config.

    Args:
        store: An instance of one of the available configs that you
            previously created

    Returns:
        None
    """
    if not store.overwrite:  # pragma: no cover
        try:
            _load_model(store)
        except Exception:
            pass
        else:
            raise RuntimeError(
                "A model already exists in the checkpoint location, you must enable overwrite mode or delete the checkpoints first."  # noqa
            )  # noqa

    text = _annotate_training_data(store)
    sp = _train_tokenizer(store)
    total_token_count, dataset = _create_dataset(store, text, sp)
    logging.info("Initializing synthetic model")
    model = build_sequential_model(
        vocab_size=len(sp), batch_size=store.batch_size, store=store
    )

    # Save checkpoints during training
    checkpoint_prefix = (Path(store.checkpoint_dir) / "synthetic").as_posix()
    if store.save_all_checkpoints:
        checkpoint_prefix = checkpoint_prefix + "-{epoch}"

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True,
        monitor=store.best_model_metric,
        save_best_only=store.save_best_model,
    )
    history_callback = _ModelHistory(total_token_count, store)

    _callbacks = [checkpoint_callback, history_callback]

    if store.early_stopping:
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor=store.best_model_metric,
            patience=store.early_stopping_patience,
            restore_best_weights=store.save_best_model,
        )
        _callbacks.append(early_stopping_callback)

    model.fit(dataset, epochs=store.epochs, callbacks=_callbacks)

    best_val = None
    if store.save_best_model:
        best_val = checkpoint_callback.best
    if store.early_stopping:
        # NOTE: In this callback, the "best" attr does not get set in the constructor, so we'll
        # set it to None if for some reason we can't get it. This also covers a test case that doesn't
        # run any epochs but accesses this attr.
        try:
            best_val = early_stopping_callback.best
        except AttributeError:
            best_val = None

    _save_history_csv(
        history_callback,
        store.checkpoint_dir,
        store.dp,
        store.best_model_metric,
        best_val,
    )
    store.save_model_params()
    logging.info(f"Saving model to {tf.train.latest_checkpoint(store.checkpoint_dir)}")


def _annotate_training_data(store: BaseConfig):
    """
    Prepare training data for tokenization with SentencePiece model.
    Including: use reserved tokens <n> to indicate end of sentences in training data.
    """
    # required for SentencePiece to tokenize newline characters
    logging.info(f"Loading training data from {store.input_data_path}")
    training_text = []
    with open(store.input_data_path, "r", encoding="utf-8", errors="replace") as infile:
        for line in infile:
            if store.max_lines and len(training_text) >= store.max_lines:
                break
            if store.field_delimiter is not None:
                line = line.strip().replace(
                    store.field_delimiter, store.field_delimiter_token
                )
            training_text.append(line)

    logging.info(f"Storing annotations to {Path(store.training_data).name}")
    labeled_text = ""
    with open(store.training_data, "w") as f:
        for sample in training_text:
            chunk = f"{sample}{NEWLINE}\n"
            f.write(chunk)
            labeled_text += chunk
    logging.info(
        f"Dataset size: {len(training_text)} lines, {len(labeled_text)} characters"
    )
    return labeled_text


def _move_tokenizer_model(store: BaseConfig):
    """
    Move SentencePiece tokenizer to model storage directory
    """
    for model in ["model", "vocab"]:
        src = Path.cwd() / f"{store.tokenizer_prefix}.{model}"
        dst = Path(store.checkpoint_dir) / f"{store.tokenizer_prefix}.{model}"
        shutil.move(src.as_posix(), dst.as_posix())


def _train_tokenizer(store: BaseConfig) -> spm.SentencePieceProcessor:
    """
    Trains SentencePiece tokenizer on training data
    """
    logging.info("Training SentencePiece tokenizer")
    spm.SentencePieceTrainer.Train(
        input=store.training_data,
        model_prefix=store.tokenizer_prefix,
        user_defined_symbols=[NEWLINE, store.field_delimiter_token],
        vocab_size=store.vocab_size,
        hard_vocab_limit=False,
        max_sentence_length=store.max_line_len,
        input_sentence_size=store.pretrain_sentence_count,
        shuffle_input_sentence=True,
        character_coverage=store.character_coverage,
    )
    _move_tokenizer_model(store)

    sp = spm.SentencePieceProcessor()
    logging.info(f"Loading tokenizer from: {Path(store.tokenizer_model).name}")
    sp.Load(store.tokenizer_model)

    # print sample output
    with open(store.training_data) as f:
        sample = f.readline().strip()
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
    logging.info(
        f"Saving SentencePiece model to {store.tokenizer_prefix}.model and {store.tokenizer_prefix}.vocab"
    )
    return sp


def _create_dataset(
    store: BaseConfig, text: str, sp: spm.SentencePieceProcessor
) -> Tuple[int, tf.data.Dataset]:
    """
    Before training, we need to map strings to a numerical representation.
    Create two lookup tables: one mapping characters to numbers,
    and another for numbers to characters.
    """
    logging.info("Tokenizing training data")
    ids = []
    total_token_count = 0
    for line in tqdm(text.split("\n")):
        _tokens = sp.EncodeAsIds(line)
        ids.extend(_tokens)
        total_token_count += len(_tokens)

    logging.info("Creating and shuffling tensorflow dataset")
    char_dataset = tf.data.Dataset.from_tensor_slices(ids)
    sequences = char_dataset.batch(store.seq_length + 1, drop_remainder=True)
    dataset = sequences.map(_split_input_target)
    dataset = dataset.shuffle(store.buffer_size).batch(
        store.batch_size, drop_remainder=True
    )
    return total_token_count, dataset


def _split_input_target(chunk: str) -> Tuple[str, str]:
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
