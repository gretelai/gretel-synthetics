"""
Train a machine learning model, based on automatically annotated input data
to generate synthetic records.

In order to use this module you must first create a config and then pass that config
to the ``train_rnn`` function.
"""

import io
import logging
import time

from contextlib import redirect_stdout
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple, TYPE_CHECKING

import pandas as pd
import tensorflow as tf

from tqdm import tqdm

from gretel_synthetics.const import (
    METRIC_ACCURACY,
    METRIC_DELTA,
    METRIC_EPSILON,
    METRIC_LOSS,
    METRIC_VAL_ACCURACY,
    METRIC_VAL_LOSS,
)
from gretel_synthetics.errors import TooFewRecordsError
from gretel_synthetics.tensorflow.dp_model import compute_epsilon
from gretel_synthetics.tensorflow.model import build_model, load_model
from gretel_synthetics.tokenizers import BaseTokenizer
from gretel_synthetics.train import EpochState

if TYPE_CHECKING:
    from gretel_synthetics.config import TensorFlowConfig
    from gretel_synthetics.train import TrainingParams
else:
    TensorFlowConfig = None
    TrainingParams = None

spm_logger = logging.getLogger("sentencepiece")
spm_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s : %(threadName)s : %(levelname)s : %(message)s",
    level=logging.INFO,
)


class _ModelHistory(tf.keras.callbacks.Callback):
    """
    Callback class to compute loss and accuracy during model training
    """

    def __init__(self, num_examples_train: int, config: TensorFlowConfig):
        self.num_examples_train = num_examples_train
        self.config = config
        self.loss = []
        self.accuracy = []
        self.val_loss = []
        self.val_accuracy = []
        self.epsilon = []
        self.delta = []
        self.best = []

    def on_epoch_end(self, epoch, logs: dict = None):
        self.loss.append(logs.get(METRIC_LOSS))
        self.accuracy.append(logs.get(METRIC_ACCURACY))
        self.val_loss.append(logs.get(METRIC_VAL_LOSS, 0))
        self.val_accuracy.append(logs.get(METRIC_VAL_ACCURACY, 0))

        if self.config.dp:
            # Account for tf-privacy library writing to stdout
            with redirect_stdout(io.StringIO()):
                eps, _ = compute_epsilon(
                    self.num_examples_train, self.config, epoch + 1
                )
                logs[METRIC_EPSILON] = eps

            # NOTE: this is just a list of the same value, but
            # is simpler for creating the history csv
            delta = 1 / float(self.num_examples_train)
            logs[METRIC_DELTA] = delta

        self.epsilon.append(logs.get(METRIC_EPSILON, 0))
        self.delta.append(logs.get(METRIC_DELTA, 0))

        # NOTE: When we do the final history saving, one of these items
        # will flip to 1 to denote the actual best model that is saved
        # This just seeds the column for now
        self.best.append(0)


class _EpochCallbackWrapper(tf.keras.callbacks.Callback):
    """Wrapper class for the generic Callable attached to the BaseConfig.  It just translates the signature
    for on_epoch_end into an EpochState which we use to invoke the BaseConfig callback.
    """

    def __init__(self, epoch_callable: Callable):
        self.epoch_callable = epoch_callable

    def on_epoch_end(self, epoch, logs: dict = None):
        logs = logs or {}
        epoch_state = EpochState(
            epoch=epoch,
            accuracy=logs.get(METRIC_ACCURACY),
            loss=logs.get(METRIC_LOSS),
            val_accuracy=logs.get(METRIC_VAL_ACCURACY, 0),
            val_loss=logs.get(METRIC_VAL_LOSS, 0),
            epsilon=logs.get(METRIC_EPSILON, 0),
            delta=logs.get(METRIC_DELTA, 0),
        )
        self.epoch_callable(epoch_state)


class _MaxTrainTimeCallback(tf.keras.callbacks.Callback):
    """
    Callback that will stop training after a maximum number
    of seconds has elapsed.
    """

    _duration_seconds: int
    _start_time: int

    def __init__(self, duration_seconds: int):
        self._duration_seconds = duration_seconds

    def on_train_begin(self, _):
        self._start_time = time.monotonic()

    def on_epoch_end(self, *args):
        elapsed = time.monotonic() - self._start_time
        if elapsed >= self._duration_seconds:
            self.model.stop_training = True


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
    df = pd.DataFrame(
        zip(
            range(len(history.loss)),
            history.loss,
            history.accuracy,
            history.val_loss,
            history.val_accuracy,
            history.epsilon,
            history.delta,
            history.best,
        ),
        columns=[
            "epoch",
            METRIC_LOSS,
            METRIC_ACCURACY,
            METRIC_VAL_LOSS,
            METRIC_VAL_ACCURACY,
            METRIC_EPSILON,
            METRIC_DELTA,
            "best",
        ],
    ).round(4)

    # Grab that last idx in case we need to use it in lieu of finding
    # a better one
    try:
        last_idx = df.iloc[[-1]].index.values.astype(int)[0]
    except IndexError as err:
        raise RuntimeError(
            "An error occurred when saving model history, this could be because training "
            "was stopped before the first epoch could finish"
        ) from err  # noqa

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

    if dp:
        # Log differential privacy settings from best training checkpoint
        epsilon = df.at[best_idx, "epsilon"]
        delta = df.at[best_idx, "delta"]
        logger.warning(
            f"Model satisfies differential privacy with epsilon ε={epsilon:.2f} "
            f"and delta δ={delta:.6f}"
        )
    else:
        df.drop(["epsilon", "delta"], axis=1, inplace=True)

    save_path = Path(save_dir) / "model_history.csv"
    logger.info(f"Saving model history to {save_path.name}")
    df.to_csv(save_path.as_posix(), index=False)


def train_rnn(params: TrainingParams):
    """
    Fit synthetic data model on training data.

    This will annotate the training data and create a new file that
    will be used to actually train on. The updated training data, model,
    checkpoints, etc will all be saved in the location specified
    by your config.

    Args:
        params: The parameters controlling model training.

    Returns:
        None
    """
    store = params.config
    # TODO: We should check that store is an instance of TensorFlowConfig, but that would currently
    # load to an import cycle.

    tokenizer = params.tokenizer
    num_lines = params.tokenizer_trainer.num_lines
    text_iter = params.tokenizer_trainer.data_iterator()

    if not store.overwrite:  # pragma: no cover
        try:
            load_model(store, tokenizer)
        except Exception:
            pass
        else:
            raise RuntimeError(
                "A model already exists in the checkpoint location, you must enable "
                "overwrite mode or delete the checkpoints first."
            )

    num_batches_train, validation_dataset, training_dataset = _create_dataset(
        store, text_iter, num_lines, tokenizer
    )
    logger.info("Initializing synthetic model", extra={"user_log": True})
    model = build_model(
        vocab_size=tokenizer.total_vocab_size,
        batch_size=store.batch_size,
        store=store,
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

    num_examples_train = store.batch_size * num_batches_train

    history_callback = _ModelHistory(num_examples_train, store)

    _callbacks = [checkpoint_callback, history_callback]

    if store.early_stopping:
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor=store.best_model_metric,
            patience=store.early_stopping_patience,
            min_delta=store.early_stopping_min_delta,
            restore_best_weights=store.save_best_model,
        )
        _callbacks.append(early_stopping_callback)

    if store.epoch_callback:
        _callbacks.append(_EpochCallbackWrapper(store.epoch_callback))

    # NOTE: This callback should go last, so any other specific
    # handling for a completed epoch would be run first before
    # stopping training.
    if store.max_training_time_seconds is not None:
        max_timeout_calback = _MaxTrainTimeCallback(store.max_training_time_seconds)
        _callbacks.append(max_timeout_calback)

    best_val = None
    try:
        model.fit(
            training_dataset,
            epochs=store.epochs,
            callbacks=_callbacks,
            validation_data=validation_dataset,
        )

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
    except (ValueError, IndexError) as exc:
        raise TooFewRecordsError(
            "Model training failed. Your training data may have too few records in it. "
            "Please try increasing your training rows and try again"
        ) from exc
    except KeyboardInterrupt:
        ...
    _save_history_csv(
        history_callback,
        store.checkpoint_dir,
        store.dp,
        store.best_model_metric,
        best_val,
    )
    logger.info(f"Saving model to {tf.train.latest_checkpoint(store.checkpoint_dir)}")


def _create_dataset(
    store: TensorFlowConfig,
    text_iter: Iterator[str],
    num_lines: int,
    tokenizer: BaseTokenizer,
) -> Tuple[int, tf.data.Dataset, tf.data.Dataset]:
    """
    Before training, we need to map strings to a numerical representation.
    Create two lookup tables: one mapping characters to numbers,
    and another for numbers to characters.
    """
    logger.info("Tokenizing input data", extra={"user_log": True})
    ids = []
    for line in tqdm(text_iter, total=num_lines):
        _tokens = tokenizer.encode_to_ids(line)
        ids.extend(_tokens)

    logger.info("Shuffling input data", extra={"user_log": True})
    char_dataset = tf.data.Dataset.from_tensor_slices(ids)
    sequences = char_dataset.batch(store.seq_length + 1, drop_remainder=True)
    full_dataset = (
        sequences.map(
            _split_input_target,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )
        .shuffle(store.buffer_size)
        .batch(store.batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Compile as Lambda functions for compatibility with AutoGraph
    def is_validation(x, y):
        return x % 5 == 0

    def is_train(x, y):
        return not is_validation(x, y)

    def recover(x, y):
        return y

    if store.validation_split:
        logger.info("Creating validation dataset", extra={"user_log": True})
        validation_dataset = (
            full_dataset.enumerate()
            .filter(is_validation)
            .map(recover, num_parallel_calls=tf.data.AUTOTUNE)
        )
        logger.info("Creating training dataset", extra={"user_log": True})
        train_dataset = (
            full_dataset.enumerate()
            .filter(is_train)
            .map(recover, num_parallel_calls=tf.data.AUTOTUNE)
        )

        num_batches_train = len(list(train_dataset.as_numpy_iterator()))
        return num_batches_train, validation_dataset, train_dataset
    else:
        num_batches_train = tf.data.experimental.cardinality(full_dataset).numpy()
        return num_batches_train, None, full_dataset


@tf.autograph.experimental.do_not_convert
def _split_input_target(chunk: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
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
