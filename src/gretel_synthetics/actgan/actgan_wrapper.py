"""Wrapper around ACTGAN model."""

from __future__ import annotations

import logging

from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import numpy as np
import pandas as pd

from rdt.transformers import BaseTransformer
from sdv.tabular.base import BaseTabularModel

from gretel_synthetics.actgan.actgan import ACTGANSynthesizer
from gretel_synthetics.actgan.columnar_df import ColumnarDF
from gretel_synthetics.actgan.structures import ConditionalVectorType
from gretel_synthetics.detectors.sdv import SDVTableMetadata
from gretel_synthetics.utils import rdt_patches, torch_utils

if TYPE_CHECKING:
    from numpy.random import RandomState
    from sdv.constraints import Constraint
    from sdv.metadata import Metadata
    from torch import Generator

    from gretel_synthetics.actgan.structures import EpochInfo

EPOCH_CALLBACK = "epoch_callback"

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class _ACTGANModel(BaseTabularModel):
    """ACTGAN Model, extends SDV base tabular model. Should not be
    used directly, instead use ``ACTGAN()``.
    """

    _MODEL_CLASS = None
    _model_kwargs = None

    _DTYPE_TRANSFORMERS = {"O": None}

    _auto_transform_datetimes: bool
    """
    Should be set by the concrete subclass constructor
    """

    _verbose: bool

    def _build_model(self):
        return self._MODEL_CLASS(**self._model_kwargs)

    def fit(self, data: Union[pd.DataFrame, str]) -> None:
        """
        Fit the ACTGAN model to the provided data. Prior to fitting,
        specific auto-detection of data types will be done if the
        provided ``data`` is a DataFrame.
        """
        if not isinstance(data, pd.DataFrame):
            return super().fit(data)

        current_field_types = self._metadata._field_types
        current_field_transformers = self._metadata._field_transformers
        detector = SDVTableMetadata(
            field_types=current_field_types,
            field_transformers=current_field_transformers,
        )

        if self._auto_transform_datetimes:
            if self._verbose:
                logger.info("Attempting datetime auto-detection...")
            detector.fit_datetime(data, with_suffix=True, must_match_all=True)

        detector.fit_empty_columns(data)
        if self._verbose:
            logger.info(f"Using field types: {detector.field_types}")
            logger.info(f"Using field transformers: {detector.field_transformers}")
        self._metadata._field_types = detector.field_types
        self._metadata._field_transformers = detector.field_transformers

        # Metadata fitting will process the data column by column, dropping original columns
        # after adding transformed ones. This is very expensive in a pd.DataFrame. Therefore,
        # transform it to our internal, columnar representation. This is a bit dicey, because
        # we don't implement the full pd.DataFrame interface, but as long as we don't change
        # the RDT dependency, this shouldn't be an issue.
        # Note that there is one internal RDT method that doesn't work with a ColumnarDF, which
        # we'll have to monkeypatch to a compatible version.
        data.reset_index(drop=True, inplace=True)
        with _patch_rdt_add_columns_to_data():
            super().fit(ColumnarDF.from_df(data))

    def _fit(self, table_data: Union[pd.DataFrame, ColumnarDF]) -> None:
        """Fit the model to the table.

        Args:
            table_data: Data to be learned.
        """
        if isinstance(table_data, ColumnarDF):
            # The ColumnarDF representation is only relevant for the metadata transform, that
            # happens between fit() and _fit(). Therefore, to minimize any required downstream
            # code changes, revert to a standard pandas dataframe now (the memory requirements are
            # the same when no changes to the structure are made).
            columnar_df = table_data
            table_data = columnar_df.to_df()
            # Python doesn't have move semantics for arguments, and since fit() is an ancestor in
            # the call tree, there is no way to let garbage collection take care of the ColumnarDF,
            # as the calling context in fit() retains a reference. Therefore, clear the contents
            # in-place to reduce memory pressure.
            columnar_df.drop(columns=columnar_df.columns, inplace=True)

        self._model: ACTGANSynthesizer = self._build_model()

        categoricals = []
        fields_before_transform = self._metadata.get_fields()
        for field in table_data.columns:
            if field in fields_before_transform:
                meta = fields_before_transform[field]
                if meta["type"] == "categorical":
                    categoricals.append(field)

            else:
                field_data = table_data[field].dropna()
                field_data_arr = field_data.to_numpy()
                if np.all((field_data_arr == 0.0) | (field_data_arr == 1.0)):
                    # booleans encoded as float values must be modeled as bool
                    field_data = field_data.astype(bool)

                dtype = field_data.infer_objects().dtype
                try:
                    kind = np.dtype(dtype).kind
                except TypeError:
                    # probably category
                    kind = "O"
                if kind in ["O", "b"]:
                    categoricals.append(field)

        self._model.fit(table_data, discrete_columns=categoricals)

    def _sample(self, num_rows: int, conditions: Optional[dict] = None) -> pd.DataFrame:
        """Sample the indicated number of rows from the model.

        Args:
            num_rows:
                Amount of rows to sample.
            conditions:
                If specified, this dictionary maps column names to the column
                value. Then, this method generates `num_rows` samples, all of
                which are conditioned on the given variables.

        Returns:
            Sampled data
        """
        return self._model.sample(num_rows, conditions)

    def _set_random_state(
        self, random_state: Union[int, Tuple[RandomState, Generator], None]
    ) -> None:
        """Set the random state of the model's random number generator.

        Args:
            random_state: Seed or tuple of random states to use.
        """
        self._model.set_random_state(random_state)

    def save(self, path: str) -> None:
        """
        Save the model to disk for re-use later. When saving, certain attributes will
        not be saved such as any epoch callbacks that are attached to the model.
        """
        self._model: ACTGANSynthesizer

        # Temporarily remove any epoch callback so pickling can be done
        _tmp_callback = self._model._epoch_callback
        self._model._epoch_callback = None
        self._model_kwargs[EPOCH_CALLBACK] = None

        super().save(path)

        # Restore our callback for continued use of the model
        self._model._epoch_callback = _tmp_callback
        self._model_kwargs[EPOCH_CALLBACK] = _tmp_callback

    @classmethod
    def load_v2(cls, path: str) -> ACTGAN:
        """
        An updated version of loading that will allow reading in a pickled model
        that can be used on a CPU or GPU for sampling.
        """
        device = torch_utils.determine_device()
        with open(path, "rb") as fin:
            loaded_model: ACTGAN = torch_utils.patched_torch_unpickle(fin, device)
            loaded_model._model.set_device(device)
        return loaded_model


class ACTGAN(_ACTGANModel):
    """
    Args:
        field_names:
            List of names of the fields that need to be modeled
            and included in the generated output data. Any additional
            fields found in the data will be ignored and will not be
            included in the generated output.
            If ``None``, all the fields found in the data are used.
        field_types:
            Dictinary specifying the data types and subtypes
            of the fields that will be modeled. Field types and subtypes
            combinations must be compatible with the SDV Metadata Schema.
        field_transformers:
            Dictinary specifying which transformers to use for each field.
            Available transformers are:

                * ``FloatFormatter``: Uses a ``FloatFormatter`` for numerical data.
                * ``FrequencyEncoder``: Uses a ``FrequencyEncoder`` without gaussian noise.
                * ``FrequencyEncoder_noised``: Uses a ``FrequencyEncoder`` adding gaussian noise.
                * ``OneHotEncoder``: Uses a ``OneHotEncoder``.
                * ``LabelEncoder``: Uses a ``LabelEncoder`` without gaussian nose.
                * ``LabelEncoder_noised``: Uses a ``LabelEncoder`` adding gaussian noise.
                * ``BinaryEncoder``: Uses a ``BinaryEncoder``.
                * ``UnixTimestampEncoder``: Uses a ``UnixTimestampEncoder``.

            NOTE: Specifically for ACTGAN, some attributes such as ``auto_transform_datetimes`` will
            automatically attempt to detect field types and will automatically set the ``field_transformers``
            dictionary at construction time. However, autodetection of ``field_types`` and ``field_transformers``
            will not be over-written by any concrete values that were provided to this constructor.
        auto_transform_datetimes: If set, prior to fitting, each column will be checked for
            being a potential "datetime" type. For each column that is discovered as a "datetime" the
            `field_types` and `field_transformers` SDV metadata dicts will be automatically updated
            such that datetimes are transformed to Unix timestamps. NOTE: if fields are already
            specified in `field_types` or `field_transformers` these fields will be skipped
            by the auto detector.
        anonymize_fields:
            Dict specifying which fields to anonymize and what faker
            category they belong to.
        primary_key:
            Name of the field which is the primary key of the table.
        constraints:
            List of Constraint objects or dicts.
        table_metadata:
            Table metadata instance or dict representation.
            If given alongside any other metadata-related arguments, an
            exception will be raised.
            If not given at all, it will be built using the other
            arguments or learned from the data.
        embedding_dim:
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim:
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim:
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr:
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay:
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr:
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay:
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size:
            Number of data samples to process in each step.
        discriminator_steps:
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        binary_encoder_cutoff:
            For any given column, the number of unique values that should exist before
            switching over to binary encoding instead of OHE. This will help reduce
            memory consumption for datasets with a lot of unique values.
        binary_encoder_nan_handler:
            Binary encoding currently may produce errant NaN values during reverse transformation. By default
            these NaN's will be left in place, however if this value is set to "mode" then those NaN' will
            be replaced by a random value that is a known mode for a given column.
        cbn_sample_size:
            Number of rows to sample from each column for identifying clusters for the cluster-based normalizer.
            This only applies to float columns. If set to ``0``, no sampling is done and all values are considered,
            which may be very slow. Defaults to 250_000.
        log_frequency:
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose:
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs:
            Number of training epochs. Defaults to 300.
        epoch_callback:
            An optional function to call after each epoch, the argument will be a
            ``EpochInfo`` instance
        pac:
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        cuda:
            If ``True``, use CUDA. If a ``str``, use the indicated device.
            If ``False``, do not use cuda at all. Defaults to ``True``.
        learn_rounding_scheme:
            Define rounding scheme for ``FloatFormatter``. If ``True``, the data returned by
            ``reverse_transform`` will be rounded to that place. Defaults to ``True``.
        enforce_min_max_values:
            Specify whether or not to clip the data returned by ``reverse_transform`` of
            the numerical transformer, ``FloatFormatter``, to the min and max values seen
            during ``fit``. Defaults to ``True``.
        conditional_vector_type:
            Type of conditional vector to include in input to the generator.
            Influences how effective and flexible the native conditional
            generation is. Options include SINGLE_DISCRETE (original CTGAN
            setup) and ANYWAY. Default is SINGLE_DISCRETE.
        conditional_select_mean_columns:
            Target number of columns to select for conditioning on average
            during training. Only used for ANYWAY conditioning. Use if typical
            number of columns to seed on is known. If set,
            conditional_select_column_prob must be None. Equivalent to setting
            conditional_select_column_prob to conditional_select_mean_columns /
            # of columns. Defaults to None.
        conditional_select_column_prob:
            Probability to select any given column to be conditioned on during
            training. Only used for ANYWAY conditioning. If set,
            conditional_select_mean_columns must be None. Defaults to None.
        reconstruction_loss_coef:
            Multiplier on reconstruction loss, higher values focus the generator
            optimization more on accurate conditional vector generation.
            Defaults to 1.0.
        force_conditioning:
            Directly set the requested conditional generation columns in
            generated data. Will bypass rejection sampling and be faster, but
            may reduce quality of the generated data and correlations between
            conditioned columns and other variables may be weaker. Defaults to
            False.
    """

    _MODEL_CLASS = ACTGANSynthesizer
    _auto_transform_datetimes: bool
    _verbose: bool

    def __init__(
        self,
        field_names: Optional[List[str]] = None,
        field_types: Optional[Dict[str, dict]] = None,
        field_transformers: Optional[Dict[str, Union[BaseTransformer, str]]] = None,
        auto_transform_datetimes: bool = False,
        anonymize_fields: Optional[Dict[str, str]] = None,
        primary_key: Optional[str] = None,
        constraints: Optional[Union[List[Constraint], List[dict]]] = None,
        table_metadata: Optional[Union[Metadata, dict]] = None,
        embedding_dim: int = 128,
        generator_dim: Sequence[int] = (256, 256),
        discriminator_dim: Sequence[int] = (256, 256),
        generator_lr: float = 2e-4,
        generator_decay: float = 1e-6,
        discriminator_lr: float = 2e-4,
        discriminator_decay: float = 1e-6,
        batch_size: int = 500,
        discriminator_steps: int = 1,
        binary_encoder_cutoff: int = 500,
        binary_encoder_nan_handler: Optional[str] = None,
        cbn_sample_size: Optional[int] = 250_000,
        log_frequency: bool = True,
        verbose: bool = False,
        epochs: int = 300,
        epoch_callback: Optional[Callable[[EpochInfo], None]] = None,
        pac: int = 10,
        cuda: bool = True,
        learn_rounding_scheme: bool = True,
        enforce_min_max_values: bool = True,
        conditional_vector_type: ConditionalVectorType = ConditionalVectorType.SINGLE_DISCRETE,
        conditional_select_mean_columns: Optional[float] = None,
        conditional_select_column_prob: Optional[float] = None,
        reconstruction_loss_coef: float = 1.0,
        force_conditioning: bool = False,
    ):
        super().__init__(
            field_names=field_names,
            primary_key=primary_key,
            field_types=field_types,
            field_transformers=field_transformers,
            anonymize_fields=anonymize_fields,
            constraints=constraints,
            table_metadata=table_metadata,
            learn_rounding_scheme=learn_rounding_scheme,
            enforce_min_max_values=enforce_min_max_values,
        )

        self._auto_transform_datetimes = auto_transform_datetimes
        self._verbose = verbose

        self._model_kwargs = {
            "embedding_dim": embedding_dim,
            "generator_dim": generator_dim,
            "discriminator_dim": discriminator_dim,
            "generator_lr": generator_lr,
            "generator_decay": generator_decay,
            "discriminator_lr": discriminator_lr,
            "discriminator_decay": discriminator_decay,
            "batch_size": batch_size,
            "discriminator_steps": discriminator_steps,
            "binary_encoder_cutoff": binary_encoder_cutoff,
            "binary_encoder_nan_handler": binary_encoder_nan_handler,
            "cbn_sample_size": cbn_sample_size,
            "log_frequency": log_frequency,
            "verbose": verbose,
            "epochs": epochs,
            "epoch_callback": epoch_callback,
            "pac": pac,
            "cuda": cuda,
            "conditional_vector_type": conditional_vector_type,
            "conditional_select_mean_columns": conditional_select_mean_columns,
            "conditional_select_column_prob": conditional_select_column_prob,
            "reconstruction_loss_coef": reconstruction_loss_coef,
            "force_conditioning": force_conditioning,
        }

    def fit(self, *args, **kwargs):
        # Float formatter should not affect anything during model fitting, but it's
        # hard to know with certainty what exactly is going on under the hood. Therefore,
        # take a conservative approach.
        with rdt_patches.patch_float_formatter_rounding_bug():
            return super().fit(*args, **kwargs)

    def sample(self, *args, **kwargs):
        with rdt_patches.patch_float_formatter_rounding_bug():
            return super().sample(*args, **kwargs)

    def sample_remaining_columns(self, *args, **kwargs):
        with rdt_patches.patch_float_formatter_rounding_bug():
            return super().sample_remaining_columns(*args, **kwargs)


@contextmanager
def _patch_rdt_add_columns_to_data():
    prev_value = BaseTransformer._add_columns_to_data
    try:
        BaseTransformer._add_columns_to_data = staticmethod(
            _add_columns_to_data_patched
        )
        yield None
    finally:
        BaseTransformer._add_columns_to_data = staticmethod(prev_value)


def _add_columns_to_data_patched(data, columns, column_names):
    """Add new columns to a ``pandas.DataFrame`` or ``ColumnarDF``.

    This is a patched version of the ``BaseTransformer._add_columns_to_data``
    static method, which avoids ``pd.concat`` and thus also works with
    a ``ColumnarDF``, as opposed to just a ``pd.DataFrame``, and also preserves
    the type of ``data``. It is fully compatible when used with a regular
    ``pd.DataFrame``, and would even be more efficient when used with that.
    """
    if columns is not None:
        if isinstance(columns, (pd.DataFrame, pd.Series)):
            columns.index = data.index

        if len(columns.shape) == 1:
            data[column_names[0]] = columns
        else:
            new_data = pd.DataFrame(columns, columns=column_names)
            data[column_names] = new_data

    return data
