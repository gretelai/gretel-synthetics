import logging
import re
import uuid
import warnings

from functools import partial
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from category_encoders import BaseNEncoder, BinaryEncoder
from gretel_synthetics.actgan.structures import (
    ActivationFn,
    ColumnIdInfo,
    ColumnTransformInfo,
    ColumnType,
    SpanInfo,
)
from gretel_synthetics.typing import DFLike, ListOrSeriesOrDF, SeriesOrDFLike
from joblib import delayed, Parallel
from rdt.transformers import BaseTransformer, ClusterBasedNormalizer, OneHotEncoder

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

warnings.simplefilter(action="ignore", category=FutureWarning)

OHE_CUTOFF = 150
"""
The max number of unique values that should be used before swtiching
away from one hot encoding
"""

ValidEncoderT = Union[BinaryEncoder, OneHotEncoder]

MODE = "mode"
VALID_ROUNDING_MODES = (MODE,)


def _new_uuid() -> str:
    return f"gretel-{uuid.uuid4().hex}"


class BinaryEncodingTransformer(BaseTransformer):
    """BinaryEncoding for categorical data."""

    INPUT_SDTYPE = "categorical"
    DETERMINISTIC_TRANSFORM = True
    DETERMINISTIC_REVERSE = True

    dummies: List[str]

    _dummy_na: bool = False
    _mode_values: FrozenSet[Any]
    """For the series we are operating on, this will contain the list of modes in the dataset
    """
    _nan_proxy: Optional[str] = None
    handle_rounding_nan: Optional[str]

    def __init__(self, handle_rounding_nan: Optional[str] = None):
        if (
            handle_rounding_nan is not None
            and handle_rounding_nan not in VALID_ROUNDING_MODES
        ):
            raise ValueError(
                f"Invalid `handle_rounding_nan`, must be one of: {VALID_ROUNDING_MODES}"
            )
        self.handle_rounding_nan = handle_rounding_nan
        self._mode_values = frozenset()
        super().__init__()

    @staticmethod
    def _prepare_data(data: ListOrSeriesOrDF) -> SeriesOrDFLike:
        """Transform data to appropriate format.
        If data is a valid list or a list of lists, transforms it into an np.array,
        otherwise returns it.

        Args:
            data: Data to prepare.

        Returns:
            pandas.Series or numpy.ndarray
        """
        if isinstance(data, list):
            data = np.array(data)

        if len(data.shape) > 2:
            raise ValueError("Unexpected format.")
        if len(data.shape) == 2 and data.shape[1] != 1:
            raise ValueError("Unexpected format.")
        if len(data.shape) == 2:
            data = data[:, 0]

        return data

    def get_output_sdtypes(self) -> Dict[str, Any]:
        """Return the output sdtypes produced by this transformer.

        Returns:
            dict:
                Mapping from the transformed column names to the produced sdtypes.
        """
        # output_sdtypes = {f'value{i}': 'float' for i in range(len(self.dummies))}
        output_sdtypes = {column_name: "float" for column_name in self.dummies}
        out = self._add_prefix(output_sdtypes)
        return out

    @staticmethod
    def fill_na_from_list(data: pd.Series, options: List[Any]) -> pd.Series:
        """
        Given a series and a list of options, fill in NaN values such that we
        do a random choice from the `options` list for each NaN. This way we aren't
        repalcing each NaN with the same value for the whole series
        """
        nan_mask = data.isnull()
        num_nans = nan_mask.sum()
        fill_list = np.random.choice(options, size=num_nans)
        data.loc[nan_mask] = fill_list
        return data

    def _fit(self, data: SeriesOrDFLike) -> None:
        """Fit the transformer to the data.

        Get the pandas `category codes` which will be used later on for BinaryEncoding.

        Args:
            data: Data to fit the transformer to.
        """
        self.encoder = BinaryEncoder()
        _patch_basen_to_integer(self.encoder.base_n_encoder)

        data = self._prepare_data(data)
        if not isinstance(data, pd.Series):
            data = pd.Series(data)

        if self.handle_rounding_nan == MODE:
            self._mode_values = frozenset(list(data.mode()))
        self._nan_proxy = _new_uuid()
        data = data.fillna(self._nan_proxy)

        self.encoder.fit(data)

        # NOTE: We set `dummies` here since a `DataTransformer` instance
        # will use the size of this list in order to determine the
        # number of categorical columns. Because both the OHE and Binary Encoder
        # can be embedded in the `DataTransformer` instance, both encoders
        # should have the `dummies` attr.
        self.dummies = self.encoder.get_feature_names().copy()

    def _transform(self, data: ListOrSeriesOrDF) -> np.ndarray:
        """Replace each category with appropiate vectors

        Args:
            data (pandas.Series, list or list of lists):
                Data to transform.

        Returns:
            numpy.ndarray
        """
        data = self._prepare_data(data)
        # unique_data = {np.nan if pd.isna(x) else x for x in pd.unique(data)}
        # unseen_categories = unique_data - set(self.dummies)
        # if unseen_categories:
        #     # Select only the first 5 unseen categories to avoid flooding the console.
        #     examples_unseen_categories = set(list(unseen_categories)[:5])
        #     warnings.warn(
        #         f'The data contains {len(unseen_categories)} new categories that were not '
        #         f'seen in the original data (examples: {examples_unseen_categories}). Creating '
        #         'a vector of all 0s. If you want to model new categories, '
        #         'please fit the transformer again with the new data.'
        #     )

        ndarray = self.encoder.transform(data).to_numpy()
        return ndarray

    def _reverse_transform(self, data: SeriesOrDFLike) -> SeriesOrDFLike:
        """Convert float values back to the original categorical values.

        Args:
            data (pd.Series or numpy.ndarray):
                Data to revert.

        Returns:
            pandas.Series
        """
        if not isinstance(data, np.ndarray):
            data = data.to_numpy()

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        threshold_data = []
        for row in data:
            rounded_row = np.array([int(x > 0.5) for x in row])
            # rounded_row = np.array([np.abs(x - 0) > np.abs(x - 1) for x in row])
            threshold_data.append(rounded_row)

        transformed_data = self.encoder.inverse_transform(np.array(threshold_data))

        # Optionally replace NaN values that might be "near misses" from predictions
        # with the mode values
        if (
            self.handle_rounding_nan == MODE
            and len(self._mode_values) > 0
            and isinstance(transformed_data, pd.DataFrame)
        ):
            column_name = transformed_data.columns[0]
            transformed_data[column_name] = self.fill_na_from_list(
                transformed_data[column_name], list(self._mode_values)
            )

        # Replace any proxy NaN values with actual NaN
        transformed_data = transformed_data.replace(
            to_replace=self._nan_proxy, value=np.nan
        )
        return transformed_data


def _patch_basen_to_integer(basen_encoder: BaseNEncoder) -> None:
    """
    FIXME(PROD-309): Temporary patch for https://github.com/scikit-learn-contrib/category_encoders/issues/392
    """

    def _patched_basen_to_integer(self, X, cols, base):
        """
        Copied from https://github.com/scikit-learn-contrib/category_encoders/blob/1def42827df4a9404553f41255878c45d754b1a0/category_encoders/basen.py#L266-L281
        and applied this fix: https://github.com/scikit-learn-contrib/category_encoders/pull/393/files
        """
        out_cols = X.columns.values.tolist()

        for col in cols:
            col_list = [
                col0
                for col0 in out_cols
                if re.match(re.escape(str(col)) + "_\\d+", str(col0))
            ]
            insert_at = out_cols.index(col_list[0])

            if base == 1:
                value_array = np.array([int(col0.split("_")[-1]) for col0 in col_list])
            else:
                len0 = len(col_list)
                value_array = np.array([base ** (len0 - 1 - i) for i in range(len0)])
            X.insert(insert_at, col, np.dot(X[col_list].values, value_array.T))
            X.drop(col_list, axis=1, inplace=True)
            out_cols = X.columns.values.tolist()

        return X

    basen_encoder.basen_to_integer = partial(_patched_basen_to_integer, basen_encoder)


class DataTransformer:
    """Data Transformer.

    Model continuous columns with a BayesianGMM and normalized to a scalar [0, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder or a Binary Encoder depending
    on field cardinality.
    """

    _max_clusters: int
    _weight_threshold: float
    _column_raw_dtypes: pd.Series
    _column_transform_info_list: List[ColumnTransformInfo]
    _binary_encoder_cutoff: int
    _binary_encoder_han_handler: Optional[str]
    _verbose: bool
    dataframe: bool
    output_dimensions: int
    output_info_list: List[SpanInfo]

    def __init__(
        self,
        max_clusters: int = 10,
        weight_threshold: float = 0.005,
        binary_encoder_cutoff: int = OHE_CUTOFF,
        binary_encoder_nan_handler: Optional[str] = None,
        verbose: bool = False,
    ):
        """Create a data transformer.

        Args:
            max_clusters:
                Maximum number of Gaussian distributions in Bayesian GMM.
            weight_threshold:
                Weight threshold for a Gaussian distribution to be kept.
            binary_encoder_cutoff:
                What column value cardinality to use to control the switch to a Binary Encoder instead of OHE
            binary_encoder_nan_handler:
                If NaN's are produced from the binary encoding reverse transform, this drives how to replace those
                NaN's with actual values
            verbose: Provide detailed logging on data transformation details.
        """
        self._max_clusters = max_clusters
        self._weight_threshold = weight_threshold
        self._binary_encoder_cutoff = binary_encoder_cutoff
        self._binary_encoder_han_handler = binary_encoder_nan_handler
        self._verbose = verbose

    def _fit_continuous(self, data: pd.DataFrame) -> ColumnTransformInfo:
        """Train Bayesian GMM for continuous columns."""
        column_name = data.columns[0]
        gm = ClusterBasedNormalizer(
            model_missing_values=True, max_clusters=min(len(data), 10)
        )
        if self._verbose:
            logger.info(
                f"Transforming continuous column: {column_name!r} with {gm.__class__.__name__}"
            )
        gm.fit(data, column_name)
        num_components = sum(gm.valid_component_indicator)

        return ColumnTransformInfo(
            column_name=column_name,
            column_type=ColumnType.CONTINUOUS,
            transform=gm,
            output_info=[
                SpanInfo(1, ActivationFn.TANH),
                SpanInfo(num_components, ActivationFn.SOFTMAX),
            ],
            output_dimensions=1 + num_components,
        )

    def _fit_discrete(self, data: pd.DataFrame) -> ColumnTransformInfo:
        """Fit one hot encoder for discrete column."""
        column_name = data.columns[0]
        if data[column_name].nunique() >= self._binary_encoder_cutoff:
            encoder = BinaryEncodingTransformer(
                handle_rounding_nan=self._binary_encoder_han_handler
            )
            activation = ActivationFn.SIGMOID
        else:
            encoder = OneHotEncoder()
            activation = ActivationFn.SOFTMAX
        if self._verbose:
            logger.info(
                f"Transforming discrete column: {column_name!r} with {encoder.__class__.__name__}"
            )
        encoder.fit(data, column_name)
        num_categories = len(encoder.dummies)

        return ColumnTransformInfo(
            column_name=column_name,
            column_type="discrete",
            transform=encoder,
            output_info=[SpanInfo(num_categories, activation)],
            output_dimensions=num_categories,
        )

    def fit(
        self, raw_data: DFLike, discrete_columns: Optional[Sequence[str]] = None
    ) -> None:
        """Fit the ``DataTransformer``.

        Fits a ``ClusterBasedNormalizer`` for continuous columns and a
        ``OneHotEncoder`` or ``BinaryEncodingTransformer`` for discrete columns depending
        on the number of unique values in columns.

        This step also counts the #columns in matrix data and span information.
        """
        self.output_info_list: List[List[SpanInfo]] = []
        self.output_dimensions = 0
        self.dataframe = True

        if discrete_columns is None:
            discrete_columns = []

        if not isinstance(raw_data, pd.DataFrame):
            self.dataframe = False
            # work around for RDT issue #328 Fitting with numerical column names fails
            discrete_columns = [str(column) for column in discrete_columns]
            column_names = [str(num) for num in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)

        self._column_raw_dtypes = raw_data.infer_objects().dtypes
        self._column_transform_info_list = []
        if self._verbose:
            logger.info(
                f"Starting data transforms on {len(raw_data.columns)} columns..."
            )
        for column_name in raw_data.columns:
            if column_name in discrete_columns:
                column_transform_info = self._fit_discrete(raw_data[[column_name]])
            else:
                column_transform_info = self._fit_continuous(raw_data[[column_name]])

            self.output_info_list.append(column_transform_info.output_info)
            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info_list.append(column_transform_info)

    def _transform_continuous(
        self, column_transform_info: ColumnTransformInfo, data: pd.DataFrame
    ) -> np.ndarray:
        column_name = data.columns[0]
        data[column_name] = data[column_name].to_numpy().flatten()
        gm = column_transform_info.transform
        transformed = gm.transform(data)

        #  Converts the transformed data to the appropriate output format.
        #  The first column (ending in '.normalized') stays the same,
        #  but the lable encoded column (ending in '.component') is one hot encoded.
        output = np.zeros((len(transformed), column_transform_info.output_dimensions))
        output[:, 0] = transformed[f"{column_name}.normalized"].to_numpy()
        index = transformed[f"{column_name}.component"].to_numpy().astype(int)
        output[np.arange(index.size), index + 1] = 1.0

        return output

    def _transform_discrete(
        self, column_transform_info: ColumnTransformInfo, data: pd.DataFrame
    ) -> np.ndarray:
        encoder = column_transform_info.transform
        return encoder.transform(data).to_numpy()

    def _synchronous_transform(
        self,
        raw_data: pd.DataFrame,
        column_transform_info_list: List[ColumnTransformInfo],
    ) -> List[np.ndarray]:
        """Take a Pandas DataFrame and transform columns synchronously"""
        column_data_list = []
        for column_transform_info in column_transform_info_list:
            column_name = column_transform_info.column_name
            data = raw_data[[column_name]]
            if column_transform_info.column_type == ColumnType.CONTINUOUS:
                column_data_list.append(
                    self._transform_continuous(column_transform_info, data)
                )
            else:
                column_data_list.append(
                    self._transform_discrete(column_transform_info, data)
                )

        return column_data_list

    def _parallel_transform(
        self,
        raw_data: pd.DataFrame,
        column_transform_info_list: List[ColumnTransformInfo],
    ) -> List[np.ndarray]:
        """Take a Pandas DataFrame and transform columns in parallel."""
        processes = []
        for column_transform_info in column_transform_info_list:
            column_name = column_transform_info.column_name
            data = raw_data[[column_name]]
            process = None
            if column_transform_info.column_type == ColumnType.CONTINUOUS:
                process = delayed(self._transform_continuous)(
                    column_transform_info, data
                )
            else:
                process = delayed(self._transform_discrete)(column_transform_info, data)
            processes.append(process)

        return Parallel(n_jobs=-1)(processes)

    def transform(self, raw_data: DFLike) -> np.ndarray:
        """Take raw data and output a matrix data."""
        if not isinstance(raw_data, pd.DataFrame):
            column_names = [str(num) for num in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)

        # Only use parallelization with larger data sizes.
        # Otherwise, the transformation will be slower.
        if raw_data.shape[0] < 500:
            column_data_list = self._synchronous_transform(
                raw_data, self._column_transform_info_list
            )
        else:
            column_data_list = self._parallel_transform(
                raw_data, self._column_transform_info_list
            )

        return np.concatenate(column_data_list, axis=1).astype(float)

    def _inverse_transform_continuous(
        self,
        column_transform_info: ColumnTransformInfo,
        column_data: pd.Series,
        sigmas,
        st,
    ):
        gm = column_transform_info.transform
        data = pd.DataFrame(column_data[:, :2], columns=list(gm.get_output_sdtypes()))
        data.iloc[:, 1] = np.argmax(column_data[:, 1:], axis=1)
        if sigmas is not None:
            selected_normalized_value = np.random.normal(data.iloc[:, 0], sigmas[st])
            data.iloc[:, 0] = selected_normalized_value

        return gm.reverse_transform(data)

    def _inverse_transform_discrete(
        self, column_transform_info: ColumnTransformInfo, column_data
    ):
        encoder: ValidEncoderT = column_transform_info.transform
        data = pd.DataFrame(column_data, columns=list(encoder.get_output_sdtypes()))
        return encoder.reverse_transform(data)[column_transform_info.column_name]

    def inverse_transform(self, data: DFLike, sigmas=None) -> DFLike:
        """Take matrix data and output raw data.

        Output uses the same type as input to the transform function.
        Either np array or pd dataframe.
        """
        st = 0
        recovered_column_data_list = []
        column_names = []
        for column_transform_info in self._column_transform_info_list:
            dim = column_transform_info.output_dimensions
            column_data = data[:, st : st + dim]  # noqa
            if column_transform_info.column_type == ColumnType.CONTINUOUS:
                recovered_column_data = self._inverse_transform_continuous(
                    column_transform_info, column_data, sigmas, st
                )
            else:
                recovered_column_data = self._inverse_transform_discrete(
                    column_transform_info, column_data
                )

            recovered_column_data_list.append(recovered_column_data)
            column_names.append(column_transform_info.column_name)
            st += dim

        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = pd.DataFrame(recovered_data, columns=column_names).astype(
            self._column_raw_dtypes
        )
        if not self.dataframe:
            recovered_data = recovered_data.to_numpy()

        return recovered_data

    def convert_column_name_value_to_id(
        self, column_name: str, value: Any
    ) -> ColumnIdInfo:
        """Get the ids of the given `column_name`."""
        discrete_counter = 0
        column_id = 0
        for column_transform_info in self._column_transform_info_list:
            if column_transform_info.column_name == column_name:
                break
            if column_transform_info.column_type == ColumnType.DISCRETE:
                discrete_counter += 1

            column_id += 1

        else:
            raise ValueError(
                f"The column_name `{column_name}` doesn't exist in the data."
            )

        encoder = column_transform_info.transform
        data = pd.DataFrame([value], columns=[column_transform_info.column_name])
        encoded = encoder.transform(data).to_numpy()[0]
        if sum(encoded) == 0:
            raise ValueError(
                f"The value `{value}` doesn't exist in the column `{column_name}`."
            )

        return ColumnIdInfo(discrete_counter, column_id, np.argmax(encoded))
