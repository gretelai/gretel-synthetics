import logging
import warnings

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from rdt.transformers import BinaryEncoder, OneHotEncoder

from gretel_synthetics.actgan.column_encodings import (
    BinaryColumnEncoding,
    FloatColumnEncoding,
    OneHotColumnEncoding,
)
from gretel_synthetics.actgan.structures import (
    ColumnIdInfo,
    ColumnTransformInfo,
    ColumnType,
)
from gretel_synthetics.actgan.train_data import TrainData
from gretel_synthetics.actgan.transformers import (
    BinaryEncodingTransformer,
    ClusterBasedNormalizer,
)
from gretel_synthetics.typing import DFLike

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
    _cbn_sample_size: Optional[int]
    _verbose: bool
    dataframe: bool

    def __init__(
        self,
        max_clusters: int = 10,
        weight_threshold: float = 0.005,
        binary_encoder_cutoff: int = OHE_CUTOFF,
        binary_encoder_nan_handler: Optional[str] = None,
        cbn_sample_size: Optional[int] = None,
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
            cbn_sample_size:
                How many rows to sample for identifying clusters in float columns. None means no sampling.
            verbose: Provide detailed logging on data transformation details.
        """
        self._max_clusters = max_clusters
        self._weight_threshold = weight_threshold
        self._binary_encoder_cutoff = binary_encoder_cutoff
        self._binary_encoder_han_handler = binary_encoder_nan_handler
        self._cbn_sample_size = cbn_sample_size
        self._verbose = verbose

    def _fit_continuous(self, data: pd.DataFrame) -> ColumnTransformInfo:
        """Train Bayesian GMM for continuous columns."""
        if self._cbn_sample_size and self._cbn_sample_size < len(data):
            # Train on only a sample of the data, if requested.
            data = data.sample(n=self._cbn_sample_size)
        column_name = data.columns[0]
        gm = ClusterBasedNormalizer(
            model_missing_values=True,
            max_clusters=min(len(data), self._max_clusters),
            weight_threshold=self._weight_threshold,
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
            encodings=[
                FloatColumnEncoding(),
                OneHotColumnEncoding(num_values=num_components),
            ],
        )

    def _fit_discrete(self, data: pd.DataFrame) -> ColumnTransformInfo:
        """Fit one hot encoder for discrete column."""
        column_name = data.columns[0]
        if data[column_name].nunique() >= self._binary_encoder_cutoff:
            encoder = BinaryEncodingTransformer(
                handle_rounding_nan=self._binary_encoder_han_handler
            )
            one_hot = False
        else:
            encoder = OneHotEncoder()
            one_hot = True
        if self._verbose:
            logger.info(
                f"Transforming discrete column: {column_name!r} with {encoder.__class__.__name__}"
            )
        encoder.fit(data, column_name)
        num_encoded_columns = len(encoder.dummies)
        encoding = (
            OneHotColumnEncoding(num_values=num_encoded_columns)
            if one_hot
            else BinaryColumnEncoding(num_bits=num_encoded_columns)
        )

        return ColumnTransformInfo(
            column_name=column_name,
            column_type=ColumnType.DISCRETE,
            transform=encoder,
            encodings=[encoding],
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
        self.dataframe = True

        if discrete_columns is None:
            discrete_columns = []

        if not isinstance(raw_data, pd.DataFrame):
            self.dataframe = False
            # work around for RDT issue #328 Fitting with numerical column names fails
            discrete_columns = [str(column) for column in discrete_columns]
            column_names = [str(num) for num in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)

        # Rather than calling infer_objects() on raw_data as a whole, call it on
        # each column individually and stitch together the result mimicking the
        # layout of raw_data.infer_objects().dtypes. This is done to reduce memory
        # pressure, as raw_data.infer_objects() might create a copy of the entire
        # data, whereas in the approach below, we only need duplicate the data from
        # a single column at a time.
        self._column_raw_dtypes = pd.Series(
            {c: s.infer_objects().dtype for c, s in raw_data.items()}
        )
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

            self._column_transform_info_list.append(column_transform_info)

    def _transform_continuous(
        self, column_transform_info: ColumnTransformInfo, data: pd.DataFrame
    ) -> List[np.ndarray]:
        column_name = data.columns[0]
        data[column_name] = data[column_name].to_numpy().flatten().astype(np.float32)
        gm = column_transform_info.transform
        transformed = gm.transform(data)

        # Pandas DataFrame block manager stores columns of the same type in a contiguous
        # NumPy array. Therefore, copy the column arrays to ensure that transformed becomes
        # eligible for garbage collection.
        # The normalized column is stored as a float32 even if the input is a float64, because
        # it will be stored in a float32 tensor anyway during learning.
        normalized = transformed[f"{column_name}.normalized"].to_numpy(
            dtype=np.float32, copy=True
        )
        component = transformed[f"{column_name}.component"].to_numpy(
            dtype=column_transform_info.encodings[1].decoded_dtype, copy=True
        )
        return [normalized, component]

    def _transform_discrete(
        self, column_transform_info: ColumnTransformInfo, data: pd.DataFrame
    ) -> List[np.ndarray]:
        # Use the RDT transformer to transform into the encoded representation
        # (binary or onehot), then use the ColumnEncoder to decode. This seems
        # a bit unnecessary, because going from categorical to decoded form
        # (ordinal ints) is more direct rather than going to onehot/binary, but
        # doing it this way minimizes code duplication and ensures that there are no
        # discrepancies caused by our binary/onehot encoding logic and the one
        # done by RDT (e.g., starting at 0 vs 1 for ordinals) - and if they are,
        # they cancel each other out through the decoding/encoding roundtrip.
        transformed = column_transform_info.transform.transform(data).to_numpy()

        decoded = column_transform_info.encodings[0].decode(transformed)
        return [decoded]

    def _transform(
        self,
        raw_data: pd.DataFrame,
        column_transform_info_list: List[ColumnTransformInfo],
    ) -> List[List[np.ndarray]]:
        """Take a Pandas DataFrame and transform columns"""
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

    def transform_decoded(self, raw_data: DFLike) -> TrainData:
        """Take raw data and output the transformed column data in decoded form."""
        if not isinstance(raw_data, pd.DataFrame):
            column_names = [str(num) for num in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)

        column_data_list = self._transform(
            raw_data,
            self._column_transform_info_list,
        )

        return TrainData(self._column_transform_info_list, column_data_list)

    def transform(self, raw_data: DFLike) -> np.ndarray:
        """Take raw data and output encoded matrix data.

        This function only exists for backwards compatibility. It is not used
        internally, as it requires a lot of memory.
        """
        return self.transform_decoded(raw_data).to_numpy_encoded(dtype=np.float32)

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

    def convert_conditions(self, conditions: Dict[str, Any]) -> np.ndarray:
        """Convert dictionary of conditions to encoded numpy array.

        Args:
            conditions: Dictionary mapping column names to column values to
            condition on. Column names and values must already be processed by
            SDV Table.transform() method for metadata transformations (e.g.,
            appends '.value' to names of numeric columns), but not processed by
            this DataTransformer instances transform methods yet.

        Returns:
            Numpy array of 1 row and encoded_dim columns. The encoding contains
            the conditioned columns and is 0 everywhere else.
        """

        known_columns = set(
            info.column_name for info in self._column_transform_info_list
        )
        unknown_columns = set(conditions.keys()) - known_columns
        if unknown_columns:
            unknown_column_str = ", ".join(sorted(unknown_columns))
            raise ValueError(
                f"Conditions includes unknown columns: {unknown_column_str}"
            )

        # TODO: Maybe reuse logic from training transforms/encode code paths,
        # keeping this logic in sync with logic scattered across
        # data_transformer.py and train_data.py is going to be a challenge.
        encodings = []
        for info in self._column_transform_info_list:
            if info.column_name in conditions.keys():
                # Make 1 row DataFrame to convert to encoded representation
                raw_data = pd.DataFrame(
                    {info.column_name: [conditions[info.column_name]]}
                )
                # Grab first element of the returned list of lists because we
                # have exactly 1 column in raw_data
                transformed_data = self._transform(raw_data, [info])[0]

                encoding = np.concatenate(
                    [
                        enc.encode(transformed_data[index])
                        for index, enc in enumerate(info.encodings)
                    ],
                    axis=1,
                    dtype="float32",
                )

            else:
                # Column is not being conditioned on. Use "default" encoding of all 0s.
                encoding = np.zeros(
                    (
                        1,
                        info.output_dimensions,
                    ),
                    dtype="float32",
                )

            encodings.append(encoding)

        return np.concatenate(encodings, axis=1)
