from __future__ import annotations

import re
import uuid

from functools import partial
from typing import Any, Dict, FrozenSet, List, Optional

import numpy as np
import pandas as pd

from category_encoders import BaseNEncoder, BinaryEncoder
from rdt.transformers import BaseTransformer
from rdt.transformers import ClusterBasedNormalizer as RDTClusterBasedNormalizer
from rdt.transformers import FloatFormatter

from gretel_synthetics.typing import ListOrSeriesOrDF, SeriesOrDFLike

MODE = "mode"
VALID_ROUNDING_MODES = (MODE,)


def _new_uuid() -> str:
    return f"gretel-{uuid.uuid4().hex}"


_MAX_CHUNK = 131072


class ClusterBasedNormalizer(RDTClusterBasedNormalizer):
    """A version of the ClusterBasedNormalizer with improved performance.

    This makes two changes to RDT's version of the ClusterBasedNormalizer:
    - To reduce memory pressure, input is split into chunks of size `_MAX_CHUNK_SIZE`,
      which are then processed individually. This is because probability prediction
      temporarily requires roughly 30x the memory of the data column (for 10 clusters),
      which can add significant memory pressure. This also means more NumPy calls, but
      the impact of this is negligible.
    - Instead of a for loop for sample selection, sample selection is done via a vectorized
      NumPy expression.
    """

    def _transform(self, data):
        """Transform the numerical data.
        Args:
            data (pandas.Series):
                Data to transform.
        Returns:
            numpy.ndarray.
        """
        data = FloatFormatter._transform(self, data)
        if data.ndim > 1:
            data, model_missing_values = data[:, 0], data[:, 1]

        output_model_missing_values = (
            self.null_transformer and self.null_transformer.models_missing_values()
        )

        data = data.reshape((len(data), 1))
        means = self._bgm_transformer.means_.reshape((1, self.max_clusters))

        stds = np.sqrt(self._bgm_transformer.covariances_).reshape(
            (1, self.max_clusters)
        )

        # Pre-allocate the result
        result = np.zeros_like(
            data, shape=(data.shape[0], 3 if output_model_missing_values else 2)
        )
        if output_model_missing_values:
            result[:, 2] = model_missing_values

        # Process the data in chhunks
        result_out = result
        while len(data) > 0:
            data_chunk = data[:_MAX_CHUNK]
            data = data[_MAX_CHUNK:]
            result_chunk = result_out[:_MAX_CHUNK]
            result_out = result_out[_MAX_CHUNK:]

            normalized_values = (data_chunk - means) / (self.STD_MULTIPLIER * stds)
            normalized_values = normalized_values[:, self.valid_component_indicator]
            component_probs = self._bgm_transformer.predict_proba(data_chunk)

            # Performance optimization: vectorize sample selection
            component_probs = component_probs[:, self.valid_component_indicator] + 1e-6
            component_probs = component_probs / component_probs.sum(axis=1).reshape(
                -1, 1
            )

            rands = np.random.rand(component_probs.shape[0]).reshape(-1, 1)
            selected_component = (rands < component_probs.cumsum(axis=1)).argmax(axis=1)
            # End performance optimization

            aranged = np.arange(len(data_chunk))
            normalized = normalized_values[aranged, selected_component].reshape([-1, 1])
            normalized = np.clip(normalized, -0.99, 0.99)
            result_chunk[:, 0] = normalized[:, 0]
            result_chunk[:, 1] = selected_component

        return result


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
            self._mode_values = frozenset(list(data.mode(dropna=False)))
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

        data = data.fillna(self._nan_proxy)
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
