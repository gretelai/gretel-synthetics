"""DataSampler module."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, TYPE_CHECKING

import numpy as np

from gretel_synthetics.actgan.column_encodings import OneHotColumnEncoding
from gretel_synthetics.actgan.structures import ColumnType
from gretel_synthetics.actgan.train_data import TrainData

if TYPE_CHECKING:
    from gretel_synthetics.actgan.structures import ColumnIdInfo, ColumnTransformInfo


@dataclass
class _DiscreteColumnInfo:
    data: np.ndarray
    num_values: int


def _is_discrete_column(column_info: ColumnTransformInfo) -> bool:
    # For the purposes of sampling, we only treat OneHot-encoded discrete columns as discrete.
    # Also note that the OneHot-encoded columns indicating which cluster a normalized float value
    # belongs to are not treated as discrete either.
    return (
        column_info.column_type == ColumnType.DISCRETE
        and len(column_info.encodings) == 1
        and isinstance(column_info.encodings[0], OneHotColumnEncoding)
    )


class DataSampler:
    """DataSampler samples the conditional vector and corresponding data for ACTGAN."""

    def __init__(
        self,
        train_data: TrainData,
        log_frequency: bool,
    ):
        self._train_data = train_data
        self._n_rows = len(train_data)

        self._discrete_columns = [
            _DiscreteColumnInfo(
                data=data_list[0], num_values=column_info.encodings[0].num_values
            )
            for column_info, data_list in train_data.columns_and_data
            if _is_discrete_column(column_info)
        ]

        n_discrete_columns = len(self._discrete_columns)

        # Store the row id for each category in each discrete column.
        # For example _rid_by_cat_cols[a][b] is a list of all rows with the
        # a-th discrete column equal value b.
        self._rid_by_cat_cols: List[List[np.ndarray]] = [
            [np.nonzero(col.data == j)[0] for j in range(col.num_values)]
            for col in self._discrete_columns
        ]

        # Prepare an interval matrix for efficiently sample conditional vector
        max_category = max(
            (col.num_values for col in self._discrete_columns), default=0
        )

        # Calculate the start position of each discrete column in a conditional
        # vector. I.e., the (ordinal) value b of the a-th discrete column is
        # stored in position _discrete_column_cond_st[a] + b in the conditional
        # vector.
        self._discrete_column_cond_st = np.array(
            ([0] + [col.num_values for col in self._discrete_columns[:-1]])[
                :n_discrete_columns
            ]
        ).cumsum()

        # Calculate the probability distributions for the values in each discrete
        # column. We store cumulative probabilities, as that is what we need for
        # sampling.
        # The probability that the (ordinal) value of the a-th discrete column is
        # less than or equal to b is _discrete_column_category_prob_cum[a, b].
        self._discrete_column_category_prob_cum = np.zeros(
            (n_discrete_columns, max_category)
        )
        self._n_discrete_columns = n_discrete_columns
        self._n_categories = sum(col.num_values for col in self._discrete_columns)

        category_freqs = []
        for i, col in enumerate(self._discrete_columns):
            category_freq = np.bincount(col.data, minlength=max_category)
            category_freqs.append(category_freq[: col.num_values])
            if log_frequency:
                category_freq = np.log(category_freq + 1)
            category_prob = category_freq / np.sum(category_freq)
            self._discrete_column_category_prob_cum[i] = category_prob.cumsum()

        self._condvec_sampler = CondVecSampler(category_freqs)

    def _random_choice_prob_index(self, discrete_column_id):
        probs_cum = self._discrete_column_category_prob_cum[discrete_column_id]
        r = np.expand_dims(np.random.rand(probs_cum.shape[0]), axis=1)
        return (probs_cum > r).argmax(axis=1)

    def sample_condvec(self, batch):
        """Generate the conditional vector for training.

        Returns:
            cond (batch x #categories):
                The conditional vector.
            mask (batch x #discrete columns):
                A one-hot vector indicating the selected discrete column.
            discrete column id (batch):
                Integer representation of mask.
            category_id_in_col (batch):
                Selected category in the selected discrete column.
        """
        if self._n_discrete_columns == 0:
            return None

        discrete_column_id = np.random.choice(
            np.arange(self._n_discrete_columns), batch
        )

        cond = np.zeros((batch, self._n_categories), dtype="float32")
        mask = np.zeros((batch, self._n_discrete_columns), dtype="float32")
        mask[np.arange(batch), discrete_column_id] = 1
        category_id_in_col = self._random_choice_prob_index(discrete_column_id)
        category_id = (
            self._discrete_column_cond_st[discrete_column_id] + category_id_in_col
        )
        cond[np.arange(batch), category_id] = 1

        return cond, mask, discrete_column_id, category_id_in_col

    def sample_data(self, n, col, opt):
        """Sample data from original training data satisfying the sampled conditional vector.

        Returns:
            n rows of matrix data.
        """
        if col is None:
            idx = np.random.randint(len(self._train_data), size=n)
            return self._train_data.to_numpy_encoded(row_indices=idx)

        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self._rid_by_cat_cols[c][o]))

        return self._train_data.to_numpy_encoded(row_indices=idx)

    def dim_cond_vec(self) -> int:
        """Return the total number of categories."""
        return self._n_categories

    @property
    def condvec_sampler(self) -> CondVecSampler:
        return self._condvec_sampler


class CondVecSampler:

    _n_discrete_columns: int
    """The total number of discrete columns."""
    _n_categories: int
    """The cumulative number of categories across all discrete columns."""
    _categories_uniform_prob_cum: np.ndarray
    """A vector with cumulative probabilities for selecting column/category pairs."""
    _discrete_column_cond_st: np.ndarray
    """Starting offset for each discrete column in the conditional vector."""

    def __init__(self, category_freqs: List[np.ndarray]):
        """Constructor.

        Args:
            category_freqs:
                For each discrete column, this list contains a 1D array with
                absolute category frequencies.
        """

        self._n_discrete_columns = len(category_freqs)
        self._n_categories = sum(len(a) for a in category_freqs)

        if self._n_discrete_columns == 0:
            return

        # Calculate a probability vector for selecting a single (column, category) pair,
        # where the column is chosen uniformly at random among all discrete columns, and the
        # category is chosen according to its relative frequency within the column.
        categories_uniform_prob = np.concatenate([a / a.sum() for a in category_freqs])
        categories_uniform_prob = (
            categories_uniform_prob / categories_uniform_prob.sum()
        )
        self._categories_uniform_prob_cum = categories_uniform_prob.cumsum()

        # Calculate the starting offset for each discrete column in the conditional vector.
        # This is the cumulative number of categories in all previous discrete columns.
        self._discrete_column_cond_st = np.array(
            [0] + [len(a) for a in category_freqs[:-1]]
        ).cumsum()

    def sample_original_condvec(self, batch_size: int):
        """Generate the conditional vector for generation use original frequency."""
        if self._n_discrete_columns == 0:
            return None

        r = np.random.rand(batch_size, 1)
        pick = np.argmax(r < self._categories_uniform_prob_cum, axis=1)

        cond = np.zeros((batch_size, self._n_categories), dtype="float32")
        cond[np.arange(batch_size), pick] = 1

        return cond

    def generate_cond_from_condition_column_info(
        self,
        condition_info: ColumnIdInfo,
        batch_size: int,
    ) -> np.ndarray:
        if condition_info.discrete_column_id >= self._n_discrete_columns:
            raise ValueError(
                f"invalid discrete column ID {condition_info.discrete_column_id}, "
                + f"there are only {self._n_discrete_columns} discrete columns"
            )
        """Generate the condition vector."""
        vec = np.zeros((batch_size, self._n_categories), dtype="float32")
        id_ = self._discrete_column_cond_st[condition_info.discrete_column_id]
        id_ += condition_info.value_id
        vec[:, id_] = 1
        return vec
