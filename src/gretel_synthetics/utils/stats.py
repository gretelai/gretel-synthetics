"""
Generates correlation reports between data sets.
"""
import math

from typing import List, Tuple

import category_encoders as ce
import numpy as np
import pandas as pd

from dython.nominal import correlation_ratio, theils_u
from joblib import delayed, Parallel
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr
from sklearn.decomposition import PCA as PCA_ANAL
from sklearn.preprocessing import StandardScaler

_DEFAULT_JOB_COUNT = 4
_DEFAULT_REPLACE_VALUE = 0.0
UNIQUENESS_THRESHOLD = 0.1


def count_memorized_lines(df1: pd.DataFrame, df2: pd.DataFrame) -> int:
    """
    Checks for overlap between training and synthesized data.

    Args:
        df1: DataFrame of training data.
        df2: DataFrame of synthetic data.

    Returns:
        int, the number of overlapping elements.
    """
    set1 = set(df1.to_csv(header=False, index=False).strip("\n").split("\n"))
    set2 = set(df2.to_csv(header=False, index=False).strip("\n").split("\n"))
    return len(set1.intersection(set2))


def get_categorical_field_distribution(field: pd.Series) -> dict:
    """
    Calculates the normalized distribution of a categorical field.

    Args:
        field: A sanitized column extracted from one of the df's.

    Returns:
        dict: keys are the unique values in the field, values are percentages (floats in [0, 100]).
    """
    distribution = {}
    if len(field) > 0:
        for v in field:
            distribution[str(v)] = distribution.get(str(v), 0) + 1
        series_len = float(len(field))
        for k in distribution.keys():
            distribution[k] = distribution[k] * 100 / series_len
    return distribution


def get_numeric_distribution_bins(training: pd.Series, synthetic: pd.Series):
    """
    To calculate the distribution distance between two numeric series a la categorical fields
    we need to bin the data.  We want the same bins between both series, based on scrubbed data.

    Args:
        training: The numeric series from the training dataframe.
        synthetic: The numeric series from the synthetic dataframe.

    Returns:
        bin_edges, numpy array of dtype float

    """
    training = training.dropna().astype("float64")
    synthetic = synthetic.dropna().astype("float64")
    # Numeric data. Want the same bins between both df's. We bin based on scrubbed data.
    if len(training) == 0:
        min_value = np.nanmin(synthetic)
        max_value = np.nanmax(synthetic)
    elif len(synthetic) == 0:
        min_value = np.nanmin(training)
        max_value = np.nanmax(training)
    else:
        min_value = min(np.nanmin(training), np.nanmin(synthetic))
        max_value = max(np.nanmax(training), np.nanmax(synthetic))
    bins = np.array([], dtype=np.float)
    try:
        # Use ‘fd’ (Freedman Diaconis Estimator) if we can.  This can produce MANY bins or MemoryErrors,
        # in that case use 'doane' instead (improved Sturges).
        bins = np.histogram_bin_edges(training, bins="fd", range=(min_value, max_value))
    except Exception:
        pass
    if len(bins) == 0 or len(bins) > 500:
        try:
            bins = np.histogram_bin_edges(
                training, bins="doane", range=(min_value, max_value)
            )
        except Exception:
            pass
    # If 'doane' still doesn't do the trick just force 500 bins.
    if len(bins) == 0 or len(bins) > 500:
        try:
            bins = np.histogram_bin_edges(
                training, bins=500, range=(min_value, max_value)
            )
        except Exception:
            pass
    return bins


def get_numeric_field_distribution(field: pd.Series, bins) -> dict:
    """
    Calculates the normalized distribution of a numeric field cut into bins.

    Args:
        field: A sanitized column extracted from one of the df's.
        bins: Usually an np.ndarray from get_bins, but can be anything that can be safely passed to pandas.cut.

    Returns:
        dict: keys are the unique values in the field, values are floats in [0, 1].
    """
    binned_data = pd.cut(field, bins, include_lowest=True)
    distribution = {}
    for d in binned_data:
        if str(d) != "nan":
            distribution[str(d)] = distribution.get(str(d), 0) + 1
    field_length = len(binned_data)
    for k in distribution.keys():
        distribution[k] = distribution[k] / field_length
    return distribution


def compute_distribution_distance(d1: dict, d2: dict) -> float:
    """
    Calculates the Jensen Shannon distance between two distributions.

    Args:
        d1: Distribution dict.  Values must be a probability vector
            (all values are floats in [0,1], sum of all values is 1.0).
        d2: Another distribution dict.

    Returns:
        float: The distance between the two vectors, range in [0, 1].
    """
    all_keys = set(d1.keys()).union(set(d2.keys()))
    if len(all_keys) == 0:
        return 0.0
    d1_values = []
    d2_values = []
    for k in all_keys:
        d1_values.append(d1.get(k, 0.0))
        d2_values.append(d2.get(k, 0.0))
    sd1 = sum(d1_values)
    sd2 = sum(d2_values)
    if sd1 == 0 or np.isnan(sd1) or sd2 == 0 or np.isnan(sd2):
        return 0.5887
    return jensenshannon(np.asarray(d1_values), np.asarray(d2_values), base=2)


def calculate_pearsons_r(x, y, opt) -> Tuple[float, float]:
    """
    Calculate the Pearson correlation coefficient for this pair of rows of our correlation matrix.
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html.

    Args:
        x: first input array.
        y: second input array.
        opt: "optimized."  If False, drop missing values when either the x or y value is null/nan.  If True,
            we've already replaced nan's with 0's for entire datafile.

    Returns:
        As per scipy, tuple of Pearson’s correlation coefficient and Two-tailed p-value.

    """
    if not opt:
        # drop missing values, when either the x or y value is null/nan
        arr = np.array([x, y]).transpose()
        arr = arr[~pd.isnull(arr).any(axis=1)].transpose()
        if (
            len(arr[0]) < 2
            or len(arr[1]) < 2
            or pd.Series(arr[0]).nunique() <= 1
            or pd.Series(arr[1]).nunique() <= 1
        ):
            return 0.0, 0.0
        return pearsonr(arr[0], arr[1])
    else:
        # else we've already replaced nan's with 0's for entire datafile
        return pearsonr(x, y)


def calculate_correlation_ratio(x, y, opt):
    """
    Calculates the Correlation Ratio for categorical-continuous association.  Used in constructing correlation matrix.
    See http://shakedzy.xyz/dython/modules/nominal/#correlation_ratio.

    Args:
        x: first input array, categorical.
        y: second input array, numeric.
        opt: "optimized."  If False, drop missing values if y (the numeric column) is null/nan.

    Returns:
        float in the range of [0,1].

    """
    if not opt:
        # Drop missing values if y (the numeric column) is null/nan
        df = pd.DataFrame({"x": x, "y": y})
        df.dropna(inplace=True)
        x = df["x"]
        y = df["y"]
    if len(x) < 2 or len(y) < 2:
        return 0.0
    else:
        # Either way, we've dealt with missing values by now, so tell dython not to do anything
        return correlation_ratio(x, y, nan_strategy="none")


def calculate_theils_u(x, y):
    """
    Calculates Theil's U statistic (Uncertainty coefficient) for categorical-categorical association.
    Used in constructing correlation matrix.
    See http://shakedzy.xyz/dython/modules/nominal/#theils_u.

    Args:
        x: first input array, categorical.
        y: second input array, categorical.

    Returns:
        float in the range of [0,1].

    """
    # Drop missing values if x or y is null/nan
    df = pd.DataFrame({"x": x, "y": y})
    df.dropna(inplace=True)
    x = df["x"]
    y = df["y"]
    if len(x) == 0 or len(y) == 0:
        return 0
    else:
        return theils_u(x, y, nan_strategy="none")


def calculate_correlation(
    df: pd.DataFrame,
    nominal_columns: List[str] = None,
    job_count: int = _DEFAULT_JOB_COUNT,
    opt: bool = False,
) -> pd.DataFrame:
    """
    Given a dataframe, calculate a matrix of the correlations between the various rows.  We use the
    calculate_pearsons_r, calculate_correlation_ratio and calculate_theils_u to fill in the matrix values.

    Args:
        df: The input dataframe.
        nominal_columns: Columns to treat as categorical.
        job_count: For parallelization of computations.
        opt: "optimized."  If opt is True, then go the faster (just not quite as accurate) route of global
            replace missing with 0.

    Returns:
        A dataframe of correlation values.

    """

    # If opt is True, then go the faster (just not quite as accurate) route of global replace missing with 0
    if opt:
        df.fillna(_DEFAULT_REPLACE_VALUE, inplace=True)

    columns = df.columns
    if nominal_columns is None:
        nominal_columns = list()

    corr = pd.DataFrame(index=columns, columns=columns)
    single_value_columns = []
    numeric_columns = []

    # Set up all the column groupings needed for correlation
    for c in columns:
        if df[c].nunique() == 1:
            single_value_columns.append(c)
        elif c not in nominal_columns:
            numeric_columns.append(c)

    nominal = [x for x in nominal_columns if x not in single_value_columns]
    df_rows = df.shape[0]
    high_unique_nominal = []
    completely_unique_nominal = []
    not_high_unique_nominal = []
    for c in nominal:
        if df[c].nunique() / df_rows == 1:
            completely_unique_nominal.append(c)
        elif df[c].nunique() / df_rows > UNIQUENESS_THRESHOLD:
            high_unique_nominal.append(c)
        else:
            not_high_unique_nominal.append(c)

    notcompletely_unique_nominal = [
        x for x in nominal if x not in completely_unique_nominal
    ]

    # Do Theil's U shortcut for 100% unique nominal (Amy invention that is 99.9% correct, and saves massive time)
    for x in completely_unique_nominal:
        corr.loc[x, :] = 1.0
        for y in columns:
            if x == y:
                corr.at[y, x] = 1.0
            # Edge case, guard against ValueError in math.log when the other column is empty
            elif df[y].nunique() == 0:
                corr.at[y, x] = 0.0
            else:
                corr.at[y, x] = math.log(df[y].nunique()) / math.log(df[x].nunique())

    for x in single_value_columns:
        corr.loc[:, x] = 0.0
        corr.loc[x, :] = 0.0
        corr.loc[x, x] = 1.0

    # Do nominal-nominal exluding any that are 100% unique (Theil's U)
    scores = Parallel(n_jobs=job_count)(
        delayed(calculate_theils_u)(df[field1], df[field2])
        for field1 in notcompletely_unique_nominal
        for field2 in notcompletely_unique_nominal
    )
    i = 0
    for field1 in notcompletely_unique_nominal:
        for field2 in notcompletely_unique_nominal:
            if field1 == field2:
                corr.at[field1, field2] = 1.0
            else:
                # looks backward, but is correct
                corr.at[field2, field1] = scores[i]
            i += 1

    # Do "not_high_unique_nominal with numeric" (Correlation Ratio)
    scores = Parallel(n_jobs=job_count)(
        delayed(calculate_correlation_ratio)(df[field1], df[field2], opt)
        for field1 in not_high_unique_nominal
        for field2 in numeric_columns
    )
    i = 0
    for field1 in not_high_unique_nominal:
        for field2 in numeric_columns:
            corr.at[field1, field2] = scores[i]
            corr.at[field2, field1] = scores[i]
            i += 1

    # Do high_unique_nominal with numeric (Theil's U) (excluding 100% unique)
    # This fixes the problem of highly unique categorical causing mass instability when using
    # the normal approach of correlation ratio.  Because there are so many categorical buckets
    # many end up with just one number is them, which causes correlation ratio's approach of
    # comparing the mean within buckets to the mean overall to give unstable, over inflated
    # correlation values.  Using Theil's U instead gives a much more realistic score
    scores = Parallel(n_jobs=job_count)(
        delayed(calculate_theils_u)(df[field1], df[field2])
        for field1 in high_unique_nominal
        for field2 in numeric_columns
    )
    i = 0
    for field1 in high_unique_nominal:
        for field2 in numeric_columns:
            corr.at[field2, field1] = scores[i]
            i += 1

    scores = Parallel(n_jobs=job_count)(
        delayed(calculate_theils_u)(df[field1], df[field2])
        for field2 in high_unique_nominal
        for field1 in numeric_columns
    )
    i = 0
    for field2 in high_unique_nominal:
        for field1 in numeric_columns:
            corr.at[field2, field1] = scores[i]
            i += 1

    # Do numeric numeric (Pearson's)
    num_len = len(numeric_columns)
    if num_len > 1:
        delayed_calls = []
        for i in range(num_len - 1):
            for j in range(i + 1, num_len):
                delayed_calls.append(
                    delayed(calculate_pearsons_r)(
                        df[numeric_columns[i]], df[numeric_columns[j]], opt
                    )
                )
        scores = Parallel(n_jobs=1)(delayed_calls)
        x = 0
        for i in range(num_len - 1):
            for j in range(i + 1, num_len):
                corr.at[numeric_columns[i], numeric_columns[j]] = scores[x][0]
                corr.at[numeric_columns[j], numeric_columns[i]] = scores[x][0]
                x += 1

    for x in numeric_columns:
        corr.at[x, x] = 1.0

    corr_final = pd.DataFrame(index=columns, columns=columns)

    for field1 in columns:
        for field2 in columns:
            if (
                not np.isnan(corr.at[field1, field2])
                and abs(corr.at[field1, field2]) < np.inf
            ):
                corr_final[field1][field2] = corr[field1][field2]
            else:
                corr_final[field1][field2] = 0

    corr_final.fillna(value=np.nan, inplace=True)

    return corr_final


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prep a dataframe for PCA.  Divide the dataframe into numeric and categorical,
    fill missing values and encode categorical columns by the frequency of each value and
    standardize all values.

    Args:
        df: The dataframe to be subjected to PCA.

    Returns:
        The dataframe, normalized.

    """
    # Divide the dataframe into numeric and categorical
    nominal_columns = list(df.select_dtypes(include=["object", "category"]).columns)
    numeric_columns = []
    for c in df.columns:
        if c not in nominal_columns:
            numeric_columns.append(c)
    df_cat = df.reindex(columns=nominal_columns)
    df_num = df.reindex(columns=numeric_columns)
    df_cat_labels = pd.DataFrame()

    # Fill missing values and encode categorical columns by the frequency of each value
    if len(numeric_columns) > 0:
        df_num = df_num.fillna(df_num.median())

    if len(nominal_columns) > 0:
        df_cat = df_cat.fillna("Missing")
        encoder = ce.count.CountEncoder()
        df_cat_labels = pd.DataFrame(encoder.fit_transform(df_cat))

    # Merge numeric and categorical back into one dataframe
    if len(nominal_columns) == 0:
        new_df = df_num
    elif len(numeric_columns) == 0:
        new_df = df_cat_labels
    else:
        new_df = pd.concat([df_num, df_cat_labels], axis=1, sort=False)

    # Finally, standardize all values
    all_columns = nominal_columns + numeric_columns
    new_df = pd.DataFrame(StandardScaler().fit_transform(new_df), columns=all_columns)

    return new_df


def compute_pca(df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """
    Do PCA on a dataframe.  See https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html.

    Args:
        df: The dataframe to analyze for principal components.
        n_components: Number of components to keep.

    Returns:
        Dataframe of principal components.

    """
    seed = 444
    df = df.dropna(axis="columns", how="all")
    df_norm = normalize_dataset(df)
    pca = PCA_ANAL(n_components=n_components, random_state=seed)
    projected = pca.fit_transform(df_norm)
    columns = [f"pc{i + 1}" for i in range(n_components)]
    return pd.DataFrame(data=projected, columns=columns)
