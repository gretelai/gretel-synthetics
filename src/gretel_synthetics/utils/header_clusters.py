import copy
import re

from functools import reduce
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

from gretel_synthetics.utils import stats

LEFT = 0
RIGHT = 1
COMPLEX_ID_PERC_UNIQ = 0.85
COMPLEX_ID_LEN = 16
TEXT_COL_LIMIT = 1500


def _is_field_complex(field: pd.Series) -> bool:
    """
    Function to determine if the field is a complex ID requiring special handling.

    Args:
        field: column values that are being evaluated to determine if the field is complex.

    Returns:
        A boolean value that signifies whether the field is complex or not.
    """

    # Return False if the field has no valid values

    field = field.dropna()
    if len(field) == 0:
        return False

    # Return False if the field is less than 85% unique

    perc_unique = field.nunique() / len(field)
    if perc_unique < COMPLEX_ID_PERC_UNIQ:
        return False

    # Return False if the field has average length less than 16 characters

    textcol = field.to_csv(header=False, index=False)
    avg_len = (len(textcol) - 2 * len(field)) / len(field)

    if avg_len < COMPLEX_ID_LEN:
        return False

    # Return False if values do not contain numbers

    contains_digit = any(map(str.isdigit, textcol[0:TEXT_COL_LIMIT]))
    if not contains_digit:
        return False

    # Return True if the field contains only numbers, letters, underscore or hyphen, else return False

    return bool(
        re.match("^[a-zA-Z0-9\-\_]+$", textcol[0:TEXT_COL_LIMIT].replace("\n", ""))
    )


def _get_correlation_matrix(df, numeric_cat: List[str] = None):

    if numeric_cat is None:
        numeric_cat = []

    nominal_columns = list(df.select_dtypes(include=["object", "category"]).columns)
    nominal_columns.extend(x for x in numeric_cat if x not in nominal_columns)

    corr_matrix = stats.calculate_correlation(df, nominal_columns=nominal_columns)

    return corr_matrix


def _average_record_length(data: pd.DataFrame) -> float:
    """
    Find the average record length of a dataset.

    Args:
        data: dataset used to calculate average record length.

    Returns:
        A float value that represents the average record length in a dataset.
    """

    def _get_row_len(elems: pd.Series) -> float:
        return reduce(
            lambda acc, val: acc + len(str(val)) if np.isscalar(val) else acc, elems, 0
        )

    return data.apply(_get_row_len, axis=1).mean() + len(data.columns) - 1


def _get_leaves(tree, node, totcolcnt):

    return_list = []
    stack = []
    curr = node

    def _walk(node: int, side: int, save=False):
        # If it's a leaf, return a list with this leaf
        if node < totcolcnt:
            if save:
                return_list.append(node)
            return None
        # else perculate
        else:
            node = int(node - totcolcnt)
            child = int(tree[node][side])
            return child

    while True:
        if curr is not None:
            stack.append(curr)
            curr = _walk(curr, LEFT)
        elif stack:
            curr = stack.pop()
            curr = _walk(curr, RIGHT, save=True)
        else:
            break

    return return_list


def _traverse_node(
    data,
    columns,
    tree,
    node,
    maxsize,
    average_record_length_threshold,
    totcolcnt,
):

    stack = []
    node_list = []
    curr = node

    def _walk(node: int, side: int):
        child = int(tree[node][side])
        child_size = 1
        idx = 0
        if child > totcolcnt:
            idx = child - totcolcnt
            child_size = tree[idx][3]

        if child_size > maxsize:
            return idx

        leaves = _get_leaves(tree, child, totcolcnt)
        cols = [columns[index] for index in leaves]
        arl = _average_record_length(data[cols])
        if (
            (arl > average_record_length_threshold)
            and (len(cols) > 1)
            and average_record_length_threshold
        ):
            return idx

        else:
            node_list.append(_get_leaves(tree, child, totcolcnt))
            return None

    while True:
        if curr is not None:
            stack.append(curr)
            curr = _walk(curr, LEFT)
        elif stack:
            curr = stack.pop()
            curr = _walk(curr, RIGHT)
        else:
            break

    return node_list


def _merge_clusters(
    data: pd.DataFrame,
    clusters: List[List[int]],
    maxlen: int,
    average_record_length_threshold: float,
    columns: List[str],
    Lopt,
    plot: bool = False,
) -> List[List[str]]:

    out = []
    tmp = []
    cluster_map = {}  # maps a column name => cluster number
    cluster_number = 0
    for cluster in clusters:
        helper = copy.deepcopy(tmp) + [columns[idx] for idx in cluster]
        arl = (
            _average_record_length(data[helper])
            if average_record_length_threshold
            else 0
        )

        # if the size of adding the next cluster
        # exceeds the max size, flush
        if ((len(helper) > maxlen) or (arl > average_record_length_threshold)) and tmp:
            for column_name in tmp:
                cluster_map[column_name] = cluster_number
            out.append(tmp)
            tmp = []
            cluster_number += 1
        tmp.extend(
            [columns[idx] for idx in cluster]
        )  # build the tmp with the actual column names

    # attach the final cluster
    if tmp:
        cluster_number += 1
        out.append(tmp)
        for column_name in tmp:
            cluster_map[column_name] = cluster_number

    if plot:
        labels = [x + "(" + str(cluster_map[x]) + ")" for x in columns]
        plt.figure(figsize=(25, 8))
        plt.title("Field Header Correlation Cluster Hierarchy")
        sch.dendrogram(Lopt, labels=labels, leaf_rotation=90.0)

    return out


def cluster(
    df: pd.DataFrame,
    header_prefix: List[str] = None,
    maxsize: int = 20,
    average_record_length_threshold: float = 0,
    method: str = "single",
    numeric_cat: List[str] = None,
    plot: bool = False,
    isolate_complex_field: bool = True,
) -> List[List[str]]:
    """
    Given an input dataframe, extract clusters of similar headers
    based on a set of heuristics.
    Args:
        df: The dataframe to cluster headers from.
        header_prefix: List of columns to remove before cluster generation.
        maxsize: The max number of fields in a cluster.
        average_record_length_threshold: Threshold for how long a cluster's records can be.
            The default, 0, turns off the average record length (arl) logic. To use arl,
            use a positive value. Based on our research we recommend setting this value
            to 250.0.
        method: Linkage method used to compute header cluster
            distances. For more information please refer to the scipy
            docs, https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy-cluster-hierarchy-linkage.  # noqa
        numeric_cat: A list of fields to define as categorical. The header
            clustering code will automatically define pandas "object" and
            "category" columns as categorical. The ``numeric_cat`` parameter
            may be used to define additional categorical fields that may
            not automatically get identified as such.
        plot: Plot header list as a dendogram.
        isolate_complex_field: Enables isolation of complex fields when clustering.

    Returns:
        A list of lists of column names, each column name list being an identified cluster.
    """

    def prepare_response(
        col_list: List[List[str]], prefix: List[str] = None
    ) -> List[List[str]]:
        if prefix is not None:
            col_list[0] = prefix + col_list[0]
        return col_list

    if numeric_cat is None:
        numeric_cat = []

    if header_prefix is not None:
        try:
            df = df.drop(header_prefix, axis=1)
        except KeyError as err:
            raise ValueError("Header prefixes do not all exist in source DF") from err

    # Bug(jm): if the number of columns left in the DF is just one
    # we just return that single column
    if df.shape[1] == 1:
        return prepare_response([list(df.columns)], header_prefix)

    # Check for complex fields which will require their batch
    single_batch_columns = []
    if isolate_complex_field:
        cluster_columns = list(df.columns)
        for col in cluster_columns:
            if _is_field_complex(df[col]):
                single_batch_columns.append(col)
                cluster_columns.remove(col)
        df = df.filter(cluster_columns)

    # Start by getting the correlation matrix
    corr_matrix = _get_correlation_matrix(df, numeric_cat)

    # Change it into a distance matrix
    X = 1 - np.array(1 - abs(corr_matrix))

    # Cluster the columns
    L = sch.linkage(X, method=method)

    # Optimize the leaf ordering to minimize the distance between adjacent leaves
    Lopt = sch.optimal_leaf_ordering(L, X)

    columns = df.columns

    start = len(Lopt) - 1

    # Start at the top of the cluster hierarchy with the final two clusters that were merged together
    # We will recursively work our way down, fetching the subclusters of a cluster if the current
    # cluster size > maxsize
    clusters = _traverse_node(
        df,
        columns,
        Lopt,
        start,
        maxsize,
        average_record_length_threshold,
        len(columns),
    )

    # At this point we have one list of column ids, where groups are separated by -1, translate it into a list of
    # header lists, and if plot=True, plot the dendogram
    col_list = _merge_clusters(
        df,
        clusters,
        maxsize,
        average_record_length_threshold,
        columns,
        Lopt,
        plot,
    )

    # Re-add columns that were isolated, as individual batches
    for col in single_batch_columns:
        col_list.append([col])

    return prepare_response(col_list, header_prefix)
