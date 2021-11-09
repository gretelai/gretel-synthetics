"""
This module enables the clustering of DataFrame headers into
like clusters based on correlations between columns
"""
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

from gretel_synthetics.utils import stats

LEFT = 0
RIGHT = 1


def _get_correlation_matrix(df, numeric_cat: List[str] = None):

    if numeric_cat is None:
        numeric_cat = []

    nominal_columns = list(df.select_dtypes(include=["object", "category"]).columns)
    nominal_columns.extend(x for x in numeric_cat if x not in nominal_columns)

    corr_matrix = stats.calculate_correlation(df, nominal_columns=nominal_columns)

    return corr_matrix


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


def _traverse_node(tree, node, maxsize, totcolcnt):

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
    clusters: List[List[int]], maxlen: int, columns: List[str], Lopt, plot=False
) -> List[List[str]]:
    out = []
    tmp = []
    cluster_map = {}  # maps a column name => cluster number
    cluster_number = 0
    for cluster in clusters:
        # if the size of adding the next cluster
        # exceeds the max size, flush
        if len(tmp) + len(cluster) > maxlen:
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
    method: str = "single",
    numeric_cat: List[str] = None,
    plot=False,
) -> List[List[str]]:
    """
    Given an input dataframe, extract clusters of similar headers
    based on a set of heuristics.

    Args:
        df: The dataframe to cluster headers from.
        header_prefix: List of columns to remove before cluster generation.
        maxsize: The max number of header clusters to generate
            from the input dataframe.
        method: Linkage method used to compute header cluster
            distances. For more information please refer to the scipy
            docs, https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy-cluster-hierarchy-linkage.
        numeric_cat: A list of fields to define as categorical. The header
            clustering code will automatically define pandas "object" and
            "category" columns as categorical. The ``numeric_cat`` parameter
            may be used to define additional categorical fields that may
            not automatically get identified as such.
        plot: Plot header list as a dendogram.
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

    # Start at the top of the cluster hierachy with the final two clusters that were merged together
    # We will recursively work our way down, fetching the subclusters of a cluster if the current
    # cluster size > maxsize
    clusters = _traverse_node(Lopt, start, maxsize, len(columns))

    # At this point we have one list of column ids, where groups are seperated by -1, translate it into a list of
    # header lists, and if plot=True, plot the dendogram
    col_list = _merge_clusters(clusters, maxsize, columns, Lopt, plot)

    return prepare_response(col_list, header_prefix)
