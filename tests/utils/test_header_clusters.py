from pathlib import Path

import pandas as pd
import pytest

from gretel_synthetics.utils import header_clusters as hc


@pytest.fixture()
def sample_df():
    return pd.read_csv(
        Path(__file__).parent / "data/cluster_test_data.csv", index_col=0
    )


@pytest.fixture()
def sample_df_2():
    dataset_path = "https://gretel-public-website.s3.amazonaws.com/datasets/experiments/complex_id_dataset.csv"
    ROUND_DECIMALS = 4
    tmp = pd.read_csv(dataset_path, low_memory=False)
    tmp = tmp.round(ROUND_DECIMALS)
    return tmp


def test_backward_compat(sample_df):
    old_clusters = hc.cluster(sample_df)
    new_clusters = hc.cluster(sample_df, average_record_length_threshold=250.0)

    assert len(old_clusters) != len(new_clusters)
    assert new_clusters == [["label"], ["text"], ["title"]]
    assert old_clusters == [["label", "text", "title"]]


def test_no_empty_clusters(sample_df):
    clusters = hc.cluster(sample_df, average_record_length_threshold=250.0)
    assert [] not in clusters


# sample_df doesn't have any fields that should be single batched,
# so independent of isolation flag, clusters should be the same
def test_no_isolation(sample_df):
    old_clusters = hc.cluster(sample_df, maxsize=20, isolate_complex_field=False)
    new_clusters = hc.cluster(sample_df, maxsize=20)

    assert len(old_clusters) == len(new_clusters)
    assert old_clusters == new_clusters


# sample_df_2 has fields that should be single batched, so shouldn't be the same depending on state of isolation flag
# 'Prospect ID' is the one field that should be single batched
def test_isolation(sample_df_2):
    old_clusters = hc.cluster(sample_df_2, maxsize=20, isolate_complex_field=False)
    new_clusters = hc.cluster(sample_df_2, maxsize=20)

    assert old_clusters != new_clusters
    assert ["Prospect ID"] in new_clusters
    assert ["Prospect ID"] not in old_clusters
