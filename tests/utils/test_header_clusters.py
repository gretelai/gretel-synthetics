from pathlib import Path

import pandas as pd
import pytest

from gretel_synthetics.utils import header_clusters as hc


@pytest.fixture()
def sample_df():
    return pd.read_csv(
        Path(__file__).parent / "data/cluster_test_data.csv", index_col=0
    )


def test_backward_compat(sample_df):
    old_clusters = hc.cluster(sample_df)
    new_clusters = hc.cluster(sample_df, average_record_length_threshold=250.0)

    assert len(old_clusters) != len(new_clusters)
    assert new_clusters == [["label"], ["text"], ["title"]]
    assert old_clusters == [["label", "text", "title"]]


def test_no_empty_clusters(sample_df):
    clusters = hc.cluster(sample_df, average_record_length_threshold=250.0)
    assert [] not in clusters
