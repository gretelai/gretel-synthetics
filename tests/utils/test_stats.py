from pathlib import Path

import faker
import numpy as np
import pandas as pd
import pytest

from gretel_synthetics.utils import stats


@pytest.fixture(scope="module")
def fake():
    fake = faker.Faker("en_US")
    return fake


def test_count_memorized_lines(fake: faker.Faker):
    records1 = []
    records2 = []
    records3 = []
    for _ in range(10):
        records1.append(
            {"foo": fake.lexify(text="????????"), "bar": fake.lexify(text="????????")}
        )
        records2.append(
            {"foo": fake.lexify(text="????????"), "bar": fake.lexify(text="????????")}
        )
        records3.append(
            {"foo": fake.lexify(text="????????"), "bar": fake.lexify(text="????????")}
        )
    df1 = pd.DataFrame(records1 + records2)
    df2 = pd.DataFrame(records2 + records3)
    df_intersection = pd.DataFrame(records2)
    assert stats.count_memorized_lines(df1, df2) == len(
        set(df_intersection.to_csv(header=False, index=False).strip("\n").split("\n"))
    )


def test_memorized_lines_differing_dtypes():
    data_source = "https://gretel-public-website.s3.us-west-2.amazonaws.com/datasets/USAdultIncome5k.csv"
    left = pd.read_csv(data_source)
    right = pd.read_csv(data_source)

    right.loc[0, "capital_gain"] = np.nan
    right.loc[len(right)] = left.loc[0].values.tolist()

    # Here, we cast within two numeric types and can find duplicates.  We drop duplicates _within_ each df to prevent overcounting.
    assert stats.count_memorized_lines(left, right) == 4998

    # If one field is already dumped directly from int to string before we can look at things, we still attempt uptying to find duplicates.
    left_stringified = left.astype({"capital_gain": str})
    assert stats.count_memorized_lines(left_stringified, right) == 4998

    # BUT if we change one entry to a "real" string, the uptyping conversion will fail and we will skip this column.
    # We proceed to floatify numeric columns (including "capital_gain" in the right df), then dump to string.
    # The int format strings in left and float formatted strings will not match and we get 0 memorized lines.
    left_stringified["capital_gain"][0] = "hello"
    assert stats.count_memorized_lines(left_stringified, right) == 0

    # BUT if the float field is stringified first, the int column will eventually match as we step down to object via float.
    left_floated = left.astype({"capital_gain": "float"})
    right_stringified = right.astype({"capital_gain": "object"})
    assert stats.count_memorized_lines(left_floated, right_stringified) == 4998


def test_get_categorical_field_distribution():
    df = pd.DataFrame(
        [{"foo": "bar"}] * 2 + [{"foo": "baz"}] * 2 + [{"foo": "barf"}] * 4
    )
    distribution = stats.get_categorical_field_distribution(df["foo"])
    assert distribution["bar"] == 25.0
    assert distribution["baz"] == 25.0
    assert distribution["barf"] == 50.0


def test_compute_distribution_distance():
    # Based on examples at
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html
    # BUT NOTE that we use base 2 throughout, some examples don't.
    # Mostly want to test that we feed values into this function correctly.
    d1 = {"foo": 1.0, "baz": 0.0}
    d2 = {"bar": 1.0, "baz": 0.0}
    assert abs(stats.compute_distribution_distance(d1, d2) - 1.0) < 0.01

    d1 = {"foo": 1.0, "baz": 0.0}
    d2 = {"foo": 0.5, "baz": 0.5}
    assert abs(stats.compute_distribution_distance(d1, d2) - 0.5579230452841438) < 0.01

    d1 = {"foo": 1.0, "bar": 0.0, "baz": 0.0}
    d2 = {"foo": 1.0}
    assert abs(stats.compute_distribution_distance(d1, d2)) < 0.01


def test_numeric_binning_sanity():
    # walk through the steps that gave us too many bins in CORE-316
    train_path = Path(__file__).parent / "data/train.csv"
    train = pd.read_csv(train_path)

    synth_path = Path(__file__).parent / "data/synth.csv"
    synth = pd.read_csv(synth_path)

    train_rows, train_cols = train.shape
    synth_rows, synth_cols = synth.shape
    max_rows = min(train_rows, synth_rows)
    train_subsample = (
        train.sample(n=max_rows, random_state=333) if train_rows > synth_rows else train
    )
    synth_subsample = (
        synth.sample(n=max_rows, random_state=333) if synth_rows > train_rows else synth
    )

    pca_train = stats.compute_pca(train_subsample)
    pca_synth = stats.compute_pca(synth_subsample)

    found_bad_column = False
    for field in pca_train.columns:
        min_value = min(min(pca_train[field]), min(pca_synth[field]))
        max_value = max(max(pca_train[field]), max(pca_synth[field]))
        # Use ‘fd’ (Freedman Diaconis Estimator), our default binning.
        # We are looking for a "bad" column that will give us too many bins.
        fd_bins = np.histogram_bin_edges(
            pca_train[field], bins="fd", range=(min_value, max_value)
        )
        if len(fd_bins) > 500:
            # We found a bad column. Set the flag and show that 'doane' will give us a more manageable number of bins.
            found_bad_column = True
            bins = stats.get_numeric_distribution_bins(
                pca_train[field], pca_synth[field]
            )
            assert len(bins) < 500

    assert found_bad_column


def test_correlations_with_inf():
    df1 = pd.DataFrame(
        [{"foo": 1, "bar": 1}, {"foo": 2, "bar": 2}, {"foo": 3, "bar": 3}] * 10
    )
    df1_corr = stats.calculate_correlation(df1)
    assert not df1_corr.empty
    df1_corr_opt = stats.calculate_correlation(df1, opt=True)
    assert not df1_corr_opt.empty

    df2 = pd.DataFrame(
        [{"foo": 1.1, "bar": 1}, {"foo": 2.2, "bar": 2}, {"foo": np.inf, "bar": 3}] * 10
    )
    df2_corr = stats.calculate_correlation(df2)
    assert not df2_corr.empty
    df2_corr_opt = stats.calculate_correlation(df2, opt=True)
    assert not df2_corr_opt.empty

    df_multi_dtype_list = []
    for i in range(100):
        entry = {"foo": i, "foobie": i, "bar": i * 1.1, "baz": str(i) + "yoooo"}
        if i % 3 == 0:
            entry["foo"] = np.inf
        if i % 11 == 0:
            entry["foo"] = np.nan

        if i % 5 == 0:
            entry["bar"] = np.inf
        if i % 13 == 0:
            entry["bar"] = np.nan

        if i % 7 == 0:
            entry["baz"] = np.inf
        if i % 17 == 0:
            entry["baz"] = np.nan

        df_multi_dtype_list.append(entry)
    df_multi_type = pd.DataFrame(df_multi_dtype_list)
    df_multi_type_corr = stats.calculate_correlation(
        df_multi_type, nominal_columns=["baz"]
    )
    assert not df_multi_type_corr.empty
    df_multi_type_corr_opt = stats.calculate_correlation(
        df_multi_type, nominal_columns=["baz"], opt=True
    )
    assert not df_multi_type_corr_opt.empty
