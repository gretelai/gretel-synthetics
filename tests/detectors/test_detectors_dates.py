import uuid

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from gretel_synthetics.detectors.dates import (
    _infer_from_series,
    _tokenize_date_str,
    detect_datetimes,
)

ROW_COUNT = 300


@pytest.fixture(scope="session")
def test_df() -> pd.DataFrame:
    dates = ["12/20/2020", "10/17/2020", "08/10/2020", "01/22/2020", "09/01/2020"]
    iso = [datetime.utcnow().isoformat() for _ in range(5)]
    random = [uuid.uuid4().hex for _ in range(5)]
    data = list(zip(dates, random, iso))
    return pd.DataFrame(data, columns=["dates", "random", "iso"])


test_dates = [
    ("12/20/2020", "##/##/####", datetime(year=2020, month=12, day=20), "%m/%d/%Y"),
    ("12.20.2020", "##.##.####", datetime(year=2020, month=12, day=20), "%m.%d.%Y"),
    ("12-20-2020", "##-##-####", datetime(year=2020, month=12, day=20), "%m-%d-%Y"),
    ("2020/12/20", "####/##/##", datetime(year=2020, month=12, day=20), "%Y/%m/%d"),
    ("Dec 20 2020", "### ## ####", datetime(year=2020, month=12, day=20), "%b %d %Y"),
    (
        "Dec 20, 2020",
        "### ##, ####",
        datetime(year=2020, month=12, day=20),
        "%b %d, %Y",
    ),
    ("2020 Dec 20", "#### ### ##", datetime(year=2020, month=12, day=20), "%Y %b %d"),
    (
        "2020-11-12T00:02:48.333811",
        "####-##-##T########.######",
        datetime(2020, 11, 12, 0, 2, 48, 333811),
        "%Y-%m-%dT%X.%f",
    ),
    (
        "2020-11-12T00:02:48",
        "####-##-##T########",
        datetime(2020, 11, 12, 0, 2, 48),
        "%Y-%m-%dT%X",
    ),
    (
        "1997-07-16T19:20:30-0700",
        "####-##-##T########-####",
        datetime(
            1997, 7, 16, 19, 20, 30, tzinfo=timezone(timedelta(days=-1, seconds=61200))
        ),
        "%Y-%m-%dT%X%z",
    ),
]


@pytest.mark.parametrize(
    "input_str, expected_mask",
    [(input_str, mask) for input_str, mask, _, _ in test_dates],
)
def test_date_str_tokenizer(input_str, expected_mask):
    assert _tokenize_date_str(input_str).masked_str == expected_mask


def test_infer_from_series():
    dates = ["12/20/2020", "10/17/2020", "08/10/2020", "01/22/2020", "09/01/2020"]
    assert _infer_from_series(dates, False) == "%m/%d/%Y"


def test_infer_from_bad_date():
    dates = ["#NAME?", "1000#", "Jim", "3", "$moola"]
    assert _infer_from_series(dates, False) is None


def test_infer_from_some_bad_date():
    dates = ["#NAME?", "1000#", "Jim", "3", "10/17/2020"]
    assert _infer_from_series(dates, False) == "%m/%d/%Y"


def test_infer_from_12_hour():
    dates = ["8:15 AM", "9:20 PM", "1:55 PM"]
    assert _infer_from_series(dates, False) == "%I:%M %p"


@pytest.mark.parametrize("with_suffix", [True, False])
def test_detect_datetimes(with_suffix, test_df):
    # Based on the values in the DF, we assert the `with_suffix` flag
    # should not change any of the results
    check = detect_datetimes(test_df, with_suffix=with_suffix)
    assert set(check.column_names) == {"dates", "iso"}
    assert check.get_column_info("random") is None

    dates = check.get_column_info("dates")
    assert dates.name == "dates"
    assert dates.inferred_format == "%m/%d/%Y"

    iso = check.get_column_info("iso")
    assert iso.name == "iso"
    assert iso.inferred_format == "%Y-%m-%dT%X.%f"


def test_infer_with_suffix():
    dates = [
        "2020-12-20T00:00:00Z",
        "2020-10-17T00:00:00Z",
        "2020-08-10T00:00:00Z",
        "2020-01-22T00:00:00Z",
        "2020-09-01T00:00:00Z",
    ]
    assert _infer_from_series(dates, True) == "%Y-%m-%dT%XZ"

    dates_2 = [d.replace("Z", "+00:00") for d in dates.copy()]
    assert _infer_from_series(dates_2, True) == "%Y-%m-%dT%X+00:00"

    dates_3 = [d.replace("Z", "-00:00") for d in dates.copy()]
    assert _infer_from_series(dates_3, True) == "%Y-%m-%dT%X-00:00"


def test_detect_datetimes_with_suffix(test_df):
    # Add a TZ suffix of "Z" to the iso strings
    test_df["iso"] = test_df["iso"].astype("string").apply(lambda val: val + "Z")
    check = detect_datetimes(test_df, with_suffix=True)
    assert set(check.column_names) == {"dates", "iso"}

    iso = check.get_column_info("iso")
    assert iso.name == "iso"

    # NOTE: We should have just a stringified "Z" as the suffix now
    assert iso.inferred_format == "%Y-%m-%dT%X.%fZ"


def test_detect_datetimes_custom_formats():
    df = pd.DataFrame(
        {
            "str": ["a", "b", "c"],
            "number": [1000, 1003, 1005],
            "dateandtime": ["2022-01-01 10:40", "2022-01-01 10:41", "2022-01-01 10:42"],
            "dateandtimeseconds": [
                "2022-01-01 10:40:00",
                "2022-01-01 10:40:01",
                "2022-01-01 10:40:02",
            ],
            "hourminute": ["11:51", "11:52", "11:53"],
            "timeonly": ["11:51:00", "11:52:00", "11:53:00"],
            "hour12": ["8:40:25 AM", "12:55:13 PM", "1:00:00 AM"],
        }
    )

    check = detect_datetimes(df)

    assert set(check.column_names) == {
        "dateandtime",
        "dateandtimeseconds",
        "hourminute",
        "timeonly",
        "hour12",
    }

    dateandtime = check.get_column_info("dateandtime")
    assert dateandtime.name == "dateandtime"
    assert dateandtime.inferred_format == "%Y-%m-%d %H:%M"

    dateandtimeseconds = check.get_column_info("dateandtimeseconds")
    assert dateandtimeseconds.name == "dateandtimeseconds"
    assert dateandtimeseconds.inferred_format == "%Y-%m-%d %X"

    hourminute = check.get_column_info("hourminute")
    assert hourminute.name == "hourminute"
    assert hourminute.inferred_format == "%H:%M"

    timeonly = check.get_column_info("timeonly")
    assert timeonly.name == "timeonly"
    assert timeonly.inferred_format == "%X"

    hour12 = check.get_column_info("hour12")
    assert hour12.name == "hour12"
    assert hour12.inferred_format == "%I:%M:%S %p"
