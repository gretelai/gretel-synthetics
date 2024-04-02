"""
Automatic datetime detection for tabluar data
"""

import itertools
import re

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union

import pandas as pd

DAYS_TO_MS = 8.64e7
MASK_CH = "#"
SAMPLE_SIZE = 100

_date_component_orders = [
    lambda y, m, d, hms, tz: f"{hms}",
    lambda y, m, d, hms, tz: f"{d} {m} {y}",
    lambda y, m, d, hms, tz: f"{m} {d} {y}",
    lambda y, m, d, hms, tz: f"{y} {m} {d}",
    lambda y, m, d, hms, tz: f"{y} {m} {d} {hms}",
    lambda y, m, d, hms, tz: f"{y} {m} {d} {hms} {tz}",
]
"""This list contains date orderings by component."""


_component_formats = {
    "y": {"%y", "%Y"},
    "m": {"%b", "%B", "%m"},
    "d": {"%a", "%A", "%d"},
    "hms": {
        "%X",
        "%X %f",
        "%H:%M",
        "%-H:%-M",
        "%-H:%M",
        "%H:%-M",
        "%I:%M %p",
        "%I:%M:%S %p",
    },
    "tz": {"%z", "%Z", "%!z"},
}
"""For every date component, there may exist multiple formats. This dictionary maps
components to any number of format variations. This used in conjunction with
``date_component_orders`` let us build up permutations of valid date string formats.
"""


_component_seperators = ["/", ".", "-", " ", ",", "T", "Z", "+"]
"""Characters from this list will be removed from a date string and used to build up
a string containing only date components that hopefully match from ``date_component_orders``.
"""

tz_suffix_re = re.compile(r"(Z|[+-]\d\d:\d\d)$")
tz_component_re = re.compile(r"(%!z)|(%[zZ])$")


def _transform_tz_suffix(m):
    suffix = m.group(1)
    if suffix == "Z":
        return "+0000"
    return suffix[:3] + suffix[4:]


@dataclass
class DateTimeColumn:
    name: str
    """
    The column name
    """

    inferred_format: Optional[str]
    """
    The possible strfmt format
    """


@dataclass
class DateTimeColumns:
    columns: Dict[str, DateTimeColumn] = field(default_factory=dict)

    @property
    def column_names(self) -> List[str]:
        return list(self.columns.keys())

    def get_column_info(self, column_name: str) -> Optional[DateTimeColumn]:
        column = self.columns.get(column_name)
        if column is None:
            return None
        return column


@dataclass
class _TokenizedStr:
    """Represents a date string that has been broken up into individual date
    components. This class is useful when trying to rebuild a new string with
    the same format.
    """

    original_str: str
    """The original source string"""

    masked_str: str
    """A masked version of the string. Masked strings only contain the mask characters
    and component seperators.
    """

    components: List[Tuple[str, Tuple[int, int]]]
    """A list of components and their string index mapped from the source string"""

    seperators: List[str]
    """A list of component seperators. Zipping this list with ``components`` yields
    the original string.
    """

    @property
    def component_str(self) -> str:
        """Returns a string containing only the components of the date. This is
        used to matched a date with a date format.
        """
        return " ".join([s for s, _ in self.components])

    def assemble_str_from_components(self, new_components: List[str]) -> str:
        """Given a new set of components, rebuild the string with formatting preserved.

        Args:
            new_components: The new set of component to reassemble the string with.
        """
        components_seperators = [
            token
            for token in itertools.chain(
                *itertools.zip_longest(new_components, self.seperators)
            )
            if token is not None or token
        ]
        return "".join(components_seperators)


@dataclass
class _ParsedDate:
    """Wrapper for a parsed date and associated metadata"""

    component_order: str
    """Matched date string format order form ``date_component_orders``. This can be
    used to reconstruct the original date string format including seperators.
    """

    date: datetime
    """The parsed datetime object"""

    tokenized_date: _TokenizedStr
    """A reference to the tokenized date string"""

    suffix: Optional[str]
    """
    The suffix of the original date string, such as "Z"
    """

    def fmt_str(self, with_suffix: bool = False) -> str:
        """The date format string used to to build the original date. This can be used
        with function like ``strftime`` or ``strptime``.

        Returns:
            Date format string such as "%m/%d/%Y".
        """
        comps = self.component_order.split(" ")
        fmt_str = self.tokenized_date.assemble_str_from_components(comps)
        if with_suffix and self.suffix is not None:
            # If we are requesting a TZ suffix to be added back in and we have a parsed
            # TZ suffix we'll replace "%z" with "Z" or "+00:00", etc.
            fmt_str = tz_component_re.sub(self.suffix, fmt_str)
        return fmt_str

    def date_to_fmt_str(self, date: datetime) -> str:
        """Given a new date object, returns that date in the parsed date format"""
        comps = date.strftime(self.component_order).split(" ")
        return self.tokenized_date.assemble_str_from_components(comps)

    def shift(self, days: int = None, ms: int = None, delta: timedelta = None) -> str:
        """Given a date shift in days or milliseconds or a ``timedelta`` object,
        will return a new date using the same original string format.
        Shifting by milliseconds is useful if the date is a timestamp.
        """
        if not isinstance(delta, timedelta):
            if not days or ms:
                raise ValueError("must specify days or ms")
            delta = timedelta(milliseconds=ms) if ms else timedelta(days=days)
        new_date = self.date + delta
        return self.date_to_fmt_str(new_date)


def _tokenize_date_str(input: str) -> _TokenizedStr:
    """Given a raw input date, will return an instance of ``TokenizedStr``. Any
    business logic, or edge cases for tokenizing a string belong in this method.
    """
    if MASK_CH in input:
        raise ValueError(f"Input date cannot be parsed. Contains mask {MASK_CH}")

    masked = list(input)
    components = []
    contig_sep = []

    cur_start = 0
    last_sep_idx = None
    for idx in range(0, len(input)):
        is_sep = input[idx] in _component_seperators
        if is_sep:
            if idx - 1 == last_sep_idx:
                contig_sep[-1] += input[idx]
            else:
                contig_sep += input[idx]
            last_sep_idx = idx

        if is_sep or idx == len(input) - 1:
            # increment the end character by one if we're at the end the input str
            stop_idx = idx + 1 if idx == len(input) - 1 and not is_sep else idx

            comp = "".join(masked[cur_start:stop_idx])
            components.append((comp, (cur_start, stop_idx)))

            masked[cur_start:stop_idx] = MASK_CH * (stop_idx - cur_start)

            if stop_idx == len(input):
                break

            cur_start = idx + 1

    # this block checks to see if the last component might be a timezone. if it is,
    # we want to merge what we originally thought was a separator, into the
    # timezone component.
    if len(components) == 5 and contig_sep[-1] in {"-", "+"}:
        sep = contig_sep.pop(-1)
        val, span = components[-1]
        components[-1] = (f"{sep}{val}", (span[0] - 1, span[1]))
    # this block checks to see if the last separator was a Z, which, in the absence of
    # a timezone component and at the end of the string, we treat as an ISO8601 abbreviated
    # timezone offset.
    elif len(components) == 4 and contig_sep[-1] == "Z":
        sep = contig_sep.pop(-1)
        components.append((sep, (last_sep_idx, last_sep_idx + 1)))

    return _TokenizedStr(input, "".join(masked), components, contig_sep)


def _strptime_extra(date_string: str, fmt: str) -> Tuple[datetime, Optional[str]]:
    """Parses a string as a datetime object, supporting ISO8601 timezone offsets.

    See the documentation on ``strftime_extra`` regarding the semantics of the new ``%!z`` format
    specifier.

    Args:
        date_string: the datetime in string format.
        fmt: the format string to use for parsing (which may include %!z).

    Returns:
        the parsed datetime.
    """
    tz_extra_pos = fmt.find("%!z")
    suffix = None
    suffix_match = tz_suffix_re.search(date_string)
    if suffix_match:
        suffix = suffix_match.groups()[0]
    if tz_extra_pos == -1:
        # If %!z isn't used, behave like strptime.
        return datetime.strptime(date_string, fmt), suffix

    # Ensure that %!z (if at all used) only occurs at the end of the string.
    if tz_extra_pos != len(fmt) - 3:
        raise ValueError(
            "%!z format modifier may only occur at the end of the format string"
        )

    # Replace a ``Z`` or ``[+-]hh:mm`` suffix with [+-]hhmm. Otherwise, just drop the timezone offset.
    # Without this, a timezone offset without a colon (such as +0000) would be parsed successfully with
    # %!z, which we want to avoid.
    date_string, nsubs = tz_suffix_re.subn(_transform_tz_suffix, date_string)
    dt = datetime.strptime(date_string, fmt[:-3] + ("%z" if nsubs else ""))

    return dt, suffix


def _date_component_permutations() -> List[Tuple[str, str, str, str, str]]:
    """Returns a list of string formats by component type. Each permutation is
    indexed by y, m, d, hms, tz and can be passed into component formatter from
    ``date_component_orders``.
    """
    return list(itertools.product(*_component_formats.values()))  # type:ignore


def _gen_date_str_fmt_permutations() -> Set[str]:
    """Returns a list of unique date string format permutations"""
    return {
        order(*str_fmt)
        for str_fmt in _date_component_permutations()
        for order in _date_component_orders
    }


_date_str_fmt_permutations = _gen_date_str_fmt_permutations()
"""A unique list of date string formats"""


def _maybe_match(date, format) -> Tuple[Optional[datetime], Optional[str]]:
    try:
        return _strptime_extra(date, format)
    except ValueError:
        return None, None


def _check_series(series: pd.Series, format: str) -> bool:
    # Remove non-standard formatting directives which are relevant for formatting
    # only, not for parsing. The first one, `!`, is introduced by us (see
    # ``_strptime_extra``), the second one, `%-`, is a directive not recognized
    # by pandas and stripped by RDT as well (see
    # https://github.com/sdv-dev/RDT/pull/458/files#r835690711 ).
    pd_format = format.replace("!", "").replace("%-", "%")
    try:
        pd.to_datetime(series, format=pd_format)
        return True
    except:
        # Conservatively ignore any error, and assume that the format
        # didn't work.
        # This is to prevent errors in the SDV code downstream.
        return False


def _parse_date_multiple(
    input_date: str,
    date_str_fmts: Union[List[str], Set[str]] = _date_str_fmt_permutations,
) -> Iterator[_ParsedDate]:
    tokenized_date = _tokenize_date_str(input_date)
    for str_fmt in date_str_fmts:
        date, suffix = _maybe_match(tokenized_date.component_str, str_fmt)
        if date:
            yield _ParsedDate(str_fmt, date, tokenized_date, suffix)


def _d_str_to_fmt_multiple(input_date: str, with_suffix: bool) -> Iterator[str]:
    """Infers all likely date format from a date string."""
    for parsed_date in _parse_date_multiple(input_date):
        yield parsed_date.fmt_str(with_suffix=with_suffix)


def _maybe_d_str_to_fmt_multiple(input_date: str, with_suffix: bool) -> Iterator[str]:
    """Infers all likely date format from a date string or nothing."""
    try:
        yield from _d_str_to_fmt_multiple(input_date, with_suffix)
    except ValueError:
        pass


def _infer_from_series_match_all(series: pd.Series, with_suffix: bool) -> Optional[str]:
    if series.empty:
        return None

    # We store the candidate formats as a list instead of a set to ensure a deterministic
    # result (the order of ``_maybe_d_str_to_fmt_multiple`` is deterministic as well).
    # This matches the behavior of ``_infer_from_series``, which - due to the above
    # property as well as ``Counter``s stable iteration based on insertion order -
    # is deterministic as well.
    candidate_fmts = list(_maybe_d_str_to_fmt_multiple(series[0], with_suffix))
    i = 1
    # Empirically, ``pd.to_datetime`` is about 8x faster than checking individual values.
    # Conservatively, we fall back to calling ``pd.to_datetime`` on the entire remaining
    # series when we have 4 or less candidate formats less.
    # In most cases, the number of candidate formats will be lower than both 4 and 8
    # after the first invocation anyway.
    while len(candidate_fmts) > 4 and i < len(series):
        value = series[i]
        candidate_fmts = [
            fmt for fmt in candidate_fmts if _maybe_match(value, fmt) != (None, None)
        ]
        i += 1

    if i < len(series):
        # If we haven't exhausted the whole series yet, do a ``pd.to_datetime``
        # call for the remaining values to weed out incorrect formats.
        remaining_series = series[i:]
        candidate_fmts = [
            fmt for fmt in candidate_fmts if _check_series(remaining_series, fmt)
        ]

    return candidate_fmts[0] if candidate_fmts else None


def _infer_from_series(
    series: pd.Series, with_suffix: bool, must_match_all: bool = False
) -> Optional[str]:
    if must_match_all:
        return _infer_from_series_match_all(series, with_suffix)

    counter = Counter()
    for value in series:
        for fmt in _maybe_d_str_to_fmt_multiple(value, with_suffix):
            counter[fmt] += 1
    highest_occurence = counter.most_common(1)
    if highest_occurence:
        return highest_occurence[0][0]

    return None


def detect_datetimes(
    df: pd.DataFrame,
    sample_size: Optional[int] = None,
    with_suffix: bool = False,
    must_match_all: bool = False,
) -> DateTimeColumns:
    if sample_size is None:
        sample_size = SAMPLE_SIZE
    column_data = DateTimeColumns()
    object_cols = [
        col for col, col_type in df.dtypes.iteritems() if col_type == "object"
    ]
    for object_col in object_cols:
        test_series: pd.Series = df[object_col].dropna(axis=0).reset_index(drop=True)
        # Only sample when we don't require the format to match all entries
        if not must_match_all and len(test_series) > sample_size:
            test_series = test_series.sample(sample_size)
        test_series_str = test_series.astype(str)
        inferred_format = _infer_from_series(
            test_series_str, with_suffix, must_match_all
        )
        if inferred_format is not None:
            inferred_format = inferred_format.replace("!", "")
            column_data.columns[object_col] = DateTimeColumn(
                object_col, inferred_format
            )
    return column_data
