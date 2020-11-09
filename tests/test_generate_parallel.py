from unittest.mock import patch, Mock
from typing import Optional
from concurrent import futures
from math import ceil
import pytest

from gretel_synthetics.generate import GenText
from gretel_synthetics.errors import TooManyInvalidError
from gretel_synthetics.generate_parallel import generate_parallel, get_num_workers


@pytest.mark.parametrize('num_cpus,total_lines,chunk_size,parallelism,expected_workers',
                         [(1, 100, 5, 0, 1),
                          (1, 100, 5, 1, 1),
                          (8, 100, 5, 0, 8),
                          (8, 100, 9, 0, 8),
                          (8, 100, 5, 1, 1),
                          (8, 100, 5, -1, 7),
                          (8, 100, 5, .5, 4),
                          (8, 100, 50, .5, 2),
                          (8, 100, 5, -.5, 4),
                          ])
@patch("loky.cpu_count")
def test_split_work(cpu_count, num_cpus, total_lines, chunk_size, parallelism, expected_workers):
    cpu_count.return_value = num_cpus

    num_workers = get_num_workers(parallelism, total_lines, chunk_size)
    assert num_workers == expected_workers


def _mock_submitter(failure_rate):
    def mock_submit(unused_fn, chunk_size: int, hard_limit: Optional[int] = None):
        target_valid = chunk_size
        target_total = int((1 + failure_rate) * chunk_size)
        if hard_limit is not None and target_total > hard_limit:
            target_valid = ceil(chunk_size * (hard_limit / target_total))
            target_total = hard_limit

        target_invalid = target_total - target_valid

        lines = [GenText(text=f'line-{i}', valid=(i >= target_invalid)) for i in range(target_total)]
        mock_future = futures.Future()
        mock_future.set_result((chunk_size, lines, target_invalid))

        return mock_future

    return mock_submit


@pytest.mark.parametrize('num_lines,num_workers,chunk_size,failure_rate',
                         [(num_lines, num_workers, chunk_size, failure_rate)
                          for num_lines in (10, 100, 1000)
                          for num_workers in (1, 4, 8, 12, 13, 16)
                          for chunk_size in (1, 5, 10, 50)
                          for failure_rate in (0.0, 0.25, 0.5)])
@patch("loky.ProcessPoolExecutor")
def test_generate_parallel(pool_mock, num_lines, num_workers, chunk_size, failure_rate):
    pool_instance = pool_mock.return_value
    pool_instance.submit.side_effect = _mock_submitter(failure_rate)

    settings_mock = Mock()
    settings_mock.max_invalid = ceil(failure_rate * num_lines)

    lines = list(generate_parallel(settings_mock, num_lines, num_workers, chunk_size))
    valid_lines = sum(1 for line in lines if line.valid)
    invalid_lines = sum(1 for line in lines if not line.valid)

    assert valid_lines == num_lines
    assert invalid_lines <= ceil(failure_rate * num_lines)


@patch("loky.ProcessPoolExecutor")
def test_generate_parallel_too_many_invalid(pool_mock):
    pool_instance = pool_mock.return_value
    pool_instance.submit.side_effect = _mock_submitter(0.5)  # 0.5 failure rate

    settings_mock = Mock()
    settings_mock.max_invalid = 250 # 0.25 failure rate budget

    with pytest.raises(TooManyInvalidError):
        for _ in generate_parallel(settings_mock, 1000, 2, 10):
            pass
