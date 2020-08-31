from unittest.mock import patch
import pytest

from gretel_synthetics.generate_parallel import get_num_workers


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
