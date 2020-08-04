from unittest.mock import patch
import pytest

from gretel_synthetics.generate_parallel import split_work


@pytest.mark.parametrize('num_cpus,total_lines,chunk_size,parallelism,expected_workers,expected_chunks',
                         [(1, 100, 5, 0, 1, 1),
                          (1, 100, 5, 1, 1, 1),
                          (8, 100, 5, 0, 8, 20),
                          (8, 100, 9, 0, 8, 12),
                          (8, 100, 5, 1, 1, 1),
                          (8, 100, 5, -1, 7, 20),
                          (8, 100, 5, .5, 4, 20),
                          (8, 100, 50, .5, 2, 2),
                          (8, 100, 5, -.5, 4, 20),
                          ])
@patch("gretel_synthetics.generate_parallel.mp.cpu_count")
def test_split_work(cpu_count, num_cpus, total_lines, chunk_size, parallelism, expected_workers, expected_chunks):
    cpu_count.return_value = num_cpus

    num_workers, chunks = split_work(parallelism, total_lines, chunk_size)
    assert num_workers == expected_workers
    assert len(chunks) == expected_chunks
    assert all(x == chunk_size for x in chunks[:-1])
    assert sum(chunks) == total_lines
