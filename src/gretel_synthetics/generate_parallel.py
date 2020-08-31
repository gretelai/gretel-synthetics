from dataclasses import dataclass, field
from typing import List, Optional, Union, Set, Tuple
import os
import sys
import loky
from concurrent import futures

from gretel_synthetics.generator import Generator, Settings, gen_text


@dataclass
class _WorkerStatus:
    """
    Status of a parallel worker to be communicated back to the main thread after each chunk.
    """

    lines: List[gen_text] = field(default_factory=list)
    """Generated lines in this chunk, if any. This contains both valid and invalid lines."""

    exception: Optional[BaseException] = None
    """Exception that was encountered during processing. Implies done=True when not None."""

    done: bool = False
    """Flag indicating whether the worker is complete."""


def get_num_workers(parallelism: Union[int, float], total_lines: int, chunk_size: int = 5) -> int:
    """
    Given a parallelism setting and a number of total lines, compute the number of parallel workers to be used.

    Args:
        parallelism: The number of concurrent workers to use. ``1`` (the default) disables parallelization,
            while a non-positive value means "number of CPUs + x" (i.e., use ``0`` for using as many workers
            as there are CPUs). A floating-point value is interpreted as a fraction of the available CPUs,
            rounded down.
        total_lines: The total number of lines to generate.
        chunk_size: the size of an individual unit of work to be distributed to workers.

    Returns:
        The number of required workers.
    """
    num_chunks = (total_lines - 1) // chunk_size + 1
    non_positive = False
    if parallelism <= 0:
        parallelism = -parallelism
        non_positive = True

    if isinstance(parallelism, float):
        num_workers = int(loky.cpu_count() * parallelism)
    else:
        num_workers = parallelism

    if non_positive:
        num_workers = loky.cpu_count() - num_workers

    num_workers = min(max(num_workers, 1), num_chunks)

    return num_workers


def generate_parallel(settings: Settings, num_lines: int, num_workers: int, chunk_size: int = 5):
    """
    Runs text generation in parallel mode.

    Text generation is performed with the given settings, using a given number of parallel workers
    and a total body of work that is split into the given list of chunks.

    Args:
        settings: the settings for text generation.
        num_lines: the number of valid lines to be generated.
        num_workers: the number of parallel workers.
        chunk_size: the maximum number of lines to be assigned to a worker at once.

    Yields:
        ``gen_text`` objects.
    """

    # Create a pool of workers that will instantiate a generator upon initialization.
    worker_pool = loky.ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_loky_init_worker,
        initargs=(settings,)
    )

    # How many valid lines we still need to generate
    remaining_lines = num_lines

    # This set tracks the currently outstanding invocations to _loky_worker_process_chunk.
    pending_tasks: Set[futures.Future[Tuple[int, List[gen_text], int]]] = set()

    # How many tasks can be pending at once. While a lower factor saves memory, it increases the
    # risk that workers sit idle because the main process is blocked on processing data and
    # therefore cannot hand out new tasks.
    max_pending_tasks = 10 * num_workers

    # How many lines to be generated have been assigned to currently active workers. This tracks
    # the nominal/target lines, and the returned number of lines may be different if workers generate
    # a lot of invalid lines.
    assigned_lines = 0

    # The _total_ number of invalid lines we have seen so far. This is used to implement a global
    # limit on the number of invalid lines, since each worker only knows the number of invalid lines
    # it has generated itself.
    total_invalid = 0

    try:
        while remaining_lines > 0:
            # If we have capacity to add new pending tasks, do so until we are at capacity or there are
            # no more lines that can be assigned to workers.
            while len(pending_tasks) < max_pending_tasks and assigned_lines < remaining_lines:
                next_chunk = min(chunk_size, remaining_lines)
                pending_tasks.add(worker_pool.submit(_loky_worker_process_chunk, next_chunk))
                assigned_lines += next_chunk

            # Wait for at least one worker to complete its current task (or fail with an exception).
            completed_tasks, pending_tasks = futures.wait(
                pending_tasks, return_when=futures.FIRST_COMPLETED)

            for task in completed_tasks:
                requested_chunk_size, lines, num_invalid = task.result(timeout=0)

                assigned_lines -= requested_chunk_size
                remaining_lines -= len(lines) - num_invalid  # Calculate number of _valid_ lines

                # Emit lines in the output
                for line in lines:
                    if line.valid is not None and not line.valid:
                        total_invalid += 1
                    if total_invalid > settings.max_invalid:
                        raise RuntimeError("Maximum number of invalid lines reached!")
                    yield line

    finally:
        # Always make sure to shut down the worker pool (no need to wait for workers to terminate).
        worker_pool.shutdown(wait=False, kill_workers=True)


###################################################################
# All code below this line is ONLY run in workers spawned by loky #
###################################################################


# Global variable for the generator used in this process. Since we are not reusing processes across
# generation tasks, we do not lose any ability of running multiple top-level generation tasks in parallel.
# Also note that each worker picks up tasks in a strictly sequential fashion and is not multi-threaded.
_loky_worker_generator : Optional[Generator] = None


def _loky_init_worker(settings: Settings):
    """
    Initializes the global state for a loky worker process.

    Args:
        settings: the settings for the generator.
    """
    # Workers should be using CPU only (note, this has no effect on the parent process)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Suppress stdout and stderr in worker threads. Do so on a best-effort basis only.
    try:
        devnull = open(os.devnull, 'w')
        os.dup2(devnull.fileno(), sys.stdout.fileno())
        os.dup2(devnull.fileno(), sys.stderr.fileno())
    except BaseException:  # pylint: disable=broad-except
        pass

    global _loky_worker_generator
    _loky_worker_generator = Generator(settings)


def _loky_worker_process_chunk(chunk_size: int) -> Tuple[int, List[gen_text], int]:
    """
    Processes a single chunk by attempting to generate the given number of lines.

    Args:
        chunk_size: the desired (target) number of valid lines to generate.

    Returns:
        3-element tuple containing:
        - the original input value of ``chunk_size``,
        - the list of all generated lines (valid and invalid), and
        - the number of invalid lines among the generated ones.

    Raises:
        RuntimeError: if _loky_init_worker has not been called yet in this process.
    """

    global _loky_worker_generator
    if not _loky_worker_generator:
        raise RuntimeError("generator has not been initialized in loky worker process")

    old_num_invalid = _loky_worker_generator.total_invalid
    lines = list(_loky_worker_generator.generate_next(chunk_size))
    num_invalid = _loky_worker_generator.total_invalid - old_num_invalid

    return chunk_size, lines, num_invalid
