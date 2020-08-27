import multiprocessing
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Union, Tuple
import queue
import os

import cloudpickle

from gretel_synthetics.generator import Generator, Settings, deserialize_settings, gen_text


mp = multiprocessing.get_context('spawn')  # fork does not work with tensorflow


MAX_QUEUE_SIZE = 30000


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

    def serialize(self) -> bytes:
        return cloudpickle.dumps(self)


def _deserialize_status(serialized: bytes) -> _WorkerStatus:
    obj = cloudpickle.loads(serialized)
    if not isinstance(obj, _WorkerStatus):
        raise TypeError("serialized object is of type {}, not _WorkerStatus".format(type(obj).__name__))
    return obj


def split_work(parallelism: Union[int, float], total_lines: int, chunk_size: int = 5) -> Tuple[int, List[int]]:
    """
    Given a parallelism setting and a number of total lines, split the work across workers.

    Args:
        parallelism: The number of concurrent workers to use. ``1`` (the default) disables parallelization,
            while a non-positive value means "number of CPUs + x" (i.e., use ``0`` for using as many workers
            as there are CPUs). A floating-point value is interpreted as a fraction of the available CPUs,
            rounded down.
        total_lines: The total number of lines to generate.
        chunk_size: the size of an individual unit of work to be distributed to workers.

    Returns:
        A pair consisting of the number of required workers, and the list of chunks of work.
    """
    num_chunks = (total_lines - 1) // chunk_size + 1
    non_positive = False
    if parallelism <= 0:
        parallelism = -parallelism
        non_positive = True

    if isinstance(parallelism, float):
        num_workers = int(mp.cpu_count() * parallelism)
    else:
        num_workers = parallelism

    if non_positive:
        num_workers = mp.cpu_count() - num_workers

    num_workers = min(max(num_workers, 1), num_chunks)

    if num_workers == 1:
        return 1, [total_lines]

    chunks = [chunk_size] * (num_chunks - 1)
    chunks.append(total_lines - sum(chunks))

    return num_workers, chunks


def generate_parallel(settings: Settings, num_workers: int, chunks: List[int], num_lines: int):
    """
    Runs text generation in parallel mode.

    Text generation is performed with the given settings, using a given number of parallel workers
    and a total body of work that is split into the given list of chunks.

    Args:
        settings: the settings for text generation.
        num_workers: the number of parallel workers.
        chunks: the list of chunks of work.

    Yields:
        ``gen_text`` objects.
    """

    # Instruct each worker to return an intermediate result (at the cost of putting a partial chunk
    # back into the queue) if it has generated 105% of the requested number of valid lines per chunk
    # (regardless of how many lines are valid in the intermediate result). This ensures that workers
    # don't get stuck for a long time generating data from bad models, where the ratio of invalid:valid
    # lines is very high.
    max_response_size = int(chunks[0] * 1.05)

    worker_input_queue = mp.Queue(MAX_QUEUE_SIZE)

    # Because of the upper limit on the max queue size, we pre-load
    # either all chunks or the max queue size amount of chunks
    # If there are still more chunks, we'll load those into the queue
    # as space becomes more available
    for _ in range(min(MAX_QUEUE_SIZE, len(chunks))):
        chunk = chunks.pop()
        worker_input_queue.put_nowait(chunk)

    # Create a queue for output produced by the worker. This queue should be large enough to buffer all
    # intermediate statuses to ensure that upstream processing doesn't block downstream workers.
    worker_output_queue = mp.Queue(MAX_QUEUE_SIZE)

    pickled_settings = cloudpickle.dumps(settings)

    workers = [
        mp.Process(
            target=_run_parallel_worker,
            args=(pickled_settings, worker_input_queue, worker_output_queue, max_response_size),
        )
        for _ in range(num_workers)
    ]

    # Start all the worker processes.
    for worker in workers:
        worker.start()

    live_workers = len(workers)
    total_invalid = 0
    total_valid = 0
    while live_workers > 0:
        output = worker_output_queue.get()

        # A worker should always send a pickled WorkerStatus after completing each chunk or the entire job.
        # However, we also allow sending a raw string as an escape hatch to communicate, e.g., exceptions while
        # serializing.
        if isinstance(output, str):
            raise RuntimeError('Fatal top-level exception from worker: {}'.format(output))

        parsed_output = _deserialize_status(output)

        if parsed_output.exception is not None:
            # First order of business: check for any exception and raise in the main program.
            raise parsed_output.exception

        # Emit lines in the output
        for line in parsed_output.lines:
            if line.valid is not None and not line.valid:
                total_invalid += 1
            else:
                total_valid += 1
            if total_invalid > settings.max_invalid:
                raise RuntimeError("Maximum number of invalid lines reached!")
            yield line

        if parsed_output.done:
            live_workers -= 1  # We aren't expecting anything more from this worker

        # if there are still chunks left, try and add them to the queue
        # if there are no more chunks, signal the workers to shutdown
        while True:
            if not chunks:
                break
            chunk = chunks.pop()
            try:
                worker_input_queue.put_nowait(chunk)
            except queue.Full:
                chunks.append(chunk)
                break

        if total_valid == num_lines:
            for _ in range(len(workers)):
                try:
                    worker_input_queue.put_nowait(None)
                except queue.Full:
                    pass

    # Join all worker processes (not strictly necessary, but cleaner).
    for worker in workers:
        worker.join()


def _run_parallel_worker(
        pickled_settings: bytes,
        input_queue: mp.Queue,
        output_queue: mp.Queue,
        max_response_size: Optional[int] = None):
    # Workers should be using CPU only (note, this has no effect on the parent process)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Suppress stdout and stderr in worker threads. Do so on a best-effort basis only.
    try:
        devnull = open(os.devnull, 'w')
        os.dup2(devnull.fileno(), sys.stdout.fileno())
        os.dup2(devnull.fileno(), sys.stderr.fileno())
    except BaseException:  # pylint: disable=broad-except
        pass

    try:
        settings = deserialize_settings(pickled_settings)

        for status in _process_all_chunks(settings, input_queue, max_response_size):
            output_queue.put(cloudpickle.dumps(status))
    except BaseException as e:
        # Catch serialization errors etc., and put into queue as a raw str to avoid triggering an exception a
        # second time that was caused by lack of serializability.
        output_queue.put(str(e))


def _process_all_chunks(settings: Settings,
                        input_queue: mp.Queue,
                        max_response_size: Optional[int] = None) -> Iterable[_WorkerStatus]:
    try:
        gen = Generator(settings)

        while True:
            chunk_size = input_queue.get()
            if chunk_size is None:
                yield _WorkerStatus(done=True)
                break
            prev_invalid = gen.total_invalid
            all_lines = list(gen.generate_next(chunk_size, hard_limit=max_response_size))
            num_valid_lines = len(all_lines) - (gen.total_invalid - prev_invalid)
            if num_valid_lines < chunk_size:
                # Return the number of lines by which we fell short to the queue.
                # This is guaranteed to succeed because every worker will only do this after
                # at least removing an element from the queue, thus ensuring that capacity
                # constraints are never violated.
                input_queue.put(chunk_size - num_valid_lines)
            yield _WorkerStatus(lines=all_lines)
    except queue.Empty:
        # NOTE: We shouldn't really get here, but leaving it for now, since we are
        # relying on signaling shutdown via a sentinel
        #
        # It is possible that further elements will be added to the queue
        # because a worker generated too many invalid lines, but in this case we can be certain
        # that the number of running workers will never fall below the number of concurrently
        # pending chunks.
        yield _WorkerStatus(done=True)
    except BaseException as e:
        # Send any exception in its own status object.
        yield _WorkerStatus(exception=e, done=True)
