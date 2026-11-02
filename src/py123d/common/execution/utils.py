"""
Execution utilities.
Code is adapted from the nuplan-devkit: https://github.com/motional/nuplan-devkit
"""

from typing import Any, Callable, List, Optional

import numpy as np
from psutil import cpu_count

from py123d.common.execution.executor import Executor, Task


def chunk_list(input_list: List[Any], num_chunks: Optional[int] = None) -> List[List[Any]]:
    """Chunks a list to equal sized lists. The size of the last list might be truncated.

    :param input_list: List to be chunked.
    :param num_chunks: Number of chunks, equals to the number of cores if set to None.
    :return: List of equal sized lists.
    """
    num_chunks = num_chunks if num_chunks else cpu_count(logical=True)
    chunks = np.array_split(input_list, num_chunks)  # type: ignore
    return [chunk.tolist() for chunk in chunks if len(chunk) != 0]


def executor_map_chunked_single(
    executor: Executor, fn: Callable[..., Any], input_objects: List[Any], name: Optional[str] = None
) -> List[Any]:
    """Map a function over individual objects through an executor. The input list is pre-chunked into
    equal-sized chunks (one per worker), and fn is called once per object within each chunk.

    :param executor: Executor to use for parallelization.
    :param fn: Function that takes a single object and returns a single result.
    :param input_objects: List of objects to map.
    :param name: Optional name for the progress bar description.
    :return: List of mapped objects.
    """

    def _chunked_fn(chunk: List[Any]) -> List[Any]:
        return [fn(obj) for obj in chunk]

    return executor_map_chunked_list(executor, _chunked_fn, input_objects, name=name)


def executor_map_chunked_list(
    executor: Executor,
    fn: Callable[..., List[Any]],
    input_objects: List[Any],
    name: Optional[str] = None,
    num_gpus: Optional[float] = None,
) -> List[Any]:
    """Map a list of objects through an executor. The input list is pre-chunked into
    equal-sized chunks (one per worker), and fn receives an entire chunk at once.

    :param executor: Executor to use for parallelization.
    :param fn: Function that takes a list of objects and returns a list of results.
    :param input_objects: List of objects to map.
    :param name: Optional name for the progress bar description.
    :param num_gpus: GPUs (or a fraction of a GPU) that each task requests from Ray; ignored by other executors.
    :return: List of mapped objects.
    """
    if executor.number_of_threads == 0:
        return fn(input_objects)

    object_chunks = chunk_list(input_objects, executor.number_of_threads)
    scattered_objects = executor.map(Task(fn=fn, num_gpus=num_gpus), object_chunks, desc=name)
    output_objects = [result for results in scattered_objects for result in results]

    return output_objects


def executor_map_queued(
    executor: Executor,
    fn: Callable[..., Any],
    input_objects: List[Any],
    name: Optional[str] = None,
    num_gpus: Optional[float] = None,
) -> List[Any]:
    """Map a function over individual objects through an executor without pre-chunking.
    Each item is submitted as an individual task, so workers dynamically pick up the next
    item when they become free. This provides better load balancing when tasks have
    varying execution times, at the cost of higher scheduling overhead.

    :param executor: Executor to use for parallelization.
    :param fn: Function that takes a single object and returns a single result.
    :param input_objects: List of objects to map.
    :param name: Optional name for the progress bar description.
    :param num_gpus: GPUs (or a fraction of a GPU) that each task requests from Ray; ignored by other executors.
    :return: List of mapped objects.
    """
    if executor.number_of_threads == 0:
        return [fn(obj) for obj in input_objects]

    return executor.map(Task(fn=fn, num_gpus=num_gpus), input_objects, desc=name)


def gpu_share_per_task(executor: Executor, requires_gpu: bool) -> Optional[float]:
    """The GPU share of each task when tasks need a GPU (e.g. for NVENC video encoding) and the executor has GPUs:
    the GPUs of a node divided by its parallel tasks, so that Ray spreads the tasks evenly over the GPUs (Ray then
    sets ``CUDA_VISIBLE_DEVICES`` per task). None otherwise.

    :param executor: The executor (only a Ray executor schedules GPUs).
    :param requires_gpu: Whether the tasks need a GPU.
    """
    from py123d.common.execution.ray_executor import RayExecutor

    if not requires_gpu or not isinstance(executor, RayExecutor):
        return None
    gpus, threads = executor.config.number_of_gpus_per_node, executor.config.number_of_cpus_per_node
    if gpus <= 0 or threads <= 0:
        return None
    return float(np.floor(gpus / threads * 1e4) / 1e4) if gpus < threads else 1.0
