from py123d.common.execution.executor import Executor, ExecutorResources, Task
from py123d.common.execution.process_pool_executor import ProcessPoolExecutor
from py123d.common.execution.sequential_executor import SequentialExecutor
from py123d.common.execution.thread_pool_executor import ThreadPoolExecutor
from py123d.common.execution.utils import (
    chunk_list,
    executor_map_chunked_list,
    executor_map_chunked_single,
    executor_map_queued,
)

__all__ = [
    "Executor",
    "ExecutorResources",
    "Task",
    "ProcessPoolExecutor",
    "SequentialExecutor",
    "ThreadPoolExecutor",
    "RayExecutor",
    "chunk_list",
    "executor_map_chunked_list",
    "executor_map_chunked_single",
    "executor_map_queued",
]


def __getattr__(name: str) -> object:
    """Import RayExecutor on first use: importing ray costs about a second, paid by every process that touches py123d.api."""
    if name == "RayExecutor":
        from py123d.common.execution.ray_executor import RayExecutor

        return RayExecutor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
