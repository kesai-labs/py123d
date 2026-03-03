"""
Process pool executor backend.
Code is adapted from the nuplan-devkit: https://github.com/motional/nuplan-devkit
"""

import logging
import multiprocessing
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor as _ProcessPoolExecutor
from typing import Any, Iterable, List, Optional

from tqdm import tqdm

from py123d.common.execution.executor import (
    Executor,
    ExecutorResources,
    Task,
    get_max_size_of_arguments,
)

logger = logging.getLogger(__name__)


class ProcessPoolExecutor(Executor):
    """
    Distributes tasks across multiple processes on a single machine.
    """

    def __init__(self, max_workers: Optional[int] = None):
        """
        Create executor with a process pool.
        :param max_workers: number of processes to use. Defaults to all available CPUs.
        """
        number_of_cpus_per_node = max_workers if max_workers else ExecutorResources.current_node_cpu_count()

        super().__init__(
            ExecutorResources(
                number_of_nodes=1,
                number_of_cpus_per_node=number_of_cpus_per_node,
                number_of_gpus_per_node=0,
            )
        )

        self._executor = _ProcessPoolExecutor(
            max_workers=max_workers, mp_context=multiprocessing.get_context("forkserver")
        )

    def _map(
        self, task: Task, *item_lists: Iterable[List[Any]], verbose: bool = False, desc: Optional[str] = None
    ) -> List[Any]:
        """Inherited, see superclass."""
        return list(
            tqdm(
                self._executor.map(task.fn, *item_lists),
                leave=False,
                total=get_max_size_of_arguments(*item_lists),
                desc=desc or "ProcessPoolExecutor",
                disable=not verbose,
            )
        )

    def submit(self, task: Task, *args: Any, **kwargs: Any) -> Future[Any]:
        """Inherited, see superclass."""
        return self._executor.submit(task.fn, *args, **kwargs)
