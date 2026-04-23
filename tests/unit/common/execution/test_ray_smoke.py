"""Minimal Ray smoke test kept in the default test run.

The rest of the Ray tests live in `test_ray_executor.py` and `test_ray_utils.py`
and are marked `slow` (deselected by default). Run them explicitly with
`pytest -m slow` or `pytest -m ""`.
"""

import pytest

ray = pytest.importorskip("ray")

from py123d.common.execution.executor import Task
from py123d.common.execution.ray_utils import ray_map


def _double(x):
    """Top-level function required for pickling in Ray."""
    return x * 2


def test_ray_map_smoke():
    """Confirm Ray can initialize and run a trivial parallel map."""
    try:
        task = Task(fn=_double)
        result = ray_map(task, [1, 2, 3])
        assert result == [2, 4, 6]
    finally:
        if ray.is_initialized():
            ray.shutdown()
