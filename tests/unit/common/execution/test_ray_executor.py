"""Tests for Ray distributed executor backend."""

from concurrent.futures import Future

import pytest

ray = pytest.importorskip("ray")

from py123d.common.execution.executor import ExecutorResources, Task
from py123d.common.execution.ray_executor import RayExecutor, initialize_ray

# ---------------------------------------------------------------------------
# Top-level pickleable functions (required for Ray serialization)
# ---------------------------------------------------------------------------


def _double(x):
    """Top-level function required for pickling in Ray."""
    return x * 2


def _add(x, y):
    """Top-level function required for pickling in Ray."""
    return x + y


def _identity(x):
    """Top-level function required for pickling in Ray."""
    return x


def _fail(x):
    """Top-level function that raises, required for pickling."""
    raise ValueError("ray executor boom")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _ensure_ray_shutdown():
    """Ensure Ray is shut down after each test to avoid interference."""
    yield
    if ray.is_initialized():
        ray.shutdown()


# ---------------------------------------------------------------------------
# Tests: initialize_ray
# ---------------------------------------------------------------------------


class TestInitializeRay:
    def test_local_init(self):
        """Local initialization returns valid ExecutorResources."""
        resources = initialize_ray()
        assert isinstance(resources, ExecutorResources)
        assert resources.number_of_nodes == 1
        assert resources.number_of_cpus_per_node >= 1

    def test_custom_threads_per_node(self):
        """threads_per_node overrides the detected CPU count."""
        resources = initialize_ray(threads_per_node=2)
        assert resources.number_of_cpus_per_node == 2

    def test_gpu_count_zero_without_cuda(self):
        """Without CUDA, GPU count is 0."""
        resources = initialize_ray(threads_per_node=2)
        assert resources.number_of_gpus_per_node == 0

    def test_no_distributed_ignores_master_ip(self):
        """With use_distributed=False, master_node_ip is ignored and Ray starts locally."""
        resources = initialize_ray(master_node_ip="127.0.0.1", use_distributed=False, threads_per_node=2)
        assert resources.number_of_nodes == 1


# ---------------------------------------------------------------------------
# Tests: RayExecutor
# ---------------------------------------------------------------------------


class TestRayExecutor:
    def test_init_default(self):
        """Default construction produces valid config."""
        executor = RayExecutor()
        assert executor.config.number_of_nodes == 1
        assert executor.config.number_of_cpus_per_node >= 1
        assert executor.config.number_of_gpus_per_node == 0

    def test_init_with_threads(self):
        """threads_per_node sets the CPU count."""
        executor = RayExecutor(threads_per_node=2)
        assert executor.config.number_of_cpus_per_node == 2
        assert executor.number_of_threads == 2

    def test_init_shuts_down_existing_ray(self):
        """Constructing RayExecutor when Ray is already running does not error."""
        ray.init(num_cpus=1, log_to_driver=False)
        assert ray.is_initialized()
        executor = RayExecutor(threads_per_node=2)
        assert ray.is_initialized()
        assert executor.config.number_of_cpus_per_node == 2

    def test_map_single_arg(self):
        """Map a function over a single argument list."""
        executor = RayExecutor(threads_per_node=2)
        task = Task(fn=_double)
        result = executor.map(task, [1, 2, 3, 4])
        assert sorted(result) == [2, 4, 6, 8]

    def test_map_multiple_args(self):
        """Map a function over multiple argument lists."""
        executor = RayExecutor(threads_per_node=2)
        task = Task(fn=_add)
        result = executor.map(task, [1, 2, 3], [10, 20, 30])
        assert sorted(result) == [11, 22, 33]

    def test_map_preserves_order(self):
        """RayExecutor.map preserves input order."""
        executor = RayExecutor(threads_per_node=2)
        task = Task(fn=_identity)
        items = list(range(20))
        result = executor.map(task, items)
        assert result == items

    def test_map_empty_list(self):
        """Mapping over an empty list returns an empty result."""
        executor = RayExecutor(threads_per_node=2)
        task = Task(fn=_identity)
        result = executor.map(task, [])
        assert result == []

    def test_submit_returns_future(self):
        """submit returns a future that resolves to the correct value."""
        executor = RayExecutor(threads_per_node=2)
        task = Task(fn=_add)
        future = executor.submit(task, 3, 4)
        assert isinstance(future, Future)
        assert future.result(timeout=10) == 7

    def test_submit_exception_propagates(self):
        """Exceptions from submitted tasks propagate via the future."""
        executor = RayExecutor(threads_per_node=2)
        task = Task(fn=_fail)
        future = executor.submit(task, 1)
        with pytest.raises(Exception):
            future.result(timeout=10)

    def test_shutdown(self):
        """shutdown() stops the Ray runtime."""
        executor = RayExecutor(threads_per_node=2)
        assert ray.is_initialized()
        executor.shutdown()
        assert not ray.is_initialized()

    def test_output_dir_sets_log_dir(self, tmp_path):
        """output_dir + logs_subdir sets _log_dir correctly."""
        executor = RayExecutor(threads_per_node=2, output_dir=tmp_path, logs_subdir="worker_logs")
        assert executor._log_dir == tmp_path / "worker_logs"

    def test_output_dir_none_sets_log_dir_none(self):
        """output_dir=None results in _log_dir=None."""
        executor = RayExecutor(threads_per_node=2, output_dir=None)
        assert executor._log_dir is None

    def test_debug_mode(self):
        """debug_mode=True sets _local_mode to True."""
        executor = RayExecutor(threads_per_node=2, debug_mode=True)
        assert executor._local_mode is True
