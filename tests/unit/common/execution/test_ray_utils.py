"""Tests for Ray execution utilities."""

import logging
from functools import partial

import pytest

ray = pytest.importorskip("ray")

from py123d.common.execution.executor import Task
from py123d.common.execution.ray_utils import _ray_map_items, _ray_object_iterator, ray_map, wrap_function

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
    raise ValueError("ray boom")


def _add_keyword(x, y=0):
    """Top-level function with keyword argument for partial tests."""
    return x + y


def _logging_fn(x):
    """Top-level function that emits a log message."""
    logging.getLogger().info(f"Processing {x}")
    return x


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ray_session():
    """Initialize Ray once for the module, shut down after all tests."""
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=2, log_to_driver=False)
    yield
    ray.shutdown()


# ---------------------------------------------------------------------------
# Tests: wrap_function (no Ray needed)
# ---------------------------------------------------------------------------


class TestWrapFunction:
    def test_no_wrapping_when_log_dir_is_none(self):
        """When log_dir is None the wrapped function behaves identically."""
        wrapped = wrap_function(_double, log_dir=None)
        result = wrapped(5)
        assert result == 10

    def test_creates_log_file_in_log_dir(self, tmp_path):
        """A .log file is created inside log_dir."""
        wrapped = wrap_function(_logging_fn, log_dir=tmp_path)
        wrapped(42)
        log_files = list(tmp_path.glob("*.log"))
        assert len(log_files) == 1

    def test_log_file_name_contains_function_name(self, tmp_path):
        """The log file name ends with __<fn_name>.log."""
        wrapped = wrap_function(_logging_fn, log_dir=tmp_path)
        wrapped(1)
        log_files = list(tmp_path.glob("*.log"))
        assert log_files[0].name.endswith("___logging_fn.log")

    def test_wrapped_function_returns_correct_result(self, tmp_path):
        """The return value passes through unchanged."""
        wrapped = wrap_function(_double, log_dir=tmp_path)
        result = wrapped(7)
        assert result == 14

    def test_log_dir_created_if_not_exists(self, tmp_path):
        """Nested log_dir is created via mkdir(parents=True)."""
        nested = tmp_path / "sub" / "dir"
        wrapped = wrap_function(_logging_fn, log_dir=nested)
        wrapped(1)
        assert nested.exists()
        assert len(list(nested.glob("*.log"))) == 1

    def test_unique_log_files_per_call(self, tmp_path):
        """Each invocation produces a distinct log file (uuid-based naming)."""
        wrapped = wrap_function(_logging_fn, log_dir=tmp_path)
        wrapped(1)
        wrapped(2)
        log_files = list(tmp_path.glob("*.log"))
        assert len(log_files) == 2
        assert log_files[0].name != log_files[1].name


# ---------------------------------------------------------------------------
# Tests: _ray_object_iterator
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("ray_session")
class TestRayObjectIterator:
    def test_iterates_all_objects(self):
        """All object refs are yielded."""
        refs = [ray.put(v) for v in [10, 20, 30]]
        results = list(_ray_object_iterator(refs))
        assert len(results) == 3

    def test_yields_correct_values(self):
        """Collected values match the input set (order not guaranteed by ray.wait)."""
        values = [10, 20, 30]
        refs = [ray.put(v) for v in values]
        collected = {val for _, val in _ray_object_iterator(refs)}
        assert collected == set(values)

    def test_empty_list(self):
        """No iterations for empty input."""
        results = list(_ray_object_iterator([]))
        assert results == []


# ---------------------------------------------------------------------------
# Tests: _ray_map_items
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("ray_session")
class TestRayMapItems:
    def test_map_single_arg_list(self):
        """Map _double over a single argument list."""
        task = Task(fn=_double)
        result = _ray_map_items(task, [1, 2, 3])
        assert result == [2, 4, 6]

    def test_map_multiple_arg_lists(self):
        """Map _add over two argument lists."""
        task = Task(fn=_add)
        result = _ray_map_items(task, [1, 2, 3], [10, 20, 30])
        assert result == [11, 22, 33]

    def test_preserves_order(self):
        """Results are returned in submission order (dict.fromkeys logic)."""
        task = Task(fn=_identity)
        items = list(range(20))
        result = _ray_map_items(task, items)
        assert result == items

    def test_rejects_no_arguments(self):
        """AssertionError when no item_lists are provided."""
        task = Task(fn=_identity)
        with pytest.raises(AssertionError, match="No map arguments received"):
            _ray_map_items(task)

    def test_rejects_non_list_arguments(self):
        """AssertionError when item_lists contain non-list iterables."""
        task = Task(fn=_identity)
        with pytest.raises(AssertionError, match="All map arguments must be lists"):
            _ray_map_items(task, (1, 2, 3))

    def test_rejects_mismatched_list_lengths(self):
        """AssertionError when lists have different lengths."""
        task = Task(fn=_add)
        with pytest.raises(AssertionError, match="All lists must have equal size"):
            _ray_map_items(task, [1, 2], [3, 4, 5])

    def test_partial_function_unpacking(self):
        """Partial functions are unpacked and keyword args forwarded."""
        fn = partial(_add_keyword, y=10)
        task = Task(fn=fn)
        result = _ray_map_items(task, [1, 2, 3])
        assert result == [11, 12, 13]


# ---------------------------------------------------------------------------
# Tests: ray_map (top-level wrapper with error handling)
# ---------------------------------------------------------------------------


class TestRayMap:
    @pytest.fixture(autouse=True)
    def _ray_per_test(self):
        """Re-initialize Ray per test because ray_map shuts it down on error."""
        if not ray.is_initialized():
            ray.init(num_cpus=2, log_to_driver=False)
        yield
        if ray.is_initialized():
            ray.shutdown()

    def test_successful_map(self):
        """Happy-path map with _double."""
        task = Task(fn=_double)
        result = ray_map(task, [1, 2, 3])
        assert result == [2, 4, 6]

    def test_exception_raises_runtime_error(self):
        """Failing tasks cause ray_map to raise RuntimeError."""
        task = Task(fn=_fail)
        with pytest.raises(RuntimeError):
            ray_map(task, [1])

    def test_ray_shutdown_on_error(self):
        """Ray is shut down when ray_map catches an error."""
        task = Task(fn=_fail)
        with pytest.raises(RuntimeError):
            ray_map(task, [1])
        assert not ray.is_initialized()
