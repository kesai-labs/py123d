"""The cache of memory-mapped Arrow tables."""

import os
from pathlib import Path

import pyarrow as pa
import pytest

from py123d.api.utils import arrow_helper
from py123d.api.utils.arrow_helper import clear_arrow_table_cache, get_lru_cached_arrow_table, write_arrow_table


@pytest.fixture(autouse=True)
def _clear_cache():
    yield
    clear_arrow_table_cache()


def _replace_table(path: Path, values: list) -> None:
    """Writes a new file and moves it over ``path``, so a memory map of the old file stays valid."""
    new_path = path.with_suffix(".new")
    write_arrow_table(pa.table({"value": values}), new_path)
    os.replace(new_path, path)


def test_cached_table_is_reused(tmp_path: Path):
    path = tmp_path / "table.arrow"
    _replace_table(path, [1, 2, 3])

    assert get_lru_cached_arrow_table(path) is get_lru_cached_arrow_table(path)


def test_clear_makes_a_rewritten_file_visible(tmp_path: Path):
    """The cache keeps returning the table it loaded first, until it is cleared."""
    path = tmp_path / "table.arrow"
    _replace_table(path, [1, 2, 3])
    assert get_lru_cached_arrow_table(path).column("value").to_pylist() == [1, 2, 3]

    _replace_table(path, [4, 5])
    assert get_lru_cached_arrow_table(path).column("value").to_pylist() == [1, 2, 3]

    clear_arrow_table_cache()
    assert get_lru_cached_arrow_table(path).column("value").to_pylist() == [4, 5]


def test_clear_closes_the_memory_maps(tmp_path: Path):
    paths = [tmp_path / f"table_{index}.arrow" for index in range(3)]
    for path in paths:
        _replace_table(path, [0])
        get_lru_cached_arrow_table(path)
    store = arrow_helper._get_store()
    sources = [store._cache[str(path)][0] for path in paths]
    assert not any(source.closed for source in sources)

    clear_arrow_table_cache()

    assert all(source.closed for source in sources)
    assert len(store._cache) == 0


def test_clear_before_first_use_does_nothing(monkeypatch):
    monkeypatch.setattr(arrow_helper, "_store", None)

    clear_arrow_table_cache()

    assert arrow_helper._store is None
