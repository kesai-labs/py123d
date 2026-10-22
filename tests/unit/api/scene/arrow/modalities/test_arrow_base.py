"""Tests for ArrowBaseModalityWriter and ArrowBaseModalityReader.read_at_timestamp."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyarrow as pa

from py123d.api.scene.arrow.modalities import sync_utils
from py123d.api.scene.arrow.modalities.arrow_base import ArrowBaseModalityReader, ArrowBaseModalityWriter
from py123d.api.scene.arrow.modalities.sync_utils import find_timestamp_index, get_timestamp_array
from py123d.datatypes import BaseModality, BaseModalityMetadata, ModalityType, Timestamp

# ---------------------------------------------------------------------------
# Concrete stub for testing the abstract reader
# ---------------------------------------------------------------------------


class _StubMetadata(BaseModalityMetadata):
    """Minimal metadata for testing."""

    def __init__(self, modality_key: str = "stub"):
        self._modality_key = modality_key

    @property
    def modality_type(self) -> ModalityType:
        return ModalityType.CUSTOM

    @property
    def modality_key(self) -> str:
        return self._modality_key

    def to_dict(self):
        return {"modality_key": self._modality_key}

    @classmethod
    def from_dict(cls, d):
        return cls(d["modality_key"])


class _StubModality(BaseModality):
    """Minimal modality for testing."""

    def __init__(self, ts_us: int, metadata: _StubMetadata):
        self._ts_us = ts_us
        self._metadata = metadata

    @property
    def timestamp(self) -> Timestamp:
        return Timestamp.from_us(self._ts_us)

    @property
    def metadata(self) -> _StubMetadata:
        return self._metadata


class _StubReader(ArrowBaseModalityReader):
    """Concrete reader that just reads the timestamp column."""

    @staticmethod
    def read_at_index(index, table, metadata, dataset, **kwargs):
        ts_col = f"{metadata.modality_key}.timestamp_us"
        ts_us = table[ts_col][index].as_py()
        return _StubModality(ts_us, metadata)

    @staticmethod
    def read_column_at_index(index, table, metadata, column, dataset, deserialize=False, **kwargs):
        full_name = f"{metadata.modality_key}.{column}"
        result: Optional[Any] = None
        if full_name in table.column_names:
            result = table[full_name][index].as_py()
        return result


def _make_timestamp_table(timestamps_us: list, key: str = "stub") -> pa.Table:
    """Create a minimal Arrow table with a single timestamp column."""
    schema = pa.schema([(f"{key}.timestamp_us", pa.int64())])
    return pa.table({f"{key}.timestamp_us": timestamps_us}, schema=schema)


# ===========================================================================
# Writer tests
# ===========================================================================


class TestArrowBaseModalityWriter:
    """Tests for ArrowBaseModalityWriter buffering, flush, close."""

    def _make_schema(self) -> pa.Schema:
        return pa.schema([("mod.timestamp_us", pa.int64()), ("mod.value", pa.string())])

    def _make_row(self, ts: int, value: str) -> dict:
        return {"mod.timestamp_us": [ts], "mod.value": [value]}

    def test_write_no_buffering(self, tmp_path: Path):
        """Without max_batch_size, each write goes directly to disk."""
        fp = tmp_path / "test.arrow"
        writer = ArrowBaseModalityWriter(fp, self._make_schema(), max_batch_size=None)
        writer.write_batch(self._make_row(100, "a"))
        writer.write_batch(self._make_row(200, "b"))
        assert writer.row_count == 2
        writer.close()

        table = pa.ipc.open_file(str(fp)).read_all()
        assert table.num_rows == 2

    def test_write_with_buffering(self, tmp_path: Path):
        """With max_batch_size=5, writing 3 rows then closing should flush all."""
        fp = tmp_path / "test.arrow"
        writer = ArrowBaseModalityWriter(fp, self._make_schema(), max_batch_size=5)
        for i in range(3):
            writer.write_batch(self._make_row(i * 100, f"v{i}"))
        assert writer.row_count == 3
        writer.close()

        table = pa.ipc.open_file(str(fp)).read_all()
        assert table.num_rows == 3

    def test_auto_flush_at_max_batch(self, tmp_path: Path):
        """Buffer flushes automatically when max_batch_size is reached."""
        fp = tmp_path / "test.arrow"
        writer = ArrowBaseModalityWriter(fp, self._make_schema(), max_batch_size=3)
        for i in range(3):
            writer.write_batch(self._make_row(i * 100, f"v{i}"))
        assert len(writer._buffer) == 0
        assert writer.row_count == 3
        writer.close()

    def test_row_count_includes_buffered(self, tmp_path: Path):
        """row_count should count buffered (unflushed) rows."""
        fp = tmp_path / "test.arrow"
        writer = ArrowBaseModalityWriter(fp, self._make_schema(), max_batch_size=10)
        writer.write_batch(self._make_row(100, "a"))
        assert writer.row_count == 1
        assert len(writer._buffer) == 1
        writer.close()

    def test_close_flushes_remaining(self, tmp_path: Path):
        """close() must flush buffered rows."""
        fp = tmp_path / "test.arrow"
        writer = ArrowBaseModalityWriter(fp, self._make_schema(), max_batch_size=100)
        for i in range(7):
            writer.write_batch(self._make_row(i * 100, f"v{i}"))
        writer.close()

        table = pa.ipc.open_file(str(fp)).read_all()
        assert table.num_rows == 7

    def test_auto_flush_at_max_batch_bytes(self, tmp_path: Path):
        """Buffer flushes on the byte budget, long before the row cap is reached.

        Guards against overflowing Arrow's 2 GiB per-column offset limit, which turns the
        flushed column into a ChunkedArray and makes the record batch unwritable.
        """
        fp = tmp_path / "test.arrow"
        schema = pa.schema([("mod.timestamp_us", pa.int64()), ("mod.data", pa.binary())])
        writer = ArrowBaseModalityWriter(fp, schema, max_batch_size=1000, max_batch_bytes=1000)
        for i in range(4):
            writer.write_batch({"mod.timestamp_us": [i * 100], "mod.data": [bytes(400)]})

        # Rows 1-3 cross the 1000-byte budget and flush; row 4 stays buffered.
        assert len(writer._buffer) == 1
        writer.close()

        reader = pa.ipc.open_file(str(fp))
        assert reader.num_record_batches == 2
        table = reader.read_all()
        assert table.num_rows == 4
        assert table["mod.timestamp_us"].to_pylist() == [0, 100, 200, 300]

    def test_set_max_batch_bytes_overrides_default(self, tmp_path: Path):
        """set_max_batch_bytes() takes effect on the next buffered row."""
        fp = tmp_path / "test.arrow"
        schema = pa.schema([("mod.timestamp_us", pa.int64()), ("mod.data", pa.binary())])
        writer = ArrowBaseModalityWriter(fp, schema, max_batch_size=1000)
        writer.set_max_batch_bytes(1000)
        for i in range(3):
            writer.write_batch({"mod.timestamp_us": [i * 100], "mod.data": [bytes(400)]})
        assert len(writer._buffer) == 0
        writer.close()

        assert pa.ipc.open_file(str(fp)).read_all().num_rows == 3

    def test_constructor_rejects_non_positive_max_batch_bytes(self, tmp_path: Path):
        """The constructor validates the byte budget too, not just the setter."""
        import pytest

        with pytest.raises(ValueError, match="must be > 0"):
            ArrowBaseModalityWriter(tmp_path / "test.arrow", self._make_schema(), max_batch_bytes=0)

    def test_max_batch_bytes_ignores_non_schema_columns(self, tmp_path: Path):
        """Keys outside the schema are dropped on flush, so they must not consume the budget."""
        fp = tmp_path / "test.arrow"
        schema = pa.schema([("mod.timestamp_us", pa.int64()), ("mod.data", pa.binary())])
        writer = ArrowBaseModalityWriter(fp, schema, max_batch_size=1000, max_batch_bytes=1000)
        for i in range(3):
            writer.write_batch(
                {
                    "mod.timestamp_us": [i * 100],
                    "mod.data": [bytes(100)],
                    "mod.not_in_schema": [bytes(10_000)],
                }
            )
        assert len(writer._buffer) == 3
        writer.close()

        assert pa.ipc.open_file(str(fp)).read_all().num_rows == 3

    def test_max_batch_bytes_counts_utf8_length_of_strings(self, tmp_path: Path):
        """Non-ASCII strings count by UTF-8 byte length, not character count."""
        fp = tmp_path / "test.arrow"
        # 300 two-byte characters = 600 bytes, so two rows cross the budget but one does not.
        writer = ArrowBaseModalityWriter(fp, self._make_schema(), max_batch_size=1000, max_batch_bytes=1000)
        writer.write_batch(self._make_row(100, "é" * 300))
        assert len(writer._buffer) == 1
        writer.write_batch(self._make_row(200, "é" * 300))
        assert len(writer._buffer) == 0
        writer.close()

        assert pa.ipc.open_file(str(fp)).read_all().num_rows == 2

    def test_set_max_batch_bytes_rejects_non_positive(self, tmp_path: Path):
        """A non-positive byte budget would flush on every row and is rejected."""
        import pytest

        fp = tmp_path / "test.arrow"
        writer = ArrowBaseModalityWriter(fp, self._make_schema(), max_batch_size=10)
        with pytest.raises(ValueError, match="must be > 0"):
            writer.set_max_batch_bytes(0)
        writer.write_batch(self._make_row(100, "a"))
        writer.close()

    def test_max_batch_bytes_ignores_fixed_width_columns(self, tmp_path: Path):
        """Fixed-width columns never trip the byte budget — only binary/string values count."""
        fp = tmp_path / "test.arrow"
        schema = pa.schema([("mod.timestamp_us", pa.int64()), ("mod.value", pa.int64())])
        writer = ArrowBaseModalityWriter(fp, schema, max_batch_size=1000, max_batch_bytes=1)
        for i in range(4):
            writer.write_batch({"mod.timestamp_us": [i * 100], "mod.value": [i]})
        assert len(writer._buffer) == 4
        writer.close()

        assert pa.ipc.open_file(str(fp)).read_all().num_rows == 4

    def test_close_idempotent(self, tmp_path: Path):
        """Calling close() twice should not crash."""
        fp = tmp_path / "test.arrow"
        writer = ArrowBaseModalityWriter(fp, self._make_schema(), max_batch_size=None)
        writer.write_batch(self._make_row(100, "a"))
        writer.close()
        writer.close()  # Second close should be safe

    def test_flush_fills_missing_columns_with_none(self, tmp_path: Path):
        """If a buffered row is missing a column, it should be filled with None."""
        fp = tmp_path / "test.arrow"
        schema = pa.schema([("mod.timestamp_us", pa.int64()), ("mod.extra", pa.int64())])
        writer = ArrowBaseModalityWriter(fp, schema, max_batch_size=5)
        writer.write_batch({"mod.timestamp_us": [100]})
        writer.close()

        table = pa.ipc.open_file(str(fp)).read_all()
        assert table.num_rows == 1
        assert table["mod.extra"][0].as_py() is None

    def test_compression_lz4(self, tmp_path: Path):
        """Writer with LZ4 compression should produce a readable file."""
        fp = tmp_path / "test.arrow"
        writer = ArrowBaseModalityWriter(fp, self._make_schema(), ipc_compression="lz4", max_batch_size=None)
        writer.write_batch(self._make_row(100, "compressed"))
        writer.close()

        table = pa.ipc.open_file(str(fp)).read_all()
        assert table["mod.timestamp_us"][0].as_py() == 100

    def test_compression_zstd(self, tmp_path: Path):
        """Writer with ZSTD compression should produce a readable file."""
        fp = tmp_path / "test.arrow"
        writer = ArrowBaseModalityWriter(fp, self._make_schema(), ipc_compression="zstd", max_batch_size=None)
        writer.write_batch(self._make_row(100, "compressed"))
        writer.close()

        table = pa.ipc.open_file(str(fp)).read_all()
        assert table["mod.timestamp_us"][0].as_py() == 100

    def test_timestamp_ordering_strictly_increasing_ok(self, tmp_path: Path):
        """Writing strictly increasing timestamps should succeed."""
        fp = tmp_path / "test.arrow"
        schema = pa.schema([("modality.timestamp_us", pa.int64()), ("modality.value", pa.int64())])
        writer = ArrowBaseModalityWriter(fp, schema, max_batch_size=None)
        writer.write_batch({"modality.timestamp_us": [100], "modality.value": [1]})
        writer.write_batch({"modality.timestamp_us": [200], "modality.value": [2]})
        writer.write_batch({"modality.timestamp_us": [300], "modality.value": [3]})
        writer.close()  # Validates on close — no error

    def test_timestamp_ordering_equal_ok(self, tmp_path: Path):
        """Duplicate timestamps should be allowed — must be increasing."""

        fp = tmp_path / "test.arrow"
        schema = pa.schema([("modality.timestamp_us", pa.int64()), ("modality.value", pa.int64())])
        writer = ArrowBaseModalityWriter(fp, schema, max_batch_size=None)
        writer.write_batch({"modality.timestamp_us": [100], "modality.value": [1]})
        writer.write_batch({"modality.timestamp_us": [100], "modality.value": [2]})
        writer.close()

    def test_timestamp_ordering_decreasing_raises(self, tmp_path: Path):
        """Writing a timestamp smaller than the previous one should raise on close()."""
        import pytest

        fp = tmp_path / "test.arrow"
        schema = pa.schema([("modality.timestamp_us", pa.int64()), ("modality.value", pa.int64())])
        writer = ArrowBaseModalityWriter(fp, schema, max_batch_size=None)
        writer.write_batch({"modality.timestamp_us": [200], "modality.value": [1]})
        writer.write_batch({"modality.timestamp_us": [100], "modality.value": [2]})
        with pytest.raises(ValueError, match="monotonically increasing"):
            writer.close()

    def test_timestamp_ordering_with_buffering(self, tmp_path: Path):
        """Timestamp ordering should be checked on close even when buffering."""
        import pytest

        fp = tmp_path / "test.arrow"
        schema = pa.schema([("modality.timestamp_us", pa.int64()), ("modality.value", pa.int64())])
        writer = ArrowBaseModalityWriter(fp, schema, max_batch_size=10)
        writer.write_batch({"modality.timestamp_us": [200], "modality.value": [1]})
        writer.write_batch({"modality.timestamp_us": [100], "modality.value": [2]})
        with pytest.raises(ValueError, match="monotonically increasing"):
            writer.close()

    def test_no_timestamp_column_raises(self, tmp_path: Path):
        """If the schema has no timestamp_us column, close() should raise."""
        import pytest

        fp = tmp_path / "test.arrow"
        schema = pa.schema([("col_a", pa.int64()), ("col_b", pa.int64())])
        writer = ArrowBaseModalityWriter(fp, schema, max_batch_size=None)
        writer.write_batch({"col_a": [1], "col_b": [1]})
        writer.write_batch({"col_a": [2], "col_b": [2]})
        with pytest.raises(ValueError, match="No timestamp column"):
            writer.close()

    def test_single_row_skips_validation(self, tmp_path: Path):
        """With only 1 row, timestamp ordering validation is skipped."""
        fp = tmp_path / "test.arrow"
        schema = pa.schema([("col_a", pa.int64())])
        writer = ArrowBaseModalityWriter(fp, schema, max_batch_size=None)
        writer.write_batch({"col_a": [1]})
        writer.close()  # No error — only 1 row, validation skipped


# ===========================================================================
# Reader timestamp matching tests
# ===========================================================================


class TestReadAtTimestamp:
    """Tests for ArrowBaseModalityReader.read_at_timestamp (exact/nearest/forward/backward)."""

    _METADATA = _StubMetadata("stub")
    _DATASET = "test-dataset"

    def test_exact_found(self):
        table = _make_timestamp_table([100, 200, 300])
        result = _StubReader.read_at_timestamp(Timestamp.from_us(200), table, self._METADATA, self._DATASET, "exact")
        assert result is not None
        assert result.timestamp.time_us == 200

    def test_exact_not_found(self):
        table = _make_timestamp_table([100, 200, 300])
        result = _StubReader.read_at_timestamp(Timestamp.from_us(150), table, self._METADATA, self._DATASET, "exact")
        assert result is None

    def test_nearest_picks_closest(self):
        table = _make_timestamp_table([100, 200, 300])
        result = _StubReader.read_at_timestamp(Timestamp.from_us(190), table, self._METADATA, self._DATASET, "nearest")
        assert result is not None
        assert result.timestamp.time_us == 200

    def test_nearest_equidistant_picks_first(self):
        """When equidistant, np.argmin picks the first occurrence (W4 documentation)."""
        table = _make_timestamp_table([100, 300])
        result = _StubReader.read_at_timestamp(Timestamp.from_us(200), table, self._METADATA, self._DATASET, "nearest")
        assert result is not None
        # argmin on equal distances returns first index
        assert result.timestamp.time_us == 100

    def test_nearest_empty_table(self):
        table = _make_timestamp_table([])
        result = _StubReader.read_at_timestamp(Timestamp.from_us(100), table, self._METADATA, self._DATASET, "nearest")
        assert result is None

    def test_forward_at_boundary(self):
        """Forward with >= should include exact match."""
        table = _make_timestamp_table([100, 200, 300])
        result = _StubReader.read_at_timestamp(Timestamp.from_us(200), table, self._METADATA, self._DATASET, "forward")
        assert result is not None
        assert result.timestamp.time_us == 200

    def test_forward_between_values(self):
        table = _make_timestamp_table([100, 200, 300])
        result = _StubReader.read_at_timestamp(Timestamp.from_us(150), table, self._METADATA, self._DATASET, "forward")
        assert result is not None
        assert result.timestamp.time_us == 200

    def test_forward_past_all(self):
        table = _make_timestamp_table([100, 200, 300])
        result = _StubReader.read_at_timestamp(Timestamp.from_us(400), table, self._METADATA, self._DATASET, "forward")
        assert result is None

    def test_backward_at_boundary(self):
        """Backward with <= should include exact match."""
        table = _make_timestamp_table([100, 200, 300])
        result = _StubReader.read_at_timestamp(Timestamp.from_us(200), table, self._METADATA, self._DATASET, "backward")
        assert result is not None
        assert result.timestamp.time_us == 200

    def test_backward_between_values(self):
        table = _make_timestamp_table([100, 200, 300])
        result = _StubReader.read_at_timestamp(Timestamp.from_us(250), table, self._METADATA, self._DATASET, "backward")
        assert result is not None
        assert result.timestamp.time_us == 200

    def test_backward_before_all(self):
        table = _make_timestamp_table([100, 200, 300])
        result = _StubReader.read_at_timestamp(Timestamp.from_us(50), table, self._METADATA, self._DATASET, "backward")
        assert result is None

    def test_forward_unsorted_timestamps(self):
        """W4: forward uses np.where which returns indices in array order, assumes sorted.

        With unsorted timestamps [300, 100, 200], forward from 150 picks the first array
        element >= 150, which is index 0 (value 300), not the chronologically first (200).
        """
        table = _make_timestamp_table([300, 100, 200])
        result = _StubReader.read_at_timestamp(Timestamp.from_us(150), table, self._METADATA, self._DATASET, "forward")
        assert result is not None
        # Bug: picks 300 (first match in array order) rather than 200 (chronologically first)
        assert result.timestamp.time_us == 300

    def test_backward_unsorted_timestamps(self):
        """W4: backward with unsorted timestamps picks last array match, not chronologically last."""
        table = _make_timestamp_table([300, 100, 200])
        result = _StubReader.read_at_timestamp(Timestamp.from_us(250), table, self._METADATA, self._DATASET, "backward")
        assert result is not None
        # Bug: picks 200 (last match in array order) rather than 300 (chronologically last <= 250)
        assert result.timestamp.time_us == 200

    def test_duplicate_timestamps(self):
        """Exact match with duplicates returns the first occurrence."""
        table = _make_timestamp_table([100, 100, 200])
        result = _StubReader.read_at_timestamp(Timestamp.from_us(100), table, self._METADATA, self._DATASET, "exact")
        assert result is not None
        assert result.timestamp.time_us == 100


class TestTimestampLookup:
    """Binary search on sorted timestamps gives the rows of the full scan; the array is computed once per table."""

    _CRITERIA = ("exact", "nearest", "forward", "backward")

    def test_binary_search_matches_full_scan(self):
        rng = np.random.default_rng(0)
        for _ in range(300):
            ts = np.sort(rng.integers(0, 20, int(rng.integers(0, 12)))) * 10  # repeated values and gaps
            for target in range(-15, 215, 5):  # before, between, on and after the values
                for criteria in self._CRITERIA:
                    expected = find_timestamp_index(ts, target, criteria, is_sorted=False)
                    assert find_timestamp_index(ts, target, criteria, is_sorted=True) == expected, (
                        ts,
                        target,
                        criteria,
                    )

    def test_array_cached_per_table_and_column(self):
        column = "stub.timestamp_us"
        table = pa.Table.from_batches(
            [pa.record_batch({column: pa.array(values, pa.int64())}) for values in ([100, 200], [200, 300])]
        )
        values, is_sorted = get_timestamp_array(table, column)
        assert values.tolist() == [100, 200, 200, 300]
        assert is_sorted
        assert not values.flags.writeable
        assert get_timestamp_array(table, column)[0] is values

    def test_cache_entry_removed_with_table(self):
        table = _make_timestamp_table([1, 2])
        key = id(table)
        get_timestamp_array(table, "stub.timestamp_us")
        assert key in sync_utils._TIMESTAMP_ARRAYS
        del table
        gc.collect()
        assert key not in sync_utils._TIMESTAMP_ARRAYS

    def test_unsorted_and_null_columns_use_the_full_scan(self):
        assert not get_timestamp_array(_make_timestamp_table([300, 100, 200]), "stub.timestamp_us")[1]
        table = pa.table({"stub.timestamp_us": pa.array([100, None, 300], pa.int64())})
        assert not get_timestamp_array(table, "stub.timestamp_us")[1]

    def test_read_at_timestamp_on_several_batches(self):
        column = "stub.timestamp_us"
        table = pa.Table.from_batches(
            [pa.record_batch({column: pa.array(values, pa.int64())}) for values in ([100, 200], [300], [400, 500])]
        )
        metadata = _StubMetadata("stub")
        result = _StubReader.read_at_timestamp(Timestamp.from_us(320), table, metadata, "test-dataset", "nearest")
        assert result is not None and result.timestamp.time_us == 300
        result = _StubReader.read_at_timestamp(Timestamp.from_us(320), table, metadata, "test-dataset", "forward")
        assert result is not None and result.timestamp.time_us == 400
