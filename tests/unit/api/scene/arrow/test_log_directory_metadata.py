"""Tests for the lazy reads of a log directory: modality metadata file by file, tables without a stat."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pytest

from py123d.api.scene.arrow.modalities.sync_utils import get_modality_table
from py123d.api.utils import arrow_helper
from py123d.api.utils.arrow_metadata_utils import parse_log_directory_metadata
from py123d.datatypes.modalities.base_modality import ModalityType

from .test_cache_source import EGO_KEY, _write_log

ROUTE_KEY = ModalityType.ROUTE_POSITION.serialize()


@pytest.fixture
def log_dir(tmp_path: Path) -> Path:
    parse_log_directory_metadata.cache_clear()
    return _write_log(tmp_path, [0.0, 1.0, 2.0, 3.0])


@pytest.fixture
def schema_reads(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    """The file names whose schema is read, in order."""
    reads: List[str] = []
    open_arrow_schema = arrow_helper.open_arrow_schema

    def counted(arrow_file_path):
        reads.append(os.path.basename(str(arrow_file_path)))
        return open_arrow_schema(arrow_file_path)

    monkeypatch.setattr(arrow_helper, "open_arrow_schema", counted)
    return reads


class TestLazyModalityMetadatas:
    def test_parse_reads_the_sync_file_alone(self, log_dir: Path, schema_reads: List[str]) -> None:
        parse_log_directory_metadata(log_dir)
        assert schema_reads == ["sync.arrow"]

    def test_one_key_reads_one_file_once(self, log_dir: Path, schema_reads: List[str]) -> None:
        metadatas = parse_log_directory_metadata(log_dir).modality_metadatas
        assert metadatas.get(EGO_KEY) is metadatas[EGO_KEY]
        assert EGO_KEY in metadatas
        assert schema_reads == ["sync.arrow", f"{EGO_KEY}.arrow"]

    def test_an_absent_key_reads_no_file(self, log_dir: Path, schema_reads: List[str]) -> None:
        metadatas = parse_log_directory_metadata(log_dir).modality_metadatas
        assert metadatas.get("lidar.absent") is None
        assert "lidar.absent" not in metadatas
        assert schema_reads == ["sync.arrow"]

    def test_get_by_type_reads_the_files_of_that_type(self, log_dir: Path, schema_reads: List[str]) -> None:
        by_type = parse_log_directory_metadata(log_dir).get_by_type(ModalityType.EGO_STATE_SE3)
        assert list(by_type) == [EGO_KEY]
        assert schema_reads == ["sync.arrow", f"{EGO_KEY}.arrow"]

    def test_iterating_reads_every_file_in_name_order(self, log_dir: Path) -> None:
        metadatas = parse_log_directory_metadata(log_dir).modality_metadatas
        assert list(metadatas) == sorted([EGO_KEY, ROUTE_KEY])
        assert len(metadatas) == 2
        assert {key: metadata.modality_type for key, metadata in metadatas.items()} == {
            EGO_KEY: ModalityType.EGO_STATE_SE3,
            ROUTE_KEY: ModalityType.ROUTE_POSITION,
        }

    def test_a_file_that_cannot_be_deserialized_is_left_out(self, log_dir: Path) -> None:
        (log_dir / "not_a_modality_type.arrow").write_bytes((log_dir / f"{EGO_KEY}.arrow").read_bytes())
        parse_log_directory_metadata.cache_clear()
        metadatas = parse_log_directory_metadata(log_dir).modality_metadatas
        assert metadatas.get("not_a_modality_type") is None
        assert list(metadatas) == sorted([EGO_KEY, ROUTE_KEY])

    def test_a_directory_without_sync_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            parse_log_directory_metadata(tmp_path)


class TestModalityTableWithoutStat:
    def test_a_cached_table_needs_no_file_system_call(self, log_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        table = get_modality_table(log_dir, EGO_KEY)
        assert table is not None

        def no_stat(*args, **kwargs):
            raise AssertionError("a cached table was looked up on the file system")

        monkeypatch.setattr(Path, "exists", no_stat)
        monkeypatch.setattr(Path, "stat", no_stat)
        assert get_modality_table(log_dir, EGO_KEY) is table

    def test_a_missing_file_is_none(self, log_dir: Path) -> None:
        assert get_modality_table(log_dir, "does_not_exist") is None
