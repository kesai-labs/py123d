"""Tests for the conversion script — maps and logs that do not complete."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import pyarrow as pa
import pytest
from omegaconf import DictConfig, OmegaConf

from py123d.datatypes import BaseMapObject, LogMetadata, MapMetadata, Timestamp
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.geometry.pose import PoseSE3
from py123d.parser.base_dataset_parser import BaseLogParser, BaseMapParser, ModalitiesSync
from py123d.script.run_conversion import _cleanup_incomplete, _convert_logs, _convert_logs_async, _convert_maps

from ..api.map.conftest import make_crosswalk
from ..api.scene.conftest import make_ego_metadata, make_log_metadata

_SPLIT = make_log_metadata().split


class _FakeLogParser(BaseLogParser):
    """Yields three ego states, optionally failing at one of them."""

    def __init__(self, log_name: str, fail_at_frame: Optional[int] = None) -> None:
        self._log_name = log_name
        self._fail_at_frame = fail_at_frame

    def get_log_metadata(self) -> LogMetadata:
        return make_log_metadata(log_name=self._log_name)

    def iter_modalities_sync(self) -> Iterator[ModalitiesSync]:
        for frame in range(3):
            if frame == self._fail_at_frame:
                raise RuntimeError("broken frame")
            timestamp = Timestamp.from_us(frame * 100_000)
            ego = EgoStateSE3.from_imu(imu_se3=PoseSE3.identity(), metadata=make_ego_metadata(), timestamp=timestamp)
            yield ModalitiesSync(timestamp=timestamp, modalities=[ego])


class _FakeMapParser(BaseMapParser):
    """Yields a single crosswalk, optionally failing after it."""

    def __init__(self, location: str, fail: bool = False) -> None:
        self._location = location
        self._fail = fail

    def get_map_metadata(self) -> MapMetadata:
        return MapMetadata(dataset="test-dataset", location=self._location, map_has_z=True, map_is_per_log=False)

    def iter_map_objects(self) -> Iterator[BaseMapObject]:
        yield make_crosswalk(object_id="cw_1")
        if self._fail:
            raise RuntimeError("broken map")


def _make_cfg(data_root: Path, keep_incomplete: bool = True) -> DictConfig:
    """The parts of the conversion config that are read after the dataset parser was built."""
    logs_root, maps_root = str(data_root / "logs"), str(data_root / "maps")
    return OmegaConf.create(
        {
            "keep_incomplete": keep_incomplete,
            "dataset_paths": {"py123d_logs_root": logs_root, "py123d_maps_root": maps_root},
            "dataset": {
                "log_writer": {
                    "_target_": "py123d.api.scene.arrow.arrow_log_writer.ArrowLogWriter",
                    "log_writer_config": {"_target_": "py123d.api.scene.arrow.utils.log_writer_config.LogWriterConfig"},
                    "logs_root": logs_root,
                    "sensors_root": str(data_root / "sensors"),
                    "sync_config": {
                        "_target_": "py123d.api.scene.arrow.arrow_log_writer.SyncConfig",
                        "reference_column": "ego_state_se3.timestamp_us",
                    },
                },
                "map_writer": {
                    "_target_": "py123d.api.map.arrow.arrow_map_writer.ArrowMapWriter",
                    "force_map_conversion": True,
                    "maps_root": maps_root,
                    "logs_root": logs_root,
                },
            },
        }
    )


@pytest.mark.parametrize("convert_fn", [_convert_logs, _convert_logs_async])
def test_failed_log_stays_incomplete(tmp_path: Path, convert_fn):
    cfg = _make_cfg(tmp_path)
    log_parsers = [_FakeLogParser("log_001"), _FakeLogParser("log_002", fail_at_frame=1), _FakeLogParser("log_003")]

    assert convert_fn(log_parsers, cfg) == [f"{_SPLIT}/log_002"]
    assert (tmp_path / "logs" / _SPLIT / "log_001" / "sync.arrow").exists()
    assert (tmp_path / "logs" / _SPLIT / "log_003" / "sync.arrow").exists()
    assert not (tmp_path / "logs" / _SPLIT / "log_002").exists()
    # The failed log is closed with the frame written before the error, e.g. for debugging.
    sync_file = tmp_path / "logs" / ".incomplete" / _SPLIT / "log_002" / "sync.arrow"
    assert pa.ipc.open_file(str(sync_file)).read_all().num_rows == 1


def test_failed_map_stays_incomplete(tmp_path: Path):
    cfg = _make_cfg(tmp_path)
    map_parsers = [_FakeMapParser("boston"), _FakeMapParser("pittsburgh", fail=True)]

    assert _convert_maps(map_parsers, cfg) == ["test-dataset/pittsburgh"]
    assert [path.name for path in (tmp_path / "maps" / "test-dataset").iterdir()] == ["test-dataset_boston.arrow"]
    assert (tmp_path / "maps" / ".incomplete" / "test-dataset" / "pittsburgh.arrow").exists()


@pytest.mark.parametrize("keep_incomplete", [True, False])
def test_cleanup_incomplete(tmp_path: Path, keep_incomplete: bool):
    cfg = _make_cfg(tmp_path, keep_incomplete)
    failed_maps = _convert_maps([_FakeMapParser("boston"), _FakeMapParser("pittsburgh", fail=True)], cfg)
    failed_logs = _convert_logs([_FakeLogParser("log_001"), _FakeLogParser("log_002", fail_at_frame=1)], cfg)
    # Another conversion may be running in the same data root, its logs must not be deleted.
    other_log_dir = tmp_path / "logs" / ".incomplete" / "other-dataset_train" / "log_running"
    other_log_dir.mkdir(parents=True)

    _cleanup_incomplete(cfg, failed_maps, failed_logs)

    assert (tmp_path / "logs" / ".incomplete" / _SPLIT / "log_002").exists() is keep_incomplete
    assert (tmp_path / "logs" / ".incomplete" / _SPLIT).exists() is keep_incomplete
    assert other_log_dir.exists()
    assert (tmp_path / "maps" / ".incomplete" / "test-dataset" / "pittsburgh.arrow").exists() is keep_incomplete
    assert (tmp_path / "maps" / ".incomplete").exists() is keep_incomplete
    assert (tmp_path / "logs" / _SPLIT / "log_001" / "sync.arrow").exists()
