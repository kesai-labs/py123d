"""Tests for ``CacheSourceInfo``: column hashing, outdated-route detection, enforcement on routes."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pyarrow as pa
import pytest

from py123d.api.scene.arrow.arrow_log_writer import ArrowLogWriter
from py123d.api.scene.arrow.lazy_scene_sequence import build_log_scene_index
from py123d.api.scene.arrow.utils.log_writer_config import LogWriterConfig
from py123d.api.scene.arrow.utils.route_utils import read_route_position
from py123d.api.scene.scene_filter import SceneFilter
from py123d.api.utils.arrow_metadata_utils import parse_log_directory_metadata
from py123d.api.utils.cache_source_utils import (
    StaleModalityError,
    build_cache_source_info,
    check_cache_source_modalities,
    get_cache_source_info,
    hash_modality_columns,
)
from py123d.datatypes import Timestamp
from py123d.datatypes.metadata.cache_source import CacheSourceInfo, CacheSourceModality
from py123d.datatypes.metadata.route_metadata import RouteMetadata
from py123d.datatypes.modalities.base_modality import ModalityType
from py123d.datatypes.vehicle_state.dynamic_state import DynamicStateSE3
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.geometry.pose import PoseSE3
from py123d.geometry.vector import Vector3D
from py123d.parser.base_dataset_parser import ModalitiesSync

from ..conftest import make_ego_metadata, make_log_metadata

TIMESTEP_US = 100_000
EGO_KEY = ModalityType.EGO_STATE_SE3.serialize()


def _make_ego(ts_us: int, x: float) -> EgoStateSE3:
    return EgoStateSE3.from_imu(
        imu_se3=PoseSE3.from_list([x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        metadata=make_ego_metadata(),
        timestamp=Timestamp.from_us(ts_us),
        dynamic_state_se3=DynamicStateSE3(
            velocity=Vector3D(1.0, 0.0, 0.0),
            acceleration=Vector3D(0.0, 0.0, 0.0),
            angular_velocity=Vector3D(0.0, 0.0, 0.0),
        ),
    )


def _write_log(tmp_path: Path, xs: List[float], write_route: bool = True) -> Path:
    log_meta = make_log_metadata()
    writer = ArrowLogWriter(LogWriterConfig(write_route=write_route), logs_root=tmp_path, sensors_root=tmp_path)
    writer.reset(log_meta)
    for i, x in enumerate(xs):
        writer.write_sync(
            ModalitiesSync(timestamp=Timestamp.from_us(i * TIMESTEP_US), modalities=[_make_ego(i * TIMESTEP_US, x)])
        )
    writer.close()
    return tmp_path / log_meta.split / log_meta.log_name


def _rewrite_ego_positions(log_dir: Path, scale: float) -> None:
    """Rewrite the ego poses in place, as a re-run motion fuser would."""
    path = log_dir / f"{EGO_KEY}.arrow"
    with pa.memory_map(str(path), "r") as source:
        table = pa.ipc.open_file(source).read_all()

    column_name = f"{EGO_KEY}.imu_se3"
    array = table.column(column_name).combine_chunks()
    poses = np.array(array.flatten().to_numpy(zero_copy_only=False)).reshape(len(table), -1)
    poses[:, 0] *= scale
    rewritten = pa.FixedSizeListArray.from_arrays(pa.array(poses.reshape(-1)), poses.shape[1])
    table = table.set_column(table.schema.get_field_index(column_name), table.schema.field(column_name), rewritten)

    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, schema=table.schema) as writer:
            writer.write_table(table)
    check_cache_source_modalities.cache_clear()
    parse_log_directory_metadata.cache_clear()


class TestCacheSourceInfo:
    def test_computed_by_format(self) -> None:
        with pytest.raises(AssertionError):
            CacheSourceInfo(computed_by="route_backfill", computed_at="2026-08-31T09:14:22Z", source_modalities=[])

    def test_dict_roundtrip(self) -> None:
        cache_source_info = CacheSourceInfo(
            computed_by="garage:route_backfill@3",
            computed_at="2026-08-31T09:14:22Z",
            source_modalities=[CacheSourceModality(modality_key=EGO_KEY, columns=["imu_se3"], hash="abc123")],
            external_sources=["nav database snapshot 2026-08-01"],
        )
        assert CacheSourceInfo.from_dict(cache_source_info.to_dict()) == cache_source_info


class TestHashModalityColumns:
    def test_same_columns_same_hash(self, tmp_path: Path) -> None:
        first = _write_log(tmp_path / "a", xs=[2.0 * i for i in range(5)])
        second = _write_log(tmp_path / "b", xs=[2.0 * i for i in range(5)])
        assert hash_modality_columns(first, EGO_KEY, ["imu_se3"]) == hash_modality_columns(second, EGO_KEY, ["imu_se3"])

    def test_changed_columns_different_hash(self, tmp_path: Path) -> None:
        first = _write_log(tmp_path / "a", xs=[2.0 * i for i in range(5)])
        second = _write_log(tmp_path / "b", xs=[3.0 * i for i in range(5)])
        assert hash_modality_columns(first, EGO_KEY, ["imu_se3"]) != hash_modality_columns(second, EGO_KEY, ["imu_se3"])

    def test_missing_modality_raises(self, tmp_path: Path) -> None:
        log_dir = _write_log(tmp_path, xs=[0.0, 1.0])
        with pytest.raises(FileNotFoundError):
            hash_modality_columns(log_dir, "lidar.lidar_merged", ["timestamp_us"])

    def test_build_cache_source_info(self, tmp_path: Path) -> None:
        log_dir = _write_log(tmp_path, xs=[0.0, 1.0])
        cache_source_info = build_cache_source_info(log_dir, "garage:probe@1", {EGO_KEY: ["imu_se3", "timestamp_us"]})
        assert [(source.modality_key, source.columns) for source in cache_source_info.source_modalities] == [
            (EGO_KEY, ["imu_se3", "timestamp_us"])
        ]
        assert cache_source_info.computed_at.endswith("Z")


class TestRouteMetadata:
    def test_cache_source_info_required(self) -> None:
        with pytest.raises(TypeError):
            RouteMetadata(resolution_m=1.0, total_arc_m=1.0, polyline_x=[], polyline_y=[], polyline_z=[])

    def test_from_dict_without_cache_source_info_raises(self) -> None:
        with pytest.raises(ValueError, match="cache_source_info"):
            RouteMetadata.from_dict(
                {"resolution_m": 1.0, "total_arc_m": 1.0, "polyline_x": [], "polyline_y": [], "polyline_z": []}
            )

    def test_writer_records_ego_columns(self, tmp_path: Path) -> None:
        log_dir = _write_log(tmp_path, xs=[2.0 * i for i in range(5)])
        route_metadata, _, _ = read_route_position(log_dir)
        cache_source_info = route_metadata.cache_source_info
        assert cache_source_info.computed_by == "py123d:route_from_ego_state_se3@1"
        assert [(source.modality_key, source.columns) for source in cache_source_info.source_modalities] == [
            (EGO_KEY, ["imu_se3"])
        ]
        assert cache_source_info.source_modalities[0].hash == hash_modality_columns(log_dir, EGO_KEY, ["imu_se3"])

    def test_provided_route_lists_external_source(self, tmp_path: Path) -> None:
        log_meta = make_log_metadata()
        writer = ArrowLogWriter(LogWriterConfig(), logs_root=tmp_path, sensors_root=tmp_path)
        writer.reset(log_meta)
        writer.set_route(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
        for i in range(3):
            writer.write_sync(
                ModalitiesSync(
                    timestamp=Timestamp.from_us(i * TIMESTEP_US), modalities=[_make_ego(i * TIMESTEP_US, float(i))]
                )
            )
        writer.close()

        route_metadata, _, _ = read_route_position(tmp_path / log_meta.split / log_meta.log_name)
        assert route_metadata.cache_source_info.external_sources == [
            "route waypoints provided via ArrowLogWriter.set_route"
        ]


class TestCheckSourceModalities:
    def test_untouched_log(self, tmp_path: Path) -> None:
        check_cache_source_modalities(_write_log(tmp_path, xs=[2.0 * i for i in range(5)]))

    def test_rewritten_ego_raises(self, tmp_path: Path) -> None:
        log_dir = _write_log(tmp_path, xs=[2.0 * i for i in range(5)])
        _rewrite_ego_positions(log_dir, scale=3.0)
        with pytest.raises(StaleModalityError, match="route_position"):
            check_cache_source_modalities(log_dir)

    def test_scene_index_raises(self, tmp_path: Path) -> None:
        log_dir = _write_log(tmp_path, xs=[2.0 * i for i in range(10)])
        _rewrite_ego_positions(log_dir, scale=3.0)
        with pytest.raises(StaleModalityError):
            build_log_scene_index(log_dir, SceneFilter(future_num_iterations=2))

    def test_log_without_route(self, tmp_path: Path) -> None:
        log_dir = _write_log(tmp_path, xs=[0.0, 1.0], write_route=False)
        check_cache_source_modalities(log_dir)
        assert get_cache_source_info(parse_log_directory_metadata(log_dir).get(EGO_KEY)) is None

    def test_unreadable_log_skipped(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "broken"
        log_dir.mkdir()
        (log_dir / "sync.arrow").write_bytes(b"not arrow")
        check_cache_source_modalities(log_dir)
