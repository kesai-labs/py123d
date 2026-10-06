"""Tests for the route polyline: writer output, filter enforcement, scene API access."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pyarrow as pa
import pytest

from py123d.api.scene.arrow.arrow_log_writer import ArrowLogWriter, SyncConfig
from py123d.api.scene.arrow.arrow_scene_api import ArrowSceneAPI
from py123d.api.scene.arrow.lazy_scene_sequence import build_log_scene_index
from py123d.api.scene.arrow.utils.log_writer_config import LogWriterConfig
from py123d.api.scene.arrow.utils.route_utils import (
    ROUTE_POSITION_KEY,
    compute_route_data,
    compute_route_data_from_waypoints,
    interpolate_route_at_arc,
    project_onto_polyline,
    read_route_position,
)
from py123d.api.scene.arrow.utils.scene_builder_utils import filter_scene_metadata_candidates
from py123d.api.scene.scene_filter import SceneFilter
from py123d.datatypes import Timestamp
from py123d.datatypes.metadata.log_metadata import LogMetadata
from py123d.datatypes.vehicle_state.dynamic_state import DynamicStateSE3
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.geometry.pose import PoseSE3
from py123d.geometry.vector import Vector3D
from py123d.parser.base_dataset_parser import ModalitiesSync

from ..conftest import make_ego_metadata, make_log_metadata

TIMESTEP_US = 100_000


def _make_ego(ts_us: int, x: float, y: float = 0.0) -> EgoStateSE3:
    dynamic = DynamicStateSE3(
        velocity=Vector3D(1.0, 0.0, 0.0),
        acceleration=Vector3D(0.0, 0.0, 0.0),
        angular_velocity=Vector3D(0.0, 0.0, 0.0),
    )
    return EgoStateSE3.from_imu(
        imu_se3=PoseSE3.from_list([x, y, 0.0, 1.0, 0.0, 0.0, 0.0]),
        metadata=make_ego_metadata(),
        timestamp=Timestamp.from_us(ts_us),
        dynamic_state_se3=dynamic,
    )


def _write_log(
    tmp_path: Path,
    xs: List[float],
    config: Optional[LogWriterConfig] = None,
    route_xyz: Optional[np.ndarray] = None,
) -> tuple[Path, LogMetadata]:
    """Write a log driving along the x-axis through the given positions."""
    log_meta = make_log_metadata()
    writer = ArrowLogWriter(config or LogWriterConfig(), logs_root=tmp_path, sensors_root=tmp_path)
    writer.reset(log_meta)
    if route_xyz is not None:
        writer.set_route(route_xyz)
    for i, x in enumerate(xs):
        writer.write_sync(
            ModalitiesSync(timestamp=Timestamp.from_us(i * TIMESTEP_US), modalities=[_make_ego(i * TIMESTEP_US, x)])
        )
    writer.close()
    return tmp_path / log_meta.split / log_meta.log_name, log_meta


def _sync_table(log_dir: Path) -> pa.Table:
    return pa.ipc.open_file(str(log_dir / "sync.arrow")).read_all()


# ===========================================================================
# compute_route_data
# ===========================================================================


class TestComputeRouteData:
    def test_straight_drive(self):
        positions = np.stack([np.arange(11) * 2.0, np.zeros(11), np.zeros(11)], axis=1)
        route = compute_route_data(positions, resolution_m=1.0)
        assert route.total_arc_m == pytest.approx(20.0)
        np.testing.assert_allclose(route.progress_m, np.arange(11) * 2.0)
        np.testing.assert_allclose(route.polyline_arc_m, np.arange(21.0))
        np.testing.assert_allclose(route.polyline_xyz[:, 0], np.arange(21.0))

    def test_standstill_jitter_accumulates_nothing(self):
        rng = np.random.default_rng(0)
        drive = np.stack([np.arange(10) * 2.0, np.zeros(10), np.zeros(10)], axis=1)
        stand = np.tile([18.0, 0.0, 0.0], (10, 1)) + rng.uniform(-0.02, 0.02, size=(10, 3))
        route = compute_route_data(np.concatenate([drive, stand]), resolution_m=1.0)
        assert route.total_arc_m == pytest.approx(18.0, abs=0.1)
        assert np.all(np.diff(route.progress_m) >= 0.0)
        assert route.progress_m[-1] == route.progress_m[10]

    def test_pure_standstill_yields_single_vertex(self):
        route = compute_route_data(np.zeros((5, 3)), resolution_m=1.0)
        assert route.total_arc_m == 0.0
        assert len(route.polyline_arc_m) == 1
        np.testing.assert_allclose(route.progress_m, np.zeros(5))

    def test_empty_positions(self):
        assert compute_route_data(np.empty((0, 3)), resolution_m=1.0) is None

    def test_interpolation_recovers_positions(self):
        angles = np.linspace(0.0, np.pi, 200)
        positions = np.stack([50.0 * np.cos(angles), 50.0 * np.sin(angles), np.zeros(200)], axis=1)
        route = compute_route_data(positions, resolution_m=1.0)
        recovered = interpolate_route_at_arc(route.polyline_arc_m, route.polyline_xyz, route.progress_m)
        assert np.linalg.norm(recovered - positions, axis=1).max() < 0.05

    def test_low_frequency_odometry_on_tight_curve(self):
        # 2 Hz keyframes at ~15 m/s on an R=15 m curve: 7.5 m between samples, the worst
        # realistic sampling. Fidelity is bounded by the data: chords undershoot the true
        # arc ~1% and cut the corner by at most the sagitta (~0.5 m).
        angles = np.arange(0.0, np.pi, 0.5)
        positions = np.stack([15.0 * np.cos(angles), 15.0 * np.sin(angles), np.zeros_like(angles)], axis=1)
        route = compute_route_data(positions, resolution_m=1.0)
        true_arc = 15.0 * (angles[-1] - angles[0])
        assert route.total_arc_m == pytest.approx(true_arc, rel=0.02)
        radial_error = np.abs(np.linalg.norm(route.polyline_xyz[:, :2], axis=1) - 15.0)
        assert radial_error.max() < 0.5


# ===========================================================================
# compute_route_data_from_waypoints / project_onto_polyline
# ===========================================================================


class TestProvidedRoute:
    def test_progress_by_projection(self):
        route_xyz = np.stack([np.arange(0.0, 101.0, 5.0), np.zeros(21), np.zeros(21)], axis=1)
        ego = np.stack([np.arange(11) * 2.0, np.full(11, 0.5), np.zeros(11)], axis=1)  # 0.5 m beside the route
        route = compute_route_data_from_waypoints(route_xyz, ego, resolution_m=1.0)
        assert route.total_arc_m == pytest.approx(100.0)
        np.testing.assert_allclose(route.progress_m, np.arange(11) * 2.0, atol=1e-6)

    def test_progress_monotone_on_self_crossing_route(self):
        # Figure-eight-like: out along +x, back along the same line.
        out = np.stack([np.arange(0.0, 51.0), np.zeros(51), np.zeros(51)], axis=1)
        back = np.stack([np.arange(49.0, -1.0, -1.0), np.zeros(50), np.zeros(50)], axis=1)
        route_xyz = np.concatenate([out, back])
        ego = np.concatenate([out[::5], back[::5]])
        route = compute_route_data_from_waypoints(route_xyz, ego, resolution_m=1.0)
        assert route.total_arc_m == pytest.approx(100.0)
        assert np.all(np.diff(route.progress_m) >= 0.0)
        # Last ego sample sits at x=4 on the return leg: 50 out + (50 - 4) back.
        assert route.progress_m[-1] == pytest.approx(96.0)

    def test_progress_around_full_roundabout_loop(self):
        # Closed circle R=20: route start and end coincide spatially, the worst case for
        # projection. The monotone window must carry progress once around, not snap back.
        angles = np.linspace(0.0, 2.0 * np.pi, 721)
        route_xyz = np.stack([20.0 * np.cos(angles), 20.0 * np.sin(angles), np.zeros_like(angles)], axis=1)
        ego = route_xyz[::36] * 0.985  # drive slightly inside the lane, ~10 deg per frame
        route = compute_route_data_from_waypoints(route_xyz, ego, resolution_m=1.0)
        assert route.total_arc_m == pytest.approx(2.0 * np.pi * 20.0, abs=0.1)
        assert np.all(np.diff(route.progress_m) >= 0.0)
        assert route.progress_m[-1] == pytest.approx(route.total_arc_m, abs=1.0)

    def test_without_ego_positions(self):
        route_xyz = np.stack([np.arange(0.0, 11.0), np.zeros(11), np.zeros(11)], axis=1)
        route = compute_route_data_from_waypoints(route_xyz, None, resolution_m=1.0)
        assert route.progress_m is None
        assert route.total_arc_m == pytest.approx(10.0)

    def test_projection_clamps_before_route_start(self):
        polyline_arc = np.arange(11.0)
        polyline_xyz = np.stack([np.arange(11.0), np.zeros(11), np.zeros(11)], axis=1)
        progress = project_onto_polyline(polyline_arc, polyline_xyz, np.array([[-5.0, 0.0, 0.0], [3.5, 0.0, 0.0]]))
        assert progress[0] == pytest.approx(0.0)
        assert progress[1] == pytest.approx(3.5)


class TestJumpWarning:
    def _positions(self, xs):
        return np.stack([np.asarray(xs, dtype=np.float64), np.zeros(len(xs)), np.zeros(len(xs))], axis=1)

    def test_teleport_warns(self, caplog):
        import logging

        from py123d.api.scene.arrow.utils.route_utils import warn_on_position_jumps

        timestamps = np.arange(5) * 100_000
        with caplog.at_level(logging.WARNING):
            warn_on_position_jumps(self._positions([0.0, 1.0, 2.0, 102.0, 103.0]), timestamps, context="log_x")
        assert "position jump" in caplog.text and "log_x" in caplog.text

    def test_plausible_motion_does_not_warn(self, caplog):
        import logging

        from py123d.api.scene.arrow.utils.route_utils import warn_on_position_jumps

        timestamps = np.arange(5) * 100_000  # 3 m per 0.1 s = 30 m/s
        with caplog.at_level(logging.WARNING):
            warn_on_position_jumps(self._positions([0.0, 3.0, 6.0, 9.0, 12.0]), timestamps, context="log_x")
        assert caplog.text == ""


# ===========================================================================
# Writer round-trip
# ===========================================================================


class TestWriterRoute:
    def test_route_position_file_and_sync_index_column(self, tmp_path: Path):
        log_dir, _ = _write_log(tmp_path, xs=[2.0 * i for i in range(11)])
        # route_position is a regular modality: sync holds row indices into its file.
        assert _sync_table(log_dir)[ROUTE_POSITION_KEY].to_pylist() == list(range(11))
        route_metadata, progress, remaining = read_route_position(log_dir)
        assert progress.tolist() == [2.0 * i for i in range(11)]
        assert remaining.tolist() == [20.0 - 2.0 * i for i in range(11)]

        # The static polyline lives in the modality metadata.
        assert route_metadata.total_arc_m == pytest.approx(20.0)
        assert route_metadata.resolution_m == 1.0
        assert route_metadata.source == "ego_state_se3"
        assert len(route_metadata.polyline_arc_m) == 21
        np.testing.assert_allclose(route_metadata.polyline_xyz[:, 0], np.arange(21.0))

    def test_write_route_disabled(self, tmp_path: Path):
        log_dir, _ = _write_log(tmp_path, xs=[0.0, 2.0], config=LogWriterConfig(write_route=False))
        assert not (log_dir / "route_position.arrow").exists()
        assert ROUTE_POSITION_KEY not in _sync_table(log_dir).column_names

    def test_deferred_sync_discovers_route_position(self, tmp_path: Path):
        log_meta = make_log_metadata()
        writer = ArrowLogWriter(
            LogWriterConfig(),
            logs_root=tmp_path,
            sensors_root=tmp_path,
            sync_config=SyncConfig(reference_column="ego_state_se3.timestamp_us"),
        )
        writer.reset(log_meta)
        for i in range(6):
            writer.write_async(_make_ego(i * TIMESTEP_US, 3.0 * i))
        writer.close()
        log_dir = tmp_path / log_meta.split / log_meta.log_name
        assert _sync_table(log_dir)[ROUTE_POSITION_KEY].to_pylist() == list(range(6))
        assert read_route_position(log_dir)[1].tolist() == [3.0 * i for i in range(6)]

    def test_provided_route_overrides_odometry(self, tmp_path: Path):
        route_xyz = np.stack([np.arange(0.0, 101.0, 5.0), np.zeros(21), np.zeros(21)], axis=1)
        log_dir, _ = _write_log(tmp_path, xs=[2.0 * i for i in range(5)], route_xyz=route_xyz)
        route_metadata, progress, _ = read_route_position(log_dir)
        assert route_metadata.source == "provided"
        assert route_metadata.total_arc_m == pytest.approx(100.0)
        assert progress.tolist() == pytest.approx([2.0 * i for i in range(5)])

    def test_provided_route_written_even_when_write_route_disabled(self, tmp_path: Path):
        route_xyz = np.stack([np.arange(0.0, 11.0), np.zeros(11), np.zeros(11)], axis=1)
        log_dir, _ = _write_log(tmp_path, xs=[0.0, 1.0], config=LogWriterConfig(write_route=False), route_xyz=route_xyz)
        route_metadata, _, _ = read_route_position(log_dir)
        assert route_metadata.source == "provided"

    def test_route_metadata_msgpack_roundtrip(self, tmp_path: Path):
        log_dir, _ = _write_log(tmp_path, xs=[2.0 * i for i in range(11)])
        route_metadata = ArrowSceneAPI(log_dir).get_route()[0]
        recovered = type(route_metadata).from_dict(route_metadata.to_dict())
        np.testing.assert_allclose(recovered.polyline_xyz, route_metadata.polyline_xyz)
        assert recovered.total_arc_m == route_metadata.total_arc_m

    def test_ego_longer_than_sensor_clip(self, tmp_path: Path):
        """physical-ai-av shape: ego motion covers far more than the synced sensor clip.

        The sync table only spans the sensor reference (here: traffic lights over 1 s),
        but the route and remaining-route filtering must use the full ego motion.
        """
        from py123d.datatypes import TrafficLightDetection, TrafficLightDetections, TrafficLightStatus

        from ..conftest import make_traffic_light_metadata

        log_meta = make_log_metadata()
        writer = ArrowLogWriter(
            LogWriterConfig(),
            logs_root=tmp_path,
            sensors_root=tmp_path,
            sync_config=SyncConfig(reference_column="traffic_light_detections.timestamp_us"),
        )
        writer.reset(log_meta)
        for i in range(100):  # ego: 100 frames, 1 m apart -> 99 m route
            writer.write_async(_make_ego(i * TIMESTEP_US, float(i)))
        tl_meta = make_traffic_light_metadata()
        for i in range(10):  # sensor reference: only the first 10 frames
            writer.write_async(
                TrafficLightDetections(
                    detections=[TrafficLightDetection(lane_id=1, status=TrafficLightStatus.GREEN)],
                    timestamp=Timestamp.from_us(i * TIMESTEP_US),
                    metadata=tl_meta,
                )
            )
        writer.close()
        log_dir = tmp_path / log_meta.split / log_meta.log_name

        sync = _sync_table(log_dir)
        assert sync.num_rows == 10  # clip-sized sync, log-sized route:
        route_metadata, _, _ = read_route_position(log_dir)
        assert route_metadata.total_arc_m == pytest.approx(99.0)
        assert ArrowSceneAPI(log_dir).get_remaining_route_m(0) == pytest.approx(99.0)

        # Remaining route counts the ego motion beyond the clip, so a 50 m requirement
        # keeps every anchor even though synced progress only reaches 9 m.
        index = build_log_scene_index(log_dir, SceneFilter(future_num_iterations=2, min_remaining_route_m=50.0))
        assert index is not None and index.anchor_indices.tolist() == list(range(8))


# ===========================================================================
# SceneFilter.min_remaining_route_m
# ===========================================================================


class TestRouteFilter:
    def test_lazy_drops_anchors_short_of_route(self, tmp_path: Path):
        # 18 m total; anchors at progress 0,2,...; remaining >= 10 keeps progress <= 8.
        log_dir, _ = _write_log(tmp_path, xs=[2.0 * i for i in range(10)])
        index = build_log_scene_index(log_dir, SceneFilter(future_num_iterations=2, min_remaining_route_m=10.0))
        assert index.anchor_indices.tolist() == [0, 1, 2, 3, 4]

    def test_lazy_boundary_is_inclusive(self, tmp_path: Path):
        log_dir, _ = _write_log(tmp_path, xs=[2.0 * i for i in range(10)])
        index = build_log_scene_index(log_dir, SceneFilter(future_num_iterations=1, min_remaining_route_m=18.0))
        assert index.anchor_indices.tolist() == [0]

    def test_eager_matches_lazy(self, tmp_path: Path):
        from py123d.api.scene.arrow.utils.scene_builder_utils import generate_scene_metadatas

        log_dir, log_meta = _write_log(tmp_path, xs=[2.0 * i for i in range(10)])
        sync = _sync_table(log_dir)
        candidates = generate_scene_metadatas(
            sync, log_meta, future_iterations=2, history_iterations=0, iteration_duration_s=0.1
        )
        kept = filter_scene_metadata_candidates(candidates, SceneFilter(min_remaining_route_m=10.0), sync, log_dir)
        assert [scene.initial_idx for scene in kept] == [0, 1, 2, 3, 4]

    def test_standstill_log_dropped_for_positive_minimum(self, tmp_path: Path):
        log_dir, _ = _write_log(tmp_path, xs=[0.0] * 5)
        assert build_log_scene_index(log_dir, SceneFilter(future_num_iterations=1, min_remaining_route_m=1.0)) is None
        index = build_log_scene_index(log_dir, SceneFilter(future_num_iterations=1, min_remaining_route_m=0.0))
        assert index is not None  # remaining 0 >= 0

    def test_log_without_column_rejected(self, tmp_path: Path):
        log_dir, _ = _write_log(tmp_path, xs=[2.0 * i for i in range(5)], config=LogWriterConfig(write_route=False))
        assert build_log_scene_index(log_dir, SceneFilter(future_num_iterations=1, min_remaining_route_m=1.0)) is None
        assert build_log_scene_index(log_dir, SceneFilter(future_num_iterations=1)) is not None

    def test_negative_minimum_raises(self):
        with pytest.raises(ValueError, match="min_remaining_route_m must be >= 0"):
            SceneFilter(min_remaining_route_m=-1.0)


# ===========================================================================
# Scene API access
# ===========================================================================


class TestSceneApiRoute:
    def test_route_accessors(self, tmp_path: Path):
        log_dir, _ = _write_log(tmp_path, xs=[2.0 * i for i in range(11)])
        api = ArrowSceneAPI(log_dir)
        route_metadata, arc, xyz = api.get_route()
        assert route_metadata.total_arc_m == pytest.approx(20.0)
        assert len(arc) == len(xyz) == 21
        assert api.get_route_progress_at_iteration(5) == pytest.approx(10.0)
        assert api.get_remaining_route_m(0) == pytest.approx(20.0)
        assert api.get_remaining_route_m(5) == pytest.approx(10.0)

    def test_route_accessors_without_route(self, tmp_path: Path):
        log_dir, _ = _write_log(tmp_path, xs=[0.0, 2.0], config=LogWriterConfig(write_route=False))
        api = ArrowSceneAPI(log_dir)
        assert api.get_route() is None
        assert api.get_route_progress_at_iteration(0) is None
        assert api.get_remaining_route_m(0) is None
