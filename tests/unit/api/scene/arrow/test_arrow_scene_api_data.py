"""Tests for ArrowSceneAPI data access methods and SceneAPI convenience methods."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from py123d.api.scene.arrow.arrow_scene_api import ArrowSceneAPI
from py123d.api.scene.scene_api import checked_optional_cast
from py123d.datatypes import (
    BoxDetectionsSE3,
    EgoStateSE3,
    ModalityType,
    Timestamp,
    TrafficLightDetections,
)
from py123d.datatypes.metadata import SceneMetadata
from py123d.datatypes.vehicle_state.ego_state_metadata import EgoStateSE3Metadata

from ..conftest import (
    make_custom_modality_metadata,
    make_ego_metadata,
    make_log_metadata,
    make_populated_log_dir,
    write_custom_modality_arrow,
    write_sync_arrow,
)


def _make_scene_metadata(**kwargs) -> SceneMetadata:
    defaults = dict(
        dataset="test-dataset",
        split="test-dataset_train",
        initial_uuid="00000000-0000-0000-0000-000000000001",
        initial_idx=0,
        num_future_iterations=9,
        num_history_iterations=0,
        future_duration_s=0.9,
        history_duration_s=0.0,
        iteration_duration_s=0.1,
        target_iteration_stride=1,
    )
    defaults.update(kwargs)
    return SceneMetadata(**defaults)


@pytest.fixture
def populated_log(tmp_path: Path):
    """Create a populated log directory with sync + ego + box_detections + traffic_lights."""
    log_dir = tmp_path / "test-dataset_train" / "log_001"
    metadatas = make_populated_log_dir(log_dir, num_rows=10, timestep_us=100_000)
    scene_meta = _make_scene_metadata()
    return log_dir, scene_meta, metadatas


# ===========================================================================
# checked_optional_cast
# ===========================================================================


class TestCheckedOptionalCast:
    def test_none_returns_none(self):
        assert checked_optional_cast(None, EgoStateSE3) is None

    def test_correct_type(self):
        meta = make_ego_metadata()
        result = checked_optional_cast(meta, EgoStateSE3Metadata)
        assert result is meta

    def test_wrong_type_raises(self):
        meta = make_ego_metadata()
        with pytest.raises(TypeError, match="Expected object of type"):
            checked_optional_cast(meta, EgoStateSE3)

    def test_subclass_passes(self):
        """isinstance handles subclasses, so a subclass should pass."""
        # EgoStateSE3Metadata is itself the class, use it directly
        meta = make_ego_metadata()
        # Check against the base class (BaseModalityMetadata) — subclass should pass
        from py123d.datatypes.modalities.base_modality import BaseModalityMetadata

        result = checked_optional_cast(meta, BaseModalityMetadata)
        assert result is meta


# ===========================================================================
# ArrowSceneAPI data access
# ===========================================================================


class TestGetModalityAtIteration:
    def test_ego_at_iteration_0(self, populated_log):
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        result = api.get_modality_at_iteration(0, ModalityType.EGO_STATE_SE3)
        assert result is not None
        assert isinstance(result, EgoStateSE3)
        assert result.timestamp.time_us == 0

    def test_ego_at_last_iteration(self, populated_log):
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        result = api.get_modality_at_iteration(9, ModalityType.EGO_STATE_SE3)
        assert result is not None
        assert result.timestamp.time_us == 9 * 100_000

    def test_negative_iteration_with_history(self, tmp_path: Path):
        """Negative iteration should access history frames."""
        log_dir = tmp_path / "test-dataset_train" / "log_001"
        make_populated_log_dir(log_dir, num_rows=10, timestep_us=100_000)
        scene_meta = _make_scene_metadata(initial_idx=2, num_future_iterations=7, num_history_iterations=2)
        api = ArrowSceneAPI(log_dir, scene_meta)
        result = api.get_modality_at_iteration(-1, ModalityType.EGO_STATE_SE3)
        assert result is not None
        # iteration -1 -> sync_index = 2 + (-1)*1 = 1 -> timestamp = 100_000
        assert result.timestamp.time_us == 100_000

    def test_missing_modality_returns_none(self, populated_log):
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        # LIDAR is not in the sync table
        result = api.get_modality_at_iteration(0, ModalityType.LIDAR)
        assert result is None

    def test_null_sync_entry_returns_none(self, tmp_path: Path):
        """When sync table has None at the requested index, return None."""
        log_dir = tmp_path / "test-dataset_train" / "log_001"
        log_dir.mkdir(parents=True)
        log_meta = make_log_metadata()
        ego_meta = make_ego_metadata()

        from ..conftest import write_ego_arrow

        write_ego_arrow(log_dir, num_rows=5, timestep_us=100_000, metadata=ego_meta)
        ego_key = ego_meta.modality_key
        write_sync_arrow(
            log_dir,
            num_rows=5,
            timestep_us=100_000,
            log_metadata=log_meta,
            modality_columns={ego_key: [None, 1, 2, 3, 4]},
        )
        scene_meta = _make_scene_metadata(num_future_iterations=4)
        api = ArrowSceneAPI(log_dir, scene_meta)
        result = api.get_modality_at_iteration(0, ModalityType.EGO_STATE_SE3)
        assert result is None

    def test_out_of_bounds_asserts(self, populated_log):
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        with pytest.raises(AssertionError, match="Iteration out of bounds"):
            api.get_modality_at_iteration(100, ModalityType.EGO_STATE_SE3)


class TestGetModalityAtTimestamp:
    def test_exact_match(self, populated_log):
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        result = api.get_modality_at_timestamp(Timestamp.from_us(300_000), ModalityType.EGO_STATE_SE3, criteria="exact")
        assert result is not None
        assert result.timestamp.time_us == 300_000

    def test_int_timestamp_treated_as_microseconds(self, populated_log):
        """W2: docstring says 'integer nanoseconds' but code calls Timestamp.from_us(timestamp).

        Passing 300000 (as microseconds) should find the frame at 0.3s.
        """
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        result = api.get_modality_at_timestamp(300_000, ModalityType.EGO_STATE_SE3, criteria="exact")
        assert result is not None
        assert result.timestamp.time_us == 300_000

    def test_nearest_match(self, populated_log):
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        result = api.get_modality_at_timestamp(
            Timestamp.from_us(310_000), ModalityType.EGO_STATE_SE3, criteria="nearest"
        )
        assert result is not None
        assert result.timestamp.time_us == 300_000

    def test_nonexistent_modality_table_returns_none(self, populated_log):
        """Modality not in log at all."""
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        result = api.get_modality_at_timestamp(Timestamp.from_us(0), ModalityType.LIDAR, criteria="exact")
        assert result is None


class TestGetModalityColumnAtIteration:
    def test_returns_raw_value(self, populated_log):
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        result = api.get_modality_column_at_iteration(0, "timestamp_us", ModalityType.EGO_STATE_SE3)
        assert result == 0
        assert isinstance(result, int)

    def test_returns_deserialized_value(self, populated_log):
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        result = api.get_modality_column_at_iteration(0, "timestamp_us", ModalityType.EGO_STATE_SE3, deserialize=True)
        assert isinstance(result, Timestamp)

    def test_returns_column_not_modality(self, populated_log):
        """W3: The variable is named `modality` but it returns a column value, not BaseModality."""
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        result = api.get_modality_column_at_iteration(0, "timestamp_us", ModalityType.EGO_STATE_SE3)
        # Should be an int (the raw column value), NOT a BaseModality
        assert not isinstance(result, EgoStateSE3)
        assert isinstance(result, int)

    def test_missing_modality_returns_none(self, populated_log):
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        result = api.get_modality_column_at_iteration(0, "timestamp_us", ModalityType.LIDAR)
        assert result is None


# ===========================================================================
# Convenience methods (typed wrappers)
# ===========================================================================


class TestConvenienceMethods:
    def test_ego_at_iteration(self, populated_log):
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        result = api.get_ego_state_se3_at_iteration(0)
        assert isinstance(result, EgoStateSE3)

    def test_ego_at_timestamp(self, populated_log):
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        result = api.get_ego_state_se3_at_timestamp(Timestamp.from_us(0), criteria="exact")
        assert isinstance(result, EgoStateSE3)

    def test_ego_metadata(self, populated_log):
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        result = api.get_ego_state_se3_metadata()
        assert isinstance(result, EgoStateSE3Metadata)

    def test_box_detections_at_iteration(self, populated_log):
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        result = api.get_box_detections_se3_at_iteration(0)
        assert isinstance(result, BoxDetectionsSE3)
        assert len(result.box_detections) == 2  # num_boxes_per_frame default

    def test_traffic_light_at_iteration(self, populated_log):
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        result = api.get_traffic_light_detections_at_iteration(0)
        assert isinstance(result, TrafficLightDetections)

    def test_custom_modality_at_iteration(self, tmp_path: Path):
        """Custom modality roundtrip through full API."""
        log_dir = tmp_path / "test-dataset_train" / "log_001"
        log_dir.mkdir(parents=True)
        log_meta = make_log_metadata()
        custom_meta = make_custom_modality_metadata("route")
        custom_key = custom_meta.modality_key
        num_rows = 5
        timestep_us = 100_000

        write_custom_modality_arrow(log_dir, num_rows, timestep_us, custom_meta)
        write_sync_arrow(
            log_dir,
            num_rows,
            timestep_us,
            log_meta,
            modality_columns={custom_key: list(range(num_rows))},
        )
        scene_meta = _make_scene_metadata(num_future_iterations=4)
        api = ArrowSceneAPI(log_dir, scene_meta)
        result = api.get_custom_modality_at_iteration(0, "route")
        assert result is not None
        assert result.data["frame"] == 0


# ===========================================================================
# Properties
# ===========================================================================


class TestSceneAPIProperties:
    def test_dataset(self, populated_log):
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        assert api.dataset == "test-dataset"

    def test_split(self, populated_log):
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        assert api.split == "test-dataset_train"

    def test_location(self, populated_log):
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        assert api.location == "boston"

    def test_log_name(self, populated_log):
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        assert api.log_name == "log_001"

    def test_scene_uuid(self, populated_log):
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        assert api.scene_uuid == scene_meta.initial_uuid

    def test_number_of_iterations(self, populated_log):
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        assert api.number_of_iterations == 10  # 9 future + 1 current

    def test_number_of_history_iterations(self, populated_log):
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        assert api.number_of_history_iterations == 0

    def test_available_camera_ids_empty(self, populated_log):
        """No cameras in the test log."""
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        assert api.available_camera_ids == []

    def test_available_camera_names_empty(self, populated_log):
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        assert api.available_camera_names == []


# ===========================================================================
# Pickling
# ===========================================================================


class TestPickling:
    def test_roundtrip(self, populated_log):
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        data = pickle.dumps(api)
        restored = pickle.loads(data)
        assert isinstance(restored, ArrowSceneAPI)
        assert restored.dataset == "test-dataset"

    def test_preserves_metadata(self, populated_log):
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        data = pickle.dumps(api)
        restored = pickle.loads(data)
        assert restored.scene_metadata == scene_meta


# ===========================================================================
# Lazy metadata loading
# ===========================================================================


class TestLazyMetadataLoading:
    def test_loads_when_none(self, populated_log):
        """When scene_metadata=None, get_scene_metadata() should load from sync.arrow."""
        log_dir, _, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_metadata=None)
        meta = api.get_scene_metadata()
        assert meta is not None
        assert meta.num_future_iterations == 9  # 10 rows - 1

    def test_cached_when_set(self, populated_log):
        """When scene_metadata is provided, it should be returned directly."""
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        assert api.get_scene_metadata() is scene_meta


# ===========================================================================
# Timestamp methods
# ===========================================================================


class TestGetAllIterationTimestamps:
    def test_basic(self, populated_log):
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        timestamps = api.get_all_iteration_timestamps()
        assert len(timestamps) == 10
        assert timestamps[0].time_us == 0
        assert timestamps[-1].time_us == 9 * 100_000

    def test_with_history(self, tmp_path: Path):
        log_dir = tmp_path / "test-dataset_train" / "log_001"
        make_populated_log_dir(log_dir, num_rows=10, timestep_us=100_000)
        scene_meta = _make_scene_metadata(initial_idx=2, num_future_iterations=7, num_history_iterations=2)
        api = ArrowSceneAPI(log_dir, scene_meta)
        timestamps = api.get_all_iteration_timestamps(include_history=True)
        assert len(timestamps) == 10  # 2 history + 1 current + 7 future


class TestGetSceneTimestampBoundaries:
    def test_basic(self, populated_log):
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        first, last = api.get_scene_timestamp_boundaries()
        assert first.time_us == 0
        assert last.time_us == 9 * 100_000


class TestGetAllModalityTimestamps:
    def test_ego_timestamps(self, populated_log):
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        timestamps = api.get_all_ego_state_se3_timestamps()
        assert len(timestamps) > 0

    def test_missing_modality_empty(self, populated_log):
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        timestamps = api.get_all_modality_timestamps(ModalityType.LIDAR)
        assert timestamps == []


# ===========================================================================
# Camera & Lidar convenience methods (require sensor data in log)
# ===========================================================================


class TestCameraConvenienceMethods:
    """Test camera convenience methods with a log containing JPEG binary camera data."""

    @pytest.fixture
    def camera_log(self, tmp_path: Path):
        import numpy as np

        from py123d.api.scene.arrow.modalities.arrow_camera import ArrowCameraWriter
        from py123d.api.scene.arrow.modalities.arrow_lidar import ArrowLidarWriter
        from py123d.datatypes.sensors.base_camera import Camera, CameraID
        from py123d.datatypes.sensors.lidar import Lidar, LidarFeature, LidarID, LidarMergedMetadata, LidarMetadata
        from py123d.datatypes.sensors.pinhole_camera import PinholeCameraMetadata, PinholeIntrinsics
        from py123d.geometry.pose import PoseSE3

        log_dir = tmp_path / "test-dataset_train" / "log_001"
        log_dir.mkdir(parents=True)

        log_meta = make_log_metadata()
        ego_meta = make_ego_metadata()
        num_rows = 5
        timestep_us = 100_000

        # Camera metadata
        cam_meta = PinholeCameraMetadata(
            camera_name="front",
            camera_id=CameraID.PCAM_F0,
            intrinsics=PinholeIntrinsics(fx=500.0, fy=500.0, cx=160.0, cy=120.0),
            distortion=None,
            width=320,
            height=240,
            camera_to_imu_se3=PoseSE3.identity(),
        )
        cam_key = cam_meta.modality_key

        # Lidar metadata (merged)
        lidar_single = LidarMetadata(lidar_name="top", lidar_id=LidarID.LIDAR_TOP)
        lidar_meta = LidarMergedMetadata(lidar_metadata_dict={LidarID.LIDAR_TOP: lidar_single})
        lidar_key = lidar_meta.modality_key

        # Write ego
        from ..conftest import write_ego_arrow

        write_ego_arrow(log_dir, num_rows, timestep_us, ego_meta)

        # Write camera (JPEG binary)
        cam_writer = ArrowCameraWriter(log_dir, cam_meta, camera_codec="jpeg_binary")
        rng = np.random.RandomState(42)
        for i in range(num_rows):
            img = rng.randint(0, 256, (240, 320, 3), dtype=np.uint8)
            cam = Camera(
                metadata=cam_meta,
                image=img,
                camera_to_global_se3=PoseSE3.identity(),
                timestamp=Timestamp.from_us(i * timestep_us),
            )
            cam_writer.write_modality(cam)
        cam_writer.close()

        # Write lidar (IPC binary)
        lidar_writer = ArrowLidarWriter(
            log_dir,
            lidar_meta,
            log_meta,
            lidar_store_option="binary",
            lidar_codec="ipc",
        )
        for i in range(num_rows):
            xyz = rng.randn(50, 3).astype(np.float32)
            features = {LidarFeature.INTENSITY.serialize(): rng.rand(50).astype(np.float32)}
            lidar = Lidar(
                timestamp=Timestamp.from_us(i * timestep_us),
                timestamp_end=Timestamp.from_us(i * timestep_us + 50_000),
                metadata=lidar_meta,
                point_cloud_3d=xyz,
                point_cloud_features=features,
            )
            lidar_writer.write_modality(lidar)
        lidar_writer.close()

        # Write sync
        modality_columns = {
            ego_meta.modality_key: list(range(num_rows)),
            cam_key: list(range(num_rows)),
            lidar_key: list(range(num_rows)),
        }
        write_sync_arrow(log_dir, num_rows, timestep_us, log_meta, modality_columns)

        scene_meta = _make_scene_metadata(num_future_iterations=4)
        return log_dir, scene_meta, cam_meta, lidar_meta

    def test_get_camera_at_iteration(self, camera_log):
        from py123d.datatypes.sensors.base_camera import Camera

        log_dir, scene_meta, cam_meta, _ = camera_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        result = api.get_camera_at_iteration(0, cam_meta.camera_id)
        assert isinstance(result, Camera)
        assert result.image.shape == (240, 320, 3)

    def test_get_camera_at_timestamp(self, camera_log):
        from py123d.datatypes.sensors.base_camera import Camera

        log_dir, scene_meta, cam_meta, _ = camera_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        result = api.get_camera_at_timestamp(Timestamp.from_us(0), cam_meta.camera_id, criteria="exact")
        assert isinstance(result, Camera)

    def test_get_camera_metadatas(self, camera_log):
        log_dir, scene_meta, cam_meta, _ = camera_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        metas = api.get_camera_metadatas()
        assert len(metas) == 1
        assert cam_meta.camera_id in metas

    def test_available_camera_ids(self, camera_log):
        log_dir, scene_meta, cam_meta, _ = camera_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        assert cam_meta.camera_id in api.available_camera_ids

    def test_available_camera_names(self, camera_log):
        log_dir, scene_meta, _, _ = camera_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        assert "front" in api.available_camera_names

    def test_get_all_camera_timestamps(self, camera_log):
        log_dir, scene_meta, cam_meta, _ = camera_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        ts = api.get_all_camera_timestamps(cam_meta.camera_id)
        assert len(ts) > 0

    def test_get_lidar_at_iteration(self, camera_log):
        from py123d.datatypes.sensors.lidar import Lidar, LidarID

        log_dir, scene_meta, _, lidar_meta = camera_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        result = api.get_lidar_at_iteration(0, LidarID.LIDAR_MERGED)
        assert isinstance(result, Lidar)
        assert result.point_cloud_3d.shape[1] == 3

    def test_get_lidar_at_timestamp(self, camera_log):
        from py123d.datatypes.sensors.lidar import Lidar, LidarID

        log_dir, scene_meta, _, _ = camera_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        result = api.get_lidar_at_timestamp(Timestamp.from_us(0), LidarID.LIDAR_MERGED, criteria="exact")
        assert isinstance(result, Lidar)

    def test_get_lidar_metadatas(self, camera_log):
        from py123d.datatypes.sensors.lidar import LidarID

        log_dir, scene_meta, _, _ = camera_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        metas = api.get_lidar_metadatas()
        assert LidarID.LIDAR_TOP in metas

    def test_available_lidar_ids(self, camera_log):
        from py123d.datatypes.sensors.lidar import LidarID

        log_dir, scene_meta, _, _ = camera_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        lidar_ids = api.available_lidar_ids
        assert LidarID.LIDAR_TOP in lidar_ids
        assert LidarID.LIDAR_MERGED in lidar_ids

    def test_available_lidar_names(self, camera_log):
        log_dir, scene_meta, _, _ = camera_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        names = api.available_lidar_names
        assert "top" in names

    def test_get_all_lidar_timestamps(self, camera_log):
        from py123d.datatypes.sensors.lidar import LidarID

        log_dir, scene_meta, _, _ = camera_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        ts = api.get_all_lidar_timestamps(LidarID.LIDAR_MERGED)
        assert len(ts) > 0


# ===========================================================================
# Map
# ===========================================================================


class TestMapAccess:
    def test_map_metadata_none(self, populated_log):
        log_dir, scene_meta, _ = populated_log
        api = ArrowSceneAPI(log_dir, scene_meta)
        assert api.get_map_metadata() is None
