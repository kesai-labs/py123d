from typing import Optional
from unittest.mock import Mock

import pytest

from py123d.api import MapAPI, SceneAPI, SceneMetadata
from py123d.datatypes import (
    BoxDetectionMetadata,
    BoxDetectionsSE3,
    EgoMetadata,
    EgoStateSE3,
    FisheyeMEICamera,
    FisheyeMEICameraID,
    FisheyeMEICameraMetadata,
    Lidar,
    LidarID,
    LidarMetadata,
    LogMetadata,
    MapMetadata,
    PinholeCamera,
    PinholeCameraID,
    PinholeCameraMetadata,
    Timestamp,
    TrafficLightDetections,
)
from py123d.datatypes.custom.custom_modality import CustomModality
from py123d.datatypes.metadata.sensor_metadata import (
    FisheyeMEICameraMetadatas,
    LidarMetadatas,
    PinholeCameraMetadatas,
)


class ConcreteSceneAPI(SceneAPI):
    """Concrete implementation for testing purposes."""

    def __init__(self):
        self._log_metadata = Mock(spec=LogMetadata)
        self._scene_metadata = Mock(spec=SceneMetadata)
        self._map_api = Mock(spec=MapAPI)
        self._map_metadata = None
        self._ego_metadata = None
        self._box_detection_metadata = None
        self._pinhole_camera_metadatas = None
        self._fisheye_mei_camera_metadatas = None
        self._lidar_metadatas = None

    def get_log_metadata(self) -> LogMetadata:
        """Inherited, see super class."""
        return self._log_metadata

    def get_scene_metadata(self) -> SceneMetadata:
        """Inherited, see super class."""
        return self._scene_metadata

    def get_map_api(self) -> Optional[MapAPI]:
        """Inherited, see super class."""
        return self._map_api

    def get_map_metadata(self) -> Optional[MapMetadata]:
        """Inherited, see super class."""
        return self._map_metadata

    def get_ego_metadata(self) -> Optional[EgoMetadata]:
        """Inherited, see super class."""
        return self._ego_metadata

    def get_box_detection_metadata(self) -> Optional[BoxDetectionMetadata]:
        """Inherited, see super class."""
        return self._box_detection_metadata

    def get_pinhole_camera_metadatas(self) -> Optional[PinholeCameraMetadatas]:
        """Inherited, see super class."""
        return self._pinhole_camera_metadatas

    def get_fisheye_mei_camera_metadatas(self) -> Optional[FisheyeMEICameraMetadatas]:
        """Inherited, see super class."""
        return self._fisheye_mei_camera_metadatas

    def get_lidar_metadatas(self) -> Optional[LidarMetadatas]:
        """Inherited, see super class."""
        return self._lidar_metadatas

    def get_timestamp_at_iteration(self, iteration: int) -> Timestamp:
        """Inherited, see super class."""
        return Mock(spec=Timestamp)

    def get_ego_state_se3_at_iteration(self, iteration: int) -> Optional[EgoStateSE3]:
        """Inherited, see super class."""
        return Mock(spec=EgoStateSE3)

    def get_box_detections_se3_at_iteration(self, iteration: int) -> Optional[BoxDetectionsSE3]:
        """Inherited, see super class."""
        return Mock(spec=BoxDetectionsSE3)

    def get_traffic_light_detections_at_iteration(self, iteration: int) -> Optional[TrafficLightDetections]:
        """Inherited, see super class."""
        return Mock(spec=TrafficLightDetections)

    def get_pinhole_camera_at_iteration(self, iteration: int, camera_id: PinholeCameraID) -> Optional[PinholeCamera]:
        """Inherited, see super class."""
        return Mock(spec=PinholeCamera)

    def get_fisheye_mei_camera_at_iteration(
        self, iteration: int, camera_id: FisheyeMEICameraID
    ) -> Optional[FisheyeMEICamera]:
        """Inherited, see super class."""
        return Mock(spec=FisheyeMEICamera)

    def get_lidar_at_iteration(self, iteration: int, lidar_id: LidarID) -> Optional[Lidar]:
        """Inherited, see super class."""
        return Mock(spec=Lidar)

    def get_custom_modality_at_iteration(self, iteration: int, name: str) -> Optional[CustomModality]:
        """Inherited, see super class."""
        return Mock(
            spec=CustomModality,
            data={"example_key": "example_value"},
            timestamp=Mock(spec=Timestamp),
        )


@pytest.fixture
def scene_api():
    """Fixture providing a concrete SceneAPI instance."""
    api = ConcreteSceneAPI()
    api._log_metadata.dataset = "test_dataset"
    api._log_metadata.split = "test_split"
    api._log_metadata.location = "test_location"
    api._log_metadata.log_name = "test_log"
    api._log_metadata.version = "1.0.0"
    api._map_metadata = Mock(spec=MapMetadata)
    api._ego_metadata = Mock(spec=EgoMetadata)

    pcam_meta = Mock(spec=PinholeCameraMetadata)
    pcam_meta.camera_name = "pcam_b0"
    api._pinhole_camera_metadatas = PinholeCameraMetadatas({PinholeCameraID.PCAM_B0: pcam_meta})

    fcam_meta = Mock(spec=FisheyeMEICameraMetadata)
    fcam_meta.camera_name = "fcam_l"
    api._fisheye_mei_camera_metadatas = FisheyeMEICameraMetadatas({FisheyeMEICameraID.FCAM_L: fcam_meta})

    lidar_meta = Mock(spec=LidarMetadata)
    lidar_meta.lidar_name = "lidar_top"
    api._lidar_metadatas = LidarMetadatas({LidarID.LIDAR_TOP: lidar_meta})

    api._scene_metadata.initial_uuid = "test-uuid-123"
    api._scene_metadata.number_of_iterations = 100
    api._scene_metadata.number_of_history_iterations = 10
    return api


class TestSceneAPIProperties:
    """Test property accessors of SceneAPI."""

    def test_log_metadata(self, scene_api):
        """Test log_metadata property."""
        assert scene_api.log_metadata == scene_api._log_metadata

    def test_scene_metadata(self, scene_api):
        """Test scene_metadata property."""
        assert scene_api.scene_metadata == scene_api._scene_metadata

    def test_map_metadata(self, scene_api):
        """Test map_metadata property."""
        assert scene_api.map_metadata == scene_api._map_metadata

    def test_map_api(self, scene_api):
        """Test map_api property."""
        assert scene_api.map_api == scene_api._map_api

    def test_dataset(self, scene_api):
        """Test dataset property."""
        assert scene_api.dataset == "test_dataset"

    def test_split(self, scene_api):
        """Test split property."""
        assert scene_api.split == "test_split"

    def test_location(self, scene_api):
        """Test location property."""
        assert scene_api.location == "test_location"

    def test_log_name(self, scene_api):
        """Test log_name property."""
        assert scene_api.log_name == "test_log"

    def test_version(self, scene_api):
        """Test version property."""
        assert scene_api.version == "1.0.0"

    def test_scene_uuid(self, scene_api):
        """Test scene_uuid property."""
        assert scene_api.scene_uuid == "test-uuid-123"

    def test_number_of_iterations(self, scene_api):
        """Test number_of_iterations property."""
        assert scene_api.number_of_iterations == 100

    def test_number_of_history_iterations(self, scene_api):
        """Test number_of_history_iterations property."""
        assert scene_api.number_of_history_iterations == 10

    def test_ego_metadata(self, scene_api):
        """Test ego_metadata property."""
        assert scene_api.ego_metadata == scene_api._ego_metadata

    def test_available_pinhole_camera_ids(self, scene_api):
        """Test available_pinhole_camera_ids property."""
        assert scene_api.available_pinhole_camera_ids == [PinholeCameraID.PCAM_B0]

    def test_available_pinhole_camera_names(self, scene_api):
        """Test available_pinhole_camera_names property."""
        assert scene_api.available_pinhole_camera_names == ["pcam_b0"]

    def test_available_fisheye_mei_camera_ids(self, scene_api):
        """Test available_fisheye_mei_camera_ids property."""
        assert scene_api.available_fisheye_mei_camera_ids == [FisheyeMEICameraID.FCAM_L]

    def test_available_fisheye_mei_camera_names(self, scene_api):
        """Test available_fisheye_mei_camera_names property."""
        assert scene_api.available_fisheye_mei_camera_names == ["fcam_l"]

    def test_available_lidar_ids(self, scene_api):
        """Test available_lidar_ids property."""
        assert scene_api.available_lidar_ids == [LidarID.LIDAR_TOP]

    def test_available_lidar_names(self, scene_api):
        """Test available_lidar_names property."""
        assert scene_api.available_lidar_names == ["lidar_top"]

    def test_available_ids_empty_when_metadatas_none(self):
        """Test that available_*_ids/names return empty lists when metadatas are None."""
        api = ConcreteSceneAPI()
        assert api.available_pinhole_camera_ids == []
        assert api.available_pinhole_camera_names == []
        assert api.available_fisheye_mei_camera_ids == []
        assert api.available_fisheye_mei_camera_names == []
        assert api.available_lidar_ids == []
        assert api.available_lidar_names == []


class TestSceneAPIMethods:
    """Test abstract method implementations."""

    def test_get_timestamp_at_iteration(self, scene_api):
        """Test get_timestamp_at_iteration method."""
        result = scene_api.get_timestamp_at_iteration(0)
        assert isinstance(result, Mock)

    def test_get_ego_state_se3_at_iteration(self, scene_api):
        """Test get_ego_state_se3_at_iteration method."""
        result = scene_api.get_ego_state_se3_at_iteration(0)
        assert result is not None

    def test_get_box_detections_se3_at_iteration(self, scene_api):
        """Test get_box_detections_se3_at_iteration method."""
        result = scene_api.get_box_detections_se3_at_iteration(0)
        assert result is not None

    def test_get_traffic_light_detections_at_iteration(self, scene_api):
        """Test get_traffic_light_detections_at_iteration method."""
        result = scene_api.get_traffic_light_detections_at_iteration(0)
        assert result is not None

    def test_get_pinhole_camera_at_iteration(self, scene_api):
        """Test get_pinhole_camera_at_iteration method."""
        result = scene_api.get_pinhole_camera_at_iteration(0, PinholeCameraID.PCAM_B0)
        assert result is not None

    def test_get_fisheye_mei_camera_at_iteration(self, scene_api):
        """Test get_fisheye_mei_camera_at_iteration method."""
        result = scene_api.get_fisheye_mei_camera_at_iteration(0, FisheyeMEICameraID.FCAM_L)
        assert result is not None

    def test_get_lidar_at_iteration(self, scene_api):
        """Test get_lidar_at_iteration method."""
        result = scene_api.get_lidar_at_iteration(0, LidarID.LIDAR_TOP)
        assert result is not None

    def test_get_custom_modality_at_iteration(self, scene_api):
        """Test get_custom_modality_at_iteration method."""
        result = scene_api.get_custom_modality_at_iteration(0, "route")
        assert result is not None
        assert result.data == {"example_key": "example_value"}


class TestSceneAPIDeprecatedMethods:
    """Test deprecated method wrappers still work."""

    def test_get_ego_state_at_iteration(self, scene_api):
        """Test deprecated get_ego_state_at_iteration delegates to get_ego_state_se3_at_iteration."""
        result = scene_api.get_ego_state_at_iteration(0)
        assert result is not None

    def test_get_box_detections_at_iteration(self, scene_api):
        """Test deprecated get_box_detections_at_iteration delegates to get_box_detections_se3_at_iteration."""
        result = scene_api.get_box_detections_at_iteration(0)
        assert result is not None

    def test_get_route_lane_group_ids(self, scene_api):
        """Test deprecated get_route_lane_group_ids returns None by default."""
        result = scene_api.get_route_lane_group_ids(0)
        assert result is None
