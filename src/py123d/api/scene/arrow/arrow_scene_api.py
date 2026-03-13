from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

from py123d.api.map.arrow.arrow_map_api import get_lru_cached_map_api
from py123d.api.map.map_api import MapAPI
from py123d.api.scene.arrow.modalities.arrow_box_detections_se3 import ArrowBoxDetectionsSE3Reader
from py123d.api.scene.arrow.modalities.arrow_camera import ArrowCameraReader
from py123d.api.scene.arrow.modalities.arrow_custom_modality import ArrowCustomModalityReader
from py123d.api.scene.arrow.modalities.arrow_ego_state_se3 import ArrowEgoStateSE3Reader
from py123d.api.scene.arrow.modalities.arrow_lidar import ArrowLidarReader
from py123d.api.scene.arrow.modalities.arrow_sync import get_timestamp_from_arrow_table
from py123d.api.scene.arrow.modalities.arrow_traffic_light_detections import ArrowTrafficLightDetectionsReader
from py123d.api.scene.arrow.modalities.sync_utils import get_sync_table
from py123d.api.scene.arrow.utils.arrow_scene_caches import (
    _get_complete_log_scene_metadata,
    _get_lru_cached_log_metadata,
)
from py123d.api.scene.scene_api import SceneAPI
from py123d.api.scene.scene_metadata import SceneMetadata
from py123d.common.dataset_paths import get_dataset_paths
from py123d.common.utils.enums import SerialIntEnum
from py123d.datatypes import (
    BoxDetectionsSE3,
    BoxDetectionsSE3Metadata,
    CustomModality,
    EgoStateSE3,
    EgoStateSE3Metadata,
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
from py123d.datatypes.custom.custom_modality import CustomModalityMetadata
from py123d.datatypes.detections.traffic_light_detections import TrafficLightDetectionsMetadata
from py123d.datatypes.modalities.base_modality import BaseModalityMetadata


class ArrowSceneAPI(SceneAPI):
    """Scene API for Arrow-based scenes. Loads each modality from a separate Arrow file in a log directory."""

    __slots__ = ("_log_dir", "_scene_metadata")

    def __init__(
        self,
        log_dir: Union[Path, str],
        scene_metadata: Optional[SceneMetadata] = None,
    ) -> None:
        """Initializes the :class:`ArrowSceneAPI`.

        :param log_dir: Path to the log directory containing per-modality Arrow files.
        :param scene_metadata: Scene metadata, defaults to None
        """
        self._log_dir: Path = Path(log_dir)
        self._scene_metadata: Optional[SceneMetadata] = scene_metadata

    def __reduce__(self):
        """Helper for pickling the object."""
        return (self.__class__, (self._log_dir, self._scene_metadata))

    # ------------------------------------------------------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------------------------------------------------------

    def _get_table_index(self, iteration: int) -> int:
        """Resolve an iteration (which may be negative for history) to an absolute table index."""
        assert -self.number_of_history_iterations <= iteration < self.number_of_iterations, "Iteration out of bounds"
        return self.get_scene_metadata().initial_idx + iteration

    # ------------------------------------------------------------------------------------------------------------------
    # 1. Scene / Log Metadata
    # ------------------------------------------------------------------------------------------------------------------

    def get_scene_metadata(self) -> SceneMetadata:
        """Inherited, see superclass."""
        if self._scene_metadata is None:
            log_metadata = self.get_log_metadata()
            self._scene_metadata = _get_complete_log_scene_metadata(self._log_dir, log_metadata)
        return self._scene_metadata

    def get_log_metadata(self) -> LogMetadata:
        """Inherited, see superclass."""
        return _get_lru_cached_log_metadata(self._log_dir)

    def get_timestamp_at_iteration(self, iteration: int) -> Timestamp:
        """Inherited, see superclass."""
        sync_table = get_sync_table(self._log_dir)
        return get_timestamp_from_arrow_table(sync_table, self._get_table_index(iteration))

    def get_all_iteration_timestamps(self) -> List[Timestamp]:
        """Inherited, see superclass."""
        sync_table = get_sync_table(self._log_dir)
        scene_metadata = self.get_scene_metadata()
        ts_column = sync_table["sync.timestamp_us"].to_numpy()
        return [Timestamp.from_us(ts_column[i]) for i in range(scene_metadata.initial_idx, scene_metadata.end_idx)]

    # ------------------------------------------------------------------------------------------------------------------
    # 2. Map
    # ------------------------------------------------------------------------------------------------------------------

    def get_map_metadata(self) -> Optional[MapMetadata]:
        """Inherited, see superclass."""
        return self.get_log_metadata().map_metadata

    def get_map_api(self) -> Optional[MapAPI]:
        """Inherited, see superclass."""
        map_file = self._resolve_map_file()
        if map_file is not None:
            return get_lru_cached_map_api(map_file)
        return None

    def _resolve_map_file(self) -> Optional[Path]:
        """Find the map file: first check per-log, then global maps directory."""
        # 1. Per-log map
        map_file = self._log_dir / "map.arrow"
        if map_file.exists():
            return map_file
        # 2. Global map
        log_metadata = self.get_log_metadata()
        dataset, location = log_metadata.dataset, log_metadata.location
        if dataset is not None and location is not None:
            maps_root = get_dataset_paths().py123d_maps_root
            if maps_root is not None:
                map_file = maps_root / dataset / f"{dataset}_{location}.arrow"
                if map_file.exists():
                    return map_file
        return None

    # ------------------------------------------------------------------------------------------------------------------
    # 4. General modality access
    # ------------------------------------------------------------------------------------------------------------------

    def get_all_modality_metadatas(self) -> Optional[BaseModalityMetadata]:
        pass

    def get_modality_metadata(
        self,
        modality_type: str,
        modality_id: Optional[Union[int, str, SerialIntEnum]] = None,
    ) -> Optional[BaseModalityMetadata]:
        pass

    def get_modality_timestamps(
        self,
        modality_type: str,
        modality_id: Optional[Union[int, str, SerialIntEnum]] = None,
    ) -> Optional[List[Timestamp]]:
        pass

    def get_modality_at_iteration(
        self,
        iteration: int,
        modality_type: str,
        modality_id: Optional[Union[int, str, SerialIntEnum]] = None,
    ) -> Optional[CustomModality]:
        pass

    def get_modality_at_timestamp(
        self,
        timestamp: Timestamp,
        modality_type: str,
        modality_id: Optional[Union[int, str, SerialIntEnum]] = None,
        criteria: Literal["exact", "forward", "backward"] = "exact",
    ) -> Optional[CustomModality]:
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # 3. EgoStateSE3
    # ------------------------------------------------------------------------------------------------------------------

    def get_ego_state_se3_metadata(self) -> Optional[EgoStateSE3Metadata]:
        """Inherited, see superclass."""
        return self.get_modality_metadata("ego_state_se3")

    def get_all_ego_state_se3_timestamps(self) -> List[Timestamp]:
        """Inherited, see superclass."""
        return ArrowEgoStateSE3Reader.read_all_timestamps(
            self._log_dir, get_sync_table(self._log_dir), self.get_scene_metadata()
        )

    def get_ego_state_se3_at_iteration(self, iteration: int) -> Optional[EgoStateSE3]:
        """Inherited, see superclass."""
        return ArrowEgoStateSE3Reader.read_at_iteration(
            self._log_dir,
            get_sync_table(self._log_dir),
            self._get_table_index(iteration),
            self.get_ego_state_se3_metadata(),
        )

    def get_ego_state_se3_at_timestamp(self, timestamp: Timestamp) -> Optional[EgoStateSE3]:
        """Get the ego state at a specific timestamp."""
        return ArrowEgoStateSE3Reader.read_at_timestamp(
            self._log_dir, get_sync_table(self._log_dir), timestamp, self.get_ego_state_se3_metadata()
        )

    # ------------------------------------------------------------------------------------------------------------------
    # 4. BoxDetectionsSE3
    # ------------------------------------------------------------------------------------------------------------------

    def get_box_detections_se3_metadata(self) -> Optional[BoxDetectionsSE3Metadata]:
        """Inherited, see superclass."""
        return self.get_log_metadata().box_detections_se3_metadata

    def get_all_box_detections_se3_timestamps(self) -> List[Timestamp]:
        """Inherited, see superclass."""
        return ArrowBoxDetectionsSE3Reader.read_all_timestamps(
            self._log_dir, get_sync_table(self._log_dir), self.get_scene_metadata()
        )

    def get_box_detections_se3_at_iteration(self, iteration: int) -> Optional[BoxDetectionsSE3]:
        """Inherited, see superclass."""
        return ArrowBoxDetectionsSE3Reader.read_at_iteration(
            self._log_dir,
            get_sync_table(self._log_dir),
            self._get_table_index(iteration),
            self.get_box_detections_se3_metadata(),
        )

    # ------------------------------------------------------------------------------------------------------------------
    # 5. Traffic Light Detections
    # ------------------------------------------------------------------------------------------------------------------

    def get_traffic_light_detections_at_iteration(self, iteration: int) -> Optional[TrafficLightDetections]:
        """Inherited, see superclass."""
        return ArrowTrafficLightDetectionsReader.read_at_iteration(
            self._log_dir,
            get_sync_table(self._log_dir),
            self._get_table_index(iteration),
        )

    def get_all_traffic_light_detections_timestamps(self) -> List[Timestamp]:
        """Inherited, see superclass."""
        return ArrowTrafficLightDetectionsReader.read_all_timestamps(
            self._log_dir, get_sync_table(self._log_dir), self.get_scene_metadata()
        )

    def get_traffic_light_detections_metadata(self) -> Optional[TrafficLightDetectionsMetadata]:
        return None

    # ------------------------------------------------------------------------------------------------------------------
    # 6. Pinhole Camera
    # ------------------------------------------------------------------------------------------------------------------

    def get_pinhole_camera_metadatas(self) -> Dict[PinholeCameraID, PinholeCameraMetadata]:
        """Inherited, see superclass."""
        return self.get_log_metadata().pinhole_cameras_metadata

    def get_all_pinhole_camera_timestamps(self, camera_id: PinholeCameraID) -> List[Timestamp]:
        """Inherited, see superclass."""
        return ArrowCameraReader.read_all_pinhole_timestamps(
            self._log_dir, get_sync_table(self._log_dir), self.get_scene_metadata(), camera_id
        )

    def get_pinhole_camera_at_iteration(self, iteration: int, camera_id: PinholeCameraID) -> Optional[PinholeCamera]:
        """Inherited, see superclass."""
        cam_metadata = self.get_pinhole_camera_metadatas().get(camera_id, None)
        return ArrowCameraReader.read_pinhole_at_iteration(
            self._log_dir,
            get_sync_table(self._log_dir),
            self._get_table_index(iteration),
            camera_id,
            cam_metadata,
            self.log_metadata,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # 7. Fisheye MEI Camera
    # ------------------------------------------------------------------------------------------------------------------

    def get_fisheye_mei_camera_metadatas(self) -> Dict[FisheyeMEICameraID, FisheyeMEICameraMetadata]:
        """Inherited, see superclass."""
        return self.get_log_metadata().fisheye_mei_cameras_metadata

    def get_all_fisheye_mei_camera_timestamps(self, camera_id: FisheyeMEICameraID) -> List[Timestamp]:
        """Inherited, see superclass."""
        return ArrowCameraReader.read_all_fisheye_mei_timestamps(
            self._log_dir, get_sync_table(self._log_dir), self.get_scene_metadata(), camera_id
        )

    def get_fisheye_mei_camera_at_iteration(
        self, iteration: int, camera_id: FisheyeMEICameraID
    ) -> Optional[FisheyeMEICamera]:
        """Inherited, see superclass."""
        cam_metadata = self.get_fisheye_mei_camera_metadatas().get(camera_id, None)
        return ArrowCameraReader.read_fisheye_mei_at_iteration(
            self._log_dir,
            get_sync_table(self._log_dir),
            self._get_table_index(iteration),
            camera_id,
            cam_metadata,
            self.log_metadata,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # 8. Lidar
    # ------------------------------------------------------------------------------------------------------------------

    def get_lidar_metadatas(self) -> Dict[LidarID, LidarMetadata]:
        """Inherited, see superclass."""
        lidar_metadatas: Dict[LidarID, LidarMetadata] = self.get_log_metadata().lidars_metadata
        if lidar_metadatas is None or len(lidar_metadatas) == 0:
            lidar_merged_metadata = self.get_log_metadata().lidar_merged_metadata
            if lidar_merged_metadata is not None:
                lidar_metadatas = lidar_merged_metadata.lidars_metadata
        return lidar_metadatas if lidar_metadatas else {}

    def get_all_lidar_timestamps(self, lidar_id: LidarID) -> List[Timestamp]:
        """Inherited, see superclass."""
        return ArrowLidarReader.read_all_timestamps(
            self._log_dir, get_sync_table(self._log_dir), self.get_scene_metadata(), lidar_id
        )

    def get_lidar_at_iteration(self, iteration: int, lidar_id: LidarID) -> Optional[Lidar]:
        """Inherited, see superclass."""
        return ArrowLidarReader.read_at_iteration(
            self._log_dir,
            get_sync_table(self._log_dir),
            self._get_table_index(iteration),
            lidar_id,
            self.get_lidar_metadatas(),
            self.get_log_metadata().lidar_merged_metadata,
            self.log_metadata,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # 9. Custom Modalities
    # ------------------------------------------------------------------------------------------------------------------

    def get_all_custom_modality_metadatas(self) -> Dict[str, CustomModalityMetadata]:
        """Inherited, see superclass."""
        log_metadata = self.get_log_metadata()
        return log_metadata._custom_modalities_metadata if log_metadata._custom_modalities_metadata is not None else {}

    def get_all_custom_modality_timestamps(self, name: str) -> List[Timestamp]:
        """Inherited, see superclass."""
        return ArrowCustomModalityReader.read_all_timestamps(
            self._log_dir, get_sync_table(self._log_dir), self.get_scene_metadata(), name
        )

    def get_custom_modality_at_iteration(self, iteration: int, name: str) -> Optional[CustomModality]:
        """Inherited, see superclass."""
        return ArrowCustomModalityReader.read_at_iteration(
            self._log_dir,
            self._get_table_index(iteration),
            name,
        )
