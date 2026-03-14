from pathlib import Path
from typing import Dict, List, Optional, Union

import pyarrow as pa

from py123d.api.map.arrow.arrow_map_api import get_lru_cached_map_api
from py123d.api.map.map_api import MapAPI
from py123d.api.scene.arrow.modalities.arrow_box_detections_se3 import ArrowBoxDetectionsSE3Reader
from py123d.api.scene.arrow.modalities.arrow_camera import ArrowCameraReader
from py123d.api.scene.arrow.modalities.arrow_custom_modality import ArrowCustomModalityReader
from py123d.api.scene.arrow.modalities.arrow_ego_state_se3 import ArrowEgoStateSE3Reader
from py123d.api.scene.arrow.modalities.arrow_lidar import ArrowLidarReader
from py123d.api.scene.arrow.modalities.arrow_sync import get_timestamp_from_arrow_table
from py123d.api.scene.arrow.modalities.arrow_traffic_light_detections import ArrowTrafficLightDetectionsReader
from py123d.api.scene.arrow.modalities.sync_utils import (
    get_all_modality_timestamps,
    get_first_sync_index,
    get_modality_table,
    get_sync_table,
)
from py123d.api.scene.arrow.utils.arrow_scene_caches import (
    _get_complete_log_scene_metadata,
)
from py123d.api.scene.scene_api import SceneAPI
from py123d.api.scene.scene_metadata import SceneMetadata
from py123d.api.utils.arrow_metadata_utils import LogDirectoryMetadata, parse_log_directory_metadata
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
from py123d.datatypes.modalities.base_modality import BaseModalityMetadata, ModalityType, get_modality_key
from py123d.datatypes.sensors.lidar import LidarMergedMetadata


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

    def _get_log_dir_metadatas(self) -> LogDirectoryMetadata:
        """Helper to get modality metadata for a given modality type and optional id."""
        return parse_log_directory_metadata(self._log_dir)

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
        return self._get_log_dir_metadatas().log_metadata

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

    def _resolve_modality_key(
        self,
        modality_type: Union[str, ModalityType],
        modality_id: Optional[Union[int, str, SerialIntEnum]] = None,
    ) -> str:
        """Resolve a modality type and optional id to a modality key string."""
        if isinstance(modality_type, str):
            modality_type = ModalityType.deserialize(modality_type)
        if isinstance(modality_id, int):
            modality_id = SerialIntEnum.from_int(modality_id)
        return get_modality_key(modality_type, modality_id)

    def get_all_modality_metadatas(self) -> Dict[str, BaseModalityMetadata]:
        """Returns all modality metadatas found in the log directory.

        :return: Mapping of modality key to its metadata.
        """
        return self._get_log_dir_metadatas().modality_metadatas

    def get_modality_metadata(
        self,
        modality_type: Union[str, ModalityType],
        modality_id: Optional[Union[int, str, SerialIntEnum]] = None,
    ) -> Optional[BaseModalityMetadata]:
        """Returns the metadata for a specific modality.

        :param modality_type: The modality type as a string or :class:`ModalityType`.
        :param modality_id: Optional modality id (e.g. sensor id).
        :return: The metadata, or None if the modality is not present.
        """
        modality_key = self._resolve_modality_key(modality_type, modality_id)
        return self._get_log_dir_metadatas().modality_metadatas.get(modality_key)

    def get_modality_timestamps(
        self,
        modality_type: Union[str, ModalityType],
        modality_id: Optional[Union[int, str, SerialIntEnum]] = None,
    ) -> List[Timestamp]:
        """Returns all timestamps for a specific modality within the scene range.

        :param modality_type: The modality type as a string or :class:`ModalityType`.
        :param modality_id: Optional modality id (e.g. sensor id).
        :return: List of timestamps, empty if the modality is not present.
        """
        modality_key = self._resolve_modality_key(modality_type, modality_id)
        sync_table = get_sync_table(self._log_dir)
        modality_table = get_modality_table(self._log_dir, modality_key)
        if modality_table is None:
            return []

        # Find the timestamp column: prefer "{key}.timestamp_us", fall back to first "*timestamp_us" column.
        ts_col_name = f"{modality_key}.timestamp_us"
        if ts_col_name not in modality_table.column_names:
            ts_col_name = next((c for c in modality_table.column_names if c.endswith("timestamp_us")), None)
        if ts_col_name is None:
            return []

        return get_all_modality_timestamps(
            self._log_dir, sync_table, self.get_scene_metadata(), modality_key, ts_col_name
        )

    def get_modality_at_iteration(
        self,
        iteration: int,
        modality_type: Union[str, ModalityType],
        modality_id: Optional[Union[int, str, SerialIntEnum]] = None,
    ) -> Optional[pa.Table]:
        """Returns the raw Arrow row(s) for a modality at the given iteration.

        This is a generic accessor that returns the raw Arrow data. For typed access,
        use the modality-specific methods (e.g. :meth:`get_ego_state_se3_at_iteration`).

        :param iteration: The iteration index (supports negative for history).
        :param modality_type: The modality type as a string or :class:`ModalityType`.
        :param modality_id: Optional modality id (e.g. sensor id).
        :return: An Arrow table slice for the matched rows, or None if unavailable.
        """
        modality_key = self._resolve_modality_key(modality_type, modality_id)
        sync_table = get_sync_table(self._log_dir)
        table_index = self._get_table_index(iteration)

        if modality_key not in sync_table.column_names:
            return None

        modality_table = get_modality_table(self._log_dir, modality_key)
        if modality_table is None:
            return None

        row_idx = get_first_sync_index(sync_table, modality_key, table_index)
        if row_idx is None:
            return None

        # Check if this is a list-typed sync column (async/deferred sync → multiple rows per iteration).
        sync_value = sync_table[modality_key][table_index].as_py()
        if isinstance(sync_value, list):
            return modality_table.slice(row_idx, len(sync_value))
        return modality_table.slice(row_idx, 1)

    # TODO: implement in the future.
    # def get_modality_at_timestamp(
    #     self,
    #     timestamp: Timestamp,
    #     modality_type: str,
    #     modality_id: Optional[Union[int, str, SerialIntEnum]] = None,
    #     criteria: Literal["exact", "forward", "backward"] = "exact",
    # ) -> Optional[CustomModality]:
    #     pass

    # ------------------------------------------------------------------------------------------------------------------
    # 3. EgoStateSE3
    # ------------------------------------------------------------------------------------------------------------------

    def get_ego_state_se3_metadata(self) -> Optional[EgoStateSE3Metadata]:
        """Inherited, see superclass."""
        return self._get_log_dir_metadatas().modality_metadatas.get("ego_state_se3")  # type: ignore

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
        return self._get_log_dir_metadatas().modality_metadatas.get("box_detections_se3")  # type: ignore

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
        by_type = self._get_log_dir_metadatas().get_by_type(ModalityType.PINHOLE_CAMERA)
        return {meta.modality_id: meta for meta in by_type.values()}  # type: ignore[misc]

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
        by_type = self._get_log_dir_metadatas().get_by_type(ModalityType.FISHEYE_MEI_CAMERA)
        return {meta.modality_id: meta for meta in by_type.values()}  # type: ignore[misc]

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
        dir_meta = self._get_log_dir_metadatas()
        by_type = dir_meta.get_by_type(ModalityType.LIDAR)
        lidar_metadatas: Dict[LidarID, LidarMetadata] = {}
        for meta in by_type.values():
            if isinstance(meta, LidarMergedMetadata):
                lidar_metadatas.update(meta.lidar_metadatas)
            elif isinstance(meta, LidarMetadata):
                lidar_metadatas[meta.lidar_id] = meta
        return lidar_metadatas

    def get_all_lidar_timestamps(self, lidar_id: LidarID) -> List[Timestamp]:
        """Inherited, see superclass."""
        return ArrowLidarReader.read_all_timestamps(
            self._log_dir, get_sync_table(self._log_dir), self.get_scene_metadata(), lidar_id
        )

    def get_lidar_at_iteration(self, iteration: int, lidar_id: LidarID) -> Optional[Lidar]:
        """Inherited, see superclass."""
        merged_meta = self._get_log_dir_metadatas().modality_metadatas.get("lidar.lidar_merged")
        return ArrowLidarReader.read_at_iteration(
            self._log_dir,
            get_sync_table(self._log_dir),
            self._get_table_index(iteration),
            lidar_id,
            self.get_lidar_metadatas(),
            merged_meta if isinstance(merged_meta, LidarMergedMetadata) else None,
            self.log_metadata,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # 9. Custom Modalities
    # ------------------------------------------------------------------------------------------------------------------

    def get_all_custom_modality_metadatas(self) -> Dict[str, CustomModalityMetadata]:
        """Inherited, see superclass."""
        by_type = self._get_log_dir_metadatas().get_by_type(ModalityType.CUSTOM)
        return {meta.modality_id: meta for meta in by_type.values()}  # type: ignore[misc]

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
