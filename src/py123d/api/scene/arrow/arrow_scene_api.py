from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from py123d.api.map.arrow.arrow_map_api import get_map_api_for_log
from py123d.api.map.map_api import MapAPI
from py123d.api.scene.arrow.modalities.arrow_base import ArrowBaseModalityReader
from py123d.api.scene.arrow.modalities.arrow_box_detections_se3 import ArrowBoxDetectionsSE3Reader
from py123d.api.scene.arrow.modalities.arrow_camera import ArrowCameraReader
from py123d.api.scene.arrow.modalities.arrow_custom_modality import ArrowCustomModalityReader
from py123d.api.scene.arrow.modalities.arrow_ego_state_se3 import ArrowEgoStateSE3Reader
from py123d.api.scene.arrow.modalities.arrow_lidar import ArrowLidarReader
from py123d.api.scene.arrow.modalities.arrow_sync import get_timestamp_from_arrow_table
from py123d.api.scene.arrow.modalities.arrow_traffic_light_detections import ArrowTrafficLightDetectionsReader
from py123d.api.scene.arrow.modalities.sync_utils import (
    get_all_modality_timestamps,
    get_modality_index_from_sync_index,
    get_modality_table,
    get_sync_table,
)
from py123d.api.scene.arrow.utils.arrow_scene_caches import (
    _get_complete_log_scene_metadata,
)
from py123d.api.scene.scene_api import SceneAPI
from py123d.api.scene.scene_metadata import SceneMetadata
from py123d.api.utils.arrow_metadata_utils import LogDirectoryMetadata, parse_log_directory_metadata
from py123d.common.utils.enums import SerialIntEnum
from py123d.datatypes import (
    BoxDetectionsSE3,
    BoxDetectionsSE3Metadata,
    CustomModality,
    Lidar,
    LidarID,
    LidarMetadata,
    LogMetadata,
    MapMetadata,
    Timestamp,
    TrafficLightDetections,
)
from py123d.datatypes.custom.custom_modality import CustomModalityMetadata
from py123d.datatypes.detections.traffic_light_detections import TrafficLightDetectionsMetadata
from py123d.datatypes.modalities.base_modality import BaseModality, BaseModalityMetadata, ModalityType, get_modality_key
from py123d.datatypes.sensors import BaseCameraMetadata, Camera, CameraID
from py123d.datatypes.sensors.lidar import LidarMergedMetadata

MODALITY_READERS: Dict[ModalityType, Type[ArrowBaseModalityReader]] = {
    ModalityType.EGO_STATE_SE3: ArrowEgoStateSE3Reader,
    ModalityType.BOX_DETECTIONS_SE3: ArrowBoxDetectionsSE3Reader,
    ModalityType.TRAFFIC_LIGHT_DETECTIONS: ArrowTrafficLightDetectionsReader,
    ModalityType.CAMERA: ArrowCameraReader,
    ModalityType.LIDAR: ArrowLidarReader,
    ModalityType.CUSTOM: ArrowCustomModalityReader,
}


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

    def _get_sync_index(self, iteration: int) -> int:
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
        return get_timestamp_from_arrow_table(sync_table, self._get_sync_index(iteration))

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
        return get_map_api_for_log(self._log_dir, self.get_log_metadata())

    # ------------------------------------------------------------------------------------------------------------------
    # 4. General modality access
    # ------------------------------------------------------------------------------------------------------------------

    def get_all_modality_metadatas(self) -> Dict[str, BaseModalityMetadata]:
        """Returns all modality metadatas found in the log directory.

        :return: Mapping of modality key to its metadata.
        """
        return self._get_log_dir_metadatas().modality_metadatas

    def get_modality_metadata(
        self,
        modality_type: Union[str, ModalityType],
        modality_id: Optional[Union[str, SerialIntEnum]] = None,
    ) -> Optional[BaseModalityMetadata]:
        """Returns the metadata for a specific modality.

        :param modality_type: The modality type as a string or :class:`ModalityType`.
        :param modality_id: Optional modality id (e.g. sensor id).
        :return: The metadata, or None if the modality is not present.
        """
        _modality_type = ModalityType.from_arbitrary(modality_type)
        _modality_key = get_modality_key(_modality_type, modality_id)
        return self._get_log_dir_metadatas().modality_metadatas.get(_modality_key)

    def get_all_modality_timestamps(
        self,
        modality_type: Union[str, ModalityType],
        modality_id: Optional[Union[str, SerialIntEnum]] = None,
    ) -> List[Timestamp]:
        """Returns all timestamps for a specific modality within the scene range.

        :param modality_type: The modality type as a string or :class:`ModalityType`.
        :param modality_id: Optional modality id (e.g. sensor id).
        :return: List of timestamps, empty if the modality is not present.
        """

        # FIXME: @DanielDauner: Clean this function
        _modality_type = ModalityType.from_arbitrary(modality_type)
        _modality_key = get_modality_key(_modality_type, modality_id)
        sync_table = get_sync_table(self._log_dir)
        modality_table = get_modality_table(self._log_dir, _modality_key)
        if modality_table is None:
            return []

        # Find the timestamp column: prefer "{key}.timestamp_us", fall back to first "*timestamp_us" column.
        ts_col_name = f"{_modality_key}.timestamp_us"
        if ts_col_name not in modality_table.column_names:
            ts_col_name = next((c for c in modality_table.column_names if c.endswith("timestamp_us")), None)
        if ts_col_name is None:
            return []

        return get_all_modality_timestamps(
            self._log_dir, sync_table, self.get_scene_metadata(), _modality_key, ts_col_name
        )

    def get_modality_at_iteration(
        self,
        iteration: int,
        modality_type: Union[str, ModalityType],
        modality_id: Optional[Union[str, SerialIntEnum]] = None,
        **kwargs,
    ) -> Optional[BaseModality]:
        """Returns the raw Arrow row(s) for a modality at the given iteration.

        This is a generic accessor that returns the raw Arrow data. For typed access,
        use the modality-specific methods (e.g. :meth:`get_ego_state_se3_at_iteration`).

        :param iteration: The iteration index (supports negative for history).
        :param modality_type: The modality type as a string or :class:`ModalityType`.
        :param modality_id: Optional modality id (e.g. sensor id).
        :param kwargs: Additional keyword arguments passed to the modality reader.
        :return: An Arrow table slice for the matched rows, or None if unavailable.
        """
        _modality_type = ModalityType.from_arbitrary(modality_type)
        _modality_key = get_modality_key(_modality_type, modality_id)

        sync_table = get_sync_table(self._log_dir)
        sync_index = self._get_sync_index(iteration)

        modality: Optional[BaseModality] = None
        if _modality_key in sync_table.column_names:
            modality_table = get_modality_table(self._log_dir, _modality_key)
            modality_index = get_modality_index_from_sync_index(sync_table, _modality_key, sync_index)
            modality_metadata = self.get_modality_metadata(_modality_type, modality_id)
            if modality_table is not None and modality_index is not None and modality_metadata is not None:
                modality = MODALITY_READERS[_modality_type].read_at_index(
                    index=modality_index,
                    table=modality_table,
                    metadata=modality_metadata,
                    dataset=self.dataset,
                    **kwargs,
                )
        return modality

    def get_modality_at_timestamp(
        self,
        timestamp: Union[Timestamp, int],
        modality_type: Union[str, ModalityType],
        modality_id: Optional[Union[str, SerialIntEnum]] = None,
        criteria: Literal["exact", "nearest", "forward", "backward"] = "exact",
        **kwargs,
    ) -> Optional[BaseModality]:
        _timestamp = Timestamp.from_us(timestamp) if not isinstance(timestamp, Timestamp) else timestamp
        _modality_type = ModalityType.from_arbitrary(modality_type)
        _modality_key = get_modality_key(_modality_type, modality_id)

        modality: Optional[BaseModality] = None
        modality_table = get_modality_table(self._log_dir, _modality_key)
        modality_metadata = self.get_modality_metadata(_modality_type, modality_id)
        if modality_table is not None and modality_metadata is not None:
            modality = MODALITY_READERS[_modality_type].read_at_timestamp(
                timestamp=_timestamp,
                table=modality_table,
                metadata=modality_metadata,
                dataset=self.dataset,
                criteria=criteria,
                **kwargs,
            )
        return modality

    def get_modality_column_at_iteration(
        self,
        iteration: int,
        column: str,
        modality_type: Union[str, ModalityType],
        modality_id: Optional[Union[str, SerialIntEnum]] = None,
        deserialize: bool = False,
    ) -> Optional[Any]:
        """Returns a single column value for a modality at the given iteration.

        :param iteration: The iteration index (supports negative for history).
        :param modality_type: The modality type as a string or :class:`ModalityType`.
        :param column: The field name (e.g. ``"imu_se3"``, ``"timestamp_us"``).
        :param modality_id: Optional modality id (e.g. sensor id).
        :param deserialize: If True, deserialize the value to its domain type (e.g. PoseSE3).
        :return: The column value (raw or deserialized), or None if unavailable.
        """
        _modality_type = ModalityType.from_arbitrary(modality_type)
        _modality_key = get_modality_key(_modality_type, modality_id)

        sync_table = get_sync_table(self._log_dir)
        sync_index = self._get_sync_index(iteration)

        if _modality_key not in sync_table.column_names:
            return None

        modality_table = get_modality_table(self._log_dir, _modality_key)
        modality_index = get_modality_index_from_sync_index(sync_table, _modality_key, sync_index)
        modality_metadata = self.get_modality_metadata(_modality_type, modality_id)

        if modality_table is None or modality_index is None or modality_metadata is None:
            return None

        reader = MODALITY_READERS[_modality_type]
        return reader.get_column_at_index(
            index=modality_index,
            table=modality_table,
            metadata=modality_metadata,
            column=column,
            deserialize=deserialize,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # 4. BoxDetectionsSE3
    # ------------------------------------------------------------------------------------------------------------------

    def get_box_detections_se3_metadata(self) -> Optional[BoxDetectionsSE3Metadata]:
        """Inherited, see superclass."""
        return self._get_log_dir_metadatas().modality_metadatas.get("box_detections_se3")  # type: ignore

    def get_all_box_detections_se3_timestamps(self) -> List[Timestamp]:
        """Inherited, see superclass."""
        return self.get_all_modality_timestamps(modality_type=ModalityType.BOX_DETECTIONS_SE3)

    def get_box_detections_se3_at_iteration(self, iteration: int) -> Optional[BoxDetectionsSE3]:
        """Inherited, see superclass."""
        modality = self.get_modality_at_iteration(iteration, ModalityType.BOX_DETECTIONS_SE3)
        assert isinstance(modality, (BoxDetectionsSE3, type(None))), (
            f"Expected BoxDetectionsSE3 or None, got {type(modality)}"
        )
        return modality

    def get_box_detections_se3_at_timestamp(
        self,
        timestamp: Union[Timestamp, int],
        criteria: Literal["exact", "nearest", "forward", "backward"] = "exact",
    ) -> Optional[BoxDetectionsSE3]:
        """Inherited, see superclass."""
        modality = self.get_modality_at_timestamp(
            timestamp=timestamp,
            modality_type=ModalityType.BOX_DETECTIONS_SE3,
            criteria=criteria,
        )
        assert isinstance(modality, (BoxDetectionsSE3, type(None))), (
            f"Expected BoxDetectionsSE3 or None, got {type(modality)}"
        )
        return modality

    # ------------------------------------------------------------------------------------------------------------------
    # 5. Traffic Light Detections
    # ------------------------------------------------------------------------------------------------------------------

    def get_traffic_light_detections_metadata(self) -> Optional[TrafficLightDetectionsMetadata]:
        """Inherited, see superclass."""
        return self._get_log_dir_metadatas().modality_metadatas.get("traffic_light_detections")  # type: ignore

    def get_all_traffic_light_detections_timestamps(self) -> List[Timestamp]:
        """Inherited, see superclass."""
        return self.get_all_modality_timestamps(modality_type=ModalityType.TRAFFIC_LIGHT_DETECTIONS)

    def get_traffic_light_detections_at_iteration(self, iteration: int) -> Optional[TrafficLightDetections]:
        """Inherited, see superclass."""
        modality = self.get_modality_at_iteration(iteration, ModalityType.TRAFFIC_LIGHT_DETECTIONS)
        assert isinstance(modality, (TrafficLightDetections, type(None))), (
            f"Expected TrafficLightDetections or None, got {type(modality)}"
        )
        return modality

    def get_traffic_light_detections_at_timestamp(
        self,
        timestamp: Union[Timestamp, int],
        criteria: Literal["exact", "nearest", "forward", "backward"] = "exact",
    ) -> Optional[TrafficLightDetections]:
        """Inherited, see superclass."""
        modality = self.get_modality_at_timestamp(
            timestamp=timestamp,
            modality_type=ModalityType.TRAFFIC_LIGHT_DETECTIONS,
            criteria=criteria,
        )
        assert isinstance(modality, (TrafficLightDetections, type(None))), (
            f"Expected TrafficLightDetections or None, got {type(modality)}"
        )
        return modality

    # ------------------------------------------------------------------------------------------------------------------
    # 6. Camera
    # ------------------------------------------------------------------------------------------------------------------

    def get_camera_metadatas(self) -> Dict[CameraID, BaseCameraMetadata]:
        """Inherited, see superclass."""
        by_type = self._get_log_dir_metadatas().get_by_type(ModalityType.CAMERA)
        return {meta.modality_id: meta for meta in by_type.values()}  # type: ignore[misc]

    def get_all_camera_timestamps(self, camera_id: CameraID) -> List[Timestamp]:
        """Inherited, see superclass."""
        return self.get_all_modality_timestamps(modality_type=ModalityType.CAMERA, modality_id=camera_id)

    def get_camera_at_iteration(
        self,
        iteration: int,
        camera_id: CameraID,
        scaling_factor: Optional[Tuple[int, int]] = None,
    ) -> Optional[Camera]:
        """Inherited, see superclass."""
        modality = self.get_modality_at_iteration(
            iteration, ModalityType.CAMERA, camera_id, scaling_factor=scaling_factor
        )
        assert isinstance(modality, (Camera, type(None))), f"Expected Camera or None, got {type(modality)}"
        return modality

    def get_camera_at_timestamp(
        self,
        timestamp: Union[Timestamp, int],
        camera_id: CameraID,
        criteria: Literal["exact", "nearest", "forward", "backward"] = "exact",
        scaling_factor: Optional[Tuple[int, int]] = None,
    ) -> Optional[Camera]:
        """Inherited, see superclass."""
        modality = self.get_modality_at_timestamp(
            timestamp=timestamp,
            modality_type=ModalityType.CAMERA,
            modality_id=camera_id,
            criteria=criteria,
            scaling_factor=scaling_factor,
        )
        assert isinstance(modality, (Camera, type(None))), f"Expected Camera or None, got {type(modality)}"
        return modality

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
        return lidar_metadatas

    def get_all_lidar_timestamps(self, lidar_id: LidarID) -> List[Timestamp]:
        """Inherited, see superclass."""
        return self.get_all_modality_timestamps(
            modality_type=ModalityType.LIDAR,
            modality_id=LidarID.LIDAR_MERGED,
        )

    def get_lidar_at_iteration(self, iteration: int, lidar_id: LidarID) -> Optional[Lidar]:
        """Inherited, see superclass."""
        merged_lidar_metadata = self.get_modality_metadata(ModalityType.LIDAR, LidarID.LIDAR_MERGED)
        _modality_id = LidarID.LIDAR_MERGED if merged_lidar_metadata is not None else lidar_id
        lidar = self.get_modality_at_iteration(
            iteration=iteration,
            modality_type=ModalityType.LIDAR,
            modality_id=_modality_id,
            lidar_id=lidar_id,
        )
        assert isinstance(lidar, (Lidar, type(None))), f"Expected Lidar or None, got {type(lidar)}"
        return lidar

    def get_lidar_at_timestamp(
        self,
        timestamp: Union[Timestamp, int],
        lidar_id: LidarID,
        criteria: Literal["exact", "nearest", "forward", "backward"] = "exact",
    ) -> Optional[Lidar]:
        """Inherited, see superclass."""
        merged_lidar_metadata = self.get_modality_metadata(ModalityType.LIDAR, LidarID.LIDAR_MERGED)
        _modality_id = LidarID.LIDAR_MERGED if merged_lidar_metadata is not None else lidar_id
        lidar = self.get_modality_at_timestamp(
            timestamp=timestamp,
            modality_type=ModalityType.LIDAR,
            modality_id=_modality_id,
            criteria=criteria,
            lidar_id=lidar_id,
        )
        assert isinstance(lidar, (Lidar, type(None))), f"Expected Lidar or None, got {type(lidar)}"
        return lidar

    # ------------------------------------------------------------------------------------------------------------------
    # 9. Custom Modalities
    # ------------------------------------------------------------------------------------------------------------------

    def get_all_custom_modality_metadatas(self) -> Dict[str, CustomModalityMetadata]:
        """Inherited, see superclass."""
        by_type = self._get_log_dir_metadatas().get_by_type(ModalityType.CUSTOM)
        return {meta.modality_id: meta for meta in by_type.values()}  # type: ignore[misc]

    def get_all_custom_modality_timestamps(self, modality_id: str) -> List[Timestamp]:
        """Inherited, see superclass."""
        return self.get_all_modality_timestamps(modality_type=ModalityType.CUSTOM, modality_id=modality_id)

    def get_custom_modality_at_iteration(self, iteration: int, modality_id: str) -> Optional[CustomModality]:
        """Inherited, see superclass."""
        modality = self.get_modality_at_iteration(iteration, ModalityType.CUSTOM, modality_id)
        assert isinstance(modality, (CustomModality, type(None))), (
            f"Expected CustomModality or None, got {type(modality)}"
        )
        return modality

    def get_custom_modality_at_timestamp(
        self,
        timestamp: Union[Timestamp, int],
        modality_id: str,
        criteria: Literal["exact", "nearest", "forward", "backward"] = "exact",
    ) -> Optional[CustomModality]:
        """Inherited, see superclass."""
        modality = self.get_modality_at_timestamp(
            timestamp=timestamp,
            modality_type=ModalityType.CUSTOM,
            modality_id=modality_id,
            criteria=criteria,
        )
        assert isinstance(modality, (CustomModality, type(None))), (
            f"Expected CustomModality or None, got {type(modality)}"
        )
        return modality
