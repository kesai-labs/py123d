from pathlib import Path
from typing import Dict, List, Optional, Type, Union

import pyarrow as pa

from py123d.api.map.arrow.arrow_map_api import get_lru_cached_map_api
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
from py123d.datatypes.modalities.base_modality import BaseModality, BaseModalityMetadata, ModalityType, get_modality_key
from py123d.datatypes.sensors.lidar import LidarMergedMetadata, get_individual_lidar, get_merged_lidar

MODALITY_READERS: Dict[ModalityType, Type[ArrowBaseModalityReader]] = {
    ModalityType.EGO_STATE_SE3: ArrowEgoStateSE3Reader,
    ModalityType.BOX_DETECTIONS_SE3: ArrowBoxDetectionsSE3Reader,
    ModalityType.TRAFFIC_LIGHT_DETECTIONS: ArrowTrafficLightDetectionsReader,
    ModalityType.PINHOLE_CAMERA: ArrowCameraReader,
    ModalityType.FISHEYE_MEI_CAMERA: ArrowCameraReader,
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
        modality_id: Optional[Union[str, SerialIntEnum]] = None,
    ) -> List[Timestamp]:
        """Returns all timestamps for a specific modality within the scene range.

        :param modality_type: The modality type as a string or :class:`ModalityType`.
        :param modality_id: Optional modality id (e.g. sensor id).
        :return: List of timestamps, empty if the modality is not present.
        """
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
    ) -> Optional[pa.Table]:
        """Returns the raw Arrow row(s) for a modality at the given iteration.

        This is a generic accessor that returns the raw Arrow data. For typed access,
        use the modality-specific methods (e.g. :meth:`get_ego_state_se3_at_iteration`).

        :param iteration: The iteration index (supports negative for history).
        :param modality_type: The modality type as a string or :class:`ModalityType`.
        :param modality_id: Optional modality id (e.g. sensor id).
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
                )
        return modality

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
        return self.get_modality_timestamps(modality_type=ModalityType.EGO_STATE_SE3)

    def get_ego_state_se3_at_iteration(self, iteration: int) -> Optional[EgoStateSE3]:
        """Inherited, see superclass."""
        modality = self.get_modality_at_iteration(iteration, ModalityType.EGO_STATE_SE3)
        assert isinstance(modality, (EgoStateSE3, type(None))), f"Expected EgoStateSE3 or None, got {type(modality)}"
        return modality

    # ------------------------------------------------------------------------------------------------------------------
    # 4. BoxDetectionsSE3
    # ------------------------------------------------------------------------------------------------------------------

    def get_box_detections_se3_metadata(self) -> Optional[BoxDetectionsSE3Metadata]:
        """Inherited, see superclass."""
        return self._get_log_dir_metadatas().modality_metadatas.get("box_detections_se3")  # type: ignore

    def get_all_box_detections_se3_timestamps(self) -> List[Timestamp]:
        """Inherited, see superclass."""
        return self.get_modality_timestamps(modality_type=ModalityType.BOX_DETECTIONS_SE3)

    def get_box_detections_se3_at_iteration(self, iteration: int) -> Optional[BoxDetectionsSE3]:
        """Inherited, see superclass."""
        modality = self.get_modality_at_iteration(iteration, ModalityType.BOX_DETECTIONS_SE3)
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

    def get_traffic_light_detections_at_iteration(self, iteration: int) -> Optional[TrafficLightDetections]:
        """Inherited, see superclass."""
        modality = self.get_modality_at_iteration(iteration, ModalityType.TRAFFIC_LIGHT_DETECTIONS)
        assert isinstance(modality, (TrafficLightDetections, type(None))), (
            f"Expected TrafficLightDetections or None, got {type(modality)}"
        )
        return modality

    def get_all_traffic_light_detections_timestamps(self) -> List[Timestamp]:
        """Inherited, see superclass."""
        return self.get_modality_timestamps(modality_type=ModalityType.TRAFFIC_LIGHT_DETECTIONS)

    # ------------------------------------------------------------------------------------------------------------------
    # 6. Pinhole Camera
    # ------------------------------------------------------------------------------------------------------------------

    def get_pinhole_camera_metadatas(self) -> Dict[PinholeCameraID, PinholeCameraMetadata]:
        """Inherited, see superclass."""
        by_type = self._get_log_dir_metadatas().get_by_type(ModalityType.PINHOLE_CAMERA)
        return {meta.modality_id: meta for meta in by_type.values()}  # type: ignore[misc]

    def get_all_pinhole_camera_timestamps(self, camera_id: PinholeCameraID) -> List[Timestamp]:
        """Inherited, see superclass."""
        return self.get_modality_timestamps(modality_type=ModalityType.PINHOLE_CAMERA, modality_id=camera_id)

    def get_pinhole_camera_at_iteration(self, iteration: int, camera_id: PinholeCameraID) -> Optional[PinholeCamera]:
        """Inherited, see superclass."""
        modality = self.get_modality_at_iteration(iteration, ModalityType.PINHOLE_CAMERA, camera_id)
        assert isinstance(modality, (PinholeCamera, type(None))), (
            f"Expected PinholeCamera or None, got {type(modality)}"
        )
        return modality

    # ------------------------------------------------------------------------------------------------------------------
    # 7. Fisheye MEI Camera
    # ------------------------------------------------------------------------------------------------------------------

    def get_fisheye_mei_camera_metadatas(self) -> Dict[FisheyeMEICameraID, FisheyeMEICameraMetadata]:
        """Inherited, see superclass."""
        by_type = self._get_log_dir_metadatas().get_by_type(ModalityType.FISHEYE_MEI_CAMERA)
        return {meta.modality_id: meta for meta in by_type.values()}  # type: ignore[misc]

    def get_all_fisheye_mei_camera_timestamps(self, camera_id: FisheyeMEICameraID) -> List[Timestamp]:
        """Inherited, see superclass."""
        return self.get_modality_timestamps(modality_type=ModalityType.FISHEYE_MEI_CAMERA, modality_id=camera_id)

    def get_fisheye_mei_camera_at_iteration(
        self, iteration: int, camera_id: FisheyeMEICameraID
    ) -> Optional[FisheyeMEICamera]:
        """Inherited, see superclass."""
        modality = self.get_modality_at_iteration(iteration, ModalityType.FISHEYE_MEI_CAMERA, camera_id)
        assert isinstance(modality, (FisheyeMEICamera, type(None))), (
            f"Expected FisheyeMEICamera or None, got {type(modality)}"
        )
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
            elif isinstance(meta, LidarMetadata):
                lidar_metadatas[meta.lidar_id] = meta
        return lidar_metadatas

    def get_all_lidar_timestamps(self, lidar_id: LidarID) -> List[Timestamp]:
        """Inherited, see superclass."""
        all_lidar_timestamps: List[Timestamp] = []
        merged_meta = self._get_log_dir_metadatas().modality_metadatas.get("lidar.lidar_merged")
        if isinstance(merged_meta, LidarMergedMetadata) and (
            lidar_id in merged_meta or lidar_id == LidarID.LIDAR_MERGED
        ):
            all_lidar_timestamps = self.get_modality_timestamps(
                modality_type=ModalityType.LIDAR,
                modality_id=LidarID.LIDAR_MERGED,
            )
        else:
            all_lidar_timestamps = self.get_modality_timestamps(
                modality_type=ModalityType.LIDAR,
                modality_id=lidar_id,
            )
        return all_lidar_timestamps

    def get_lidar_at_iteration(self, iteration: int, lidar_id: LidarID) -> Optional[Lidar]:
        """Inherited, see superclass.

        Handles three cases:
        1. Merged lidar table exists and contains the requested sensor -> read from merged table.
        2. Individual lidar table exists -> read from individual table.
        3. LIDAR_MERGED requested but no merged table -> merge all individual lidars on the fly.
        """
        lidar: Optional[Lidar] = None
        merged_meta = self._get_log_dir_metadatas().modality_metadatas.get("lidar.lidar_merged")

        if isinstance(merged_meta, LidarMergedMetadata) and (
            lidar_id in merged_meta or lidar_id == LidarID.LIDAR_MERGED
        ):
            # Case 1: Read from pre-merged lidar table, and split
            merged_lidar = self.get_modality_at_iteration(
                iteration=iteration,
                modality_type=ModalityType.LIDAR,
                modality_id=LidarID.LIDAR_MERGED,
            )
            lidar = merged_lidar if lidar_id == LidarID.LIDAR_MERGED else get_individual_lidar(merged_lidar, lidar_id)
        else:
            if lidar_id != LidarID.LIDAR_MERGED:
                # Case 2: Merge lidar from individual tables on the fly
                all_lidars = []
                for individual_lidar_id in self.get_lidar_metadatas().keys():
                    individual_lidar = self.get_modality_at_iteration(
                        iteration=iteration,
                        modality_type=ModalityType.LIDAR,
                        modality_id=individual_lidar_id,
                    )
                    if individual_lidar is not None:
                        all_lidars.append(individual_lidar)
                lidar = get_merged_lidar(all_lidars)
            else:
                # Case 3: Read from individual lidar table
                lidar = self.get_modality_at_iteration(
                    iteration=iteration,
                    modality_type=ModalityType.LIDAR,
                    modality_id=lidar_id,
                )

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
        return self.get_modality_timestamps(modality_type=ModalityType.CUSTOM, modality_id=modality_id)

    def get_custom_modality_at_iteration(self, iteration: int, modality_id: str) -> Optional[CustomModality]:
        """Inherited, see superclass."""
        modality = self.get_modality_at_iteration(iteration, ModalityType.CUSTOM, modality_id)
        assert isinstance(modality, (CustomModality, type(None))), (
            f"Expected CustomModality or None, got {type(modality)}"
        )
        return modality
