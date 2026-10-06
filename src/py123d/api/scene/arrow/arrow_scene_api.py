from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Type, Union

import numpy as np

from py123d.api.map.arrow.arrow_map_api import get_map_api_for_log
from py123d.api.map.map_api import MapAPI
from py123d.api.scene.arrow.modalities.arrow_barometer import ArrowBarometerReader
from py123d.api.scene.arrow.modalities.arrow_base import ArrowBaseModalityReader
from py123d.api.scene.arrow.modalities.arrow_box_detections_se3 import ArrowBoxDetectionsSE3Reader
from py123d.api.scene.arrow.modalities.arrow_camera import ArrowCameraReader
from py123d.api.scene.arrow.modalities.arrow_custom_modality import ArrowCustomModalityReader
from py123d.api.scene.arrow.modalities.arrow_ego_state_se3 import ArrowEgoStateSE3Reader
from py123d.api.scene.arrow.modalities.arrow_gnss import ArrowGnssReader
from py123d.api.scene.arrow.modalities.arrow_imu import ArrowImuReader
from py123d.api.scene.arrow.modalities.arrow_lidar import ArrowLidarReader
from py123d.api.scene.arrow.modalities.arrow_magnetometer import ArrowMagnetometerReader
from py123d.api.scene.arrow.modalities.arrow_radar import ArrowRadarReader
from py123d.api.scene.arrow.modalities.arrow_route_position import ArrowRoutePositionReader
from py123d.api.scene.arrow.modalities.arrow_sync import get_timestamp_from_arrow_table
from py123d.api.scene.arrow.modalities.arrow_traffic_light_detections import ArrowTrafficLightDetectionsReader
from py123d.api.scene.arrow.modalities.sync_utils import (
    _get_scene_sync_range,
    get_all_modality_timestamps,
    get_modality_index_from_sync_index,
    get_modality_table,
    get_sync_table,
)
from py123d.api.scene.arrow.utils.arrow_scene_caches import _get_complete_log_scene_metadata
from py123d.api.scene.scene_api import SceneAPI
from py123d.api.utils.arrow_metadata_utils import LogDirectoryMetadata, parse_log_directory_metadata
from py123d.common.utils.enums import SerialIntEnum
from py123d.datatypes import (
    BaseModality,
    BaseModalityMetadata,
    LogMetadata,
    MapMetadata,
    ModalityType,
    RouteMetadata,
    Timestamp,
    get_modality_key,
)
from py123d.datatypes.metadata import SceneMetadata

MODALITY_READERS: Dict[ModalityType, Type[ArrowBaseModalityReader]] = {
    ModalityType.EGO_STATE_SE3: ArrowEgoStateSE3Reader,
    ModalityType.BOX_DETECTIONS_SE3: ArrowBoxDetectionsSE3Reader,
    ModalityType.TRAFFIC_LIGHT_DETECTIONS: ArrowTrafficLightDetectionsReader,
    ModalityType.CAMERA: ArrowCameraReader,
    ModalityType.CAMERA_SEMANTIC: ArrowCameraReader,
    ModalityType.CAMERA_INSTANCE: ArrowCameraReader,
    ModalityType.CAMERA_DEPTH: ArrowCameraReader,
    ModalityType.LIDAR: ArrowLidarReader,
    ModalityType.RADAR: ArrowRadarReader,
    ModalityType.IMU: ArrowImuReader,
    ModalityType.GNSS: ArrowGnssReader,
    ModalityType.BAROMETER: ArrowBarometerReader,
    ModalityType.MAGNETOMETER: ArrowMagnetometerReader,
    ModalityType.ROUTE_POSITION: ArrowRoutePositionReader,
    ModalityType.CUSTOM: ArrowCustomModalityReader,
}


class ArrowSceneAPI(SceneAPI):
    """Scene API for Arrow-based scenes. Loads each modality from a separate Arrow file in a log directory."""

    __slots__ = ("_log_dir", "_scene_metadata", "_maps_root")

    def __init__(
        self,
        log_dir: Union[Path, str],
        scene_metadata: Optional[SceneMetadata] = None,
        maps_root: Optional[Union[Path, str]] = None,
    ) -> None:
        """Initializes the :class:`ArrowSceneAPI`.

        :param log_dir: Path to the log directory containing per-modality Arrow files.
        :param scene_metadata: Scene metadata, defaults to None
        :param maps_root: Root directory the scene's map is resolved under before the
            global dataset paths, defaults to None
        """
        self._log_dir: Path = Path(log_dir)
        self._scene_metadata: Optional[SceneMetadata] = scene_metadata
        self._maps_root: Optional[Path] = Path(maps_root) if maps_root is not None else None

    def __reduce__(self):
        """Helper for pickling the object."""
        return (self.__class__, (self._log_dir, self._scene_metadata, self._maps_root))

    # ------------------------------------------------------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------------------------------------------------------

    def _get_sync_index(self, iteration: int) -> int:
        """Resolve an iteration (which may be negative for history) to an absolute table index."""
        assert -self.number_of_history_iterations <= iteration < self.number_of_iterations, "Iteration out of bounds"
        metadata = self.get_scene_metadata()
        return metadata.initial_idx + iteration * metadata.target_iteration_stride

    def _get_log_dir_metadatas(self) -> LogDirectoryMetadata:
        """Helper to get modality metadata for a given modality type and optional id."""
        return parse_log_directory_metadata(self._log_dir)

    def _get_modality_timestamp_boundaries(self, modality_key: str) -> Optional[Tuple[Timestamp, Timestamp]]:
        """Helper to get the first and last timestamps for a modality in the log directory."""
        return None

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

    def get_all_iteration_timestamps(self, include_history: bool = False) -> List[Timestamp]:
        """Inherited, see superclass."""
        sync_table = get_sync_table(self._log_dir)
        scene_metadata = self.get_scene_metadata()
        start_idx, end_idx = _get_scene_sync_range(scene_metadata, include_history)
        stride = scene_metadata.target_iteration_stride
        ts_column = sync_table["sync.timestamp_us"].to_numpy()
        return [Timestamp.from_us(ts_column[i]) for i in range(start_idx, end_idx, stride)]

    def get_scene_timestamp_boundaries(self, include_history: bool = False) -> Tuple[Timestamp, Timestamp]:
        """Inherited, see superclass."""
        sync_table = get_sync_table(self._log_dir)
        scene_metadata = self.get_scene_metadata()
        start_idx, end_idx = _get_scene_sync_range(scene_metadata, include_history)
        return (
            get_timestamp_from_arrow_table(sync_table, start_idx),
            get_timestamp_from_arrow_table(sync_table, end_idx - 1),
        )

    # ------------------------------------------------------------------------------------------------------------------
    # 2. Map
    # ------------------------------------------------------------------------------------------------------------------

    def get_map_metadata(self) -> Optional[MapMetadata]:
        """Inherited, see superclass."""
        return self.get_log_metadata().map_metadata

    def get_map_api(self) -> Optional[MapAPI]:
        """Inherited, see superclass."""
        return get_map_api_for_log(self._log_dir, self.get_log_metadata(), maps_root=self._maps_root)

    # ------------------------------------------------------------------------------------------------------------------
    # 3. Route
    # ------------------------------------------------------------------------------------------------------------------

    def get_route(self) -> Optional[Tuple[RouteMetadata, np.ndarray, np.ndarray]]:
        """Inherited, see superclass."""
        route_metadata = self.get_modality_metadata(ModalityType.ROUTE_POSITION)
        if not isinstance(route_metadata, RouteMetadata):
            return None
        return route_metadata, route_metadata.polyline_arc_m, route_metadata.polyline_xyz

    def get_route_progress_at_iteration(self, iteration: int) -> Optional[float]:
        """Inherited, see superclass."""
        return self.get_modality_column_at_iteration(
            iteration,
            column="progress_m",
            modality_type=ModalityType.ROUTE_POSITION,
        )

    def get_remaining_route_m(self, iteration: int = 0) -> Optional[float]:
        """Inherited, see superclass. Reads the stored ``remaining_m`` column directly,
        so the polyline metadata stays undecoded."""
        return self.get_modality_column_at_iteration(
            iteration,
            column="remaining_m",
            modality_type=ModalityType.ROUTE_POSITION,
        )

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
        include_history: bool = False,
    ) -> List[Timestamp]:
        """Returns all timestamps for a specific modality within the scene range.

        :param modality_type: The modality type as a string or :class:`ModalityType`.
        :param modality_id: Optional modality id (e.g. sensor id).
        :param include_history: If True, include history iterations before the scene start.
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
            self._log_dir,
            sync_table,
            self.get_scene_metadata(),
            _modality_key,
            ts_col_name,
            include_history,
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
                    log_dir=self._log_dir,
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
                log_dir=self._log_dir,
                **kwargs,
            )
        return modality

    def get_modality_between_timestamps(
        self,
        start_timestamp: Union[Timestamp, int],
        end_timestamp: Union[Timestamp, int],
        modality_type: Union[str, ModalityType],
        modality_id: Optional[Union[str, SerialIntEnum]] = None,
        inclusive: Literal["left", "right", "both", "neither"] = "left",
        **kwargs,
    ) -> Iterator[BaseModality]:
        """Inherited, see superclass."""
        start_us = start_timestamp.time_us if isinstance(start_timestamp, Timestamp) else int(start_timestamp)
        end_us = end_timestamp.time_us if isinstance(end_timestamp, Timestamp) else int(end_timestamp)

        _modality_type = ModalityType.from_arbitrary(modality_type)
        _modality_key = get_modality_key(_modality_type, modality_id)

        modality_table = get_modality_table(self._log_dir, _modality_key)
        modality_metadata = self.get_modality_metadata(_modality_type, modality_id)
        if modality_table is not None and modality_metadata is not None:
            # NOTE @DanielDauner:
            # searchsorted(side="left") gives an inclusive bound,
            # side="right" gives an exclusive bound.
            sides: Dict[str, Tuple[Literal["left", "right"], Literal["left", "right"]]] = {
                "left": ("left", "left"),
                "right": ("right", "right"),
                "both": ("left", "right"),
                "neither": ("right", "left"),
            }
            start_side, end_side = sides[inclusive]

            ts_array = modality_table[f"{_modality_key}.timestamp_us"].to_numpy()
            lo = int(np.searchsorted(ts_array, start_us, side=start_side))
            hi = int(np.searchsorted(ts_array, end_us, side=end_side))

            reader = MODALITY_READERS[_modality_type]
            for index in range(lo, hi):
                modality = reader.read_at_index(
                    index=index,
                    table=modality_table,
                    metadata=modality_metadata,
                    dataset=self.dataset,
                    log_dir=self._log_dir,
                    **kwargs,
                )
                if modality is not None:
                    yield modality

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

        modality: Optional[BaseModality] = None
        if _modality_key in sync_table.column_names:
            modality_table = get_modality_table(self._log_dir, _modality_key)
            modality_index = get_modality_index_from_sync_index(sync_table, _modality_key, sync_index)
            modality_metadata = self.get_modality_metadata(_modality_type, modality_id)
            if modality_table is not None and modality_index is not None and modality_metadata is not None:
                modality = MODALITY_READERS[_modality_type].read_column_at_index(
                    index=modality_index,
                    table=modality_table,
                    metadata=modality_metadata,
                    column=column,
                    dataset=self.dataset,
                    deserialize=deserialize,
                    log_dir=self._log_dir,
                )

        return modality
