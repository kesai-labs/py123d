from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Union

import pyarrow as pa

from py123d.common.utils.uuid_utils import convert_to_str_uuid
from py123d.datatypes.detections import BoxDetectionsSE3, TrafficLights
from py123d.datatypes.metadata.log_metadata import LogMetadata
from py123d.datatypes.sensors import (
    FisheyeMEICamera,
    FisheyeMEICameraID,
    Lidar,
    LidarID,
    PinholeCamera,
    PinholeCameraID,
)
from py123d.datatypes.time import Timestamp
from py123d.datatypes.vehicle_state import EgoStateSE3
from py123d.api.map.arrow_map_api import get_global_map_api, get_local_map_api
from py123d.api.map.map_api import MapAPI
from py123d.api.scene.arrow.utils.arrow_getters import (
    get_box_detections_se3_from_arrow_table,
    get_camera_from_arrow_table,
    get_ego_state_se3_from_arrow_table,
    get_lidar_from_arrow_table,
    get_route_lane_group_ids_from_arrow_table,
    get_timestamp_from_arrow_table,
    get_traffic_light_detections_from_arrow_table,
)
from py123d.api.scene.arrow.utils.arrow_metadata_utils import (
    get_log_metadata_from_arrow_table,
)
from py123d.api.scene.scene_api import SceneAPI
from py123d.api.scene.scene_metadata import SceneMetadata
from py123d.api.utils.arrow_helper import get_lru_cached_arrow_table
from py123d.api.utils.arrow_schema import UUID_COLUMN


def _get_complete_log_scene_metadata(arrow_file_path: Union[Path, str], log_metadata: LogMetadata) -> SceneMetadata:
    """Helper function to get the scene metadata for a complete log of an Arrow file."""
    table = get_lru_cached_arrow_table(arrow_file_path)
    initial_uuid = convert_to_str_uuid(table[UUID_COLUMN][0].as_py())
    num_rows = table.num_rows
    return SceneMetadata(
        initial_uuid=initial_uuid,
        initial_idx=0,
        duration_s=log_metadata.timestep_seconds * num_rows,
        history_s=0.0,
        iteration_duration_s=log_metadata.timestep_seconds,
    )


@lru_cache(maxsize=1_000)
def _get_lru_cached_log_metadata(arrow_file_path: Union[Path, str]) -> LogMetadata:
    """Helper function to get the LRU cached log metadata from an Arrow file."""
    table = get_lru_cached_arrow_table(arrow_file_path)
    return get_log_metadata_from_arrow_table(table)


class ArrowSceneAPI(SceneAPI):
    """Scene API for Arrow-based scenes. Provides access to all data modalities in an Arrow scene."""

    def __init__(
        self,
        arrow_file_path: Union[Path, str],
        scene_metadata: Optional[SceneMetadata] = None,
    ) -> None:
        """Initializes the :class:`ArrowSceneAPI`.

        :param arrow_file_path: Path to the Arrow file.
        :param scene_metadata: Scene metadata, defaults to None
        """

        self._arrow_file_path: Path = Path(arrow_file_path)
        self._scene_metadata: Optional[SceneMetadata] = scene_metadata

        # NOTE: Lazy load a log-specific map API, and keep reference.
        # Global maps are LRU cached internally.
        self._local_map_api: Optional[MapAPI] = None

    # Helper methods
    # ------------------------------------------------------------------------------------------------------------------

    def __reduce__(self):
        """Helper for pickling the object."""
        return (
            self.__class__,
            (
                self._arrow_file_path,
                self._scene_metadata,
            ),
        )

    def _get_recording_table(self) -> pa.Table:
        """Helper function to return an LRU cached reference to the arrow table."""
        return get_lru_cached_arrow_table(self._arrow_file_path)

    def _get_table_index(self, iteration: int) -> int:
        """Helper function to get the table index for a given iteration."""
        assert -self.number_of_history_iterations <= iteration < self.number_of_iterations, "Iteration out of bounds"
        table_index = self.get_scene_metadata().initial_idx + iteration
        return table_index

    # Implementation of abstract methods
    # ------------------------------------------------------------------------------------------------------------------

    def get_log_metadata(self) -> LogMetadata:
        """Inherited, see superclass."""
        return _get_lru_cached_log_metadata(self._arrow_file_path)

    def get_scene_metadata(self) -> SceneMetadata:
        """Inherited, see superclass."""
        if self._scene_metadata is None:
            log_metadata = self.get_log_metadata()
            self._scene_metadata = _get_complete_log_scene_metadata(self._arrow_file_path, log_metadata)
        return self._scene_metadata

    def get_map_api(self) -> Optional[MapAPI]:
        """Inherited, see superclass."""
        map_api: Optional[MapAPI] = None
        if self.log_metadata.map_metadata is not None:
            if self.log_metadata.map_metadata.map_is_local:
                if self._local_map_api is None:
                    map_api = get_local_map_api(self.log_metadata.split, self.log_name)
                    self._local_map_api = map_api
                else:
                    map_api = self._local_map_api
            else:
                map_api = get_global_map_api(self.log_metadata.dataset, self.log_metadata.location)
        return map_api

    def get_timestamp_at_iteration(self, iteration: int) -> Timestamp:
        """Inherited, see superclass."""
        return get_timestamp_from_arrow_table(self._get_recording_table(), self._get_table_index(iteration))

    def get_ego_state_at_iteration(self, iteration: int) -> Optional[EgoStateSE3]:
        """Inherited, see superclass."""
        return get_ego_state_se3_from_arrow_table(
            self._get_recording_table(),
            self._get_table_index(iteration),
            self.log_metadata.vehicle_parameters,
        )

    def get_box_detections_at_iteration(self, iteration: int) -> Optional[BoxDetectionsSE3]:
        """Inherited, see superclass."""
        return get_box_detections_se3_from_arrow_table(
            self._get_recording_table(),
            self._get_table_index(iteration),
            self.log_metadata,
        )

    def get_traffic_light_detections_at_iteration(self, iteration: int) -> Optional[TrafficLights]:
        """Inherited, see superclass."""
        return get_traffic_light_detections_from_arrow_table(
            self._get_recording_table(), self._get_table_index(iteration)
        )

    def get_route_lane_group_ids(self, iteration: int) -> Optional[List[int]]:
        """Inherited, see superclass."""
        return get_route_lane_group_ids_from_arrow_table(self._get_recording_table(), self._get_table_index(iteration))

    def get_pinhole_camera_at_iteration(self, iteration: int, camera_id: PinholeCameraID) -> Optional[PinholeCamera]:
        """Inherited, see superclass."""
        pinhole_camera: Optional[PinholeCamera] = None
        if camera_id in self.available_pinhole_camera_ids:
            pinhole_camera_ = get_camera_from_arrow_table(
                self._get_recording_table(),
                self._get_table_index(iteration),
                camera_id,
                self.log_metadata,
            )
            assert isinstance(pinhole_camera_, PinholeCamera) or pinhole_camera_ is None
            pinhole_camera = pinhole_camera_
        return pinhole_camera

    def get_fisheye_mei_camera_at_iteration(
        self, iteration: int, camera_id: FisheyeMEICameraID
    ) -> Optional[FisheyeMEICamera]:
        """Inherited, see superclass."""
        fisheye_mei_camera: Optional[FisheyeMEICamera] = None
        if camera_id in self.available_fisheye_mei_camera_ids:
            fisheye_mei_camera_ = get_camera_from_arrow_table(
                self._get_recording_table(),
                self._get_table_index(iteration),
                camera_id,
                self.log_metadata,
            )
            assert isinstance(fisheye_mei_camera_, FisheyeMEICamera) or fisheye_mei_camera_ is None
            fisheye_mei_camera = fisheye_mei_camera_
        return fisheye_mei_camera

    def get_lidar_at_iteration(self, iteration: int, lidar_id: LidarID) -> Optional[Lidar]:
        """Inherited, see superclass."""
        lidar: Optional[Lidar] = None
        if lidar_id in self.available_lidar_ids or lidar_id == LidarID.LIDAR_MERGED:
            lidar = get_lidar_from_arrow_table(
                self._get_recording_table(),
                self._get_table_index(iteration),
                lidar_id,
                self.log_metadata,
            )
        return lidar
