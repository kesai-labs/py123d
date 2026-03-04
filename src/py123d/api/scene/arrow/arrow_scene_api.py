from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Union

import pyarrow as pa

from py123d.api.map.arrow.arrow_map_api import get_global_map_api, get_local_map_api
from py123d.api.map.map_api import MapAPI
from py123d.api.scene.arrow.utils.arrow_getters import (
    get_box_detections_se3_from_arrow_table,
    get_camera_from_arrow_table,
    get_ego_state_se3_from_arrow_table,
    get_lidar_from_arrow_table,
    get_timestamp_from_arrow_table,
    get_traffic_light_detections_from_arrow_table,
)
from py123d.api.scene.arrow.utils.arrow_metadata_utils import (
    get_box_detection_metadata_from_arrow_schema,
    get_ego_metadata_from_arrow_schema,
    get_fisheye_mei_camera_metadatas_from_arrow_schema,
    get_lidar_metadatas_from_arrow_schema,
    get_log_metadata_from_arrow_table,
    get_pinhole_camera_metadatas_from_arrow_schema,
)
from py123d.api.scene.scene_api import SceneAPI
from py123d.api.scene.scene_metadata import SceneMetadata
from py123d.api.utils.arrow_helper import get_lru_cached_arrow_table
from py123d.api.utils.arrow_schema import (
    BOX_DETECTIONS_SE3,
    CUSTOM_MODALITY,
    EGO_STATE_SE3,
    FISHEYE_MEI,
    LIDAR,
    PINHOLE_CAMERA,
    SYNC,
    TRAFFIC_LIGHTS,
)
from py123d.common.utils.msgpack_utils import msgpack_decode_with_numpy
from py123d.common.utils.uuid_utils import convert_to_str_uuid
from py123d.datatypes.custom import CustomModality
from py123d.datatypes.detections import BoxDetectionsSE3, TrafficLightDetections
from py123d.datatypes.detections.box_detection_label_metadata import BoxDetectionMetadata
from py123d.datatypes.metadata.log_metadata import LogMetadata
from py123d.datatypes.metadata.map_metadata import MapMetadata
from py123d.datatypes.sensors import (
    FisheyeMEICamera,
    FisheyeMEICameraID,
    FisheyeMEICameraMetadata,
    Lidar,
    LidarID,
    LidarMetadata,
    PinholeCamera,
    PinholeCameraID,
    PinholeCameraMetadata,
)
from py123d.datatypes.time import Timestamp
from py123d.datatypes.vehicle_state import EgoStateSE3
from py123d.datatypes.vehicle_state.ego_metadata import EgoMetadata


def _get_complete_log_scene_metadata(log_dir: Union[Path, str], log_metadata: LogMetadata) -> SceneMetadata:
    """Helper function to get the scene metadata for a complete log from a log directory."""
    sync_path = Path(log_dir) / f"{SYNC.prefix()}.arrow"
    table = get_lru_cached_arrow_table(sync_path)
    initial_uuid = convert_to_str_uuid(table[SYNC.col("uuid")][0].as_py())
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
    """Scene API for Arrow-based scenes. Loads each modality from a separate Arrow file in a log directory."""

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
                self._log_dir,
                self._scene_metadata,
            ),
        )

    def _get_modality_table(self, modality_name: str) -> Optional[pa.Table]:
        """Load the Arrow table for the given modality, or None if the file does not exist."""
        file_path = self._log_dir / f"{modality_name}.arrow"
        if not file_path.exists():
            return None
        return get_lru_cached_arrow_table(file_path)

    def _get_sync_table(self) -> pa.Table:
        """Load the sync table. This must always exist."""
        table = self._get_modality_table(SYNC.prefix())
        assert table is not None, f"sync.arrow not found in {self._log_dir}"
        return table

    def _get_table_index(self, iteration: int) -> int:
        """Helper function to get the table index for a given iteration."""
        assert -self.number_of_history_iterations <= iteration < self.number_of_iterations, "Iteration out of bounds"
        table_index = self.get_scene_metadata().initial_idx + iteration
        return table_index

    # ------------------------------------------------------------------------------------------------------------------
    # Per-modality metadata retrieval (read from the corresponding Arrow schema)
    # ------------------------------------------------------------------------------------------------------------------

    def get_ego_metadata(self) -> Optional[EgoMetadata]:
        """Returns the :class:`~py123d.datatypes.vehicle_state.EgoMetadata` read from ``ego_state_se3.arrow``."""
        ego_table = self._get_modality_table(EGO_STATE_SE3.prefix())
        if ego_table is None:
            return None
        return get_ego_metadata_from_arrow_schema(ego_table.schema)

    def get_box_detection_metadata(self) -> Optional[BoxDetectionMetadata]:
        """Returns the :class:`~py123d.datatypes.detections.BoxDetectionMetadata` from ``box_detections_se3.arrow``."""
        box_table = self._get_modality_table(BOX_DETECTIONS_SE3.prefix())
        if box_table is None:
            return None
        return get_box_detection_metadata_from_arrow_schema(box_table.schema)

    def get_pinhole_camera_metadatas(self) -> Dict[PinholeCameraID, PinholeCameraMetadata]:
        """Returns per-camera :class:`~py123d.datatypes.sensors.PinholeCameraMetadata` from the Arrow schema."""
        cam_table = self._get_modality_table(PINHOLE_CAMERA.prefix())
        if cam_table is None:
            return {}
        return get_pinhole_camera_metadatas_from_arrow_schema(cam_table.schema) or {}

    def get_fisheye_mei_camera_metadatas(self) -> Dict[FisheyeMEICameraID, FisheyeMEICameraMetadata]:
        """Returns per-camera :class:`~py123d.datatypes.sensors.FisheyeMEICameraMetadata` from the Arrow schema."""
        cam_table = self._get_modality_table(FISHEYE_MEI.prefix())
        if cam_table is None:
            return {}
        return get_fisheye_mei_camera_metadatas_from_arrow_schema(cam_table.schema) or {}

    def get_lidar_metadatas(self) -> Dict[LidarID, LidarMetadata]:
        """Returns per-lidar :class:`~py123d.datatypes.sensors.LidarMetadata` from the lidar Arrow schema."""
        lidar_table = self._get_modality_table(LIDAR.prefix())
        if lidar_table is None:
            return {}
        return get_lidar_metadatas_from_arrow_schema(lidar_table.schema) or {}

    # Implementation of abstract methods
    # ------------------------------------------------------------------------------------------------------------------

    def get_log_metadata(self) -> LogMetadata:
        """Inherited, see superclass."""
        return _get_lru_cached_log_metadata(self._log_dir / f"{SYNC.prefix()}.arrow")

    def get_scene_metadata(self) -> SceneMetadata:
        """Inherited, see superclass."""
        if self._scene_metadata is None:
            log_metadata = self.get_log_metadata()
            self._scene_metadata = _get_complete_log_scene_metadata(self._log_dir, log_metadata)
        return self._scene_metadata

    def get_map_api(self) -> Optional[MapAPI]:
        """Inherited, see superclass."""
        map_api: Optional[MapAPI] = None
        map_metadata = self.get_map_metadata()
        if map_metadata is not None:
            if map_metadata.map_is_local:
                if self._local_map_api is None:
                    map_api = get_local_map_api(self.log_metadata.split, self.log_name)
                    self._local_map_api = map_api
                else:
                    map_api = self._local_map_api
            else:
                map_api = get_global_map_api(self.log_metadata.dataset, self.log_metadata.location)
        return map_api

    def get_map_metadata(self) -> Optional[MapMetadata]:
        """Inherited, see superclass."""
        # TODO: Map metadata should be stored in the map Arrow file once map refactoring is done.
        #       For now, check if a local map directory exists as a heuristic.
        return None

    def get_timestamp_at_iteration(self, iteration: int) -> Timestamp:
        """Inherited, see superclass."""
        return get_timestamp_from_arrow_table(self._get_sync_table(), self._get_table_index(iteration))

    def get_ego_state_se3_at_iteration(self, iteration: int) -> Optional[EgoStateSE3]:
        """Inherited, see superclass."""
        ego_table = self._get_modality_table(EGO_STATE_SE3.prefix())
        if ego_table is None:
            return None
        sync_table = self._get_sync_table()
        idx = self._get_table_index(iteration)
        count = sync_table[SYNC.col("ego_state_se3_count")][idx].as_py()
        if count == 0:
            return None
        offset = sync_table[SYNC.col("ego_state_se3_offset")][idx].as_py()
        ego_metadata = get_ego_metadata_from_arrow_schema(ego_table.schema)
        return get_ego_state_se3_from_arrow_table(ego_table, offset, ego_metadata)

    def get_box_detections_se3_at_iteration(self, iteration: int) -> Optional[BoxDetectionsSE3]:
        """Inherited, see superclass."""
        box_table = self._get_modality_table(BOX_DETECTIONS_SE3.prefix())
        if box_table is None:
            return None
        sync_table = self._get_sync_table()
        idx = self._get_table_index(iteration)
        count = sync_table[SYNC.col("box_detections_se3_count")][idx].as_py()
        if count == 0:
            return None
        offset = sync_table[SYNC.col("box_detections_se3_offset")][idx].as_py()
        box_detection_metadata = get_box_detection_metadata_from_arrow_schema(box_table.schema)
        if box_detection_metadata is None:
            return None
        timestamp = self.get_timestamp_at_iteration(iteration)
        return get_box_detections_se3_from_arrow_table(box_table, offset, box_detection_metadata, timestamp)

    def get_traffic_light_detections_at_iteration(self, iteration: int) -> Optional[TrafficLightDetections]:
        """Inherited, see superclass."""
        tl_table = self._get_modality_table(TRAFFIC_LIGHTS.prefix())
        if tl_table is None:
            return None
        sync_table = self._get_sync_table()
        idx = self._get_table_index(iteration)
        count = sync_table[SYNC.col("traffic_lights_count")][idx].as_py()
        if count == 0:
            return None
        offset = sync_table[SYNC.col("traffic_lights_offset")][idx].as_py()
        return get_traffic_light_detections_from_arrow_table(tl_table, offset)

    def get_pinhole_camera_at_iteration(self, iteration: int, camera_id: PinholeCameraID) -> Optional[PinholeCamera]:
        """Inherited, see superclass."""
        cam_table = self._get_modality_table(PINHOLE_CAMERA.prefix())
        if cam_table is None:
            return None
        cam_metadatas = self.get_pinhole_camera_metadatas()
        cam_meta = cam_metadatas.get(camera_id)
        if cam_meta is None:
            return None
        sync_table = self._get_sync_table()
        idx = self._get_table_index(iteration)
        offset = sync_table[SYNC.col("pinhole_camera_offset")][idx].as_py()
        count = sync_table[SYNC.col("pinhole_camera_count")][idx].as_py()
        for i in range(offset, offset + count):
            row_cam_id = cam_table[PINHOLE_CAMERA.col("camera_id")][i].as_py()
            if row_cam_id == int(camera_id):
                pinhole_camera_ = get_camera_from_arrow_table(cam_table, i, camera_id, cam_meta, self.log_metadata)
                assert isinstance(pinhole_camera_, PinholeCamera) or pinhole_camera_ is None
                return pinhole_camera_
        return None

    def get_fisheye_mei_camera_at_iteration(
        self, iteration: int, camera_id: FisheyeMEICameraID
    ) -> Optional[FisheyeMEICamera]:
        """Inherited, see superclass."""
        cam_table = self._get_modality_table(FISHEYE_MEI.prefix())
        if cam_table is None:
            return None
        cam_metadatas = self.get_fisheye_mei_camera_metadatas()
        cam_meta = cam_metadatas.get(camera_id)
        if cam_meta is None:
            return None
        sync_table = self._get_sync_table()
        idx = self._get_table_index(iteration)
        offset = sync_table[SYNC.col("fisheye_mei_offset")][idx].as_py()
        count = sync_table[SYNC.col("fisheye_mei_count")][idx].as_py()
        for i in range(offset, offset + count):
            row_cam_id = cam_table[FISHEYE_MEI.col("camera_id")][i].as_py()
            if row_cam_id == int(camera_id):
                fisheye_mei_camera_ = get_camera_from_arrow_table(cam_table, i, camera_id, cam_meta, self.log_metadata)
                assert isinstance(fisheye_mei_camera_, FisheyeMEICamera) or fisheye_mei_camera_ is None
                return fisheye_mei_camera_
        return None

    def get_lidar_at_iteration(self, iteration: int, lidar_id: LidarID) -> Optional[Lidar]:
        """Inherited, see superclass."""
        lidar_table = self._get_modality_table(LIDAR.prefix())
        if lidar_table is None:
            return None
        lidar_metadatas = self.get_lidar_metadatas()
        if lidar_id not in lidar_metadatas:
            return None
        sync_table = self._get_sync_table()
        idx = self._get_table_index(iteration)
        offset = sync_table[SYNC.col("lidar_offset")][idx].as_py()
        count = sync_table[SYNC.col("lidar_count")][idx].as_py()
        for i in range(offset, offset + count):
            row_lid_id = lidar_table[LIDAR.col("lidar_id")][i].as_py()
            if row_lid_id == int(lidar_id):
                return get_lidar_from_arrow_table(lidar_table, i, lidar_id, lidar_metadatas, self.log_metadata)
        return None

    def get_custom_modality_at_iteration(self, iteration: int, name: str) -> Optional[CustomModality]:
        """Inherited, see superclass."""
        table = self._get_modality_table(CUSTOM_MODALITY.prefix(name))
        if table is None:
            return None
        idx = self._get_table_index(iteration)
        encoded_data: bytes = table[CUSTOM_MODALITY.col("data", name)][idx].as_py()
        timestamp_us: int = table[CUSTOM_MODALITY.col("timestamp_us", name)][idx].as_py()
        data = msgpack_decode_with_numpy(encoded_data)
        return CustomModality(data=data, timestamp=Timestamp.from_us(timestamp_us))
