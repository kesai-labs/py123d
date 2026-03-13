from pathlib import Path
from typing import List, Literal, Optional

import pyarrow as pa

from py123d.api.scene.arrow.modalities.arrow_base import ArrowBaseModalityWriter
from py123d.api.scene.arrow.modalities.sync_utils import (
    get_all_modality_timestamps,
    get_first_sync_index,
    get_modality_table,
)
from py123d.api.scene.arrow.modalities.utils import all_columns_in_schema
from py123d.api.scene.scene_metadata import SceneMetadata
from py123d.datatypes import (
    Timestamp,
    TrafficLightDetection,
    TrafficLightDetections,
    TrafficLightDetectionsMetadata,
    TrafficLightStatus,
)
from py123d.datatypes.modalities.base_modality import BaseModality

_modality_key = "FIXME"
# ------------------------------------------------------------------------------------------------------------------
# Writer
# ------------------------------------------------------------------------------------------------------------------


class ArrowTrafficLightDetectionsWriter(ArrowBaseModalityWriter):
    def __init__(
        self,
        log_dir: Path,
        metadata: TrafficLightDetectionsMetadata,
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
    ) -> None:
        self._modality_metadata = metadata
        self._modality_key = metadata.modality_type

        file_path = log_dir / f"{metadata.modality_type}.arrow"

        schema = pa.schema(
            [
                (f"{self._modality_key}.timestamp_us", pa.int64()),
                (f"{self._modality_key}.lane_id", pa.list_(pa.int32())),
                (f"{self._modality_key}.status", pa.list_(pa.uint8())),
            ]
        )
        super().__init__(
            file_path=file_path,
            schema=schema,
            ipc_compression=ipc_compression,
            ipc_compression_level=ipc_compression_level,
            max_batch_size=1000,
        )

    def write_modality(self, modality: BaseModality) -> None:
        assert isinstance(modality, TrafficLightDetections), f"Expected TrafficLightDetections, got {type(modality)}"
        lane_id_list = []
        status_list = []

        for traffic_light_detection in modality:
            lane_id_list.append(traffic_light_detection.lane_id)
            status_list.append(traffic_light_detection.status)

        self.write_batch(
            {
                f"{self._modality_key}.timestamp_us": [modality.timestamp.time_us],
                f"{self._modality_key}.lane_id": [lane_id_list],
                f"{self._modality_key}.status": [status_list],
            }
        )


# ------------------------------------------------------------------------------------------------------------------
# Reader
# ------------------------------------------------------------------------------------------------------------------


class ArrowTrafficLightDetectionsReader:
    """Stateless reader for traffic light detections from Arrow tables."""

    @staticmethod
    def read_at_iteration(
        log_dir: Path,
        sync_table: pa.Table,
        table_index: int,
    ) -> Optional[TrafficLightDetections]:
        """Read traffic light detections at a specific sync table index.

        :param log_dir: Path to the log directory.
        :param sync_table: The sync Arrow table.
        :param table_index: The resolved sync table index.
        :return: The traffic light detections, or None if unavailable.
        """
        tl_table = get_modality_table(log_dir, _modality_key)
        if tl_table is None:
            return None
        row_idx = get_first_sync_index(sync_table, _modality_key, table_index)
        if row_idx is None:
            return None
        return _deserialize_traffic_light_detections(tl_table, row_idx)

    @staticmethod
    def read_all_timestamps(
        log_dir: Path,
        sync_table: pa.Table,
        scene_metadata: SceneMetadata,
    ) -> List[Timestamp]:
        """Read all traffic light detection timestamps within the scene range.

        :param log_dir: Path to the log directory.
        :param sync_table: The sync Arrow table.
        :param scene_metadata: Scene metadata defining the iteration range.
        :return: All traffic light detection timestamps in the scene, ordered by time.
        """
        return get_all_modality_timestamps(
            log_dir, sync_table, scene_metadata, _modality_key, f"{_modality_key}.timestamp_us"
        )


def _deserialize_traffic_light_detections(
    arrow_table: pa.Table,
    index: int,
) -> Optional[TrafficLightDetections]:
    """Deserialize traffic light detections from Arrow table columns at the given row index."""
    tl_columns = [
        f"{_modality_key}.timestamp_us",
        f"{_modality_key}.lane_id",
        f"{_modality_key}.status",
    ]
    if not all_columns_in_schema(arrow_table, tl_columns):
        return None

    timestamp = Timestamp.from_us(arrow_table[f"{_modality_key}.timestamp_us"][index].as_py())
    detections: List[TrafficLightDetection] = []
    for lane_id, status in zip(
        arrow_table[f"{_modality_key}.lane_id"][index].as_py(),
        arrow_table[f"{_modality_key}.status"][index].as_py(),
    ):
        detections.append(
            TrafficLightDetection(
                lane_id=lane_id,
                status=TrafficLightStatus(status),
            )
        )
    return TrafficLightDetections(detections=detections, timestamp=timestamp)
