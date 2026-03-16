from pathlib import Path
from typing import Any, List, Literal, Optional

import pyarrow as pa

from py123d.api.scene.arrow.modalities.arrow_base import ArrowBaseModalityReader, ArrowBaseModalityWriter
from py123d.api.scene.arrow.modalities.utils import all_columns_in_schema
from py123d.api.utils.arrow_metadata_utils import add_metadata_to_arrow_schema
from py123d.datatypes import (
    Timestamp,
    TrafficLightDetection,
    TrafficLightDetections,
    TrafficLightDetectionsMetadata,
    TrafficLightStatus,
)
from py123d.datatypes.modalities.base_modality import BaseModality, BaseModalityMetadata

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
        self._modality_key = metadata.modality_key

        file_path = log_dir / f"{metadata.modality_key}.arrow"

        schema = pa.schema(
            [
                (f"{self._modality_key}.timestamp_us", pa.int64()),
                (f"{self._modality_key}.lane_id", pa.list_(pa.int32())),
                (f"{self._modality_key}.status", pa.list_(pa.uint8())),
            ]
        )
        schema = add_metadata_to_arrow_schema(schema, metadata)
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


class ArrowTrafficLightDetectionsReader(ArrowBaseModalityReader):
    """Stateless reader for traffic light detections from Arrow tables."""

    @staticmethod
    def read_at_index(
        index: int,
        table: pa.Table,
        metadata: BaseModalityMetadata,
        dataset: str,
        **kwargs,
    ) -> Optional[TrafficLightDetections]:
        assert isinstance(metadata, TrafficLightDetectionsMetadata)
        return _deserialize_traffic_light_detections(table, index, metadata)

    @staticmethod
    def read_column_at_index(
        index: int,
        table: pa.Table,
        metadata: BaseModalityMetadata,
        column: str,
        dataset: str,
        deserialize: bool = False,
        **kwargs,
    ) -> Optional[Any]:
        """Return a single column value at the given row index. For traffic light detections, we support reading timestamp_us, lane_id, and status columns."""
        assert isinstance(metadata, TrafficLightDetectionsMetadata)
        full_column_name = f"{metadata.modality_key}.{column}"
        column_at_iteration: Optional[Any] = None
        if full_column_name in table.column_names:
            column_at_iteration = table[full_column_name][index].as_py()
            if deserialize and column == "timestamp_us":
                column_at_iteration = Timestamp.from_us(column_at_iteration)  # type: ignore
            elif deserialize and column == "status":
                column_at_iteration = [TrafficLightStatus(s) for s in column_at_iteration]  # type: ignore
            elif deserialize and column == "lane_id":
                column_at_iteration = [int(lane_id) for lane_id in column_at_iteration]  # type: ignore
        return column_at_iteration


def _deserialize_traffic_light_detections(
    arrow_table: pa.Table,
    index: int,
    metadata: TrafficLightDetectionsMetadata,
) -> Optional[TrafficLightDetections]:
    """Deserialize traffic light detections from Arrow table columns at the given row index."""
    modality_key = metadata.modality_key
    tl_columns = [
        f"{modality_key}.timestamp_us",
        f"{modality_key}.lane_id",
        f"{modality_key}.status",
    ]
    if not all_columns_in_schema(arrow_table, tl_columns):
        return None

    timestamp = Timestamp.from_us(arrow_table[f"{modality_key}.timestamp_us"][index].as_py())
    detections: List[TrafficLightDetection] = []
    for lane_id, status in zip(
        arrow_table[f"{modality_key}.lane_id"][index].as_py(),
        arrow_table[f"{modality_key}.status"][index].as_py(),
    ):
        detections.append(
            TrafficLightDetection(
                lane_id=lane_id,
                status=TrafficLightStatus(status),
            )
        )
    return TrafficLightDetections(detections=detections, timestamp=timestamp, metadata=metadata)
