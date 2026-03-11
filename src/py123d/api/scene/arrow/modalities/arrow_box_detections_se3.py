from pathlib import Path
from typing import List, Literal, Optional

import pyarrow as pa

from py123d.api.scene.arrow.modalities.base_modality import BaseModalityWriter
from py123d.api.scene.arrow.modalities.sync_utils import (
    get_all_modality_timestamps,
    get_first_sync_index,
    get_modality_table,
)
from py123d.api.scene.arrow.modalities.utils import all_columns_in_schema, get_optional_array_mixin
from py123d.api.scene.scene_metadata import SceneMetadata
from py123d.api.utils.arrow_metadata_utils import add_metadata_to_arrow_schema
from py123d.datatypes.detections.box_detections import BoxDetectionAttributes, BoxDetectionSE3, BoxDetectionsSE3
from py123d.datatypes.detections.box_detections_metadata import BoxDetectionsSE3Metadata
from py123d.datatypes.time.timestamp import Timestamp
from py123d.geometry.bounding_box import BoundingBoxSE3
from py123d.geometry.geometry_index import BoundingBoxSE3Index, Vector3DIndex
from py123d.geometry.vector import Vector3D

_MODALITY_NAME = "box_detections_se3"


# ------------------------------------------------------------------------------------------------------------------
# Writer
# ------------------------------------------------------------------------------------------------------------------


class ArrowBoxDetectionsSE3Writer(BaseModalityWriter):
    def __init__(
        self,
        log_dir: Path,
        metadata: BoxDetectionsSE3Metadata,
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
    ) -> None:
        self._modality_metadata = metadata
        self._modality_name = metadata.modality_name

        file_path = log_dir / f"{metadata.modality_name}.arrow"

        schema = pa.schema(
            [
                (f"{self._modality_name}.timestamp_us", pa.int64()),
                (f"{self._modality_name}.bounding_box_se3", pa.list_(pa.list_(pa.float64(), len(BoundingBoxSE3Index)))),
                (f"{self._modality_name}.track_token", pa.list_(pa.string())),
                (f"{self._modality_name}.label", pa.list_(pa.uint16())),
                (f"{self._modality_name}.velocity_3d", pa.list_(pa.list_(pa.float64(), len(Vector3DIndex)))),
                (f"{self._modality_name}.num_lidar_points", pa.list_(pa.int32())),
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

    def write_modality(self, box_detections_se3: BoxDetectionsSE3):
        assert isinstance(box_detections_se3, BoxDetectionsSE3), (
            f"Expected BoxDetectionsSE3, got {type(box_detections_se3)}"
        )
        bounding_box_se3_list = []
        token_list = []
        label_list = []
        velocity_3d_list = []
        num_lidar_points_list = []
        for box_detection in box_detections_se3:
            bounding_box_se3_list.append(box_detection.bounding_box_se3)
            token_list.append(box_detection.attributes.track_token)
            label_list.append(box_detection.attributes.label)
            velocity_3d_list.append(box_detection.velocity_3d)
            num_lidar_points_list.append(box_detection.attributes.num_lidar_points)

        self.write_batch(
            {
                f"{self._modality_name}.timestamp_us": [box_detections_se3.timestamp.time_us],
                f"{self._modality_name}.bounding_box_se3": [bounding_box_se3_list],
                f"{self._modality_name}.track_token": [token_list],
                f"{self._modality_name}.label": [label_list],
                f"{self._modality_name}.velocity_3d": [velocity_3d_list],
                f"{self._modality_name}.num_lidar_points": [num_lidar_points_list],
            }
        )


# ------------------------------------------------------------------------------------------------------------------
# Reader
# ------------------------------------------------------------------------------------------------------------------


class ArrowBoxDetectionsSE3Reader:
    """Stateless reader for box detections SE3 data from Arrow tables."""

    @staticmethod
    def read_at_iteration(
        log_dir: Path,
        sync_table: pa.Table,
        table_index: int,
        metadata: Optional[BoxDetectionsSE3Metadata],
    ) -> Optional[BoxDetectionsSE3]:
        """Read box detections at a specific sync table index.

        :param log_dir: Path to the log directory.
        :param sync_table: The sync Arrow table.
        :param table_index: The resolved sync table index.
        :param metadata: Box detections metadata (contains label class info).
        :return: The box detections, or None if unavailable.
        """
        if metadata is None:
            return None
        box_table = get_modality_table(log_dir, _MODALITY_NAME)
        if box_table is None:
            return None
        row_idx = get_first_sync_index(sync_table, _MODALITY_NAME, table_index)
        if row_idx is None:
            return None
        return _deserialize_box_detections_se3(box_table, row_idx, metadata)

    @staticmethod
    def read_all_timestamps(
        log_dir: Path,
        sync_table: pa.Table,
        scene_metadata: SceneMetadata,
    ) -> List[Timestamp]:
        """Read all box detection timestamps within the scene range.

        :param log_dir: Path to the log directory.
        :param sync_table: The sync Arrow table.
        :param scene_metadata: Scene metadata defining the iteration range.
        :return: All box detection timestamps in the scene, ordered by time.
        """
        return get_all_modality_timestamps(
            log_dir, sync_table, scene_metadata, _MODALITY_NAME, f"{_MODALITY_NAME}.timestamp_us"
        )


def _deserialize_box_detections_se3(
    modality_table: pa.Table,
    index: int,
    modality_metadata: BoxDetectionsSE3Metadata,
) -> Optional[BoxDetectionsSE3]:
    """Deserialize box detections from Arrow table columns at the given row index."""
    bd_columns = [
        f"{_MODALITY_NAME}.timestamp_us",
        f"{_MODALITY_NAME}.bounding_box_se3",
        f"{_MODALITY_NAME}.track_token",
        f"{_MODALITY_NAME}.label",
        f"{_MODALITY_NAME}.velocity_3d",
        f"{_MODALITY_NAME}.num_lidar_points",
    ]
    if not all_columns_in_schema(modality_table, bd_columns):
        return None

    timestamp = Timestamp.from_us(modality_table[f"{_MODALITY_NAME}.timestamp_us"][index].as_py())
    box_detections_list: List[BoxDetectionSE3] = []
    box_detection_label_class = modality_metadata.box_detection_label_class
    for _bounding_box_se3, _token, _label, _velocity, _num_lidar_points in zip(
        modality_table[f"{_MODALITY_NAME}.bounding_box_se3"][index].as_py(),
        modality_table[f"{_MODALITY_NAME}.track_token"][index].as_py(),
        modality_table[f"{_MODALITY_NAME}.label"][index].as_py(),
        modality_table[f"{_MODALITY_NAME}.velocity_3d"][index].as_py(),
        modality_table[f"{_MODALITY_NAME}.num_lidar_points"][index].as_py(),
    ):
        box_detections_list.append(
            BoxDetectionSE3(
                attributes=BoxDetectionAttributes(
                    label=box_detection_label_class(_label),
                    track_token=_token,
                    num_lidar_points=_num_lidar_points,
                ),
                bounding_box_se3=BoundingBoxSE3.from_list(_bounding_box_se3),
                velocity_3d=get_optional_array_mixin(_velocity, Vector3D),  # type: ignore
            )
        )
    return BoxDetectionsSE3(box_detections=box_detections_list, timestamp=timestamp)
