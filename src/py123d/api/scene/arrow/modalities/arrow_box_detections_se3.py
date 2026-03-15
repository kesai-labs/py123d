from pathlib import Path
from typing import Any, List, Literal, Optional, Type
from warnings import warn

import pyarrow as pa

from py123d.api.scene.arrow.modalities.arrow_base import ArrowBaseModalityReader, ArrowBaseModalityWriter
from py123d.api.scene.arrow.modalities.utils import all_columns_in_schema, get_optional_array_mixin
from py123d.api.utils.arrow_metadata_utils import add_metadata_to_arrow_schema
from py123d.datatypes.detections.box_detection_label import BoxDetectionLabel
from py123d.datatypes.detections.box_detections import BoxDetectionAttributes, BoxDetectionSE3, BoxDetectionsSE3
from py123d.datatypes.detections.box_detections_metadata import BoxDetectionsSE3Metadata
from py123d.datatypes.modalities.base_modality import BaseModality, BaseModalityMetadata
from py123d.datatypes.time.timestamp import Timestamp
from py123d.geometry.bounding_box import BoundingBoxSE3
from py123d.geometry.geometry_index import BoundingBoxSE3Index, Vector3DIndex
from py123d.geometry.vector import Vector3D

# ------------------------------------------------------------------------------------------------------------------
# Writer
# ------------------------------------------------------------------------------------------------------------------


class ArrowBoxDetectionsSE3Writer(ArrowBaseModalityWriter):
    def __init__(
        self,
        log_dir: Path,
        metadata: BoxDetectionsSE3Metadata,
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
    ) -> None:
        self._modality_metadata = metadata
        self._modality_key = metadata.modality_key

        file_path = log_dir / f"{metadata.modality_key}.arrow"

        schema = pa.schema(
            [
                (f"{self._modality_key}.timestamp_us", pa.int64()),
                (f"{self._modality_key}.bounding_box_se3", pa.list_(pa.list_(pa.float64(), len(BoundingBoxSE3Index)))),
                (f"{self._modality_key}.track_token", pa.list_(pa.string())),
                (f"{self._modality_key}.label", pa.list_(pa.uint16())),
                (f"{self._modality_key}.velocity_3d", pa.list_(pa.list_(pa.float64(), len(Vector3DIndex)))),
                (f"{self._modality_key}.num_lidar_points", pa.list_(pa.int32())),
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

    def write_modality(self, modality: BaseModality):
        assert isinstance(modality, BoxDetectionsSE3), f"Expected BoxDetectionsSE3, got {type(modality)}"
        bounding_box_se3_list = []
        token_list = []
        label_list = []
        velocity_3d_list = []
        num_lidar_points_list = []
        for box_detection in modality:
            bounding_box_se3_list.append(box_detection.bounding_box_se3)
            token_list.append(box_detection.attributes.track_token)
            label_list.append(box_detection.attributes.label)
            velocity_3d_list.append(box_detection.velocity_3d)
            num_lidar_points_list.append(box_detection.attributes.num_lidar_points)

        self.write_batch(
            {
                f"{self._modality_key}.timestamp_us": [modality.timestamp.time_us],
                f"{self._modality_key}.bounding_box_se3": [bounding_box_se3_list],
                f"{self._modality_key}.track_token": [token_list],
                f"{self._modality_key}.label": [label_list],
                f"{self._modality_key}.velocity_3d": [velocity_3d_list],
                f"{self._modality_key}.num_lidar_points": [num_lidar_points_list],
            }
        )


# ------------------------------------------------------------------------------------------------------------------
# Reader
# ------------------------------------------------------------------------------------------------------------------


class ArrowBoxDetectionsSE3Reader(ArrowBaseModalityReader):
    """Stateless reader for box detections SE3 data from Arrow tables."""

    @staticmethod
    def read_at_index(
        index: int,
        table: pa.Table,
        metadata: BaseModalityMetadata,
        dataset: str,
        **kwargs,
    ) -> Optional[BoxDetectionsSE3]:
        assert isinstance(metadata, BoxDetectionsSE3Metadata)
        return _deserialize_box_detections_se3(table, index, metadata)

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
        column_at_iteration: Optional[Any] = None

        full_column_name = f"{metadata.modality_key}.{column}"
        if full_column_name in table.column_names:
            column_at_iteration = table[full_column_name][index].as_py()  # type: ignore
        if deserialize:
            if column == "bounding_box_se3":
                column_at_iteration = _deserialize_bounding_box_se3_list(column_at_iteration)  # type: ignore
            elif column == "velocity_3d":
                column_at_iteration = _deserialize_velocity_3d_list(column_at_iteration)  # type: ignore
            elif column == "label":
                column_at_iteration = _deserialize_label_list(
                    column_at_iteration,  # type: ignore
                    metadata.box_detection_label_class,  # type: ignore
                )  # type: ignore
            else:
                warn(
                    f"Deserialization for column '{column}' is not implemented. Returning raw value.",
                    category=UserWarning,
                )

        return column_at_iteration


def _deserialize_bounding_box_se3_list(bounding_box_se3_list: List[List[float]]) -> List[BoundingBoxSE3]:
    """Deserialize a list of bounding boxes from a list of lists of floats."""
    deserialized_list = []
    for bounding_box_se3 in bounding_box_se3_list:
        deserialized_list.append(BoundingBoxSE3.from_list(bounding_box_se3))
    return deserialized_list


def _deserialize_velocity_3d_list(velocity_3d_list: List[List[float]]) -> List[Optional[Vector3D]]:
    """Deserialize a list of 3D velocities from a list of lists of floats."""
    deserialized_list = []
    for velocity_3d in velocity_3d_list:
        deserialized_list.append(get_optional_array_mixin(velocity_3d, Vector3D))  # type: ignore
    return deserialized_list


def _deserialize_label_list(label_list: List[str], label_class: Type[BoxDetectionLabel]) -> List[BoxDetectionLabel]:
    """Deserialize a list of labels from a list of strings."""
    return [label_class(label) for label in label_list]


def _deserialize_box_detections_se3(
    modality_table: pa.Table,
    index: int,
    modality_metadata: BoxDetectionsSE3Metadata,
) -> Optional[BoxDetectionsSE3]:
    """Deserialize box detections from Arrow table columns at the given row index."""
    bd_columns = [
        f"{modality_metadata.modality_key}.timestamp_us",
        f"{modality_metadata.modality_key}.bounding_box_se3",
        f"{modality_metadata.modality_key}.track_token",
        f"{modality_metadata.modality_key}.label",
        f"{modality_metadata.modality_key}.velocity_3d",
        f"{modality_metadata.modality_key}.num_lidar_points",
    ]
    if not all_columns_in_schema(modality_table, bd_columns):
        return None

    timestamp = Timestamp.from_us(modality_table[f"{modality_metadata.modality_key}.timestamp_us"][index].as_py())
    box_detections_list: List[BoxDetectionSE3] = []
    box_detection_label_class = modality_metadata.box_detection_label_class
    for _bounding_box_se3, _token, _label, _velocity, _num_lidar_points in zip(
        modality_table[f"{modality_metadata.modality_key}.bounding_box_se3"][index].as_py(),
        modality_table[f"{modality_metadata.modality_key}.track_token"][index].as_py(),
        modality_table[f"{modality_metadata.modality_key}.label"][index].as_py(),
        modality_table[f"{modality_metadata.modality_key}.velocity_3d"][index].as_py(),
        modality_table[f"{modality_metadata.modality_key}.num_lidar_points"][index].as_py(),
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
    return BoxDetectionsSE3(
        box_detections=box_detections_list,
        timestamp=timestamp,
        metadata=modality_metadata,
    )
