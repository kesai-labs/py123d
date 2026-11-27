"""Arrow writer and reader of 2D box streams (``box_detections_2d.<camera_id>.arrow``).

Columns (per row, i.e. per camera image; the box columns are lists with one entry per box):

- ``timestamp_us`` (int64): the time stamp of the camera image;
- ``box`` (list of 4 float32): the object box ``x1, y1, x2, y2`` in pixels of the stored image;
- ``label`` (list of uint16): the value of a member of the metadata's label class;
- ``score`` (list of float32);
- ``origin`` (list of uint8): an index into the metadata's origin names;
- ``region`` (list of 4 int32 or null): the pixel region a processing step applied to (``x2``, ``y2`` exclusive);
- ``track_token`` (list of string or null): the track token of the box.

The metadata (:class:`~py123d.datatypes.detections.box_detections_2d.BoxDetections2DMetadata`) is stored in the
schema metadata.
"""

from pathlib import Path
from typing import Any, List, Literal, Optional

import pyarrow as pa

from py123d.api.scene.arrow.modalities.arrow_base import ArrowBaseModalityReader, ArrowBaseModalityWriter
from py123d.api.scene.arrow.modalities.utils import all_columns_in_schema
from py123d.api.utils.arrow_metadata_utils import add_metadata_to_arrow_schema
from py123d.datatypes.detections.box_detections_2d import BoxDetection2D, BoxDetections2D, BoxDetections2DMetadata
from py123d.datatypes.modalities.base_modality import BaseModality, BaseModalityMetadata
from py123d.datatypes.time.timestamp import Timestamp

BOX_DETECTIONS_2D_COLUMNS = ("timestamp_us", "box", "label", "score", "origin", "region", "track_token")


def box_detections_2d_schema(modality_key: str) -> pa.Schema:
    """The Arrow schema of a 2D box stream (without its metadata)."""
    return pa.schema(
        [
            (f"{modality_key}.timestamp_us", pa.int64()),
            (f"{modality_key}.box", pa.list_(pa.list_(pa.float32(), 4))),
            (f"{modality_key}.label", pa.list_(pa.uint16())),
            (f"{modality_key}.score", pa.list_(pa.float32())),
            (f"{modality_key}.origin", pa.list_(pa.uint8())),
            (f"{modality_key}.region", pa.list_(pa.list_(pa.int32(), 4))),
            (f"{modality_key}.track_token", pa.list_(pa.string())),
        ]
    )


# ------------------------------------------------------------------------------------------------------------------
# Writer
# ------------------------------------------------------------------------------------------------------------------


class ArrowBoxDetections2DWriter(ArrowBaseModalityWriter):
    """Writes one 2D box stream: one row per camera image, in time order."""

    def __init__(
        self,
        log_dir: Path,
        metadata: BoxDetections2DMetadata,
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
    ) -> None:
        self._modality_metadata = metadata
        self._modality_key = metadata.modality_key
        schema = add_metadata_to_arrow_schema(box_detections_2d_schema(self._modality_key), metadata)
        super().__init__(
            file_path=Path(log_dir) / f"{self._modality_key}.arrow",
            schema=schema,
            ipc_compression=ipc_compression,
            ipc_compression_level=ipc_compression_level,
            max_batch_size=1000,
        )

    def write_modality(self, modality: BaseModality) -> None:
        assert isinstance(modality, BoxDetections2D), f"Expected BoxDetections2D, got {type(modality)}"
        metadata = self._modality_metadata
        for detection in modality:
            if not isinstance(detection.label, metadata.box_detection_label_class):
                raise ValueError(f"Label {detection.label!r} is not a {metadata.box_detection_label_class.__name__}.")
            if not 0 <= detection.origin < len(metadata.origin_names):
                raise ValueError(f"Origin {detection.origin} is not an index into {list(metadata.origin_names)}.")
        key = self._modality_key
        self.write_batch(
            {
                f"{key}.timestamp_us": [modality.timestamp.time_us],
                f"{key}.box": [[list(d.box) for d in modality]],
                f"{key}.label": [[int(d.label) for d in modality]],
                f"{key}.score": [[d.score for d in modality]],
                f"{key}.origin": [[d.origin for d in modality]],
                f"{key}.region": [[None if d.region is None else list(d.region) for d in modality]],
                f"{key}.track_token": [[d.track_token for d in modality]],
            }
        )


# ------------------------------------------------------------------------------------------------------------------
# Reader
# ------------------------------------------------------------------------------------------------------------------


class ArrowBoxDetections2DReader(ArrowBaseModalityReader):
    """Stateless reader of 2D box streams."""

    @staticmethod
    def read_at_index(
        index: int,
        table: pa.Table,
        metadata: BaseModalityMetadata,
        dataset: str,
        **kwargs,
    ) -> Optional[BoxDetections2D]:
        assert isinstance(metadata, BoxDetections2DMetadata)
        return _deserialize_box_detections_2d(table, index, metadata)

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
        """A single column at a row: ``timestamp_us`` (a :class:`Timestamp` if ``deserialize``) or one of the box
        columns as plain lists (``label`` as members of the label class if ``deserialize``)."""
        assert isinstance(metadata, BoxDetections2DMetadata)
        full_column_name = f"{metadata.modality_key}.{column}"
        if full_column_name not in table.column_names:
            return None
        value = table[full_column_name][index].as_py()
        if deserialize and column == "timestamp_us":
            return Timestamp.from_us(value)
        if deserialize and column == "label":
            return [metadata.box_detection_label_class(label) for label in value]
        return value


def _deserialize_box_detections_2d(
    table: pa.Table, index: int, metadata: BoxDetections2DMetadata
) -> Optional[BoxDetections2D]:
    key = metadata.modality_key
    columns = [f"{key}.{name}" for name in BOX_DETECTIONS_2D_COLUMNS]
    if not all_columns_in_schema(table, columns):
        return None
    values = {name: table[f"{key}.{name}"][index].as_py() for name in BOX_DETECTIONS_2D_COLUMNS}
    label_class = metadata.box_detection_label_class
    detections: List[BoxDetection2D] = [
        BoxDetection2D(
            box=box, label=label_class(label), score=score, origin=origin, region=region, track_token=track_token
        )
        for box, label, score, origin, region, track_token in zip(
            values["box"], values["label"], values["score"], values["origin"], values["region"], values["track_token"]
        )
    ]
    return BoxDetections2D(
        box_detections=detections, timestamp=Timestamp.from_us(values["timestamp_us"]), metadata=metadata
    )
