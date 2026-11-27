"""Roundtrip tests of the 2D box stream (box_detections_2d.<camera_id>)."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from py123d.api.scene.arrow.arrow_scene_api import MODALITY_READERS
from py123d.api.scene.arrow.modalities.arrow_box_detections_2d import (
    ArrowBoxDetections2DReader,
    ArrowBoxDetections2DWriter,
)
from py123d.api.utils.arrow_metadata_utils import get_metadata_from_arrow_schema, resolve_metadata_class
from py123d.datatypes import (
    AnonymizationBoxDetectionLabel,
    BoxDetection2D,
    BoxDetections2D,
    BoxDetections2DMetadata,
    CameraID,
    DefaultBoxDetectionLabel,
    ModalityType,
    Timestamp,
)
from py123d.parser.registry import WODPerceptionBoxDetectionLabel

T0 = 1_784_208_918_000_000
FACE, PLATE = AnonymizationBoxDetectionLabel.FACE, AnonymizationBoxDetectionLabel.LICENSE_PLATE


def _metadata() -> BoxDetections2DMetadata:
    return BoxDetections2DMetadata(
        camera_id="ftcam_f0",
        box_detection_label_class=AnonymizationBoxDetectionLabel,
        origin_names=["detection", "filled_gap", "extension"],
        info={"package": "test", "version": "1.0", "settings": {"threshold": 0.3}},
    )


def _frames(metadata: BoxDetections2DMetadata):
    """3 images: 2 boxes (one without a region, one with a track token), none, 1 box."""
    return [
        BoxDetections2D(
            [
                BoxDetection2D((10.5, 20.25, 30.75, 40.0), label=FACE, score=0.5, origin=0, region=(9, 18, 33, 43)),
                BoxDetection2D((100.0, 200.0, 150.5, 230.5), label=PLATE, score=0.875, origin=2, track_token="t7"),
            ],
            Timestamp.from_us(T0),
            metadata,
        ),
        BoxDetections2D([], Timestamp.from_us(T0 + 100_000), metadata),
        BoxDetections2D(
            [BoxDetection2D((1.0, 2.0, 3.0, 4.0), label=PLATE, score=0.25, origin=1, region=(0, 1, 4, 5))],
            Timestamp.from_us(T0 + 200_000),
            metadata,
        ),
    ]


def _write(log_dir: Path, frames) -> pa.Table:
    metadata = frames[0].metadata
    writer = ArrowBoxDetections2DWriter(log_dir=log_dir, metadata=metadata)
    for frame in frames:
        writer.write_modality(frame)
    writer.close()
    return pa.ipc.open_file(str(log_dir / f"{metadata.modality_key}.arrow")).read_all()


def test_metadata_keys_and_dict_roundtrip():
    metadata = _metadata()
    assert metadata.modality_type == ModalityType.BOX_DETECTIONS_2D
    assert metadata.modality_key == "box_detections_2d.ftcam_f0"
    assert metadata.camera_key == "camera.ftcam_f0"
    assert metadata.camera_id is CameraID.FTCAM_F0
    assert BoxDetections2DMetadata(CameraID.FTCAM_F0, AnonymizationBoxDetectionLabel).camera_id is CameraID.FTCAM_F0
    assert BoxDetections2DMetadata.from_dict(metadata.to_dict()) == metadata
    assert resolve_metadata_class("box_detections_2d.ftcam_f0") is BoxDetections2DMetadata
    assert MODALITY_READERS[ModalityType.BOX_DETECTIONS_2D] is ArrowBoxDetections2DReader
    with pytest.raises(KeyError):
        BoxDetections2DMetadata("camera.ftcam_f0", AnonymizationBoxDetectionLabel)


def test_write_and_read_every_row(tmp_path: Path):
    frames = _frames(_metadata())
    table = _write(tmp_path, frames)
    assert table.num_rows == 3
    metadata = get_metadata_from_arrow_schema(table.schema, BoxDetections2DMetadata)
    assert metadata == frames[0].metadata and metadata.info["settings"] == {"threshold": 0.3}
    for row, frame in enumerate(frames):
        read = ArrowBoxDetections2DReader.read_at_index(row, table, metadata, "test")
        assert read is not None
        assert read.timestamp.time_us == frame.timestamp.time_us
        assert list(read) == list(frame)  # the box values are exact in float32
    first = ArrowBoxDetections2DReader.read_at_index(0, table, metadata, "test")
    assert [b.label for b in first] == [FACE, PLATE]
    assert [b.default_label for b in first] == [DefaultBoxDetectionLabel.OTHER] * 2
    assert [first.origin_name(b) for b in first] == ["detection", "extension"]
    assert first[1].region is None
    assert [b.track_token for b in first] == [None, "t7"]


def test_read_column_and_timestamp_lookup(tmp_path: Path):
    frames = _frames(_metadata())
    table = _write(tmp_path, frames)
    metadata = frames[0].metadata
    reader = ArrowBoxDetections2DReader
    assert reader.read_column_at_index(2, table, metadata, "label", "test") == [1]
    assert reader.read_column_at_index(2, table, metadata, "label", "test", deserialize=True) == [PLATE]
    assert reader.read_column_at_index(1, table, metadata, "box", "test") == []
    assert reader.read_column_at_index(0, table, metadata, "timestamp_us", "test", deserialize=True).time_us == T0
    assert reader.read_column_at_index(0, table, metadata, "missing", "test") is None
    found = reader.read_at_timestamp(Timestamp.from_us(T0 + 190_000), table, metadata, "test", criteria="nearest")
    assert found is not None and found.timestamp.time_us == T0 + 200_000


def test_labels_must_be_of_the_label_class_and_origins_indices(tmp_path: Path):
    metadata = _metadata()
    writer = ArrowBoxDetections2DWriter(log_dir=tmp_path, metadata=metadata)
    for label in (0, WODPerceptionBoxDetectionLabel.TYPE_VEHICLE):  # a plain index, a label of another class
        with pytest.raises(ValueError):
            writer.write_modality(
                BoxDetections2D([BoxDetection2D((0, 0, 1, 1), label=label)], Timestamp.from_us(T0), metadata)
            )
    with pytest.raises(ValueError):
        writer.write_modality(
            BoxDetections2D([BoxDetection2D((0, 0, 1, 1), label=FACE, origin=3)], Timestamp.from_us(T0), metadata)
        )
    writer.close()
    with pytest.raises(ValueError):
        BoxDetection2D((0, 0, 1), label=FACE)
