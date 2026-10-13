"""Shared fixtures and factory functions for scene API tests."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pytest

from py123d.api.utils.arrow_metadata_utils import add_metadata_to_arrow_schema
from py123d.datatypes import (
    BoxDetectionsSE3Metadata,
    CustomModalityMetadata,
    LogMetadata,
    MapMetadata,
    TrafficLightDetectionsMetadata,
)
from py123d.datatypes.detections.box_detection_label import DefaultBoxDetectionLabel
from py123d.datatypes.vehicle_state.dynamic_state import DynamicStateSE3Index
from py123d.datatypes.vehicle_state.ego_state_metadata import EgoStateSE3Metadata
from py123d.geometry.geometry_index import BoundingBoxSE3Index, PoseSE3Index, Vector3DIndex
from py123d.geometry.pose import PoseSE3

# ---------------------------------------------------------------------------
# Cache cleanup
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_arrow_cache():
    """Clear the LRU mmap cache after each test to prevent stale data."""
    yield
    from py123d.api.utils.arrow_helper import clear_arrow_table_cache

    clear_arrow_table_cache()


# ---------------------------------------------------------------------------
# Metadata factories
# ---------------------------------------------------------------------------


def make_log_metadata(
    dataset: str = "test-dataset",
    split: str = "test-dataset_train",
    log_name: str = "log_001",
    location: Optional[str] = "boston",
    map_metadata: Optional[MapMetadata] = None,
) -> LogMetadata:
    """Create a LogMetadata with sensible defaults."""
    return LogMetadata(
        dataset=dataset,
        split=split,
        log_name=log_name,
        location=location,
        map_metadata=map_metadata,
    )


def make_map_metadata(
    dataset: str = "test-dataset",
    location: str = "boston",
    map_has_z: bool = True,
) -> MapMetadata:
    """Create a MapMetadata with sensible defaults."""
    return MapMetadata(
        dataset=dataset,
        location=location,
        map_has_z=map_has_z,
        map_is_per_log=False,
    )


def make_ego_metadata() -> EgoStateSE3Metadata:
    """Create an EgoStateSE3Metadata with identity poses."""
    return EgoStateSE3Metadata(
        vehicle_name="test_vehicle",
        width=2.0,
        length=4.5,
        height=1.6,
        wheel_base=2.8,
        center_to_imu_se3=PoseSE3.identity(),
        rear_axle_to_imu_se3=PoseSE3.identity(),
    )


def make_box_detections_metadata() -> BoxDetectionsSE3Metadata:
    """Create a BoxDetectionsSE3Metadata with default labels."""
    return BoxDetectionsSE3Metadata(
        box_detection_label_class=DefaultBoxDetectionLabel,
    )


def make_traffic_light_metadata() -> TrafficLightDetectionsMetadata:
    """Create a TrafficLightDetectionsMetadata."""
    return TrafficLightDetectionsMetadata()


def make_custom_modality_metadata(modality_id: str = "route") -> CustomModalityMetadata:
    """Create a CustomModalityMetadata."""
    return CustomModalityMetadata(modality_id=modality_id)


# ---------------------------------------------------------------------------
# Arrow file writers
# ---------------------------------------------------------------------------


def write_sync_arrow(
    log_dir: Path,
    num_rows: int,
    timestep_us: int,
    log_metadata: LogMetadata,
    modality_columns: Optional[Dict[str, List[Optional[int]]]] = None,
) -> pa.Table:
    """Write a sync.arrow file to ``log_dir`` and return the table.

    :param modality_columns: Mapping of modality key to list of row indices (None for missing).
    """
    timestamps = (np.arange(num_rows, dtype=np.int64) * timestep_us).tolist()
    uuids = [uuid.uuid4().bytes for _ in range(num_rows)]

    fields = [
        pa.field("sync.uuid", pa.binary(16)),
        pa.field("sync.timestamp_us", pa.int64()),
    ]
    data: Dict[str, list] = {
        "sync.uuid": uuids,
        "sync.timestamp_us": timestamps,
    }

    if modality_columns:
        for key, indices in modality_columns.items():
            fields.append(pa.field(key, pa.int64()))
            data[key] = indices

    schema = pa.schema(fields)
    schema = add_metadata_to_arrow_schema(schema, log_metadata)

    table = pa.table(data, schema=schema)

    sync_path = log_dir / "sync.arrow"
    with pa.OSFile(str(sync_path), "wb") as sink:
        writer = pa.ipc.new_file(sink, table.schema)
        writer.write_table(table)
        writer.close()

    return table


def write_ego_arrow(
    log_dir: Path,
    num_rows: int,
    timestep_us: int,
    metadata: EgoStateSE3Metadata,
) -> pa.Table:
    """Write ego_state_se3.arrow with real schema (PoseSE3 as list of float64)."""
    key = metadata.modality_key
    identity_pose = PoseSE3.identity().tolist()
    zero_dynamic = [0.0] * len(DynamicStateSE3Index)

    timestamps = []
    poses = []
    dynamics = []
    for i in range(num_rows):
        timestamps.append(i * timestep_us)
        poses.append(identity_pose)
        dynamics.append(zero_dynamic)

    schema = pa.schema(
        [
            (f"{key}.timestamp_us", pa.int64()),
            (f"{key}.imu_se3", pa.list_(pa.float64(), len(PoseSE3Index))),
            (f"{key}.dynamic_state_se3", pa.list_(pa.float64(), len(DynamicStateSE3Index))),
        ]
    )
    schema = add_metadata_to_arrow_schema(schema, metadata)

    table = pa.table(
        {
            f"{key}.timestamp_us": timestamps,
            f"{key}.imu_se3": poses,
            f"{key}.dynamic_state_se3": dynamics,
        },
        schema=schema,
    )

    file_path = log_dir / f"{key}.arrow"
    with pa.OSFile(str(file_path), "wb") as sink:
        writer = pa.ipc.new_file(sink, table.schema)
        writer.write_table(table)
        writer.close()

    return table


def write_box_detections_arrow(
    log_dir: Path,
    num_rows: int,
    timestep_us: int,
    metadata: BoxDetectionsSE3Metadata,
    num_boxes_per_frame: int = 2,
) -> pa.Table:
    """Write box_detections_se3.arrow."""
    key = metadata.modality_key

    timestamps = []
    bboxes = []
    tokens = []
    labels = []
    velocities = []
    num_points = []

    for i in range(num_rows):
        timestamps.append(i * timestep_us)
        frame_bboxes = []
        frame_tokens = []
        frame_labels = []
        frame_velocities = []
        frame_points = []
        for j in range(num_boxes_per_frame):
            frame_bboxes.append([float(j), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 4.5, 2.0, 1.6])
            frame_tokens.append(f"track_{j}")
            frame_labels.append(int(DefaultBoxDetectionLabel.VEHICLE))
            frame_velocities.append([1.0, 0.0, 0.0])
            frame_points.append(100)
        bboxes.append(frame_bboxes)
        tokens.append(frame_tokens)
        labels.append(frame_labels)
        velocities.append(frame_velocities)
        num_points.append(frame_points)

    schema = pa.schema(
        [
            (f"{key}.timestamp_us", pa.int64()),
            (f"{key}.bounding_box_se3", pa.list_(pa.list_(pa.float64(), len(BoundingBoxSE3Index)))),
            (f"{key}.track_token", pa.list_(pa.string())),
            (f"{key}.label", pa.list_(pa.uint16())),
            (f"{key}.velocity_3d", pa.list_(pa.list_(pa.float64(), len(Vector3DIndex)))),
            (f"{key}.num_lidar_points", pa.list_(pa.int32())),
        ]
    )
    schema = add_metadata_to_arrow_schema(schema, metadata)

    table = pa.table(
        {
            f"{key}.timestamp_us": timestamps,
            f"{key}.bounding_box_se3": bboxes,
            f"{key}.track_token": tokens,
            f"{key}.label": labels,
            f"{key}.velocity_3d": velocities,
            f"{key}.num_lidar_points": num_points,
        },
        schema=schema,
    )

    file_path = log_dir / f"{key}.arrow"
    with pa.OSFile(str(file_path), "wb") as sink:
        writer = pa.ipc.new_file(sink, table.schema)
        writer.write_table(table)
        writer.close()

    return table


def write_traffic_light_arrow(
    log_dir: Path,
    num_rows: int,
    timestep_us: int,
    metadata: TrafficLightDetectionsMetadata,
    num_lights_per_frame: int = 2,
) -> pa.Table:
    """Write traffic_light_detections.arrow."""
    key = metadata.modality_key

    timestamps = []
    lane_ids = []
    statuses = []

    for i in range(num_rows):
        timestamps.append(i * timestep_us)
        frame_lanes = []
        frame_statuses = []
        for j in range(num_lights_per_frame):
            frame_lanes.append(j + 1)
            frame_statuses.append(j % 3)  # GREEN, YELLOW, RED cycle
        lane_ids.append(frame_lanes)
        statuses.append(frame_statuses)

    schema = pa.schema(
        [
            (f"{key}.timestamp_us", pa.int64()),
            (f"{key}.lane_id", pa.list_(pa.int32())),
            (f"{key}.status", pa.list_(pa.uint8())),
        ]
    )
    schema = add_metadata_to_arrow_schema(schema, metadata)

    table = pa.table(
        {
            f"{key}.timestamp_us": timestamps,
            f"{key}.lane_id": lane_ids,
            f"{key}.status": statuses,
        },
        schema=schema,
    )

    file_path = log_dir / f"{key}.arrow"
    with pa.OSFile(str(file_path), "wb") as sink:
        writer = pa.ipc.new_file(sink, table.schema)
        writer.write_table(table)
        writer.close()

    return table


def write_custom_modality_arrow(
    log_dir: Path,
    num_rows: int,
    timestep_us: int,
    metadata: CustomModalityMetadata,
) -> pa.Table:
    """Write custom.{modality_id}.arrow with msgpack-encoded data."""
    from py123d.common.utils.msgpack_utils import msgpack_encode_with_numpy

    key = metadata.modality_key

    timestamps = []
    data_blobs = []
    for i in range(num_rows):
        timestamps.append(i * timestep_us)
        data_blobs.append(msgpack_encode_with_numpy({"frame": i, "value": float(i) * 0.1}))

    schema = pa.schema(
        [
            (f"{key}.timestamp_us", pa.int64()),
            (f"{key}.data", pa.binary()),
        ]
    )
    schema = add_metadata_to_arrow_schema(schema, metadata)

    table = pa.table(
        {
            f"{key}.timestamp_us": timestamps,
            f"{key}.data": data_blobs,
        },
        schema=schema,
    )

    file_path = log_dir / f"{key}.arrow"
    with pa.OSFile(str(file_path), "wb") as sink:
        writer = pa.ipc.new_file(sink, table.schema)
        writer.write_table(table)
        writer.close()

    return table


# ---------------------------------------------------------------------------
# Composite fixtures
# ---------------------------------------------------------------------------


def make_populated_log_dir(
    log_dir: Path,
    num_rows: int = 10,
    timestep_us: int = 100_000,
) -> Tuple[LogMetadata, EgoStateSE3Metadata, BoxDetectionsSE3Metadata, TrafficLightDetectionsMetadata]:
    """Create a log directory with sync + ego + box_detections + traffic_lights.

    Returns all the metadata objects for further use.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    log_metadata = make_log_metadata()
    ego_metadata = make_ego_metadata()
    box_metadata = make_box_detections_metadata()
    tl_metadata = make_traffic_light_metadata()

    ego_key = ego_metadata.modality_key
    box_key = box_metadata.modality_key
    tl_key = tl_metadata.modality_key

    # All modalities present at every frame
    modality_columns = {
        ego_key: list(range(num_rows)),
        box_key: list(range(num_rows)),
        tl_key: list(range(num_rows)),
    }

    write_sync_arrow(log_dir, num_rows, timestep_us, log_metadata, modality_columns)
    write_ego_arrow(log_dir, num_rows, timestep_us, ego_metadata)
    write_box_detections_arrow(log_dir, num_rows, timestep_us, box_metadata)
    write_traffic_light_arrow(log_dir, num_rows, timestep_us, tl_metadata)

    return log_metadata, ego_metadata, box_metadata, tl_metadata
