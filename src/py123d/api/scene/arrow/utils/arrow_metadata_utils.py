import json
from pathlib import Path
from typing import Dict, Optional, Union

import pyarrow as pa

from py123d.api.utils.arrow_helper import get_lru_cached_arrow_table
from py123d.datatypes.detections.box_detection_label_metadata import BoxDetectionMetadata
from py123d.datatypes.metadata import LogMetadata, MapMetadata
from py123d.datatypes.metadata.abstract_metadata import AbstractMetadata
from py123d.datatypes.sensors.fisheye_mei_camera import FisheyeMEICameraID, FisheyeMEICameraMetadata
from py123d.datatypes.sensors.lidar import LidarID, LidarMetadata
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraID, PinholeCameraMetadata
from py123d.datatypes.vehicle_state.ego_metadata import EgoMetadata

# ------------------------------------------------------------------------------------------------------------------
# Keys used to store per-modality metadata in Arrow schema metadata
# ------------------------------------------------------------------------------------------------------------------

_LOG_METADATA_KEY = b"log_metadata"
_MAP_METADATA_KEY = b"map_metadata"
_EGO_METADATA_KEY = b"ego_metadata"
_BOX_DETECTION_METADATA_KEY = b"box_detection_metadata"
_PINHOLE_CAMERA_METADATA_KEY = b"pinhole_camera_metadata"
_FISHEYE_MEI_CAMERA_METADATA_KEY = b"fisheye_mei_camera_metadata"
_LIDAR_METADATA_KEY = b"lidar_metadata"


# ------------------------------------------------------------------------------------------------------------------
# LogMetadata
# ------------------------------------------------------------------------------------------------------------------


def get_log_metadata_from_arrow_file(arrow_file_path: Union[Path, str]) -> LogMetadata:
    """Gets the log metadata from an Arrow file."""
    table = get_lru_cached_arrow_table(arrow_file_path)
    return get_log_metadata_from_arrow_table(table)


def get_log_metadata_from_arrow_table(arrow_table: pa.Table) -> LogMetadata:
    """Gets the log metadata from an Arrow table."""
    return get_log_metadata_from_arrow_schema(arrow_table.schema)


def get_log_metadata_from_arrow_schema(arrow_schema: pa.Schema) -> LogMetadata:
    """Gets the log metadata from an Arrow schema."""
    return LogMetadata.from_dict(json.loads(arrow_schema.metadata[_LOG_METADATA_KEY].decode()))


def add_log_metadata_to_arrow_schema(schema: pa.Schema, log_metadata: LogMetadata) -> pa.Schema:
    """Adds log metadata to an Arrow schema."""
    existing = dict(schema.metadata) if schema.metadata else {}
    existing[_LOG_METADATA_KEY] = json.dumps(log_metadata.to_dict()).encode()
    return schema.with_metadata(existing)


# ------------------------------------------------------------------------------------------------------------------
# MapMetadata
# ------------------------------------------------------------------------------------------------------------------


def get_map_metadata_from_arrow_table(arrow_table: pa.Table) -> MapMetadata:
    """Gets the map metadata from an Arrow table."""
    return get_map_metadata_from_arrow_schema(arrow_table.schema)


def get_map_metadata_from_arrow_schema(arrow_schema: pa.Schema) -> MapMetadata:
    """Gets the map metadata from an Arrow schema."""
    return MapMetadata.from_dict(json.loads(arrow_schema.metadata[_MAP_METADATA_KEY].decode()))


# ------------------------------------------------------------------------------------------------------------------
# EgoMetadata
# ------------------------------------------------------------------------------------------------------------------


def add_ego_metadata_to_arrow_schema(schema: pa.Schema, ego_metadata: EgoMetadata) -> pa.Schema:
    """Adds ego metadata to an Arrow schema."""
    existing = dict(schema.metadata) if schema.metadata else {}
    existing[_EGO_METADATA_KEY] = json.dumps(ego_metadata.to_dict()).encode()
    return schema.with_metadata(existing)


def get_ego_metadata_from_arrow_schema(arrow_schema: pa.Schema) -> Optional[EgoMetadata]:
    """Gets the ego metadata from an Arrow schema, or None if not present."""
    if arrow_schema.metadata and _EGO_METADATA_KEY in arrow_schema.metadata:
        return EgoMetadata.from_dict(json.loads(arrow_schema.metadata[_EGO_METADATA_KEY].decode()))
    return None


# ------------------------------------------------------------------------------------------------------------------
# BoxDetectionMetadata
# ------------------------------------------------------------------------------------------------------------------


def add_box_detection_metadata_to_arrow_schema(
    schema: pa.Schema, box_detection_metadata: BoxDetectionMetadata
) -> pa.Schema:
    """Adds box detection metadata to an Arrow schema."""
    existing = dict(schema.metadata) if schema.metadata else {}
    existing[_BOX_DETECTION_METADATA_KEY] = json.dumps(box_detection_metadata.to_dict()).encode()
    return schema.with_metadata(existing)


def get_box_detection_metadata_from_arrow_schema(arrow_schema: pa.Schema) -> Optional[BoxDetectionMetadata]:
    """Gets the box detection metadata from an Arrow schema, or None if not present."""
    if arrow_schema.metadata and _BOX_DETECTION_METADATA_KEY in arrow_schema.metadata:
        return BoxDetectionMetadata.from_dict(json.loads(arrow_schema.metadata[_BOX_DETECTION_METADATA_KEY].decode()))
    return None


# ------------------------------------------------------------------------------------------------------------------
# PinholeCameraMetadata (dict of all cameras, keyed by PinholeCameraID int value)
# ------------------------------------------------------------------------------------------------------------------


def add_pinhole_camera_metadatas_to_arrow_schema(
    schema: pa.Schema, camera_metadatas: Dict[PinholeCameraID, PinholeCameraMetadata]
) -> pa.Schema:
    """Adds pinhole camera metadatas dict to an Arrow schema."""
    existing = dict(schema.metadata) if schema.metadata else {}
    serialized = {str(int(cid)): meta.to_dict() for cid, meta in camera_metadatas.items()}
    existing[_PINHOLE_CAMERA_METADATA_KEY] = json.dumps(serialized).encode()
    return schema.with_metadata(existing)


def get_pinhole_camera_metadatas_from_arrow_schema(
    arrow_schema: pa.Schema,
) -> Optional[Dict[PinholeCameraID, PinholeCameraMetadata]]:
    """Gets the pinhole camera metadatas dict from an Arrow schema, or None if not present."""
    if arrow_schema.metadata and _PINHOLE_CAMERA_METADATA_KEY in arrow_schema.metadata:
        raw = json.loads(arrow_schema.metadata[_PINHOLE_CAMERA_METADATA_KEY].decode())
        return {PinholeCameraID(int(k)): PinholeCameraMetadata.from_dict(v) for k, v in raw.items()}
    return None


# ------------------------------------------------------------------------------------------------------------------
# FisheyeMEICameraMetadata (dict of all cameras, keyed by FisheyeMEICameraID int value)
# ------------------------------------------------------------------------------------------------------------------


def add_fisheye_mei_camera_metadatas_to_arrow_schema(
    schema: pa.Schema, camera_metadatas: Dict[FisheyeMEICameraID, FisheyeMEICameraMetadata]
) -> pa.Schema:
    """Adds fisheye MEI camera metadatas dict to an Arrow schema."""
    existing = dict(schema.metadata) if schema.metadata else {}
    serialized = {str(int(cid)): meta.to_dict() for cid, meta in camera_metadatas.items()}
    existing[_FISHEYE_MEI_CAMERA_METADATA_KEY] = json.dumps(serialized).encode()
    return schema.with_metadata(existing)


def get_fisheye_mei_camera_metadatas_from_arrow_schema(
    arrow_schema: pa.Schema,
) -> Optional[Dict[FisheyeMEICameraID, FisheyeMEICameraMetadata]]:
    """Gets the fisheye MEI camera metadatas dict from an Arrow schema, or None if not present."""
    if arrow_schema.metadata and _FISHEYE_MEI_CAMERA_METADATA_KEY in arrow_schema.metadata:
        raw = json.loads(arrow_schema.metadata[_FISHEYE_MEI_CAMERA_METADATA_KEY].decode())
        return {FisheyeMEICameraID(int(k)): FisheyeMEICameraMetadata.from_dict(v) for k, v in raw.items()}
    return None


# ------------------------------------------------------------------------------------------------------------------
# LidarMetadata (dict of all lidars, keyed by LidarID int value)
# ------------------------------------------------------------------------------------------------------------------


def add_lidar_metadatas_to_arrow_schema(schema: pa.Schema, lidar_metadatas: Dict[LidarID, LidarMetadata]) -> pa.Schema:
    """Adds lidar metadata dict to an Arrow schema."""
    existing = dict(schema.metadata) if schema.metadata else {}
    serialized = {str(int(lid)): meta.to_dict() for lid, meta in lidar_metadatas.items()}
    existing[_LIDAR_METADATA_KEY] = json.dumps(serialized).encode()
    return schema.with_metadata(existing)


def get_lidar_metadatas_from_arrow_schema(arrow_schema: pa.Schema) -> Optional[Dict[LidarID, LidarMetadata]]:
    """Gets the lidar metadata dict from an Arrow schema, or None if not present."""
    if arrow_schema.metadata and _LIDAR_METADATA_KEY in arrow_schema.metadata:
        raw = json.loads(arrow_schema.metadata[_LIDAR_METADATA_KEY].decode())
        return {LidarID(int(k)): LidarMetadata.from_dict(v) for k, v in raw.items()}
    return None


# ------------------------------------------------------------------------------------------------------------------
# Generic helper (kept for backward compatibility)
# ------------------------------------------------------------------------------------------------------------------


def add_metadata_to_arrow_schema(schema: pa.Schema, metadata: AbstractMetadata) -> pa.Schema:
    """Adds metadata to an Arrow schema under the generic 'metadata' key."""
    existing = dict(schema.metadata) if schema.metadata else {}
    existing[b"metadata"] = json.dumps(metadata.to_dict()).encode()
    return schema.with_metadata(existing)
