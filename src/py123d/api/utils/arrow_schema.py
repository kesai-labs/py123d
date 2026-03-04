from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pyarrow as pa

from py123d.datatypes import DynamicStateSE3Index
from py123d.geometry import BoundingBoxSE3Index, PoseSE3Index
from py123d.geometry.geometry_index import Vector3DIndex


def _get_uuid_arrow_type():
    """Gets the appropriate Arrow UUID data type based on pyarrow version."""
    # NOTE @DanielDauner: pyarrow introduced native UUID type in version 18.0.0
    if pa.__version__ >= "18.0.0":
        return pa.uuid()
    else:
        return pa.binary(16)


# ----------------------------------------------------------------------------------------------------------------------
# ModalitySchema descriptor
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModalitySchema:
    """Describes one modality's Arrow schema.

    Generates fully-qualified column names, Arrow schema dicts, and column lists
    from a single definition. Supports both fixed modalities (e.g. ``sync``) and
    parametric modalities that take a sensor instance name (e.g. ``pinhole_camera``).
    """

    modality_name: str
    columns: Dict[str, pa.DataType] = field(default_factory=dict)
    parametric: bool = False

    def prefix(self, instance: Optional[str] = None) -> str:
        """Returns the modality prefix, used as file stem and writer key.

        :param instance: Sensor instance name (required for parametric modalities).
        :return: E.g. ``"sync"`` or ``"pinhole_camera.pcam_f0"``.
        """
        if self.parametric:
            assert instance is not None, f"Parametric modality '{self.modality_name}' requires an instance name."
            return f"{self.modality_name}.{instance}"
        return self.modality_name

    def col(self, field_name: str, instance: Optional[str] = None) -> str:
        """Returns a fully-qualified column name.

        :param field_name: The column field name (e.g. ``"uuid"``, ``"data"``).
        :param instance: Sensor instance name (required for parametric modalities).
        :return: E.g. ``"sync.uuid"`` or ``"pinhole_camera.pcam_f0.data"``.
        """
        return f"{self.prefix(instance)}.{field_name}"

    def all_columns(self, instance: Optional[str] = None) -> List[str]:
        """Returns all column names for this modality based on the columns dict.

        :param instance: Sensor instance name (required for parametric modalities).
        :return: List of fully-qualified column names.
        """
        return [self.col(f, instance) for f in self.columns]

    def schema_dict(
        self,
        instance: Optional[str] = None,
        type_overrides: Optional[Dict[str, Optional[pa.DataType]]] = None,
    ) -> Dict[str, pa.DataType]:
        """Generates a ``{qualified_column_name: arrow_type}`` dict for Arrow schema creation.

        :param instance: Sensor instance name (required for parametric modalities).
        :param type_overrides: Optional overrides. Set a value to ``None`` to remove a column,
            or provide a new ``pa.DataType`` to override the default type. New keys add columns.
        :return: Dict suitable for ``pa.schema(list(d.items()))``.
        """
        types = {**self.columns, **(type_overrides or {})}
        types = {k: v for k, v in types.items() if v is not None}
        return {self.col(f, instance): t for f, t in types.items()}


# ----------------------------------------------------------------------------------------------------------------------
# Modality definitions
# ----------------------------------------------------------------------------------------------------------------------

SYNC = ModalitySchema(
    "sync",
    {
        "uuid": _get_uuid_arrow_type(),
        "timestamp_us": pa.int64(),
    },
)

EGO_STATE_SE3 = ModalitySchema(
    "ego_state_se3",
    {
        "imu_se3": pa.list_(pa.float64(), len(PoseSE3Index)),
        "dynamic_state_se3": pa.list_(pa.float64(), len(DynamicStateSE3Index)),
        "timestamp_us": pa.int64(),
    },
)

BOX_DETECTIONS_SE3 = ModalitySchema(
    "box_detections_se3",
    {
        "bounding_box_se3": pa.list_(pa.list_(pa.float64(), len(BoundingBoxSE3Index))),
        "token": pa.list_(pa.string()),
        "label": pa.list_(pa.uint16()),
        "velocity_3d": pa.list_(pa.list_(pa.float64(), len(Vector3DIndex))),
        "num_lidar_points": pa.list_(pa.int32()),
    },
)

TRAFFIC_LIGHTS = ModalitySchema(
    "traffic_lights",
    {
        "lane_id": pa.list_(pa.int32()),
        "status": pa.list_(pa.uint8()),
        "timestamp_us": pa.int64(),
    },
)

PINHOLE_CAMERA = ModalitySchema(
    "pinhole_camera",
    {
        "camera_id": pa.uint8(),
        "data": pa.string(),
        "state_se3": pa.list_(pa.float64(), len(PoseSE3Index)),
        "timestamp_us": pa.int64(),
    },
)

FISHEYE_MEI = ModalitySchema(
    "fisheye_mei",
    {
        "camera_id": pa.uint8(),
        "data": pa.string(),
        "state_se3": pa.list_(pa.float64(), len(PoseSE3Index)),
        "timestamp_us": pa.int64(),
    },
)

LIDAR = ModalitySchema(
    "lidar",
    {
        "lidar_id": pa.uint8(),
        "data": pa.string(),
        "start_timestamp_us": pa.int64(),
        "end_timestamp_us": pa.int64(),
    },
)

CUSTOM_MODALITY = ModalitySchema(
    "custom",
    {
        "data": pa.binary(),
        "timestamp_us": pa.int64(),
    },
    parametric=True,
)


# ----------------------------------------------------------------------------------------------------------------------
# Storage-variant type overrides
# ----------------------------------------------------------------------------------------------------------------------

CAMERA_STORE_TYPES: Dict[str, Dict[str, pa.DataType]] = {
    "path": {},
    "jpeg_binary": {"data": pa.binary()},
    "png_binary": {"data": pa.binary()},
}

LIDAR_STORE_TYPES: Dict[str, Dict[str, Optional[pa.DataType]]] = {
    "path": {},
    "binary": {
        "data": None,
        "point_cloud_3d": pa.binary(),
        "point_cloud_features": pa.binary(),
    },
}
