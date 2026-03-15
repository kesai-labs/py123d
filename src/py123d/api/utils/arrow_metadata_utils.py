import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

import msgpack
import pyarrow as pa

from py123d.datatypes.metadata.base_metadata import BaseMetadata
from py123d.datatypes.metadata.log_metadata import LogMetadata
from py123d.datatypes.modalities.base_modality import BaseModalityMetadata, ModalityType, get_modality_type_from_key

T = TypeVar("T", bound=BaseMetadata)

logger = logging.getLogger(__name__)

# _LOG_METADATA_KEY = b"log_metadata"
_METADATA_KEY = b"metadata"

# Files in a log directory that are not per-modality data.
_NON_MODALITY_FILES = {"sync.arrow", "map.arrow"}


def get_metadata_from_arrow_schema(
    arrow_schema: pa.Schema,
    metadata_class: type[T],
    modality_key: bytes = _METADATA_KEY,
) -> T:
    """Gets metadata for a specific modality from an Arrow schema."""

    deserialized_metadata = None
    if arrow_schema.metadata is not None and modality_key in arrow_schema.metadata:
        deserialized_metadata = metadata_class.from_dict(
            msgpack.unpackb(arrow_schema.metadata[modality_key], raw=False)
        )

    try:
        assert deserialized_metadata is not None, (
            f"Metadata for modality key '{modality_key.decode()}' not found in Arrow schema."
        )
    except AssertionError as e:
        available_keys = [k.decode() for k in arrow_schema.metadata.keys()] if arrow_schema.metadata else []
        raise ValueError(f"{str(e)} Available metadata keys: {available_keys}") from e
    return deserialized_metadata  # type: ignore


def add_metadata_to_arrow_schema(
    schema: pa.Schema,
    metadata: BaseMetadata,
    modality_key: bytes = _METADATA_KEY,
) -> pa.Schema:
    """Adds metadata for a specific modality to an Arrow schema."""
    existing = dict(schema.metadata) if schema.metadata else {}
    existing[modality_key] = msgpack.packb(metadata.to_dict(), use_bin_type=True)
    return schema.with_metadata(existing)


# ------------------------------------------------------------------------------------------------------------------
# Log Directory Metadata Parsing
# ------------------------------------------------------------------------------------------------------------------


def _get_modality_metadata_registry() -> Dict[ModalityType, Any]:
    """Returns the registry mapping ModalityType to its default metadata class.

    Imports are deferred to avoid circular dependencies and to keep the module
    lightweight when only the basic get/add helpers are needed.
    """
    from py123d.datatypes.custom.custom_modality import CustomModalityMetadata
    from py123d.datatypes.detections.box_detections_metadata import BoxDetectionsSE3Metadata
    from py123d.datatypes.detections.traffic_light_detections import TrafficLightDetectionsMetadata
    from py123d.datatypes.sensors.base_camera import camera_metadata_from_dict
    from py123d.datatypes.sensors.lidar import LidarMetadata
    from py123d.datatypes.vehicle_state.ego_state_metadata import EgoStateSE3Metadata

    # _CameraMetadataFactory acts as a drop-in for a metadata class: its from_dict()
    # reads the "camera_model" discriminator and dispatches to the correct subclass.
    class _CameraMetadataFactory:
        from_dict = staticmethod(camera_metadata_from_dict)

    return {
        ModalityType.EGO_STATE_SE3: EgoStateSE3Metadata,
        ModalityType.BOX_DETECTIONS_SE3: BoxDetectionsSE3Metadata,
        ModalityType.TRAFFIC_LIGHT_DETECTIONS: TrafficLightDetectionsMetadata,
        ModalityType.CAMERA: _CameraMetadataFactory,
        ModalityType.LIDAR: LidarMetadata,
        ModalityType.CUSTOM: CustomModalityMetadata,
    }


def _get_modality_key_overrides() -> Dict[str, Type[BaseModalityMetadata]]:
    """Returns overrides for specific modality keys where ModalityType alone is ambiguous."""
    from py123d.datatypes.sensors.lidar import LidarMergedMetadata

    return {
        "lidar.lidar_merged": LidarMergedMetadata,
    }


def resolve_metadata_class(modality_key: str) -> Any:
    """Resolve the metadata class for a given modality key.

    Uses the modality type encoded in the key (the part before the first ``"."``)
    to look up the appropriate metadata class from a registry. Specific modality
    keys (e.g. ``"lidar.lidar_merged"``) can override the default for their type.

    :param modality_key: The modality key, e.g. ``"ego_state_se3"`` or ``"pinhole_camera.pcam_f0"``.
    :return: The metadata class to use for deserialization.
    :raises ValueError: If no metadata class is registered for the modality type.
    """
    overrides = _get_modality_key_overrides()
    if modality_key in overrides:
        return overrides[modality_key]

    modality_type = get_modality_type_from_key(modality_key)
    registry = _get_modality_metadata_registry()
    if modality_type not in registry:
        raise ValueError(f"No metadata class registered for modality type '{modality_type}' (key: '{modality_key}')")
    return registry[modality_type]


@dataclass
class LogDirectoryMetadata:
    """All metadata parsed from a log directory's Arrow files.

    :param log_metadata: The :class:`~py123d.datatypes.LogMetadata` from ``sync.arrow``.
    :param modality_metadatas: Mapping of modality key to its deserialized metadata object.
    """

    log_metadata: LogMetadata  # noqa: F821 — forward ref to avoid top-level import
    modality_metadatas: Dict[str, BaseModalityMetadata] = field(default_factory=dict)

    def get(self, modality_key: str) -> BaseModalityMetadata:
        """Get metadata for a specific modality key.

        :param modality_key: e.g. ``"ego_state_se3"`` or ``"pinhole_camera.pcam_f0"``.
        :raises KeyError: If the modality key is not found.
        """
        return self.modality_metadatas[modality_key]

    def get_by_type(self, modality_type: ModalityType) -> Dict[str, BaseModalityMetadata]:
        """Get all metadata entries matching a given modality type.

        :param modality_type: The :class:`ModalityType` to filter by.
        :return: Dict of matching modality keys to their metadata.
        """
        result: Dict[str, BaseModalityMetadata] = {}
        for key, meta in self.modality_metadatas.items():
            if meta.modality_type == modality_type:
                result[key] = meta
        return result

    @property
    def modality_keys(self) -> list:
        """List of all modality keys found in the log directory."""
        return list(self.modality_metadatas.keys())


@lru_cache(maxsize=1000)  # NOTE: @DanielDauner Needs refactoring
def parse_log_directory_metadata(log_dir: Path) -> LogDirectoryMetadata:
    """Parse all metadata from Arrow files in a log directory.

    Reads the :class:`~py123d.datatypes.LogMetadata` from ``sync.arrow`` and discovers
    all per-modality Arrow files, deserializing each file's schema metadata into the
    appropriate metadata class based on the modality type encoded in the filename.

    The modality type is derived from the Arrow filename (e.g. ``ego_state_se3.arrow``
    → :class:`~py123d.datatypes.EgoStateSE3Metadata`). For ambiguous cases like lidar,
    the full modality key (including the ID suffix) is used to select the correct class
    (e.g. ``lidar.lidar_merged`` → :class:`~py123d.datatypes.LidarMergedMetadata`).

    :param log_dir: Path to the log directory containing Arrow files.
    :return: A :class:`LogDirectoryMetadata` with log metadata and all modality metadatas.
    :raises FileNotFoundError: If ``sync.arrow`` does not exist.
    """
    from py123d.api.utils.arrow_helper import open_arrow_schema
    from py123d.datatypes.metadata.log_metadata import LogMetadata

    log_dir = Path(log_dir)

    sync_path = log_dir / "sync.arrow"
    if not sync_path.exists():
        raise FileNotFoundError(f"sync.arrow not found in {log_dir}")

    log_metadata = get_metadata_from_arrow_schema(open_arrow_schema(sync_path), LogMetadata)

    modality_metadatas: Dict[str, BaseModalityMetadata] = {}
    for arrow_file in sorted(log_dir.glob("*.arrow")):
        if arrow_file.name in _NON_MODALITY_FILES:
            continue

        modality_key = arrow_file.stem
        try:
            metadata_class = resolve_metadata_class(modality_key)
            schema = open_arrow_schema(arrow_file)
            modality_metadatas[modality_key] = get_metadata_from_arrow_schema(schema, metadata_class)
        except (ValueError, KeyError) as e:
            logger.warning("Skipping '%s': %s", arrow_file.name, e)

    return LogDirectoryMetadata(log_metadata=log_metadata, modality_metadatas=modality_metadatas)
