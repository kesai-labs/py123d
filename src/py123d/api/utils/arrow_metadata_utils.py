import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Type, TypeVar

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
# route_map.arrow: an alternative route file with the layout of route_position.arrow (route_utils.ROUTE_MAP_FILE),
# read only when SceneFilter.route_file selects it.
_NON_MODALITY_FILES = {"sync.arrow", "map.arrow", "route_map.arrow"}


def get_metadata_from_arrow_schema(
    arrow_schema: pa.Schema,
    metadata_class: type[T],
    modality_key: bytes = _METADATA_KEY,
) -> T:
    """Gets metadata for a specific modality from an Arrow schema."""

    deserialized_metadata = None
    if arrow_schema.metadata is not None and modality_key in arrow_schema.metadata:
        deserialized_metadata = metadata_class.from_dict(
            msgpack.unpackb(arrow_schema.metadata[modality_key], raw=False, strict_map_key=False)
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
    from py123d.datatypes.metadata.route_metadata import RouteMetadata
    from py123d.datatypes.sensors.barometer import BarometerMetadata
    from py123d.datatypes.sensors.base_camera import camera_metadata_from_dict
    from py123d.datatypes.sensors.depth_camera import DepthCameraMetadata
    from py123d.datatypes.sensors.gnss import GnssMetadata
    from py123d.datatypes.sensors.imu import ImuMetadata
    from py123d.datatypes.sensors.lidar import LidarMetadata
    from py123d.datatypes.sensors.magnetometer import MagnetometerMetadata
    from py123d.datatypes.sensors.radar import RadarMetadata
    from py123d.datatypes.sensors.segmentation_camera import SegmentationCameraMetadata
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
        ModalityType.CAMERA_SEMANTIC: SegmentationCameraMetadata,
        ModalityType.CAMERA_INSTANCE: SegmentationCameraMetadata,
        ModalityType.CAMERA_DEPTH: DepthCameraMetadata,
        ModalityType.LIDAR: LidarMetadata,
        ModalityType.RADAR: RadarMetadata,
        ModalityType.IMU: ImuMetadata,
        ModalityType.GNSS: GnssMetadata,
        ModalityType.BAROMETER: BarometerMetadata,
        ModalityType.MAGNETOMETER: MagnetometerMetadata,
        ModalityType.ROUTE_POSITION: RouteMetadata,
        ModalityType.CUSTOM: CustomModalityMetadata,
    }


def _get_modality_key_overrides() -> Dict[str, Type[BaseModalityMetadata]]:
    """Returns overrides for specific modality keys where ModalityType alone is ambiguous."""
    from py123d.datatypes.sensors.lidar import LidarMergedMetadata
    from py123d.datatypes.sensors.radar import RadarMergedMetadata

    return {
        "lidar.lidar_merged": LidarMergedMetadata,
        "radar.radar_merged": RadarMergedMetadata,
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


class _ModalityMetadatas(Mapping):
    """The modality metadatas of a log directory by modality key, each read from its file on first use.

    A lookup of one key reads that file's schema only; a log holds tens of Arrow files, and a reader of the ego
    states needs one of them. Iterating reads every file, and leaves out the files whose metadata cannot be
    deserialized, as the eager parse did.
    """

    def __init__(self, log_dir: Path, modality_keys: List[str]) -> None:
        """
        :param log_dir: The log directory.
        :param modality_keys: The stems of the directory's modality Arrow files, sorted.
        """
        self._log_dir = log_dir
        self._modality_keys = modality_keys
        self._unread = set(modality_keys)
        self._metadatas: Dict[str, BaseModalityMetadata] = {}

    def _read(self, modality_key: str) -> None:
        """Read one file's metadata if it was not read before; a file that cannot be deserialized is skipped."""
        if modality_key not in self._unread:
            return
        from py123d.api.utils.arrow_helper import open_arrow_schema

        try:
            metadata_class = resolve_metadata_class(modality_key)
            schema = open_arrow_schema(os.path.join(self._log_dir, f"{modality_key}.arrow"))
            self._metadatas[modality_key] = get_metadata_from_arrow_schema(schema, metadata_class)
        except (ValueError, KeyError) as e:
            logger.warning("Skipping '%s.arrow': %s", modality_key, e)
        self._unread.discard(modality_key)

    def __getitem__(self, modality_key: str) -> BaseModalityMetadata:
        self._read(modality_key)
        return self._metadatas[modality_key]

    def of_type(self, modality_type: ModalityType) -> Dict[str, BaseModalityMetadata]:
        """The metadatas of one modality type, reading the files of that type alone: the type is in the file name.

        :param modality_type: The :class:`ModalityType` to filter by.
        :return: Dict of matching modality keys to their metadata.
        """
        result: Dict[str, BaseModalityMetadata] = {}
        for modality_key in self._modality_keys:
            try:
                if get_modality_type_from_key(modality_key) != modality_type:
                    continue
            except (ValueError, KeyError):
                pass  # A name that states no known type: the metadata decides.
            self._read(modality_key)
            metadata = self._metadatas.get(modality_key)
            if metadata is not None and metadata.modality_type == modality_type:
                result[modality_key] = metadata
        return result

    def __iter__(self) -> Iterator[str]:
        for modality_key in self._modality_keys:
            self._read(modality_key)
        return (modality_key for modality_key in self._modality_keys if modality_key in self._metadatas)

    def __len__(self) -> int:
        return sum(1 for _ in self)


@dataclass
class LogDirectoryMetadata:
    """All metadata parsed from a log directory's Arrow files.

    :param log_metadata: The :class:`~py123d.datatypes.LogMetadata` from ``sync.arrow``.
    :param modality_metadatas: Mapping of modality key to its deserialized metadata object.
    """

    log_metadata: LogMetadata  # noqa: F821 — forward ref to avoid top-level import
    modality_metadatas: Mapping[str, BaseModalityMetadata] = field(default_factory=dict)

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
        if isinstance(self.modality_metadatas, _ModalityMetadatas):
            return self.modality_metadatas.of_type(modality_type)
        return {key: meta for key, meta in self.modality_metadatas.items() if meta.modality_type == modality_type}

    @property
    def modality_keys(self) -> list:
        """List of all modality keys found in the log directory."""
        return list(self.modality_metadatas.keys())


@lru_cache(maxsize=1000)  # NOTE: @DanielDauner Needs refactoring
def parse_log_directory_metadata(log_dir: Path) -> LogDirectoryMetadata:
    """Parse all metadata from Arrow files in a log directory.

    Reads the :class:`~py123d.datatypes.LogMetadata` from ``sync.arrow`` and discovers
    all per-modality Arrow files. Each file's schema metadata is deserialized on first
    use, into the metadata class of the modality type encoded in the filename.

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

    # One directory listing names every file; no stat per file.
    arrow_files = sorted(name for name in os.listdir(log_dir) if name.endswith(".arrow"))
    if "sync.arrow" not in arrow_files:
        raise FileNotFoundError(f"sync.arrow not found in {log_dir}")

    log_metadata = get_metadata_from_arrow_schema(open_arrow_schema(log_dir / "sync.arrow"), LogMetadata)
    modality_metadatas = _ModalityMetadatas(
        log_dir,
        [name[: -len(".arrow")] for name in arrow_files if name not in _NON_MODALITY_FILES],
    )
    return LogDirectoryMetadata(log_metadata=log_metadata, modality_metadatas=modality_metadatas)
