from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from py123d.datatypes import BaseMapObject, LogMetadata, MapMetadata, Timestamp
from py123d.datatypes.modalities.base_modality import BaseModality, BaseModalityMetadata
from py123d.datatypes.sensors.fisheye_mei_camera import FisheyeMEICameraMetadata
from py123d.datatypes.sensors.ftheta_camera import FThetaCameraMetadata
from py123d.datatypes.sensors.lidar import LidarMergedMetadata, LidarMetadata
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraMetadata
from py123d.datatypes.sensors.radar import RadarMergedMetadata, RadarMetadata
from py123d.geometry.pose import PoseSE3


class BaseDatasetParser(abc.ABC):
    """Top-level parser that produces per-log and per-map containers.

    An orchestrator calls :meth:`get_log_parsers` / :meth:`get_map_parsers` once on
    the main process, then distributes the resulting lightweight containers to workers.
    """

    @abc.abstractmethod
    def get_map_parsers(self) -> List[BaseMapParser]:
        """Returns one :class:`MapParser` per map region in the dataset."""

    @abc.abstractmethod
    def get_log_parsers(self) -> List[BaseLogParser]:
        """Returns one :class:`LogParser` per log in the dataset."""


class BaseMapParser(abc.ABC):
    """Lightweight, picklable handle to one map's data."""

    @abc.abstractmethod
    def get_map_metadata(self) -> MapMetadata:
        """Returns metadata describing this map (location, coordinate system, etc.)."""

    @abc.abstractmethod
    def iter_map_objects(self) -> Iterator[BaseMapObject]:
        """Yields map objects lazily, one at a time."""


class BaseLogParser(abc.ABC):
    """Lightweight, picklable handle to one log's data.

    Implementations hold only the paths and parameters needed to read the raw data.
    The heavy I/O happens lazily inside :meth:`iter_frames`.
    """

    @abc.abstractmethod
    def get_log_metadata(self) -> LogMetadata:
        """Returns metadata describing this log (associated map, time range, etc.)."""

    @abc.abstractmethod
    def iter_modalities_sync(self) -> Iterator[ModalitiesSync]:
        """Yields synchronized modalities of data, one at a time."""

    def iter_modalities_async(self) -> Iterator[BaseModality]:
        """Yields modality data objects independently for asynchronous writing.

        Each modality is yielded as it is produced, rather than waiting for a full
        synchronized frame. The orchestrator forwards these to
        ``writer.write_async(modality, modality_metadata)``.

        The default implementation unwraps each :class:`ModalitiesSync` frame from
        :meth:`iter_modalities_sync`. Override this method to yield modalities at
        their native sensor rates (e.g., cameras at 20 Hz, lidar at 10 Hz).
        """
        for modalities_sync in self.iter_modalities_sync():
            yield from modalities_sync.modalities

    def get_route_xyz(self):
        """Optionally returns the log's route as XYZ waypoints (N, 3) in the ego odometry
        frame — a planned route, or full-horizon odometry a synced conversion would
        otherwise truncate. The orchestrator forwards it to ``writer.set_route``; the
        default None derives the route from the written ego poses instead.
        """
        return None


class ModalitiesSync:
    """Helper class for passing synchronized modalities to log writers, without loading all data into memory at once."""

    def __init__(self, timestamp: Timestamp, modalities: List[BaseModality]) -> None:
        self._timestamp = timestamp
        self._modalities = modalities

    @property
    def timestamp(self) -> Timestamp:
        """Returns the timestamp associated with this synchronized frame."""
        return self._timestamp

    @property
    def modalities(self) -> List[BaseModality]:
        """Returns the list of modalities in this synchronized frame."""
        return self._modalities


class ParsedLidar(BaseModality):
    """Helper modality for passing a lidar observation to log writers, without loading the full point cloud into memory."""

    def __init__(
        self,
        metadata: Union[LidarMetadata, LidarMergedMetadata],
        start_timestamp: Timestamp,
        end_timestamp: Timestamp,
        dataset_root: Union[str, Path],
        relative_path: Union[str, Path],
        iteration: Optional[int] = None,
        load_kwargs: Optional[Dict[str, Any]] = None,
        arrival_timestamp: Optional[Timestamp] = None,
    ) -> None:
        self._metadata: Union[LidarMetadata, LidarMergedMetadata] = metadata
        self._start_timestamp: Timestamp = start_timestamp
        self._end_timestamp: Timestamp = end_timestamp
        self._arrival_timestamp: Optional[Timestamp] = arrival_timestamp

        self._dataset_root: Optional[Union[str, Path]] = dataset_root
        self._relative_path: Optional[Union[str, Path]] = relative_path
        self._iteration: Optional[int] = iteration
        # Generic, dataset-specific extras forwarded verbatim to the point-cloud loader (and persisted
        # in the Arrow lidar frame under the "path" store option, so they survive to API read time).
        # nuScenes uses this to carry the keyframe's panoptic-label relative path; see
        # :func:`py123d.common.io.lidar.path_lidar_io.load_point_cloud_data_from_path`.
        self._load_kwargs: Optional[Dict[str, Any]] = load_kwargs

        assert self._dataset_root is not None and self._relative_path is not None, (
            "File path must be provided for ParsedLidar."
        )

    @property
    def timestamp(self) -> Timestamp:
        """Returns the timestamp associated with this lidar data. For synchronization purposes, we use the start timestamp."""
        return self._start_timestamp

    @property
    def start_timestamp(self) -> Timestamp:
        """Returns the start timestamp associated with this lidar data."""
        return self._start_timestamp

    @property
    def end_timestamp(self) -> Timestamp:
        """Returns the end timestamp associated with this lidar data."""
        return self._end_timestamp

    @property
    def arrival_timestamp(self) -> Optional[Timestamp]:
        """The time the recording system received this lidar data, if recorded."""
        return self._arrival_timestamp

    @property
    def metadata(self) -> BaseModalityMetadata:
        """Returns the metadata associated with this lidar data."""
        return self._metadata

    @property
    def load_kwargs(self) -> Optional[Dict[str, Any]]:
        """Returns the dataset-specific extras forwarded to the point-cloud loader, if any."""
        return self._load_kwargs


class ParsedRadar(BaseModality):
    """Helper modality for passing a radar observation to log writers, without loading the full point cloud into memory."""

    def __init__(
        self,
        metadata: Union[RadarMetadata, RadarMergedMetadata],
        timestamp: Timestamp,
        dataset_root: Union[str, Path],
        relative_path: Union[str, Path],
        iteration: Optional[int] = None,
        load_kwargs: Optional[Dict[str, Any]] = None,
        arrival_timestamp: Optional[Timestamp] = None,
    ) -> None:
        self._metadata: Union[RadarMetadata, RadarMergedMetadata] = metadata
        # A radar scan is treated as an instantaneous snapshot (no rolling shutter), so it carries a
        # single timestamp rather than a sweep window like lidar.
        self._timestamp: Timestamp = timestamp
        self._arrival_timestamp: Optional[Timestamp] = arrival_timestamp

        self._dataset_root: Optional[Union[str, Path]] = dataset_root
        self._relative_path: Optional[Union[str, Path]] = relative_path
        self._iteration: Optional[int] = iteration
        # Generic, dataset-specific extras forwarded verbatim to the point-cloud loader (and persisted in the
        # Arrow radar frame under the "path" store option). e.g. nuScenes carries the per-RadarID relative
        # paths here, since its radars live in separate files that are merged on load.
        self._load_kwargs: Optional[Dict[str, Any]] = load_kwargs

        assert self._dataset_root is not None and self._relative_path is not None, (
            "File path must be provided for ParsedRadar."
        )

    @property
    def timestamp(self) -> Timestamp:
        """Returns the timestamp associated with this radar data."""
        return self._timestamp

    @property
    def arrival_timestamp(self) -> Optional[Timestamp]:
        """The time the recording system received this radar data, if recorded."""
        return self._arrival_timestamp

    @property
    def metadata(self) -> BaseModalityMetadata:
        """Returns the metadata associated with this radar data."""
        return self._metadata

    @property
    def load_kwargs(self) -> Optional[Dict[str, Any]]:
        """Returns the dataset-specific extras forwarded to the point-cloud loader, if any."""
        return self._load_kwargs


class ParsedCamera(BaseModality):
    """Helper modality to pass cameras to log writer without loading loading an image/video or decoding the bytestring.

    ``camera_to_global_se3`` may be None, in which case the writer stores a null and the reader
    composes the pose from the log's ego trajectory and the camera's own extrinsic.
    """

    def __init__(
        self,
        metadata: Union[PinholeCameraMetadata, FisheyeMEICameraMetadata, FThetaCameraMetadata],
        timestamp: Timestamp,
        camera_to_global_se3: Optional[PoseSE3],
        dataset_root: Optional[Union[str, Path]] = None,
        relative_path: Optional[Union[str, Path]] = None,
        byte_string: Optional[bytes] = None,
        exposure_factor: Optional[float] = None,
        arrival_timestamp: Optional[Timestamp] = None,
    ) -> None:
        self._metadata = metadata
        self._timestamp = timestamp
        self._camera_to_global_se3 = camera_to_global_se3

        self._dataset_root = dataset_root
        self._relative_path = relative_path
        self._byte_string = byte_string
        self._exposure_factor = exposure_factor
        self._arrival_timestamp = arrival_timestamp

        assert self.has_file_path or self.has_byte_string, (
            "Either file path or byte string must be provided for ParsedCamera."
        )

    @property
    def timestamp(self) -> Timestamp:
        """Returns the timestamp associated with this camera data."""
        return self._timestamp

    @property
    def arrival_timestamp(self) -> Optional[Timestamp]:
        """The time the recording system received this camera data, if recorded."""
        return self._arrival_timestamp

    @property
    def metadata(self) -> BaseModalityMetadata:
        """Returns the metadata associated with this camera data."""
        return self._metadata

    @property
    def camera_to_global_se3(self) -> Optional[PoseSE3]:
        """Returns the camera-to-global pose associated with this camera data.

        None where the dataset stores the pose implicitly. It is then written as a null and
        composed on read from ``ego_state_se3`` and the camera extrinsic, which is what lets a
        re-estimated ego trajectory reach the camera poses without rewriting the image tables.
        """
        return self._camera_to_global_se3

    @property
    def relative_path(self) -> Optional[Union[str, Path]]:
        """Returns the relative file path to the camera data, if available."""
        return self._relative_path

    @property
    def has_file_path(self) -> bool:
        return self._dataset_root is not None and self._relative_path is not None

    @property
    def has_jpeg_file_path(self) -> bool:
        return self.has_file_path and str(self._relative_path).lower().endswith((".jpg", ".jpeg"))

    @property
    def has_png_file_path(self) -> bool:
        return self.has_file_path and str(self._relative_path).lower().endswith((".png",))

    @property
    def has_byte_string(self) -> bool:
        return self._byte_string is not None

    @property
    def exposure_factor(self) -> Optional[float]:
        """Per-frame exposure normalization gain applied upstream of the stored image, if known.

        Datasets whose recording pipeline auto-normalizes exposure (e.g. Kesai) record the
        applied gain here so absolute radiance can be recovered; None when not provided.
        """
        return self._exposure_factor
