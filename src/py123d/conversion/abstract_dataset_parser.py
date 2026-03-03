from __future__ import annotations

import abc
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import numpy as np
import numpy.typing as npt

from py123d.datatypes.custom.custom_modality import CustomModality
from py123d.datatypes.detections.box_detections import BoxDetectionsSE3
from py123d.datatypes.detections.traffic_light_detections import TrafficLightDetections
from py123d.datatypes.map_objects.base_map_objects import BaseMapObject
from py123d.datatypes.metadata import LogMetadata, MapMetadata
from py123d.datatypes.sensors.fisheye_mei_camera import FisheyeMEICameraID
from py123d.datatypes.sensors.lidar import LidarID
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraID
from py123d.datatypes.time.timestamp import Timestamp
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.geometry.pose import PoseSE3


@dataclass
class LidarData:
    """Helper dataclass to pass Lidar data to log writers."""

    lidar_name: str
    lidar_type: LidarID
    start_timestamp: Timestamp
    end_timestamp: Timestamp

    iteration: Optional[int] = None
    dataset_root: Optional[Union[str, Path]] = None
    relative_path: Optional[Union[str, Path]] = None
    point_cloud_3d: Optional[npt.NDArray] = None
    point_cloud_features: Optional[Dict[str, npt.NDArray]] = None

    def __post_init__(self):
        assert self.has_file_path or self.has_point_cloud_3d, (
            "Either file path (dataset_root and relative_path) or point_cloud must be provided for LidarData."
        )

    @property
    def has_file_path(self) -> bool:
        return self.dataset_root is not None and self.relative_path is not None

    @property
    def has_point_cloud_3d(self) -> bool:
        return self.point_cloud_3d is not None

    @property
    def has_point_cloud_features(self) -> bool:
        return self.point_cloud_features is not None


@dataclass
class CameraData:
    """Helper dataclass to pass Camera data to log writers."""

    camera_name: str
    camera_id: Union[PinholeCameraID, FisheyeMEICameraID]
    extrinsic: PoseSE3
    timestamp: Timestamp

    jpeg_binary: Optional[bytes] = None
    numpy_image: Optional[npt.NDArray[np.uint8]] = None
    dataset_root: Optional[Union[str, Path]] = None
    relative_path: Optional[Union[str, Path]] = None

    def __post_init__(self):
        assert self.has_file_path or self.has_jpeg_binary or self.has_numpy_image, (
            "Either file path (dataset_root and relative_path) or jpeg_binary or numpy_image must be provided for CameraData."
        )

        if self.has_file_path:
            absolute_path = Path(self.dataset_root) / self.relative_path  # type: ignore
            assert absolute_path.exists(), f"Camera file not found: {absolute_path}"

    @property
    def has_file_path(self) -> bool:
        return self.dataset_root is not None and self.relative_path is not None

    @property
    def has_jpeg_file_path(self) -> bool:
        return self.has_file_path and str(self.relative_path).lower().endswith((".jpg", ".jpeg"))

    @property
    def has_png_file_path(self) -> bool:
        return self.has_file_path and str(self.relative_path).lower().endswith((".png",))

    @property
    def has_jpeg_binary(self) -> bool:
        return self.jpeg_binary is not None

    @property
    def has_numpy_image(self) -> bool:
        return self.numpy_image is not None


@dataclass
class FrameData:
    """One synchronized frame of data, as produced by a :class:`LogParser`.

    Fields mirror the ``AbstractLogWriter.write()`` signature so that an orchestrator
    can forward them directly::

        for frame in log_parser.iter_frames():
            writer.write(**frame.to_writer_kwargs())
    """

    timestamp: Timestamp
    uuid: Optional[uuid.UUID] = None
    ego_state_se3: Optional[EgoStateSE3] = None
    box_detections_se3: Optional[BoxDetectionsSE3] = None
    traffic_lights: Optional[TrafficLightDetections] = None
    pinhole_cameras: Optional[List[CameraData]] = None
    fisheye_mei_cameras: Optional[List[CameraData]] = None
    lidar: Optional[LidarData] = None
    custom_modalities: Optional[Dict[str, CustomModality]] = None

    def to_writer_kwargs(self) -> dict:
        """Returns a dict suitable for ``writer.write(**frame.to_writer_kwargs())``."""
        kwargs: dict = {"timestamp": self.timestamp}
        if self.uuid is not None:
            kwargs["uuid"] = self.uuid
        if self.ego_state_se3 is not None:
            kwargs["ego_state_se3"] = self.ego_state_se3
        if self.box_detections_se3 is not None:
            kwargs["box_detections_se3"] = self.box_detections_se3
        if self.traffic_lights is not None:
            kwargs["traffic_lights"] = self.traffic_lights
        if self.pinhole_cameras is not None:
            kwargs["pinhole_cameras"] = self.pinhole_cameras
        if self.fisheye_mei_cameras is not None:
            kwargs["fisheye_mei_cameras"] = self.fisheye_mei_cameras
        if self.lidar is not None:
            kwargs["lidar"] = self.lidar
        if self.custom_modalities is not None:
            kwargs["custom_modalities"] = self.custom_modalities
        return kwargs


class LogParser(abc.ABC):
    """Lightweight, picklable handle to one log's data.

    Implementations hold only the paths and parameters needed to read the raw data.
    The heavy I/O happens lazily inside :meth:`iter_frames`.
    """

    @abc.abstractmethod
    def get_log_metadata(self) -> LogMetadata:
        """Returns metadata describing this log (sensors, calibration, etc.)."""

    @abc.abstractmethod
    def iter_frames(self) -> Iterator[FrameData]:
        """Yields synchronized frames in chronological order.

        Each :class:`FrameData` contains one timestamp and all modalities available
        at that timestamp. The generator performs I/O lazily — one frame at a time.
        """


class MapParser(abc.ABC):
    """Lightweight, picklable handle to one map's data."""

    @abc.abstractmethod
    def get_map_metadata(self) -> MapMetadata:
        """Returns metadata describing this map (location, coordinate system, etc.)."""

    @abc.abstractmethod
    def iter_map_objects(self) -> Iterator[BaseMapObject]:
        """Yields map objects lazily, one at a time."""


class DatasetParser(abc.ABC):
    """Top-level parser that produces per-log and per-map containers.

    An orchestrator calls :meth:`get_log_parsers` / :meth:`get_map_parsers` once on
    the main process, then distributes the resulting lightweight containers to workers.
    """

    @abc.abstractmethod
    def get_log_parsers(self) -> List[LogParser]:
        """Returns one :class:`LogParser` per log in the dataset."""

    @abc.abstractmethod
    def get_map_parsers(self) -> List[MapParser]:
        """Returns one :class:`MapParser` per map region in the dataset."""
