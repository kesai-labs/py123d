from __future__ import annotations

import abc
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt

from py123d.conversion.dataset_converter_config import DatasetConverterConfig
from py123d.datatypes.detections.box_detections import BoxDetectionsSE3
from py123d.datatypes.detections.traffic_light_detections import TrafficLights
from py123d.datatypes.metadata import LogMetadata
from py123d.datatypes.sensors.fisheye_mei_camera import FisheyeMEICameraID
from py123d.datatypes.sensors.lidar import LidarID
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraID
from py123d.datatypes.time.time_point import Timestamp
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.geometry import PoseSE3


class AbstractLogWriter(abc.ABC):
    """Abstract base class for log writers.

    A log writer is responsible specifying the output format of a converted log.
    This includes how data is organized, how it is serialized, and how it is stored.
    """

    @abc.abstractmethod
    def reset(
        self,
        dataset_converter_config: DatasetConverterConfig,
        log_metadata: LogMetadata,
    ) -> bool:
        """Resets the log writer to start writing a new log according to the provided configuration and metadata.

        :param dataset_converter_config: The dataset converter configuration.
        :param log_metadata: The metadata for the log.
        :return: True if the current logs needs to be written, False otherwise.
        """

    def write(
        self,
        timestamp: Timestamp,
        uuid: Optional[uuid.UUID] = None,
        ego_state_se3: Optional[EgoStateSE3] = None,
        box_detections_se3: Optional[BoxDetectionsSE3] = None,
        traffic_lights: Optional[TrafficLights] = None,
        pinhole_cameras: Optional[List[CameraData]] = None,
        fisheye_mei_cameras: Optional[List[CameraData]] = None,
        lidar: Optional[LidarData] = None,
    ) -> None:
        """Writes a single iteration of data to the log.

        :param timestamp: Required, the timestamp of the iteration.
        :param ego_state: Optional, the ego state of the vehicle, defaults to None.
        :param box_detections: Optional, the box detections, defaults to None
        :param traffic_lights: Optional, the traffic light detections, defaults to None
        :param pinhole_cameras: Optional, the pinhole camera data, defaults to None
        :param fisheye_mei_cameras: Optional, the fisheye MEI camera data, defaults to None
        :param lidar: Optional, the Lidar data, defaults to None
        """
        pass

    @abc.abstractmethod
    def write_ego_state_se3(self, ego_state_se3: EgoStateSE3) -> None:
        pass

    @abc.abstractmethod
    def write_box_detections_se3(self, box_detections_se3: BoxDetectionsSE3) -> None:
        pass

    @abc.abstractmethod
    def write_traffic_lights(self, traffic_lights: TrafficLights) -> None:
        pass

    @abc.abstractmethod
    def write_pinhole_camera(self, camera_data: CameraData) -> None:
        pass

    @abc.abstractmethod
    def write_fisheye_mei_camera(self, camera_data: CameraData) -> None:
        pass

    @abc.abstractmethod
    def write_lidar(self, lidar_data: LidarData) -> None:
        pass

    @abc.abstractmethod
    def write_aux_dict(self, aux_dict: Dict[str, Union[str, int, float, bool]]) -> None:
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """Closes the log writer and finalizes the log io operations."""


@dataclass
class LidarData:
    """Helper dataclass to pass Lidar data to log writers."""

    lidar_name: str
    lidar_type: LidarID

    timestamp: Optional[Timestamp] = None
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

    timestamp: Optional[Timestamp] = None
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
        return self.relative_path is not None and str(self.relative_path).lower().endswith((".jpg", ".jpeg"))

    @property
    def has_png_file_path(self) -> bool:
        return self.relative_path is not None and str(self.relative_path).lower().endswith((".png",))

    @property
    def has_jpeg_binary(self) -> bool:
        return self.jpeg_binary is not None

    @property
    def has_numpy_image(self) -> bool:
        return self.numpy_image is not None
