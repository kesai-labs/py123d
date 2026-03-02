from __future__ import annotations

import abc
from typing import List, Optional

from py123d.api.map.map_api import MapAPI
from py123d.api.scene.scene_metadata import SceneMetadata
from py123d.datatypes import (
    BoxDetectionsSE3,
    EgoStateSE3,
    FisheyeMEICamera,
    FisheyeMEICameraID,
    Lidar,
    LidarID,
    LogMetadata,
    MapMetadata,
    PinholeCamera,
    PinholeCameraID,
    Timestamp,
    TrafficLights,
    VehicleParameters,
)


class SceneAPI(abc.ABC):
    """Base class for all scene APIs. The scene API provides access to all data modalities at in a scene."""

    # Abstract Methods, to be implemented by subclasses
    # ------------------------------------------------------------------------------------------------------------------

    @abc.abstractmethod
    def get_log_metadata(self) -> LogMetadata:
        """Returns the :class:`~py123d.datatypes.metadata.LogMetadata` of the scene.

        :return: The log metadata.
        """

    @abc.abstractmethod
    def get_scene_metadata(self) -> SceneMetadata:
        """Returns the :class:`~py123d.store.scene.scene_metadata.SceneMetadata` of the scene.

        :return: The scene metadata.
        """

    @abc.abstractmethod
    def get_map_api(self) -> Optional[MapAPI]:
        """Returns the :class:`~py123d.store.MapAPI` of the scene, if available.

        :return: The map API, or None if not available.
        """

    @abc.abstractmethod
    def get_timestamp_at_iteration(self, iteration: int) -> Timestamp:
        """Returns the :class:`~py123d.datatypes.time.Timestamp` at a given iteration.

        :param iteration: The iteration to get the timestamp for.
        :return: The timestamp at the given iteration.
        """

    @abc.abstractmethod
    def get_ego_state_se3_at_iteration(self, iteration: int) -> Optional[EgoStateSE3]:
        """Returns the :class:`~py123d.datatypes.vehicle_state.EgoStateSE3` at a given iteration, if available.

        :param iteration: The iteration to get the ego state for.
        :return: The ego state at the given iteration, or None if not available.
        """

    @abc.abstractmethod
    def get_box_detections_at_iteration(self, iteration: int) -> Optional[BoxDetectionsSE3]:
        """Returns the :class:`~py123d.datatypes.detections.BoxDetectionsSE3` at a given iteration, if available.

        :param iteration: The iteration to get the box detections for.
        :return: The box detections at the given iteration, or None if not available.
        """

    @abc.abstractmethod
    def get_traffic_light_detections_at_iteration(self, iteration: int) -> Optional[TrafficLights]:
        """Returns the :class:`~py123d.datatypes.detections.TrafficLights` at a given iteration,
            if available.

        :param iteration: The iteration to get the traffic light detections for.
        :return: The traffic light detections at the given iteration, or None if not available.
        """

    @abc.abstractmethod
    def get_route_lane_group_ids(self, iteration: int) -> Optional[List[int]]:
        """Returns the list of route lane group IDs at a given iteration, if available.

        :param iteration: The iteration to get the route lane group IDs for.
        :return: The list of route lane group IDs at the given iteration, or None if not available.
        """

    @abc.abstractmethod
    def get_pinhole_camera_at_iteration(
        self,
        iteration: int,
        camera_id: PinholeCameraID,
    ) -> Optional[PinholeCamera]:
        """Returns the :class:`~py123d.datatypes.sensors.PinholeCamera` of a given \
            :class:`~py123d.datatypes.sensors.PinholeCameraID` at a given iteration, if available.

        :param iteration: The iteration to get the pinhole camera for.
        :param camera_id: The :type:`~py123d.datatypes.sensors.PinholeCameraID` of the pinhole camera.
        :return: The pinhole camera, or None if not available.
        """

    @abc.abstractmethod
    def get_fisheye_mei_camera_at_iteration(
        self, iteration: int, camera_id: FisheyeMEICameraID
    ) -> Optional[FisheyeMEICamera]:
        """Returns the :class:`~py123d.datatypes.sensors.FisheyeMEICamera` of a given \
            :class:`~py123d.datatypes.sensors.FisheyeMEICameraID` at a given iteration, if available.

        :param iteration: The iteration to get the fisheye MEI camera for.
        :param camera_id: The :type:`~py123d.datatypes.sensors.FisheyeMEICameraID` of the fisheye MEI camera.
        :return: The fisheye MEI camera, or None if not available.
        """

    @abc.abstractmethod
    def get_lidar_at_iteration(self, iteration: int, lidar_id: LidarID) -> Optional[Lidar]:
        """Returns the :class:`~py123d.datatypes.sensors.Lidar` of a given :class:`~py123d.datatypes.sensors.LidarID`\
            at a given iteration, if available.

        :param iteration: The iteration to get the Lidar for.
        :param lidar_id: The :type:`~py123d.datatypes.sensors.LidarID` of the Lidar.
        :return: The Lidar, or None if not available.
        """

    # Deprecated Methods, to be removed in future versions
    # ------------------------------------------------------------------------------------------------------------------
    def get_ego_state_at_iteration(self, iteration: int) -> Optional[EgoStateSE3]:
        """Returns the :class:`~py123d.datatypes.vehicle_state.EgoStateSE3` at a given iteration, if available.

        :param iteration: The iteration to get the ego state for.
        :return: The ego state at the given iteration, or None if not available.
        """
        return self.get_ego_state_se3_at_iteration(iteration)

    # Syntactic Sugar / Properties, that are convenient to access and pass to subclasses
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def log_metadata(self) -> LogMetadata:
        """The :class:`~py123d.datatypes.metadata.LogMetadata` of the scene."""
        return self.get_log_metadata()

    @property
    def scene_metadata(self) -> SceneMetadata:
        """The :class:`~py123d.store.scene.SceneMetadata` of the scene."""
        return self.get_scene_metadata()

    @property
    def map_metadata(self) -> Optional[MapMetadata]:
        """The :class:`~py123d.datatypes.metadata.MapMetadata` of the scene, if available."""
        return self.log_metadata.map_metadata

    @property
    def map_api(self) -> Optional[MapAPI]:
        """The :class:`~py123d.store.map.MapAPI` of the scene, if available."""
        return self.get_map_api()

    @property
    def dataset(self) -> str:
        """The dataset name from the log metadata."""
        return self.log_metadata.dataset

    @property
    def split(self) -> str:
        """The data split name from the log metadata."""
        return self.log_metadata.split

    @property
    def location(self) -> Optional[str]:
        """The location from the log metadata."""
        return self.log_metadata.location

    @property
    def log_name(self) -> str:
        """The log name from the log metadata."""
        return self.log_metadata.log_name

    @property
    def version(self) -> str:
        """The version of the py123d library used to create this log metadata."""
        return self.log_metadata.version

    @property
    def scene_uuid(self) -> str:
        """The UUID of the scene."""
        return self.scene_metadata.initial_uuid

    @property
    def number_of_iterations(self) -> int:
        """The number of iterations in the scene."""
        return self.scene_metadata.number_of_iterations

    @property
    def number_of_history_iterations(self) -> int:
        """The number of history iterations in the scene."""
        return self.scene_metadata.number_of_history_iterations

    @property
    def vehicle_parameters(self) -> Optional[VehicleParameters]:
        """The :class:`~py123d.datatypes.vehicle_state.VehicleParameters` of the ego vehicle, if available."""
        return self.log_metadata.vehicle_parameters

    @property
    def available_pinhole_camera_ids(self) -> List[PinholeCameraID]:
        """List of available :class:`~py123d.datatypes.sensors.PinholeCameraID` in the log metadata."""
        return list(self.log_metadata.pinhole_camera_metadata.keys())

    @property
    def available_pinhole_camera_names(self) -> List[str]:
        """List of available :class:`~py123d.datatypes.sensors.PinholeCameraID` in the log metadata."""
        return [camera.camera_name for camera in self.log_metadata.pinhole_camera_metadata.values()]

    @property
    def available_fisheye_mei_camera_ids(self) -> List[FisheyeMEICameraID]:
        """List of available :class:`~py123d.datatypes.sensors.FisheyeMEICameraType` in the log metadata."""
        return list(self.log_metadata.fisheye_mei_camera_metadata.keys())

    @property
    def available_fisheye_mei_camera_names(self) -> List[str]:
        """List of available :class:`~py123d.datatypes.sensors.FisheyeMEICameraType` in the log metadata."""
        return [camera.camera_name for camera in self.log_metadata.fisheye_mei_camera_metadata.values()]

    @property
    def available_lidar_ids(self) -> List[LidarID]:
        """List of available :class:`~py123d.datatypes.sensors.LidarID` in the log metadata."""
        return list(self.log_metadata.lidar_metadata.keys())

    @property
    def available_lidar_names(self) -> List[str]:
        """List of available Lidar names in the log metadata."""
        return [lidar.lidar_name for lidar in self.log_metadata.lidar_metadata.values()]
