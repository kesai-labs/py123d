from __future__ import annotations

import abc
from typing import List, Optional

from py123d.api.map.map_api import MapAPI
from py123d.api.scene.scene_metadata import SceneMetadata
from py123d.datatypes import (
    BoxDetectionMetadata,
    BoxDetectionsSE3,
    CustomModality,
    EgoMetadata,
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
    TrafficLightDetections,
)
from py123d.datatypes.metadata.sensor_metadata import (
    FisheyeMEICameraMetadatas,
    LidarMetadatas,
    PinholeCameraMetadatas,
)


class SceneAPI(abc.ABC):
    """Base class for all scene APIs. The scene API provides access to all data modalities at in a scene."""

    # ------------------------------------------------------------------------------------------------------------------
    # 1. Abstract Methods, to be implemented by subclasses
    # ------------------------------------------------------------------------------------------------------------------

    # 1.1 Static Metadata
    # ------------------------------------------------------------------------------------------------------------------

    @abc.abstractmethod
    def get_scene_metadata(self) -> SceneMetadata:
        """Returns the :class:`~py123d.api.scene.scene_metadata.SceneMetadata` of the scene.

        :return: The scene metadata.
        """

    @abc.abstractmethod
    def get_log_metadata(self) -> LogMetadata:
        """Returns the :class:`~py123d.datatypes.metadata.LogMetadata` of the scene.

        :return: The log metadata.
        """

    @abc.abstractmethod
    def get_map_metadata(self) -> Optional[MapMetadata]:
        """Returns the :class:`~py123d.datatypes.metadata.MapMetadata` of the scene, if available.

        :return: The map metadata, or None if not available.
        """

    @abc.abstractmethod
    def get_ego_metadata(self) -> Optional[EgoMetadata]:
        """Returns the :class:`~py123d.datatypes.EgoMetadata` of the ego vehicle, if available.

        :return: The ego metadata, or None if not available.
        """

    @abc.abstractmethod
    def get_box_detection_metadata(self) -> Optional[BoxDetectionMetadata]:
        """Returns the :class:`~py123d.datatypes.detections.BoxDetectionMetadata` of the scene, if available.

        :return: The box detection metadata, or None if not available.
        """

    @abc.abstractmethod
    def get_pinhole_camera_metadatas(self) -> Optional[PinholeCameraMetadatas]:
        """Returns the :class:`~py123d.datatypes.metadata.sensor_metadata.PinholeCameraMetadatas` for all available \
            pinhole cameras in the scene, if available.

        :return: The pinhole camera metadatas, or None if not available.
        """

    @abc.abstractmethod
    def get_fisheye_mei_camera_metadatas(self) -> Optional[FisheyeMEICameraMetadatas]:
        """Returns the :class:`~py123d.datatypes.metadata.sensor_metadata.FisheyeMEICameraMetadatas` for all available \
            fisheye MEI cameras in the scene, if available.

        :return: The fisheye MEI camera metadatas, or None if not available.
        """

    @abc.abstractmethod
    def get_lidar_metadatas(self) -> Optional[LidarMetadatas]:
        """Returns the :class:`~py123d.datatypes.metadata.sensor_metadata.LidarMetadatas` for all available \
            lidars in the scene, if available.

        :return: The lidar metadatas, or None if not available.
        """

    # 1.2 Map
    # ------------------------------------------------------------------------------------------------------------------

    @abc.abstractmethod
    def get_map_api(self) -> Optional[MapAPI]:
        """Returns the :class:`~py123d.api.MapAPI` of the scene, if available.

        :return: The map API, or None if not available.
        """

    # 1.2 Dynamic Log Data.
    # ------------------------------------------------------------------------------------------------------------------

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
    def get_box_detections_se3_at_iteration(self, iteration: int) -> Optional[BoxDetectionsSE3]:
        """Returns the :class:`~py123d.datatypes.detections.BoxDetectionsSE3` at a given iteration, if available.

        :param iteration: The iteration to get the box detections for.
        :return: The box detections at the given iteration, or None if not available.
        """

    @abc.abstractmethod
    def get_traffic_light_detections_at_iteration(self, iteration: int) -> Optional[TrafficLightDetections]:
        """Returns the :class:`~py123d.datatypes.detections.TrafficLightDetections` at a given iteration,
            if available.

        :param iteration: The iteration to get the traffic light detections for.
        :return: The traffic light detections at the given iteration, or None if not available.
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

    @abc.abstractmethod
    def get_custom_modality_at_iteration(self, iteration: int, name: str) -> Optional[CustomModality]:
        """Returns the :class:`~py123d.datatypes.custom.CustomModality` with the given name at a given iteration,
            if available.

        :param iteration: The iteration to get the custom modality for.
        :param name: The name of the custom modality (e.g. ``"route"``, ``"predictions"``).
        :return: The custom modality, or None if not available.
        """

    # ------------------------------------------------------------------------------------------------------------------
    # Deprecated Methods, to be removed in future versions
    # ------------------------------------------------------------------------------------------------------------------
    def get_ego_state_at_iteration(self, iteration: int) -> Optional[EgoStateSE3]:
        """Returns the :class:`~py123d.datatypes.vehicle_state.EgoStateSE3` at a given iteration, if available.

        :param iteration: The iteration to get the ego state for.
        :return: The ego state at the given iteration, or None if not available.
        """
        return self.get_ego_state_se3_at_iteration(iteration)

    def get_box_detections_at_iteration(self, iteration: int) -> Optional[BoxDetectionsSE3]:
        """Returns the :class:`~py123d.datatypes.detections.BoxDetectionsSE3` at a given iteration, if available.

        :param iteration: The iteration to get the box detections for.
        :return: The box detections at the given iteration, or None if not available.
        """
        return self.get_box_detections_se3_at_iteration(iteration)

    def get_route_lane_group_ids(self, iteration: int) -> Optional[List[int]]:
        """Returns the list of route lane group IDs at a given iteration, if available.

        :param iteration: The iteration to get the route lane group IDs for.
        :return: The list of route lane group IDs at the given iteration, or None if not available.
        """

    # Syntactic Sugar / Properties, that are convenient to access and pass to subclasses
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def log_metadata(self) -> LogMetadata:
        """The :class:`~py123d.datatypes.metadata.LogMetadata` of the scene."""
        return self.get_log_metadata()

    @property
    def scene_metadata(self) -> SceneMetadata:
        """The :class:`~py123d.api.scene.SceneMetadata` of the scene."""
        return self.get_scene_metadata()

    @property
    def map_metadata(self) -> Optional[MapMetadata]:
        """The :class:`~py123d.datatypes.metadata.MapMetadata` of the scene, if available."""
        return self.get_map_metadata()

    @property
    def ego_metadata(self) -> Optional[EgoMetadata]:
        """The :class:`~py123d.datatypes.vehicle_state.EgoMetadata` of the ego vehicle, if available."""
        return self.get_ego_metadata()

    @property
    def map_api(self) -> Optional[MapAPI]:
        """The :class:`~py123d.api.map.MapAPI` of the scene, if available."""
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
    def available_pinhole_camera_ids(self) -> List[PinholeCameraID]:
        """List of available :class:`~py123d.datatypes.sensors.PinholeCameraID`."""
        metadatas = self.get_pinhole_camera_metadatas()
        return list(metadatas.keys()) if metadatas is not None else []

    @property
    def available_pinhole_camera_names(self) -> List[str]:
        """List of available pinhole camera names."""
        metadatas = self.get_pinhole_camera_metadatas()
        return [camera.camera_name for camera in metadatas.values()] if metadatas is not None else []

    @property
    def available_fisheye_mei_camera_ids(self) -> List[FisheyeMEICameraID]:
        """List of available :class:`~py123d.datatypes.sensors.FisheyeMEICameraID`."""
        metadatas = self.get_fisheye_mei_camera_metadatas()
        return list(metadatas.keys()) if metadatas is not None else []

    @property
    def available_fisheye_mei_camera_names(self) -> List[str]:
        """List of available fisheye MEI camera names."""
        metadatas = self.get_fisheye_mei_camera_metadatas()
        return [camera.camera_name for camera in metadatas.values()] if metadatas is not None else []

    @property
    def available_lidar_ids(self) -> List[LidarID]:
        """List of available :class:`~py123d.datatypes.sensors.LidarID`."""
        metadatas = self.get_lidar_metadatas()
        return list(metadatas.keys()) if metadatas is not None else []

    @property
    def available_lidar_names(self) -> List[str]:
        """List of available Lidar names."""
        metadatas = self.get_lidar_metadatas()
        return [lidar.lidar_name for lidar in metadatas.values()] if metadatas is not None else []
