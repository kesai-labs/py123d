from __future__ import annotations

import abc
from typing import Dict, List, Literal, Optional, Tuple, TypeVar, Union

from py123d.api.map.map_api import MapAPI
from py123d.common.utils.enums import SerialIntEnum
from py123d.datatypes import (
    BaseCameraMetadata,
    BaseModality,
    BaseModalityMetadata,
    BoxDetectionsSE3,
    BoxDetectionsSE3Metadata,
    Camera,
    CameraID,
    CustomModality,
    CustomModalityMetadata,
    EgoStateSE3,
    EgoStateSE3Metadata,
    Lidar,
    LidarID,
    LidarMergedMetadata,
    LidarMetadata,
    LogMetadata,
    MapMetadata,
    ModalityType,
    Timestamp,
    TrafficLightDetections,
    TrafficLightDetectionsMetadata,
)
from py123d.datatypes.metadata import SceneMetadata

T = TypeVar("T")


def checked_optional_cast(obj: object, cls: type[T]) -> Optional[T]:
    """Checks that ``obj`` is an instance of ``cls`` or None, then returns it with the narrowed type."""
    if obj is None:
        return None
    if not isinstance(obj, cls):
        raise TypeError(f"Expected object of type {cls} or None, but got {type(obj)}")
    return obj


class SceneAPI(abc.ABC):
    """Base class for all scene APIs. The scene API provides access to all data modalities at in a scene."""

    __slots__ = ()

    # ------------------------------------------------------------------------------------------------------------------
    # 1. Abstract Methods, to be implemented by subclasses
    # ------------------------------------------------------------------------------------------------------------------

    # 1.1 Scene / Log Metadata
    # ------------------------------------------------------------------------------------------------------------------

    @abc.abstractmethod
    def get_scene_metadata(self) -> SceneMetadata:
        """Returns the :class:`~py123d.datatypes.metadata.SceneMetadata` of the scene.

        :return: The scene metadata.
        """

    @abc.abstractmethod
    def get_log_metadata(self) -> LogMetadata:
        """Returns the :class:`~py123d.datatypes.metadata.LogMetadata` of the scene.

        :return: The log metadata.
        """

    @abc.abstractmethod
    def get_timestamp_at_iteration(self, iteration: int) -> Timestamp:
        """Returns the :class:`~py123d.datatypes.time.Timestamp` at a given iteration.

        :param iteration: The iteration to get the timestamp for.
        :return: The timestamp at the given iteration.
        """

    @abc.abstractmethod
    def get_all_iteration_timestamps(self, include_history: bool = False) -> List[Timestamp]:
        """Returns all sync timestamps for the current scene.

        :param include_history: If True, include history iterations before the scene start.
        :return: List of timestamps, one per iteration in the scene.
        """

    @abc.abstractmethod
    def get_scene_timestamp_boundaries(self, include_history: bool = False) -> Tuple[Timestamp, Timestamp]:
        """Returns the first and last sync timestamps of the scene.

        :param include_history: If True, extend the range to include history iterations.
        :return: Tuple of (first_timestamp, last_timestamp).
        """

    # 1.2 Map
    # ------------------------------------------------------------------------------------------------------------------

    @abc.abstractmethod
    def get_map_metadata(self) -> Optional[MapMetadata]:
        """Returns the :class:`~py123d.datatypes.metadata.MapMetadata` of the scene, if available.

        :return: The map metadata, or None if not available.
        """

    @abc.abstractmethod
    def get_map_api(self) -> Optional[MapAPI]:
        """Returns the :class:`~py123d.api.MapAPI` of the scene, if available.

        :return: The map API, or None if not available.
        """

    # 1.3 General Modalities
    # ------------------------------------------------------------------------------------------------------------------

    @abc.abstractmethod
    def get_all_modality_metadatas(self) -> Dict[str, BaseModalityMetadata]:
        """Returns all modality metadatas found in the log directory.

        :return: Mapping of modality key to its metadata.
        """

    @abc.abstractmethod
    def get_modality_metadata(
        self,
        modality_type: Union[str, ModalityType],
        modality_id: Optional[Union[str, SerialIntEnum]] = None,
    ) -> Optional[BaseModalityMetadata]:
        """Returns the metadata for a specific modality.

        :param modality_type: The type of the modality.
        :param modality_id: The ID of the modality, if applicable.
        :return: The modality metadata, or None if not found.
        """

    @abc.abstractmethod
    def get_all_modality_timestamps(
        self,
        modality_type: Union[str, ModalityType],
        modality_id: Optional[Union[str, SerialIntEnum]] = None,
        include_history: bool = False,
    ) -> List[Timestamp]:
        """Returns all timestamps for a specific modality within the scene range.

        :param modality_type: The modality type as a string or :class:`ModalityType`.
        :param modality_id: Optional modality id (e.g. sensor id).
        :param include_history: If True, include history iterations before the scene start.
        :return: List of timestamps, empty if the modality is not present.
        """

    @abc.abstractmethod
    def get_modality_at_iteration(
        self,
        iteration: int,
        modality_type: Union[str, ModalityType],
        modality_id: Optional[Union[str, SerialIntEnum]] = None,
        **kwargs,
    ) -> Optional[BaseModality]:
        """Returns the modality data at a given iteration, if available.

        :param iteration: The iteration to get the modality data for.
        :param modality_type: The modality type as a string or :class:`ModalityType`.
        :param modality_id: Optional modality id (e.g. sensor id).
        :return: The modality data at the given iteration, or None if not available.
        """

    @abc.abstractmethod
    def get_modality_at_timestamp(
        self,
        timestamp: Union[Timestamp, int],
        modality_type: Union[str, ModalityType],
        modality_id: Optional[Union[str, SerialIntEnum]] = None,
        criteria: Literal["exact", "nearest", "forward", "backward"] = "exact",
        **kwargs,
    ) -> Optional[BaseModality]:
        """Returns the modality data at a given timestamp, if available.

        :param timestamp: The timestamp to get the modality data for, as a Timestamp object or integer microseconds.
        :param modality_type: The modality type as a string or :class:`ModalityType`.
        :param modality_id: Optional modality id (e.g. sensor id).
        :param criteria: Criteria for matching the timestamp if an exact match is not found. One of:
            - "exact": Only return data if an exact timestamp match is found.
            - "nearest": Return data from the nearest timestamp.
            - "forward": Return data from the nearest timestamp that is greater than or equal to the requested timestamp.
            - "backward": Return data from the nearest timestamp that is less than or equal to the requested timestamp.
        :return: The modality data at the given timestamp, or None if not available.
        """

    # ------------------------------------------------------------------------------------------------------------------
    # 2. Per-modality access methods.
    # ------------------------------------------------------------------------------------------------------------------

    # 2.1 Ego State SE3
    # ------------------------------------------------------------------------------------------------------------------

    def get_ego_state_se3_metadata(self) -> Optional[EgoStateSE3Metadata]:
        """Returns the :class:`~py123d.datatypes.EgoStateSE3Metadata` of the ego vehicle, if available.

        :return: The ego metadata, or None if not available.
        """
        ego_state_se3_metadata = self.get_modality_metadata(ModalityType.EGO_STATE_SE3)
        return checked_optional_cast(ego_state_se3_metadata, EgoStateSE3Metadata)

    def get_all_ego_state_se3_timestamps(self, include_history: bool = False) -> List[Timestamp]:
        """Returns all ego state timestamps within the current scene.

        :param include_history: If True, include history iterations before the scene start.
        :return: All ego state timestamps in the scene, ordered by time.
        """
        return self.get_all_modality_timestamps(
            modality_type=ModalityType.EGO_STATE_SE3, include_history=include_history
        )

    def get_ego_state_se3_at_iteration(self, iteration: int) -> Optional[EgoStateSE3]:
        """Returns the :class:`~py123d.datatypes.vehicle_state.EgoStateSE3` at a given iteration, if available.

        :param iteration: The iteration to get the ego state for.
        :return: The ego state at the given iteration, or None if not available.
        """
        ego_state_se3 = self.get_modality_at_iteration(iteration, modality_type=ModalityType.EGO_STATE_SE3)
        return checked_optional_cast(ego_state_se3, EgoStateSE3)

    def get_ego_state_se3_at_timestamp(
        self,
        timestamp: Union[Timestamp, int],
        criteria: Literal["exact", "nearest", "forward", "backward"] = "exact",
    ) -> Optional[EgoStateSE3]:
        """Returns the :class:`~py123d.datatypes.vehicle_state.EgoStateSE3` at a given timestamp, if available.

        :param timestamp: The timestamp to get the ego state for, as a Timestamp object or integer microseconds.
        :param criteria: Criteria for matching the timestamp if an exact match is not found. One of:
            - "exact": Only return data if an exact timestamp match is found.
            - "nearest": Return data from the nearest timestamp.
            - "forward": Return data from the nearest timestamp that is greater than or equal to the requested timestamp.
            - "backward": Return data from the nearest timestamp that is less than or equal to the requested timestamp.
        :return: The ego state at the given timestamp, or None if not available.
        """
        ego_state_se3 = self.get_modality_at_timestamp(
            timestamp, modality_type=ModalityType.EGO_STATE_SE3, criteria=criteria
        )
        return checked_optional_cast(ego_state_se3, EgoStateSE3)

    # 2.2 Box Detections SE3
    # ------------------------------------------------------------------------------------------------------------------

    def get_box_detections_se3_metadata(self) -> Optional[BoxDetectionsSE3Metadata]:
        """Returns the :class:`~py123d.datatypes.detections.BoxDetectionsSE3Metadata` of the scene, if available.

        :return: The box detection metadata, or None if not available.
        """
        box_detections_se3_metadata = self.get_modality_metadata(ModalityType.BOX_DETECTIONS_SE3)
        return checked_optional_cast(box_detections_se3_metadata, BoxDetectionsSE3Metadata)

    def get_all_box_detections_se3_timestamps(self, include_history: bool = False) -> List[Timestamp]:
        """Returns all box detection timestamps within the current scene.

        :param include_history: If True, include history iterations before the scene start.
        :return: All box detection timestamps in the scene, ordered by time.
        """
        return self.get_all_modality_timestamps(
            modality_type=ModalityType.BOX_DETECTIONS_SE3, include_history=include_history
        )

    def get_box_detections_se3_at_iteration(self, iteration: int) -> Optional[BoxDetectionsSE3]:
        """Returns the :class:`~py123d.datatypes.detections.BoxDetectionsSE3` at a given iteration, if available.

        :param iteration: The iteration to get the box detections for.
        :return: The box detections at the given iteration, or None if not available.
        """
        box_detections_se3 = self.get_modality_at_iteration(iteration, modality_type=ModalityType.BOX_DETECTIONS_SE3)
        return checked_optional_cast(box_detections_se3, BoxDetectionsSE3)

    def get_box_detections_se3_at_timestamp(
        self,
        timestamp: Union[Timestamp, int],
        criteria: Literal["exact", "nearest", "forward", "backward"] = "exact",
    ) -> Optional[BoxDetectionsSE3]:
        """Returns the :class:`~py123d.datatypes.detections.BoxDetectionsSE3` at a given timestamp, if available.

        :param timestamp: The timestamp to get the box detections for, as a Timestamp object or integer microseconds.
        :param criteria: Criteria for matching the timestamp if an exact match is not found. One of:
            - "exact": Only return data if an exact timestamp match is found.
            - "nearest": Return data from the nearest timestamp.
            - "forward": Return data from the nearest timestamp that is greater than or equal to the requested timestamp.
            - "backward": Return data from the nearest timestamp that is less than or equal to the requested timestamp.
        :return: The box detections at the given timestamp, or None if not available.
        """
        box_detections_se3 = self.get_modality_at_timestamp(
            timestamp, modality_type=ModalityType.BOX_DETECTIONS_SE3, criteria=criteria
        )
        return checked_optional_cast(box_detections_se3, BoxDetectionsSE3)

    # 2.3 Traffic Light Detections
    # ------------------------------------------------------------------------------------------------------------------

    def get_traffic_light_detections_metadata(self) -> Optional[TrafficLightDetectionsMetadata]:
        """Returns the :class:`~py123d.datatypes.detections.TrafficLightDetectionsMetadata` of the scene, if available.

        :return: The traffic light detection metadata, or None if not available.
        """
        traffic_light_detections_metadata = self.get_modality_metadata(ModalityType.TRAFFIC_LIGHT_DETECTIONS)
        return checked_optional_cast(traffic_light_detections_metadata, TrafficLightDetectionsMetadata)

    def get_all_traffic_light_detections_timestamps(self, include_history: bool = False) -> List[Timestamp]:
        """Returns all traffic light detection timestamps within the current scene.

        :param include_history: If True, include history iterations before the scene start.
        :return: All traffic light detection timestamps in the scene, ordered by time.
        """
        return self.get_all_modality_timestamps(
            modality_type=ModalityType.TRAFFIC_LIGHT_DETECTIONS, include_history=include_history
        )

    def get_traffic_light_detections_at_iteration(self, iteration: int) -> Optional[TrafficLightDetections]:
        """Returns the :class:`~py123d.datatypes.detections.TrafficLightDetections` at a given iteration,
            if available.

        :param iteration: The iteration to get the traffic light detections for.
        :return: The traffic light detections at the given iteration, or None if not available.
        """
        traffic_light_detections = self.get_modality_at_iteration(
            iteration, modality_type=ModalityType.TRAFFIC_LIGHT_DETECTIONS
        )
        return checked_optional_cast(traffic_light_detections, TrafficLightDetections)

    def get_traffic_light_detections_at_timestamp(
        self,
        timestamp: Union[Timestamp, int],
        criteria: Literal["exact", "nearest", "forward", "backward"] = "exact",
    ) -> Optional[TrafficLightDetections]:
        traffic_light_detections = self.get_modality_at_timestamp(
            timestamp,
            modality_type=ModalityType.TRAFFIC_LIGHT_DETECTIONS,
            criteria=criteria,
        )
        return checked_optional_cast(traffic_light_detections, TrafficLightDetections)

    # 2.4 Camera
    # ------------------------------------------------------------------------------------------------------------------

    def get_camera_metadatas(self) -> Dict[CameraID, BaseCameraMetadata]:
        """Returns per-camera metadata for all cameras in the scene.

        :return: A dictionary mapping camera IDs to their metadata.
        """
        camera_metadatas = {
            metadata.camera_id: metadata
            for metadata in self.get_all_modality_metadatas().values()
            if metadata.modality_type == ModalityType.CAMERA and isinstance(metadata, BaseCameraMetadata)
        }
        return camera_metadatas

    def get_all_camera_timestamps(self, camera_id: CameraID, include_history: bool = False) -> List[Timestamp]:
        """Returns all camera timestamps within the current scene.

        :param camera_id: The camera ID.
        :param include_history: If True, include history iterations before the scene start.
        :return: All camera timestamps in the scene, ordered by time.
        """
        return self.get_all_modality_timestamps(
            modality_type=ModalityType.CAMERA, modality_id=camera_id, include_history=include_history
        )

    def get_camera_at_iteration(
        self,
        iteration: int,
        camera_id: CameraID,
        scale: Optional[int] = None,
    ) -> Optional[Camera]:
        """Returns a :class:`~py123d.datatypes.sensors.Camera` at a given iteration, if available.

        :param iteration: The iteration to get the camera for.
        :param camera_id: The camera ID.
        :param scale: Optional downscale denominator, e.g. 2 for half size, 4 for quarter size.
        :return: The camera, or None if not available.
        """
        camera = self.get_modality_at_iteration(
            iteration,
            modality_type=ModalityType.CAMERA,
            modality_id=camera_id,
            scale=scale,
        )
        return checked_optional_cast(camera, Camera)

    def get_camera_at_timestamp(
        self,
        timestamp: Union[Timestamp, int],
        camera_id: CameraID,
        criteria: Literal["exact", "nearest", "forward", "backward"] = "exact",
        scale: Optional[int] = None,
    ) -> Optional[Camera]:
        """Returns a :class:`~py123d.datatypes.sensors.Camera` at a given timestamp, if available.

        :param timestamp: The timestamp to get the camera for, as a Timestamp object or integer microseconds.
        :param camera_id: The camera ID.
        :param criteria: Criteria for matching the timestamp if an exact match is not found. One of:
            - "exact": Only return data if an exact timestamp match is found.
            - "nearest": Return data from the nearest timestamp.
            - "forward": Return data from the nearest timestamp that is greater than or equal to the requested timestamp.
            - "backward": Return data from the nearest timestamp that is less than or equal to the requested timestamp.
        :param scale: Optional downscale denominator, e.g. 2 for half size, 4 for quarter size.
        :return: The camera, or None if not available.
        """
        camera = self.get_modality_at_timestamp(
            timestamp,
            modality_type=ModalityType.CAMERA,
            modality_id=camera_id,
            criteria=criteria,
            scale=scale,
        )
        return checked_optional_cast(camera, Camera)

    # 2.5 Lidar
    # ------------------------------------------------------------------------------------------------------------------

    def get_lidar_metadatas(self) -> Dict[LidarID, LidarMetadata]:
        """Returns per-lidar metadata, if available.

        :return: The lidar metadatas, or None if not available.
        """
        lidar_metadatas: Dict[LidarID, LidarMetadata] = {}
        merged_lidar_metadata = self.get_modality_metadata(ModalityType.LIDAR, LidarID.LIDAR_MERGED)
        if merged_lidar_metadata is not None and isinstance(merged_lidar_metadata, LidarMergedMetadata):
            lidar_metadatas.update(merged_lidar_metadata.lidar_metadatas)
        else:
            for metadata in self.get_all_modality_metadatas().values():
                if metadata.modality_type == ModalityType.LIDAR and isinstance(metadata, LidarMetadata):
                    lidar_metadatas[metadata.lidar_id] = metadata
        return lidar_metadatas

    def get_all_lidar_timestamps(self, lidar_id: LidarID, include_history: bool = False) -> List[Timestamp]:
        """Returns all lidar start timestamps within the current scene.

        :param lidar_id: The :type:`~py123d.datatypes.sensors.LidarID` of the Lidar.
        :param include_history: If True, include history iterations before the scene start.
        :return: All lidar start timestamps in the scene, ordered by time.
        """
        return self.get_all_modality_timestamps(
            modality_type=ModalityType.LIDAR, modality_id=lidar_id, include_history=include_history
        )

    def get_lidar_at_iteration(self, iteration: int, lidar_id: LidarID) -> Optional[Lidar]:
        """Returns the :class:`~py123d.datatypes.sensors.Lidar` of a given :class:`~py123d.datatypes.sensors.LidarID`\
            at a given iteration, if available.

        :param iteration: The iteration to get the Lidar for.
        :param lidar_id: The :type:`~py123d.datatypes.sensors.LidarID` of the Lidar.
        :return: The Lidar, or None if not available.
        """
        merged_lidar_metadata = self.get_modality_metadata(ModalityType.LIDAR, LidarID.LIDAR_MERGED)
        _modality_id = LidarID.LIDAR_MERGED if merged_lidar_metadata is not None else lidar_id
        lidar = self.get_modality_at_iteration(
            iteration=iteration,
            modality_type=ModalityType.LIDAR,
            modality_id=_modality_id,
            lidar_id=lidar_id,
        )
        return checked_optional_cast(lidar, Lidar)

    def get_lidar_at_timestamp(
        self,
        timestamp: Union[Timestamp, int],
        lidar_id: LidarID,
        criteria: Literal["exact", "nearest", "forward", "backward"] = "exact",
    ) -> Optional[Lidar]:
        """Returns the :class:`~py123d.datatypes.sensors.Lidar` of a given :class:`~py123d.datatypes.sensors.LidarID`\
            at a given timestamp, if available.

        :param timestamp: The timestamp to get the Lidar for, as a Timestamp object or integer microseconds.
        :param lidar_id: The :type:`~py123d.datatypes.sensors.LidarID` of the Lidar.
        :param criteria: Criteria for matching the timestamp if an exact match is not found. One of:
            - "exact": Only return data if an exact timestamp match is found.
            - "nearest": Return data from the nearest timestamp.
            - "forward": Return data from the nearest timestamp that is greater than or equal to the requested timestamp.
            - "backward": Return data from the nearest timestamp that is less than or equal to the requested timestamp.
        :return: The Lidar, or None if not available.
        """
        merged_lidar_metadata = self.get_modality_metadata(ModalityType.LIDAR, LidarID.LIDAR_MERGED)
        _modality_id = LidarID.LIDAR_MERGED if merged_lidar_metadata is not None else lidar_id
        lidar = self.get_modality_at_timestamp(
            timestamp=timestamp,
            modality_type=ModalityType.LIDAR,
            modality_id=_modality_id,
            criteria=criteria,
        )
        return checked_optional_cast(lidar, Lidar)

    # 2.6 Custom Modalities
    # ------------------------------------------------------------------------------------------------------------------

    def get_all_custom_modality_metadatas(self) -> Dict[str, CustomModalityMetadata]:
        """Returns the metadata for all custom modalities in the scene, keyed by modality ID.

        :return: A dictionary of custom modality metadata, keyed by modality ID.
        """
        custom_modality_metadatas: Dict[str, CustomModalityMetadata] = {}
        for metadata in self.get_all_modality_metadatas().values():
            if metadata.modality_type == ModalityType.CUSTOM and isinstance(metadata, CustomModalityMetadata):
                custom_modality_metadatas[str(metadata.modality_id)] = metadata
        return custom_modality_metadatas

    def get_all_custom_modality_timestamps(self, modality_id: str, include_history: bool = False) -> List[Timestamp]:
        """Returns all custom modality timestamps within the current scene.

        :param modality_id: The ID of the custom modality.
        :param include_history: If True, include history iterations before the scene start.
        :return: All custom modality timestamps in the scene, ordered by time.
        """
        return self.get_all_modality_timestamps(
            modality_type=ModalityType.CUSTOM, modality_id=modality_id, include_history=include_history
        )

    def get_custom_modality_at_iteration(self, iteration: int, modality_id: str) -> Optional[CustomModality]:
        """Returns the :class:`~py123d.datatypes.custom.CustomModality` with the given ID at a given iteration,
            if available.

        :param iteration: The iteration to get the custom modality for.
        :param modality_id: The ID of the custom modality (e.g. ``"route"``, ``"predictions"``).
        :return: The custom modality, or None if not available.
        """
        custom_modality = self.get_modality_at_iteration(
            iteration=iteration, modality_type=ModalityType.CUSTOM, modality_id=modality_id
        )
        return checked_optional_cast(custom_modality, CustomModality)

    def get_custom_modality_at_timestamp(
        self,
        timestamp: Union[Timestamp, int],
        modality_id: str,
        criteria: Literal["exact", "nearest", "forward", "backward"] = "exact",
    ) -> Optional[CustomModality]:
        """Returns the :class:`~py123d.datatypes.custom.CustomModality` with the given ID at a given timestamp,
            if available.

        :param timestamp: The timestamp to get the custom modality for, as a Timestamp object or integer microseconds.
        :param modality_id: The ID of the custom modality (e.g. ``"route"``, ``"predictions"``).
        :param criteria: Criteria for matching the timestamp if an exact match is not found. One of:
            - "exact": Only return data if an exact timestamp match is found.
            - "nearest": Return data from the nearest timestamp.
            - "forward": Return data from the nearest timestamp that is greater than or equal to the requested timestamp.
            - "backward": Return data from the nearest timestamp that is less than or equal to the requested timestamp.
        :return: The custom modality, or None if not available.
        """
        custom_modality = self.get_modality_at_timestamp(
            timestamp=timestamp,
            modality_type=ModalityType.CUSTOM,
            modality_id=modality_id,
            criteria=criteria,
        )
        return checked_optional_cast(custom_modality, CustomModality)

    # Syntactic Sugar / Properties, that are convenient to access and pass to subclasses
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def log_metadata(self) -> LogMetadata:
        """The :class:`~py123d.datatypes.metadata.LogMetadata` of the scene."""
        return self.get_log_metadata()

    @property
    def scene_metadata(self) -> SceneMetadata:
        """The :class:`~py123d.datatypes.metadata.SceneMetadata` of the scene."""
        return self.get_scene_metadata()

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
    def scene_uuid(self) -> str:
        """The UUID of the scene."""
        return self.scene_metadata.initial_uuid

    @property
    def number_of_iterations(self) -> int:
        """The number of iterations in the scene (includes current frame + future)."""
        return self.scene_metadata.num_future_iterations + 1

    @property
    def number_of_history_iterations(self) -> int:
        """The number of history iterations in the scene."""
        return self.scene_metadata.num_history_iterations

    @property
    def available_camera_ids(self) -> List[CameraID]:
        """List of available camera IDs."""
        return list(self.get_camera_metadatas().keys())

    @property
    def available_camera_names(self) -> List[str]:
        """List of available camera names."""
        return [camera.camera_name for camera in self.get_camera_metadatas().values()]

    @property
    def available_lidar_ids(self) -> List[LidarID]:
        """List of available :class:`~py123d.datatypes.sensors.LidarID`."""
        available_lidar_ids = list(self.get_lidar_metadatas().keys())
        if self.get_modality_metadata(ModalityType.LIDAR, LidarID.LIDAR_MERGED) is not None:
            available_lidar_ids += [LidarID.LIDAR_MERGED]
        return available_lidar_ids

    @property
    def available_lidar_names(self) -> List[str]:
        """List of available Lidar names."""
        available_lidar_names: List[str] = [
            lidar_metadata.lidar_name for lidar_metadata in self.get_lidar_metadatas().values()
        ]
        if self.get_modality_metadata(ModalityType.LIDAR, LidarID.LIDAR_MERGED) is not None:
            available_lidar_names.append(LidarID.LIDAR_MERGED.serialize())
        return available_lidar_names
