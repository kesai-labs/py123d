from __future__ import annotations

from typing import Dict, List, Optional

import py123d
from py123d.datatypes.custom.custom_modality import CustomModalityMetadata
from py123d.datatypes.detections.box_detections_metadata import BoxDetectionsSE3Metadata
from py123d.datatypes.detections.traffic_light_detections import TrafficLightDetectionsMetadata
from py123d.datatypes.metadata.base_metadata import BaseMetadata, BaseModalityMetadata
from py123d.datatypes.metadata.map_metadata import MapMetadata
from py123d.datatypes.sensors.fisheye_mei_camera import FisheyeMEICameraID, FisheyeMEICameraMetadata
from py123d.datatypes.sensors.lidar import LidarID, LidarMergedMetadata, LidarMetadata
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraID, PinholeCameraMetadata
from py123d.datatypes.vehicle_state.ego_metadata import EgoStateSE3Metadata


class LogMetadata(BaseMetadata):
    """Class to hold metadata information about a log."""

    __slots__ = (
        "_dataset",
        "_split",
        "_log_name",
        "_location",
        "_timestep_seconds",
        "_map_metadata",
        "_ego_state_se3_metadata",
        "_box_detections_se3_metadata",
        "_traffic_light_detections_metadata",
        "_pinhole_cameras_metadata",
        "_fisheye_mei_cameras_metadata",
        "_lidars_metadata",
        "_lidar_merged_metadata",
        "_custom_modalities_metadata",
        "_version",
    )

    def __init__(
        self,
        dataset: str,
        split: str,
        log_name: str,
        location: Optional[str],
        timestep_seconds: float,
        map_metadata: Optional[MapMetadata] = None,
        ego_state_se3_metadata: Optional[EgoStateSE3Metadata] = None,
        box_detections_se3_metadata: Optional[BoxDetectionsSE3Metadata] = None,
        traffic_light_detections_metadata: Optional[TrafficLightDetectionsMetadata] = None,
        pinhole_cameras_metadata: Optional[Dict[PinholeCameraID, PinholeCameraMetadata]] = None,
        fisheye_mei_cameras_metadata: Optional[Dict[FisheyeMEICameraID, FisheyeMEICameraMetadata]] = None,
        lidars_metadata: Optional[Dict[LidarID, LidarMetadata]] = None,
        lidar_merged_metadata: Optional[LidarMergedMetadata] = None,
        custom_modalities_metadata: Optional[Dict[str, CustomModalityMetadata]] = None,
        version: str = str(py123d.__version__),
    ):
        """Create a :class:`LogMetadata` instance from a dictionary.

        :param dataset: The dataset name in lowercase.
        :param split: Data split name, typically ``{dataset_name}_{train/val/test}``.
        :param log_name: Name of the log file.
        :param location: Location of the log data.
        :param timestep_seconds: The time interval between consecutive frames in seconds.
        """

        # Basic log info
        self._dataset = dataset
        self._split = split
        self._log_name = log_name
        self._location = location
        self._timestep_seconds = timestep_seconds

        # Map metadata
        self._map_metadata: Optional[MapMetadata] = map_metadata

        # Modality Meta
        self._ego_state_se3_metadata: Optional[EgoStateSE3Metadata] = ego_state_se3_metadata
        self._box_detections_se3_metadata: Optional[BoxDetectionsSE3Metadata] = box_detections_se3_metadata
        self._traffic_light_detections_metadata: Optional[TrafficLightDetectionsMetadata] = (
            traffic_light_detections_metadata
        )
        self._pinhole_cameras_metadata: Optional[Dict[PinholeCameraID, PinholeCameraMetadata]] = (
            pinhole_cameras_metadata
        )
        self._fisheye_mei_cameras_metadata: Optional[Dict[FisheyeMEICameraID, FisheyeMEICameraMetadata]] = (
            fisheye_mei_cameras_metadata
        )
        self._lidars_metadata: Optional[Dict[LidarID, LidarMetadata]] = lidars_metadata
        self._lidar_merged_metadata: Optional[LidarMergedMetadata] = lidar_merged_metadata
        self._custom_modalities_metadata: Optional[Dict[str, CustomModalityMetadata]] = custom_modalities_metadata

        # Currently not used, but can be helpful for tracking library version used to create the log metadata
        self._version = version

    @property
    def dataset(self) -> str:
        """The dataset name in lowercase."""
        return self._dataset

    @property
    def split(self) -> str:
        """Data split name, typically ``{dataset_name}_{train/val/test}``."""
        return self._split

    @property
    def log_name(self) -> str:
        """Name of the log file."""
        return self._log_name

    @property
    def location(self) -> Optional[str]:
        """Location of the log data."""
        return self._location

    @property
    def timestep_seconds(self) -> float:
        """The time interval between consecutive frames in seconds."""
        return self._timestep_seconds

    @property
    def version(self) -> str:
        """Version of the py123d library used to create this log metadata (not used currently)."""
        return self._version

    @property
    def map_metadata(self) -> Optional[MapMetadata]:
        """Map metadata for this log, if available."""
        return self._map_metadata

    @property
    def ego_state_se3_metadata(self) -> Optional[EgoStateSE3Metadata]:
        """Ego state SE3 metadata for this log, if available."""
        return self._ego_state_se3_metadata

    @property
    def box_detections_se3_metadata(self) -> Optional[BoxDetectionsSE3Metadata]:
        """Box detections SE3 metadata for this log, if available."""
        return self._box_detections_se3_metadata

    @property
    def traffic_light_detections_metadata(self) -> Optional[TrafficLightDetectionsMetadata]:
        """Traffic light detections metadata for this log, if available."""
        return self._traffic_light_detections_metadata

    @property
    def pinhole_cameras_metadata(self) -> Dict[PinholeCameraID, PinholeCameraMetadata]:
        """Pinhole camera metadata for this log, if available. Returns a dictionary mapping camera IDs to metadata."""
        return self._pinhole_cameras_metadata if self._pinhole_cameras_metadata is not None else {}

    @property
    def fisheye_mei_cameras_metadata(self) -> Dict[FisheyeMEICameraID, FisheyeMEICameraMetadata]:
        """Fisheye MEI camera metadata for this log, if available. Returns a dictionary mapping camera IDs to metadata."""
        return self._fisheye_mei_cameras_metadata if self._fisheye_mei_cameras_metadata is not None else {}

    @property
    def lidars_metadata(self) -> Dict[LidarID, LidarMetadata]:
        """Lidar metadata for this log, if available. Returns a dictionary mapping lidar IDs to metadata."""
        return self._lidars_metadata if self._lidars_metadata is not None else {}

    @property
    def lidar_merged_metadata(self) -> Optional[LidarMergedMetadata]:
        """Lidar merged metadata for this log, if available."""
        return self._lidar_merged_metadata

    @property
    def all_modality_metadatas(self) -> List[BaseModalityMetadata]:
        """Returns a flat list of all modality metadata present in this log."""
        result: List[BaseModalityMetadata] = []
        if self._ego_state_se3_metadata is not None:
            result.append(self._ego_state_se3_metadata)
        if self._box_detections_se3_metadata is not None:
            result.append(self._box_detections_se3_metadata)
        if self._traffic_light_detections_metadata is not None:
            result.append(self._traffic_light_detections_metadata)
        if self._pinhole_cameras_metadata is not None:
            for cam_meta in self._pinhole_cameras_metadata.values():
                result.append(cam_meta)
        if self._fisheye_mei_cameras_metadata is not None:
            for cam_meta in self._fisheye_mei_cameras_metadata.values():
                result.append(cam_meta)
        if self._lidars_metadata is not None:
            for lidar_meta in self._lidars_metadata.values():
                result.append(lidar_meta)
        if self._lidar_merged_metadata is not None:
            result.append(self._lidar_merged_metadata)
        if self._custom_modalities_metadata is not None:
            for custom_meta in self._custom_modalities_metadata.values():
                result.append(custom_meta)
        return result

    # Basic fields that are always serialized.
    _BASIC_FIELDS = ("dataset", "split", "log_name", "location", "timestep_seconds", "version")

    @classmethod
    def from_dict(cls, data_dict: Dict) -> LogMetadata:
        """Create a :class:`LogMetadata` instance from a Python dictionary.

        Deserializes both basic log fields and modality metadata (if present).
        Older dictionaries that only contain basic fields are handled gracefully.

        :param data_dict: Dictionary containing log metadata.
        :return: A :class:`LogMetadata` instance.
        """
        # Map metadata
        map_meta_raw = data_dict.get("map_metadata")
        map_metadata = MapMetadata.from_dict(map_meta_raw) if map_meta_raw is not None else None

        # Ego state
        ego_raw = data_dict.get("ego_state_se3_metadata")
        ego_metadata = EgoStateSE3Metadata.from_dict(ego_raw) if ego_raw is not None else None

        # Box detections
        box_raw = data_dict.get("box_detections_se3_metadata")
        box_metadata = BoxDetectionsSE3Metadata.from_dict(box_raw) if box_raw is not None else None

        # Traffic lights
        tl_raw = data_dict.get("traffic_light_detections_metadata")
        tl_metadata = TrafficLightDetectionsMetadata.from_dict(tl_raw) if tl_raw is not None else None

        # Pinhole cameras: serialized as {str(int(cam_id)): cam_meta_dict, ...}
        pcam_raw = data_dict.get("pinhole_cameras_metadata")
        pinhole_cameras_metadata: Optional[Dict[PinholeCameraID, PinholeCameraMetadata]] = None
        if pcam_raw is not None:
            pinhole_cameras_metadata = {
                PinholeCameraID(int(k)): PinholeCameraMetadata.from_dict(v) for k, v in pcam_raw.items()
            }

        # Fisheye MEI cameras
        fcam_raw = data_dict.get("fisheye_mei_cameras_metadata")
        fisheye_mei_cameras_metadata: Optional[Dict[FisheyeMEICameraID, FisheyeMEICameraMetadata]] = None
        if fcam_raw is not None:
            fisheye_mei_cameras_metadata = {
                FisheyeMEICameraID(int(k)): FisheyeMEICameraMetadata.from_dict(v) for k, v in fcam_raw.items()
            }

        # Lidars
        lidar_raw = data_dict.get("lidars_metadata")
        lidars_metadata: Optional[Dict[LidarID, LidarMetadata]] = None
        if lidar_raw is not None:
            lidars_metadata = {LidarID(int(k)): LidarMetadata.from_dict(v) for k, v in lidar_raw.items()}

        # Lidar merged
        lm_raw = data_dict.get("lidar_merged_metadata")
        lidar_merged_metadata = LidarMergedMetadata.from_dict(lm_raw) if lm_raw is not None else None

        # Custom modalities
        custom_raw = data_dict.get("custom_modalities_metadata")
        custom_modalities_metadata: Optional[Dict[str, CustomModalityMetadata]] = None
        if custom_raw is not None:
            custom_modalities_metadata = {k: CustomModalityMetadata.from_dict(v) for k, v in custom_raw.items()}

        return LogMetadata(
            dataset=data_dict["dataset"],
            split=data_dict["split"],
            log_name=data_dict["log_name"],
            location=data_dict.get("location"),
            timestep_seconds=data_dict["timestep_seconds"],
            version=data_dict.get("version", "unknown"),
            map_metadata=map_metadata,
            ego_state_se3_metadata=ego_metadata,
            box_detections_se3_metadata=box_metadata,
            traffic_light_detections_metadata=tl_metadata,
            pinhole_cameras_metadata=pinhole_cameras_metadata,
            fisheye_mei_cameras_metadata=fisheye_mei_cameras_metadata,
            lidars_metadata=lidars_metadata,
            lidar_merged_metadata=lidar_merged_metadata,
            custom_modalities_metadata=custom_modalities_metadata,
        )

    def to_dict(self) -> Dict:
        """Convert the :class:`LogMetadata` instance to a JSON-serializable dictionary.

        Serializes both basic log fields and all modality metadata.

        :return: A dictionary representation of the log metadata.
        """
        result: Dict = {f: getattr(self, f"_{f}") for f in self._BASIC_FIELDS}

        result["map_metadata"] = self._map_metadata.to_dict() if self._map_metadata is not None else None
        result["ego_state_se3_metadata"] = (
            self._ego_state_se3_metadata.to_dict() if self._ego_state_se3_metadata is not None else None
        )
        result["box_detections_se3_metadata"] = (
            self._box_detections_se3_metadata.to_dict() if self._box_detections_se3_metadata is not None else None
        )
        result["traffic_light_detections_metadata"] = (
            self._traffic_light_detections_metadata.to_dict()
            if self._traffic_light_detections_metadata is not None
            else None
        )
        result["pinhole_cameras_metadata"] = (
            {str(int(k)): v.to_dict() for k, v in self._pinhole_cameras_metadata.items()}
            if self._pinhole_cameras_metadata is not None
            else None
        )
        result["fisheye_mei_cameras_metadata"] = (
            {str(int(k)): v.to_dict() for k, v in self._fisheye_mei_cameras_metadata.items()}
            if self._fisheye_mei_cameras_metadata is not None
            else None
        )
        result["lidars_metadata"] = (
            {str(int(k)): v.to_dict() for k, v in self._lidars_metadata.items()}
            if self._lidars_metadata is not None
            else None
        )
        result["lidar_merged_metadata"] = (
            self._lidar_merged_metadata.to_dict() if self._lidar_merged_metadata is not None else None
        )
        result["custom_modalities_metadata"] = (
            {k: v.to_dict() for k, v in self._custom_modalities_metadata.items()}
            if self._custom_modalities_metadata is not None
            else None
        )

        return result

    def __repr__(self) -> str:
        return (
            f"LogMetadata(dataset={self.dataset}, split={self.split}, log_name={self.log_name}, "
            f"location={self.location}, timestep_seconds={self.timestep_seconds}, "
            f"version={self.version})"
        )
