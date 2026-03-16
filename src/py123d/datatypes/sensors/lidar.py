from __future__ import annotations

from typing import Any, Dict, Iterator, List, Mapping, Optional, Type, Union

import numpy as np
import numpy.typing as npt

from py123d.common.utils.enums import SerialIntEnum
from py123d.datatypes.modalities.base_modality import BaseModality, BaseModalityMetadata, ModalityType
from py123d.datatypes.time.timestamp import Timestamp
from py123d.geometry import Point3DIndex, PoseSE3


class LidarID(SerialIntEnum):
    """Enumeration of Lidar sensors, in multi-sensor setups."""

    LIDAR_UNKNOWN = 0
    """Unknown Lidar type."""

    LIDAR_MERGED = 1
    """Merged sensor Lidar type"""

    LIDAR_TOP = 2
    """Top-facing Lidar type."""

    LIDAR_FRONT = 3
    """Front-facing Lidar type."""

    LIDAR_SIDE_LEFT = 4
    """Left-side Lidar type."""

    LIDAR_SIDE_RIGHT = 5
    """Right-side Lidar type."""

    LIDAR_BACK = 6
    """Back-facing Lidar type."""

    LIDAR_DOWN = 7
    """Down-facing Lidar type."""


class LidarFeature(SerialIntEnum):
    """Enumeration of common Lidar point cloud features"""

    IDS = 0
    """Point IDs feature index."""

    INTENSITY = 1
    """Intensity feature index."""

    CHANNEL = 2
    """Ring feature index."""

    TIMESTAMPS = 3
    """Timestamp feature index."""

    RANGE = 4
    """Range feature index."""

    ELONGATION = 5
    """Elongation feature index."""


LIDAR_FEATURE_DTYPES: Dict[LidarFeature, Type] = {
    LidarFeature.IDS: np.uint8,
    LidarFeature.INTENSITY: np.uint8,
    LidarFeature.CHANNEL: np.uint8,
    LidarFeature.TIMESTAMPS: np.int64,
    LidarFeature.RANGE: np.float32,
    LidarFeature.ELONGATION: np.float32,
}


class LidarMetadata(BaseModalityMetadata):
    """Metadata for Lidar sensor, static for a given sensor."""

    __slots__ = ("_lidar_name", "_lidar_id", "_lidar_to_imu_se3")

    def __init__(
        self,
        lidar_name: str,
        lidar_id: LidarID,
        lidar_to_imu_se3: PoseSE3 = PoseSE3.identity(),
    ):
        """Initialize Lidar metadata.

        :param lidar_name: The name of the Lidar sensor from the dataset.
        :param lidar_id: The ID of the Lidar sensor.
        :param lidar_to_imu_se3: The extrinsic pose of the Lidar sensor relative to the IMU
        """
        self._lidar_name = lidar_name
        self._lidar_id = lidar_id
        self._lidar_to_imu_se3 = lidar_to_imu_se3

    @property
    def lidar_name(self) -> str:
        """The name of the Lidar sensor from the dataset."""
        return self._lidar_name

    @property
    def lidar_id(self) -> LidarID:
        """The ID of the Lidar sensor."""
        return self._lidar_id

    @property
    def lidar_to_imu_se3(self) -> PoseSE3:
        """The extrinsic :class:`~py123d.geometry.PoseSE3` of the Lidar sensor, relative to the IMU frame."""
        return self._lidar_to_imu_se3

    @property
    def modality_type(self) -> ModalityType:
        return ModalityType.LIDAR

    @property
    def modality_id(self) -> Optional[Union[str, SerialIntEnum]]:
        return self._lidar_id

    @classmethod
    def from_dict(cls, data_dict: dict) -> LidarMetadata:
        """Construct the Lidar metadata from a dictionary.

        :param data_dict: A dictionary containing Lidar metadata.
        :raises ValueError: If the dictionary is missing required fields or contains invalid data.
        :return: An instance of LidarMetadata.
        """
        return LidarMetadata(
            lidar_name=data_dict["lidar_name"],
            lidar_id=LidarID(data_dict["lidar_id"]),
            lidar_to_imu_se3=PoseSE3.from_list(data_dict["lidar_to_imu_se3"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Lidar metadata to a dictionary.

        :return: A dictionary representation of the Lidar metadata.
        """
        return {
            "lidar_name": self.lidar_name,
            "lidar_id": int(self.lidar_id),
            "lidar_to_imu_se3": self.lidar_to_imu_se3.tolist(),
        }


class LidarMergedMetadata(BaseModalityMetadata, Mapping[LidarID, LidarMetadata]):
    __slots__ = ("_lidar_metadata_dict",)

    def __init__(self, lidar_metadata_dict: Dict[LidarID, LidarMetadata]):
        self._lidar_metadata_dict = lidar_metadata_dict

    def __getitem__(self, key: LidarID) -> LidarMetadata:
        return self._lidar_metadata_dict[key]

    def __iter__(self) -> Iterator[LidarID]:
        return iter(self._lidar_metadata_dict)

    def __len__(self) -> int:
        return len(self._lidar_metadata_dict)

    @property
    def modality_type(self) -> ModalityType:
        return ModalityType.LIDAR

    @property
    def modality_id(self) -> Optional[Union[str, SerialIntEnum]]:
        return LidarID.LIDAR_MERGED

    @property
    def lidar_metadatas(self) -> Dict[LidarID, LidarMetadata]:
        """Returns the dictionary of per-lidar metadata contained in this merged metadata."""
        return self._lidar_metadata_dict

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the metadata instance to a plain Python dictionary.

        :return: A dictionary representation using only default Python types.
        """
        return {str(int(lid)): meta.to_dict() for lid, meta in self._lidar_metadata_dict.items()}

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> LidarMergedMetadata:
        """Construct a metadata instance from a plain Python dictionary.

        :param data_dict: A dictionary containing the metadata fields.
        :return: A metadata instance.
        """
        return LidarMergedMetadata(
            lidar_metadata_dict={LidarID(int(k)): LidarMetadata.from_dict(v) for k, v in data_dict.items()}
        )


class Lidar(BaseModality):
    """Data structure for Lidar point cloud data and associated metadata."""

    __slots__ = (
        "_timestamp",
        "_timestamp_end",
        "_metadata",
        "_point_cloud_3d",
        "_point_cloud_features",
    )

    def __init__(
        self,
        timestamp: Timestamp,
        timestamp_end: Timestamp,
        metadata: Union[LidarMetadata, LidarMergedMetadata],
        point_cloud_3d: npt.NDArray[np.float32],
        point_cloud_features: Optional[Dict[str, npt.NDArray]] = None,
    ) -> None:
        """Initialize Lidar data structure.

        :param metadata: Lidar metadata.
        :param point_cloud_3d: Lidar point cloud as an Nx3 numpy array, where N is the number of points, \
            and the (x, y, z), indexed by :class:`~py123d.geometry.Point3DIndex`.
        :param point_cloud_features: Optional dictionary of point cloud features.
        """
        self._timestamp = timestamp
        self._timestamp_end = timestamp_end
        self._metadata = metadata
        self._point_cloud_3d = point_cloud_3d
        self._point_cloud_features = point_cloud_features

    @property
    def lidar_metadatas(self) -> Dict[LidarID, LidarMetadata]:
        """Returns the dictionary of per-lidar metadata contained in this Lidar's metadata.

        If the metadata is a :class:`LidarMergedMetadata`, returns its internal dictionary.
        If the metadata is a single :class:`LidarMetadata`, returns a dictionary with one entry.
        """
        lidar_metadatas = {}
        if isinstance(self._metadata, LidarMergedMetadata):
            lidar_metadatas = self._metadata.lidar_metadatas
        else:
            lidar_metadatas = {self._metadata.lidar_id: self._metadata}
        return lidar_metadatas

    @property
    def is_merged(self) -> bool:
        """Returns True if this Lidar contains merged data from multiple lidars, False if it contains data from a single lidar."""
        return isinstance(self._metadata, LidarMergedMetadata)

    @property
    def metadata(self) -> Union[LidarMetadata, LidarMergedMetadata]:
        """The :class:`LidarMetadata` associated with this Lidar recording."""
        return self._metadata

    @property
    def timestamp(self) -> Timestamp:
        """The timestamp associated with this Lidar recording."""
        return self._timestamp

    @property
    def timestamp_end(self) -> Timestamp:
        """The end timestamp associated with this Lidar recording."""
        return self._timestamp_end

    @property
    def point_cloud_3d(self) -> npt.NDArray[np.float32]:
        """Lidar point cloud as an Nx3 numpy array, where N is the number of points, \
            and the (x, y, z), indexed by :class:`~py123d.geometry.Point3DIndex`
        """
        return self._point_cloud_3d

    @property
    def point_cloud_features(self) -> Optional[Dict[str, npt.NDArray]]:
        """The point cloud features as a dictionary of numpy arrays."""
        return self._point_cloud_features

    @property
    def xyz(self) -> npt.NDArray[np.float32]:
        """The point cloud as an Nx3 array of x, y, z coordinates."""
        return self._point_cloud_3d

    @property
    def xy(self) -> npt.NDArray[np.float32]:
        """The point cloud as an Nx2 array of x, y coordinates."""
        return self._point_cloud_3d[..., Point3DIndex.XY]  # type: ignore

    @property
    def ids(self) -> Optional[npt.NDArray[np.uint8]]:
        """The point cloud as an Nx1 array of point IDs, if available."""
        ids: Optional[npt.NDArray[np.uint8]] = None
        key = LidarFeature.IDS.serialize()
        if self._point_cloud_features is not None and key in self._point_cloud_features:
            ids = self._point_cloud_features[key].astype(np.uint8)  # type: ignore
        return ids

    @property
    def intensity(self) -> Optional[npt.NDArray[np.uint8]]:
        """The point cloud as an Nx1 array of intensity values, if available."""
        intensity: Optional[npt.NDArray[np.uint8]] = None
        key = LidarFeature.INTENSITY.serialize()
        if self._point_cloud_features is not None and key in self._point_cloud_features:
            intensity = self._point_cloud_features[key].astype(np.uint8)  # type: ignore
        return intensity

    @property
    def channel(self) -> Optional[npt.NDArray[np.uint8]]:
        """The point cloud as an Nx1 array of channel/ring values, if available."""
        channel: Optional[npt.NDArray[np.uint8]] = None
        key = LidarFeature.CHANNEL.serialize()
        if self._point_cloud_features is not None and key in self._point_cloud_features:
            channel = self._point_cloud_features[key].astype(np.uint8)  # type: ignore
        return channel

    @property
    def timestamps(self) -> Optional[npt.NDArray[np.int64]]:
        """The point cloud as an Nx1 array of timestamps in microseconds, if available."""
        timestamp: Optional[npt.NDArray[np.int64]] = None
        key = LidarFeature.TIMESTAMPS.serialize()
        if self._point_cloud_features is not None and key in self._point_cloud_features:
            timestamp = self._point_cloud_features[key].astype(np.int64)  # type: ignore
        return timestamp

    @property
    def range(self) -> Optional[npt.NDArray[np.float32]]:
        """The point cloud as an Nx1 array of range values, if available."""
        range: Optional[npt.NDArray[np.float32]] = None
        key = LidarFeature.RANGE.serialize()
        if self._point_cloud_features is not None and key in self._point_cloud_features:
            range = self._point_cloud_features[key].astype(np.float32)  # type: ignore
        return range

    @property
    def elongation(self) -> Optional[npt.NDArray[np.float32]]:
        """The point cloud as an Nx1 array of elongation values, if available."""
        elongation: Optional[npt.NDArray[np.float32]] = None
        key = LidarFeature.ELONGATION.serialize()
        if self._point_cloud_features is not None and key in self._point_cloud_features:
            elongation = self._point_cloud_features[key].astype(np.float32)  # type: ignore
        return elongation


def get_merged_lidar(lidars: List[Lidar]) -> Optional[Lidar]:
    """Merges multiple Lidar objects into a single Lidar object with concatenated point clouds and features."""

    lidar_merged: Optional[Lidar] = None

    if len(lidars) >= 1:
        lidar_metadatas: Dict[LidarID, LidarMetadata] = {}
        for lidar in lidars:
            lidar_metadatas.update(lidar.lidar_metadatas)

        # Use earliest start timestamp and latest end timestamp from the individual lidars
        timestamp = min(lidars, key=lambda l: l.timestamp.time_us).timestamp
        timestamp_end = max(lidars, key=lambda l: l.timestamp_end.time_us).timestamp_end

        point_cloud_3d = np.concatenate([lidar.point_cloud_3d for lidar in lidars], axis=0)
        point_cloud_features_list: Dict[str, List[np.ndarray]] = {}
        for lidar in lidars:
            if lidar.point_cloud_features is not None:
                for feature_name, feature_values in lidar.point_cloud_features.items():
                    if feature_name not in point_cloud_features_list:
                        point_cloud_features_list[feature_name] = []
                    point_cloud_features_list[feature_name].append(feature_values)

        point_cloud_features = {
            feature_name: np.concatenate(features_list, axis=0)
            for feature_name, features_list in point_cloud_features_list.items()
        }
        lidar_merged = Lidar(
            timestamp=timestamp,
            timestamp_end=timestamp_end,
            metadata=LidarMergedMetadata(lidar_metadata_dict=lidar_metadatas),
            point_cloud_3d=point_cloud_3d,
            point_cloud_features=point_cloud_features,
        )

    return lidar_merged


def get_individual_lidar(lidar_merged: Optional[Lidar], lidar_id: LidarID) -> Optional[Lidar]:
    """Splits a merged Lidar object into an individual Lidar object for a specific LidarID."""
    assert lidar_id != LidarID.LIDAR_MERGED, "Cannot split merged lidar with LIDAR_MERGED ID"

    target_lidar: Optional[Lidar] = None

    if lidar_merged is not None:
        target_metadata = lidar_merged.lidar_metadatas.get(lidar_id, None)
        point_cloud_ids = lidar_merged.ids
        if target_metadata is not None and point_cloud_ids is not None:
            target_mask = point_cloud_ids == lidar_id.value
            target_point_cloud_3d = lidar_merged.point_cloud_3d[target_mask]
            target_point_cloud_features = {
                feature_name: feature_values[target_mask]
                for feature_name, feature_values in (lidar_merged.point_cloud_features or {}).items()
            }
            target_metadata = lidar_merged.lidar_metadatas[lidar_id]
            target_lidar = Lidar(
                timestamp=lidar_merged.timestamp,
                timestamp_end=lidar_merged.timestamp_end,
                metadata=target_metadata,
                point_cloud_3d=target_point_cloud_3d,
                point_cloud_features=target_point_cloud_features,
            )

    return target_lidar
