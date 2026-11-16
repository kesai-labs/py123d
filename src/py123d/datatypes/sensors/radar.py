from __future__ import annotations

from typing import Any, Dict, Iterator, List, Mapping, Optional, Type, Union

import numpy as np
import numpy.typing as npt

from py123d.common.utils.enums import SerialIntEnum
from py123d.datatypes.modalities.base_modality import BaseModality, BaseModalityMetadata, ModalityType
from py123d.datatypes.time.timestamp import Timestamp
from py123d.geometry import Point3DIndex, PoseSE3


class RadarID(SerialIntEnum):
    """Enumeration of Radar sensors, in multi-sensor setups."""

    RADAR_UNKNOWN = 0
    """Unknown Radar type."""

    RADAR_MERGED = 1
    """Merged sensor Radar type."""

    RADAR_FRONT = 2
    """Front-facing (center) Radar type."""

    RADAR_FRONT_LEFT = 3
    """Front-left corner Radar type."""

    RADAR_FRONT_RIGHT = 4
    """Front-right corner Radar type."""

    RADAR_SIDE_LEFT = 5
    """Left-side Radar type."""

    RADAR_SIDE_RIGHT = 6
    """Right-side Radar type."""

    RADAR_BACK_LEFT = 7
    """Back-left corner Radar type."""

    RADAR_BACK_RIGHT = 8
    """Back-right corner Radar type."""

    RADAR_BACK = 9
    """Back-facing (center) Radar type."""

    RADAR_REAR_LEFT = 10
    """Rear-left-facing Radar type (distinct from the back-left corner radar)."""

    RADAR_REAR_RIGHT = 11
    """Rear-right-facing Radar type (distinct from the back-right corner radar)."""


class RadarFeature(SerialIntEnum):
    """Enumeration of common Radar point cloud features.

    Aligned with the (rich) nuScenes radar schema; datasets populate the subset they provide.
    Extending is additive: append a member here and a dtype in :data:`RADAR_FEATURE_DTYPES`.
    """

    IDS = 0
    """Per-point originating :class:`RadarID` value. Used for merge/split, same role as
    :attr:`~py123d.datatypes.sensors.lidar.LidarFeature.IDS`."""

    TIMESTAMPS = 1
    """Per-point timestamp feature index, in microseconds."""

    CLUSTER_ID = 2
    """Per-point cluster id feature index (the nuScenes ``id`` field)."""

    DYN_PROP = 3
    """Dynamic property feature index (moving/stationary classification, dataset-native)."""

    RCS = 4
    """Radar cross section feature index, in dBsm."""

    VELOCITY_X = 5
    """Raw radial velocity x feature index, in m/s."""

    VELOCITY_Y = 6
    """Raw radial velocity y feature index, in m/s."""

    VELOCITY_X_COMP = 7
    """Ego-motion compensated velocity x feature index, in m/s."""

    VELOCITY_Y_COMP = 8
    """Ego-motion compensated velocity y feature index, in m/s."""

    IS_QUALITY_VALID = 9
    """Quality-valid state flag feature index."""

    AMBIG_STATE = 10
    """Doppler ambiguity state feature index."""

    X_RMS = 11
    """x position RMS feature index."""

    Y_RMS = 12
    """y position RMS feature index."""

    INVALID_STATE = 13
    """Invalid state flag feature index."""

    PDH0 = 14
    """False-alarm probability feature index."""

    VX_RMS = 15
    """Velocity x RMS feature index."""

    VY_RMS = 16
    """Velocity y RMS feature index."""

    SNR = 17
    """Signal-to-noise ratio feature index, in dB."""

    EXIST_PROBABILITY = 18
    """Detection existence probability feature index."""

    CONFIDENCE = 19
    """Detection confidence feature index (dataset-native scale)."""

    MULTI_TARGET_PROBABILITY = 20
    """Probability that the detection covers multiple targets feature index."""

    AZIMUTH_STD = 21
    """Azimuth angle standard deviation feature index, in radians."""

    ELEVATION_STD = 22
    """Elevation angle standard deviation feature index, in radians."""

    RADIAL_VELOCITY = 23
    """Raw radial (line-of-sight) velocity feature index, in m/s. Unprojected scalar measurement;
    datasets that only provide this may additionally derive :attr:`VELOCITY_X`/:attr:`VELOCITY_Y`
    by projecting it onto the detection azimuth."""

    RECEIVED_SIGNAL_STRENGTH = 24
    """Received signal strength feature index, in dBm."""


RADAR_FEATURE_DTYPES: Dict[RadarFeature, Type] = {
    RadarFeature.IDS: np.uint8,
    RadarFeature.TIMESTAMPS: np.int64,
    RadarFeature.CLUSTER_ID: np.uint16,
    RadarFeature.DYN_PROP: np.uint8,
    RadarFeature.RCS: np.float32,
    RadarFeature.VELOCITY_X: np.float32,
    RadarFeature.VELOCITY_Y: np.float32,
    RadarFeature.VELOCITY_X_COMP: np.float32,
    RadarFeature.VELOCITY_Y_COMP: np.float32,
    RadarFeature.IS_QUALITY_VALID: np.uint8,
    RadarFeature.AMBIG_STATE: np.uint8,
    RadarFeature.X_RMS: np.float32,
    RadarFeature.Y_RMS: np.float32,
    RadarFeature.INVALID_STATE: np.uint8,
    RadarFeature.PDH0: np.uint8,
    RadarFeature.VX_RMS: np.float32,
    RadarFeature.VY_RMS: np.float32,
    RadarFeature.SNR: np.float32,
    RadarFeature.EXIST_PROBABILITY: np.float32,
    RadarFeature.CONFIDENCE: np.float32,
    RadarFeature.MULTI_TARGET_PROBABILITY: np.float32,
    RadarFeature.AZIMUTH_STD: np.float32,
    RadarFeature.ELEVATION_STD: np.float32,
    RadarFeature.RADIAL_VELOCITY: np.float32,
    RadarFeature.RECEIVED_SIGNAL_STRENGTH: np.float32,
}


class RadarMetadata(BaseModalityMetadata):
    """Metadata for Radar sensor, static for a given sensor."""

    __slots__ = ("_radar_name", "_radar_id", "_radar_to_imu_se3", "_has_arrival_time")

    def __init__(
        self,
        radar_name: str,
        radar_id: RadarID,
        radar_to_imu_se3: PoseSE3 = PoseSE3.identity(),
        has_arrival_time: bool = False,
    ):
        """Initialize Radar metadata.

        :param radar_name: The name of the Radar sensor from the dataset.
        :param radar_id: The ID of the Radar sensor.
        :param radar_to_imu_se3: The extrinsic pose of the Radar sensor relative to the IMU.
        :param has_arrival_time: Whether the log stores the time each measurement was received
            (see :attr:`~py123d.datatypes.BaseModality.arrival_timestamp`) in the column
            ``<modality_key>.arrival_us``. Logs written without it leave it False and have no
            such column.
        """
        self._radar_name = radar_name
        self._radar_id = radar_id
        self._radar_to_imu_se3 = radar_to_imu_se3
        self._has_arrival_time = has_arrival_time

    @property
    def radar_name(self) -> str:
        """The name of the Radar sensor from the dataset."""
        return self._radar_name

    @property
    def radar_id(self) -> RadarID:
        """The ID of the Radar sensor."""
        return self._radar_id

    @property
    def radar_to_imu_se3(self) -> PoseSE3:
        """The extrinsic :class:`~py123d.geometry.PoseSE3` of the Radar sensor, relative to the IMU frame."""
        return self._radar_to_imu_se3

    @property
    def has_arrival_time(self) -> bool:
        """Whether the log stores the time each measurement was received."""
        return self._has_arrival_time

    @property
    def modality_type(self) -> ModalityType:
        return ModalityType.RADAR

    @property
    def modality_id(self) -> Optional[Union[str, SerialIntEnum]]:
        return self._radar_id

    @classmethod
    def from_dict(cls, data_dict: dict) -> RadarMetadata:
        """Construct the Radar metadata from a dictionary.

        :param data_dict: A dictionary containing Radar metadata.
        :return: An instance of RadarMetadata.
        """
        return RadarMetadata(
            radar_name=data_dict["radar_name"],
            radar_id=RadarID(data_dict["radar_id"]),
            radar_to_imu_se3=PoseSE3.from_list(data_dict["radar_to_imu_se3"]),
            has_arrival_time=data_dict.get("has_arrival_time", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Radar metadata to a dictionary.

        :return: A dictionary representation of the Radar metadata.
        """
        data_dict = {
            "radar_name": self.radar_name,
            "radar_id": int(self.radar_id),
            "radar_to_imu_se3": self.radar_to_imu_se3.tolist(),
        }
        # Only written when set, so the metadata of a log without arrival times is unchanged.
        if self._has_arrival_time:
            data_dict["has_arrival_time"] = True
        return data_dict


class RadarMergedMetadata(BaseModalityMetadata, Mapping[RadarID, RadarMetadata]):
    __slots__ = ("_radar_metadata_dict",)

    def __init__(self, radar_metadata_dict: Dict[RadarID, RadarMetadata]):
        self._radar_metadata_dict = radar_metadata_dict

    def __getitem__(self, key: RadarID) -> RadarMetadata:
        return self._radar_metadata_dict[key]

    def __iter__(self) -> Iterator[RadarID]:
        return iter(self._radar_metadata_dict)

    def __len__(self) -> int:
        return len(self._radar_metadata_dict)

    @property
    def modality_type(self) -> ModalityType:
        return ModalityType.RADAR

    @property
    def modality_id(self) -> Optional[Union[str, SerialIntEnum]]:
        return RadarID.RADAR_MERGED

    @property
    def radar_metadatas(self) -> Dict[RadarID, RadarMetadata]:
        """Returns the dictionary of per-radar metadata contained in this merged metadata."""
        return self._radar_metadata_dict

    @property
    def has_arrival_time(self) -> bool:
        """Whether every contained radar stores arrival times. A merged row then carries the latest
        of them, the time at which the whole merged scan had been received."""
        return len(self._radar_metadata_dict) > 0 and all(
            metadata.has_arrival_time for metadata in self._radar_metadata_dict.values()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the metadata instance to a plain Python dictionary.

        :return: A dictionary representation using only default Python types.
        """
        return {str(int(rid)): meta.to_dict() for rid, meta in self._radar_metadata_dict.items()}

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> RadarMergedMetadata:
        """Construct a metadata instance from a plain Python dictionary.

        :param data_dict: A dictionary containing the metadata fields.
        :return: A metadata instance.
        """
        return RadarMergedMetadata(
            radar_metadata_dict={RadarID(int(k)): RadarMetadata.from_dict(v) for k, v in data_dict.items()}
        )


class Radar(BaseModality):
    """Data structure for Radar point cloud data and associated metadata."""

    __slots__ = (
        "_timestamp",
        "_metadata",
        "_point_cloud_3d",
        "_point_cloud_features",
        "_arrival_timestamp",
    )

    def __init__(
        self,
        timestamp: Timestamp,
        metadata: Union[RadarMetadata, RadarMergedMetadata],
        point_cloud_3d: npt.NDArray[np.float32],
        point_cloud_features: Optional[Dict[str, npt.NDArray]] = None,
        arrival_timestamp: Optional[Timestamp] = None,
    ) -> None:
        """Initialize Radar data structure.

        Unlike a spinning lidar, a radar scan is treated as an instantaneous snapshot (no mechanical
        rotation / rolling shutter), so it carries a single :attr:`timestamp` rather than a sweep window.

        :param metadata: Radar metadata.
        :param point_cloud_3d: Radar point cloud as an Nx3 numpy array, where N is the number of points, \
            and the (x, y, z), indexed by :class:`~py123d.geometry.Point3DIndex`.
        :param point_cloud_features: Optional dictionary of point cloud features.
        :param arrival_timestamp: Optional time the recording system received the measurement
            (see :attr:`~py123d.datatypes.BaseModality.arrival_timestamp`).
        """
        self._timestamp = timestamp
        self._metadata = metadata
        self._point_cloud_3d = point_cloud_3d
        self._point_cloud_features = point_cloud_features
        self._arrival_timestamp = arrival_timestamp

    @property
    def radar_metadatas(self) -> Dict[RadarID, RadarMetadata]:
        """Returns the dictionary of per-radar metadata contained in this Radar's metadata.

        If the metadata is a :class:`RadarMergedMetadata`, returns its internal dictionary.
        If the metadata is a single :class:`RadarMetadata`, returns a dictionary with one entry.
        """
        radar_metadatas = {}
        if isinstance(self._metadata, RadarMergedMetadata):
            radar_metadatas = self._metadata.radar_metadatas
        else:
            radar_metadatas = {self._metadata.radar_id: self._metadata}
        return radar_metadatas

    @property
    def is_merged(self) -> bool:
        """Returns True if this Radar contains merged data from multiple radars, False if it contains data from a single radar."""
        return isinstance(self._metadata, RadarMergedMetadata)

    @property
    def metadata(self) -> Union[RadarMetadata, RadarMergedMetadata]:
        """The :class:`RadarMetadata` associated with this Radar recording."""
        return self._metadata

    @property
    def arrival_timestamp(self) -> Optional[Timestamp]:
        """The time the recording system received this measurement, if recorded."""
        return self._arrival_timestamp

    @property
    def timestamp(self) -> Timestamp:
        """The timestamp associated with this Radar recording."""
        return self._timestamp

    @property
    def point_cloud_3d(self) -> npt.NDArray[np.float32]:
        """Radar point cloud as an Nx3 numpy array, where N is the number of points, \
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

    def _feature(self, feature: RadarFeature) -> Optional[npt.NDArray]:
        """Returns the requested feature array cast to its canonical dtype, if available."""
        value: Optional[npt.NDArray] = None
        key = feature.serialize()
        if self._point_cloud_features is not None and key in self._point_cloud_features:
            value = self._point_cloud_features[key].astype(RADAR_FEATURE_DTYPES[feature])
        return value

    @property
    def ids(self) -> Optional[npt.NDArray[np.uint8]]:
        """The point cloud as an Nx1 array of originating :class:`RadarID` values, if available."""
        return self._feature(RadarFeature.IDS)  # type: ignore

    @property
    def timestamps(self) -> Optional[npt.NDArray[np.int64]]:
        """The point cloud as an Nx1 array of timestamps in microseconds, if available."""
        return self._feature(RadarFeature.TIMESTAMPS)  # type: ignore

    @property
    def cluster_id(self) -> Optional[npt.NDArray[np.uint16]]:
        """The point cloud as an Nx1 array of cluster ids, if available."""
        return self._feature(RadarFeature.CLUSTER_ID)  # type: ignore

    @property
    def rcs(self) -> Optional[npt.NDArray[np.float32]]:
        """The point cloud as an Nx1 array of radar cross section values (dBsm), if available."""
        return self._feature(RadarFeature.RCS)  # type: ignore

    @property
    def snr(self) -> Optional[npt.NDArray[np.float32]]:
        """The point cloud as an Nx1 array of signal-to-noise ratios, if available."""
        return self._feature(RadarFeature.SNR)  # type: ignore

    @property
    def confidence(self) -> Optional[npt.NDArray[np.float32]]:
        """The point cloud as an Nx1 array of detection confidences (dataset-native scale), if available."""
        return self._feature(RadarFeature.CONFIDENCE)  # type: ignore

    @property
    def velocity(self) -> Optional[npt.NDArray[np.float32]]:
        """The point cloud as an Nx2 array of raw (vx, vy) velocities in m/s, if available."""
        velocity: Optional[npt.NDArray[np.float32]] = None
        vx = self._feature(RadarFeature.VELOCITY_X)
        vy = self._feature(RadarFeature.VELOCITY_Y)
        if vx is not None and vy is not None:
            velocity = np.stack([vx, vy], axis=-1)
        return velocity

    @property
    def velocity_comp(self) -> Optional[npt.NDArray[np.float32]]:
        """The point cloud as an Nx2 array of ego-motion compensated (vx, vy) velocities in m/s, if available."""
        velocity_comp: Optional[npt.NDArray[np.float32]] = None
        vx = self._feature(RadarFeature.VELOCITY_X_COMP)
        vy = self._feature(RadarFeature.VELOCITY_Y_COMP)
        if vx is not None and vy is not None:
            velocity_comp = np.stack([vx, vy], axis=-1)
        return velocity_comp


def get_merged_radar(radars: List[Radar]) -> Optional[Radar]:
    """Merges multiple Radar objects into a single Radar object with concatenated point clouds and features."""

    radar_merged: Optional[Radar] = None

    if len(radars) >= 1:
        radar_metadatas: Dict[RadarID, RadarMetadata] = {}
        for radar in radars:
            radar_metadatas.update(radar.radar_metadatas)

        # Use the earliest timestamp among the individual radars as the merged snapshot time.
        timestamp = min(radars, key=lambda r: r.timestamp.time_us).timestamp

        # The merged scan is complete when its last part has arrived; unknown if any part lacks a time.
        known_arrivals = [radar.arrival_timestamp.time_us for radar in radars if radar.arrival_timestamp is not None]
        arrival_timestamp = Timestamp.from_us(max(known_arrivals)) if len(known_arrivals) == len(radars) else None

        point_cloud_3d = np.concatenate([radar.point_cloud_3d for radar in radars], axis=0)
        point_cloud_features_list: Dict[str, List[np.ndarray]] = {}
        for radar in radars:
            if radar.point_cloud_features is not None:
                for feature_name, feature_values in radar.point_cloud_features.items():
                    if feature_name not in point_cloud_features_list:
                        point_cloud_features_list[feature_name] = []
                    point_cloud_features_list[feature_name].append(feature_values)

        point_cloud_features = {
            feature_name: np.concatenate(features_list, axis=0)
            for feature_name, features_list in point_cloud_features_list.items()
        }
        radar_merged = Radar(
            timestamp=timestamp,
            metadata=RadarMergedMetadata(radar_metadata_dict=radar_metadatas),
            point_cloud_3d=point_cloud_3d,
            point_cloud_features=point_cloud_features,
            arrival_timestamp=arrival_timestamp,
        )

    return radar_merged


def get_individual_radar(radar_merged: Optional[Radar], radar_id: RadarID) -> Optional[Radar]:
    """Splits a merged Radar object into an individual Radar object for a specific RadarID."""
    assert radar_id != RadarID.RADAR_MERGED, "Cannot split merged radar with RADAR_MERGED ID"

    target_radar: Optional[Radar] = None

    if radar_merged is not None:
        target_metadata = radar_merged.radar_metadatas.get(radar_id, None)
        point_cloud_ids = radar_merged.ids
        if target_metadata is not None and point_cloud_ids is not None:
            target_mask = point_cloud_ids == radar_id.value
            target_point_cloud_3d = radar_merged.point_cloud_3d[target_mask]
            target_point_cloud_features = {
                feature_name: feature_values[target_mask]
                for feature_name, feature_values in (radar_merged.point_cloud_features or {}).items()
            }
            target_radar = Radar(
                timestamp=radar_merged.timestamp,
                metadata=target_metadata,
                point_cloud_3d=target_point_cloud_3d,
                point_cloud_features=target_point_cloud_features,
                arrival_timestamp=radar_merged.arrival_timestamp,
            )

    return target_radar
