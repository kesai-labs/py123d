from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import numpy.typing as npt

from py123d.common.utils.enums import SerialIntEnum
from py123d.datatypes.modalities.base_modality import BaseModality, BaseModalityMetadata, ModalityType
from py123d.datatypes.time.timestamp import Timestamp
from py123d.geometry.pose import PoseSE3
from py123d.geometry.rotation import Quaternion
from py123d.geometry.vector import Vector3D


class ImuMetadata(BaseModalityMetadata):
    """Metadata for an IMU sensor, static for a given sensor.

    Mirrors the shape of ``sensor_msgs/msg/Imu``: gyroscope and accelerometer are always
    present, while the fused orientation and the covariance matrices are optional and
    declared up front via :attr:`has_orientation` / :attr:`has_covariances`. Columns for
    undeclared fields are omitted from the Arrow file entirely.
    """

    __slots__ = ("_imu_name", "_imu_id", "_imu_to_imu_se3", "_has_orientation", "_has_covariances", "_has_arrival_time")

    def __init__(
        self,
        imu_name: str,
        imu_id: Optional[str] = None,
        imu_to_imu_se3: PoseSE3 = PoseSE3.identity(),
        has_orientation: bool = False,
        has_covariances: bool = False,
        has_arrival_time: bool = False,
    ):
        """Initialize IMU metadata.

        :param imu_name: The name of the IMU sensor from the dataset.
        :param imu_id: Optional identifier to distinguish multiple IMUs in one rig. None (the
            default) means the log has a single IMU and the modality key is just ``imu``.
        :param imu_to_imu_se3: The extrinsic pose of this IMU relative to the rig's reference
            IMU frame. Identity for the reference IMU itself.
        :param has_orientation: Whether this sensor provides a fused orientation quaternion.
        :param has_covariances: Whether this sensor provides the three covariance matrices.
        :param has_arrival_time: Whether the log stores the time each measurement was received
            (see :attr:`~py123d.datatypes.BaseModality.arrival_timestamp`) in the column
            ``<modality_key>.arrival_us``. Logs written without it leave it False and have no
            such column.
        """
        self._imu_name = imu_name
        self._imu_id = imu_id
        self._imu_to_imu_se3 = imu_to_imu_se3
        self._has_orientation = has_orientation
        self._has_covariances = has_covariances
        self._has_arrival_time = has_arrival_time

    @property
    def imu_name(self) -> str:
        """The name of the IMU sensor from the dataset."""
        return self._imu_name

    @property
    def imu_id(self) -> Optional[str]:
        """Optional identifier to distinguish multiple IMUs in one rig."""
        return self._imu_id

    @property
    def imu_to_imu_se3(self) -> PoseSE3:
        """The extrinsic :class:`~py123d.geometry.PoseSE3` of this IMU, relative to the rig's reference IMU frame."""
        return self._imu_to_imu_se3

    @property
    def has_orientation(self) -> bool:
        """Whether this sensor provides a fused orientation quaternion."""
        return self._has_orientation

    @property
    def has_covariances(self) -> bool:
        """Whether this sensor provides covariance matrices."""
        return self._has_covariances

    @property
    def has_arrival_time(self) -> bool:
        """Whether the log stores the time each measurement was received."""
        return self._has_arrival_time

    @property
    def modality_type(self) -> ModalityType:
        return ModalityType.IMU

    @property
    def modality_id(self) -> Optional[Union[str, SerialIntEnum]]:
        return self._imu_id

    @classmethod
    def from_dict(cls, data_dict: dict) -> ImuMetadata:
        """Construct the IMU metadata from a dictionary.

        :param data_dict: A dictionary containing IMU metadata.
        :return: An instance of ImuMetadata.
        """
        return ImuMetadata(
            imu_name=data_dict["imu_name"],
            imu_id=data_dict.get("imu_id"),
            imu_to_imu_se3=PoseSE3.from_list(data_dict["imu_to_imu_se3"]),
            has_orientation=data_dict.get("has_orientation", False),
            has_covariances=data_dict.get("has_covariances", False),
            has_arrival_time=data_dict.get("has_arrival_time", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the IMU metadata to a dictionary.

        :return: A dictionary representation of the IMU metadata.
        """
        data_dict = {
            "imu_name": self._imu_name,
            "imu_id": self._imu_id,
            "imu_to_imu_se3": self._imu_to_imu_se3.tolist(),
            "has_orientation": self._has_orientation,
            "has_covariances": self._has_covariances,
        }
        # Only written when set, so the metadata of a log without arrival times is unchanged.
        if self._has_arrival_time:
            data_dict["has_arrival_time"] = True
        return data_dict


class Imu(BaseModality):
    """Data structure for a single IMU measurement and associated metadata.

    Angular velocity and linear acceleration are expressed in the sensor frame, following
    ``sensor_msgs/msg/Imu`` conventions (rad/s and m/s^2, acceleration includes gravity).
    """

    __slots__ = (
        "_timestamp",
        "_metadata",
        "_angular_velocity",
        "_linear_acceleration",
        "_orientation",
        "_orientation_covariance",
        "_angular_velocity_covariance",
        "_linear_acceleration_covariance",
        "_arrival_timestamp",
    )

    def __init__(
        self,
        timestamp: Timestamp,
        metadata: ImuMetadata,
        angular_velocity: Vector3D,
        linear_acceleration: Vector3D,
        orientation: Optional[Quaternion] = None,
        orientation_covariance: Optional[npt.NDArray[np.float64]] = None,
        angular_velocity_covariance: Optional[npt.NDArray[np.float64]] = None,
        linear_acceleration_covariance: Optional[npt.NDArray[np.float64]] = None,
        arrival_timestamp: Optional[Timestamp] = None,
    ) -> None:
        """Initialize an IMU measurement.

        :param timestamp: The timestamp of the measurement.
        :param metadata: The IMU metadata.
        :param angular_velocity: Gyroscope reading in rad/s, in the sensor frame.
        :param linear_acceleration: Accelerometer reading in m/s^2, in the sensor frame.
        :param orientation: Optional fused orientation quaternion.
        :param orientation_covariance: Optional row-major 3x3 covariance, flattened to (9,).
        :param angular_velocity_covariance: Optional row-major 3x3 covariance, flattened to (9,).
        :param linear_acceleration_covariance: Optional row-major 3x3 covariance, flattened to (9,).
        :param arrival_timestamp: Optional time the recording system received the measurement
            (see :attr:`~py123d.datatypes.BaseModality.arrival_timestamp`).
        """
        self._timestamp = timestamp
        self._metadata = metadata
        self._angular_velocity = angular_velocity
        self._linear_acceleration = linear_acceleration
        self._orientation = orientation
        self._orientation_covariance = orientation_covariance
        self._angular_velocity_covariance = angular_velocity_covariance
        self._linear_acceleration_covariance = linear_acceleration_covariance
        self._arrival_timestamp = arrival_timestamp

    @property
    def timestamp(self) -> Timestamp:
        """The timestamp associated with this IMU measurement."""
        return self._timestamp

    @property
    def metadata(self) -> ImuMetadata:
        """The :class:`ImuMetadata` associated with this IMU measurement."""
        return self._metadata

    @property
    def arrival_timestamp(self) -> Optional[Timestamp]:
        """The time the recording system received this measurement, if recorded."""
        return self._arrival_timestamp

    @property
    def angular_velocity(self) -> Vector3D:
        """Gyroscope reading in rad/s, in the sensor frame."""
        return self._angular_velocity

    @property
    def linear_acceleration(self) -> Vector3D:
        """Accelerometer reading in m/s^2, in the sensor frame."""
        return self._linear_acceleration

    @property
    def orientation(self) -> Optional[Quaternion]:
        """Fused orientation quaternion, or None if the sensor does not provide one."""
        return self._orientation

    @property
    def orientation_covariance(self) -> Optional[npt.NDArray[np.float64]]:
        """Row-major 3x3 orientation covariance flattened to (9,), or None if not provided."""
        return self._orientation_covariance

    @property
    def angular_velocity_covariance(self) -> Optional[npt.NDArray[np.float64]]:
        """Row-major 3x3 angular velocity covariance flattened to (9,), or None if not provided."""
        return self._angular_velocity_covariance

    @property
    def linear_acceleration_covariance(self) -> Optional[npt.NDArray[np.float64]]:
        """Row-major 3x3 linear acceleration covariance flattened to (9,), or None if not provided."""
        return self._linear_acceleration_covariance
