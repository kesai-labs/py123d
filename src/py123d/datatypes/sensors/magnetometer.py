from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import numpy.typing as npt

from py123d.common.utils.enums import SerialIntEnum
from py123d.datatypes.modalities.base_modality import BaseModality, BaseModalityMetadata, ModalityType
from py123d.datatypes.time.timestamp import Timestamp
from py123d.geometry.pose import PoseSE3
from py123d.geometry.vector import Vector3D


class MagnetometerMetadata(BaseModalityMetadata):
    """Metadata for a magnetometer sensor, static for a given sensor."""

    __slots__ = ("_magnetometer_name", "_magnetometer_id", "_magnetometer_to_imu_se3", "_has_arrival_time")

    def __init__(
        self,
        magnetometer_name: str,
        magnetometer_id: Optional[str] = None,
        magnetometer_to_imu_se3: PoseSE3 = PoseSE3.identity(),
        has_arrival_time: bool = False,
    ):
        """Initialize magnetometer metadata.

        :param magnetometer_name: The name of the magnetometer sensor from the dataset.
        :param magnetometer_id: Optional identifier to distinguish multiple magnetometers in one
            rig. None (the default) means the log has a single magnetometer and the modality key
            is ``magnetometer``.
        :param magnetometer_to_imu_se3: The extrinsic pose of the sensor relative to the IMU frame.
        :param has_arrival_time: Whether the log stores the time each measurement was received
            (see :attr:`~py123d.datatypes.BaseModality.arrival_timestamp`) in the column
            ``<modality_key>.arrival_us``. Logs written without it leave it False and have no
            such column.
        """
        self._magnetometer_name = magnetometer_name
        self._magnetometer_id = magnetometer_id
        self._magnetometer_to_imu_se3 = magnetometer_to_imu_se3
        self._has_arrival_time = has_arrival_time

    @property
    def magnetometer_name(self) -> str:
        """The name of the magnetometer sensor from the dataset."""
        return self._magnetometer_name

    @property
    def magnetometer_id(self) -> Optional[str]:
        """Optional identifier to distinguish multiple magnetometers in one rig."""
        return self._magnetometer_id

    @property
    def magnetometer_to_imu_se3(self) -> PoseSE3:
        """The extrinsic :class:`~py123d.geometry.PoseSE3` of the sensor, relative to the IMU frame."""
        return self._magnetometer_to_imu_se3

    @property
    def has_arrival_time(self) -> bool:
        """Whether the log stores the time each measurement was received."""
        return self._has_arrival_time

    @property
    def modality_type(self) -> ModalityType:
        return ModalityType.MAGNETOMETER

    @property
    def modality_id(self) -> Optional[Union[str, SerialIntEnum]]:
        return self._magnetometer_id

    @classmethod
    def from_dict(cls, data_dict: dict) -> MagnetometerMetadata:
        """Construct the magnetometer metadata from a dictionary.

        :param data_dict: A dictionary containing magnetometer metadata.
        :return: An instance of MagnetometerMetadata.
        """
        return MagnetometerMetadata(
            magnetometer_name=data_dict["magnetometer_name"],
            magnetometer_id=data_dict.get("magnetometer_id"),
            magnetometer_to_imu_se3=PoseSE3.from_list(data_dict["magnetometer_to_imu_se3"]),
            has_arrival_time=data_dict.get("has_arrival_time", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the magnetometer metadata to a dictionary.

        :return: A dictionary representation of the magnetometer metadata.
        """
        data_dict = {
            "magnetometer_name": self._magnetometer_name,
            "magnetometer_id": self._magnetometer_id,
            "magnetometer_to_imu_se3": self._magnetometer_to_imu_se3.tolist(),
        }
        # Only written when set, so the metadata of a log without arrival times is unchanged.
        if self._has_arrival_time:
            data_dict["has_arrival_time"] = True
        return data_dict


class Magnetometer(BaseModality):
    """Data structure for a single magnetometer measurement and associated metadata.

    The magnetic field is in Tesla, in the sensor frame, following ``sensor_msgs/msg/MagneticField``.
    """

    __slots__ = ("_timestamp", "_metadata", "_magnetic_field", "_magnetic_field_covariance", "_arrival_timestamp")

    def __init__(
        self,
        timestamp: Timestamp,
        metadata: MagnetometerMetadata,
        magnetic_field: Vector3D,
        magnetic_field_covariance: Optional[npt.NDArray[np.float64]] = None,
        arrival_timestamp: Optional[Timestamp] = None,
    ) -> None:
        """Initialize a magnetometer measurement.

        :param timestamp: The timestamp of the measurement.
        :param metadata: The magnetometer metadata.
        :param magnetic_field: Magnetic field in Tesla, in the sensor frame.
        :param magnetic_field_covariance: Optional row-major 3x3 covariance, flattened to (9,).
        :param arrival_timestamp: Optional time the recording system received the measurement
            (see :attr:`~py123d.datatypes.BaseModality.arrival_timestamp`).
        """
        self._timestamp = timestamp
        self._metadata = metadata
        self._magnetic_field = magnetic_field
        self._magnetic_field_covariance = magnetic_field_covariance
        self._arrival_timestamp = arrival_timestamp

    @property
    def timestamp(self) -> Timestamp:
        """The timestamp associated with this magnetometer measurement."""
        return self._timestamp

    @property
    def metadata(self) -> MagnetometerMetadata:
        """The :class:`MagnetometerMetadata` associated with this magnetometer measurement."""
        return self._metadata

    @property
    def arrival_timestamp(self) -> Optional[Timestamp]:
        """The time the recording system received this measurement, if recorded."""
        return self._arrival_timestamp

    @property
    def magnetic_field(self) -> Vector3D:
        """Magnetic field in Tesla, in the sensor frame."""
        return self._magnetic_field

    @property
    def magnetic_field_covariance(self) -> Optional[npt.NDArray[np.float64]]:
        """Row-major 3x3 magnetic field covariance flattened to (9,), or None if not provided."""
        return self._magnetic_field_covariance
