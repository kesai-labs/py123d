from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from py123d.common.utils.enums import SerialIntEnum
from py123d.datatypes.modalities.base_modality import BaseModality, BaseModalityMetadata, ModalityType
from py123d.datatypes.time.timestamp import Timestamp
from py123d.geometry.pose import PoseSE3


class GnssMetadata(BaseModalityMetadata):
    """Metadata for a GNSS receiver, static for a given sensor.

    Mirrors the shape of ``sensor_msgs/msg/NavSatFix``. The optional :attr:`datum_lla` records
    the geodetic reference point that the log's local metric frame was anchored to (typically
    the first valid fix), so local x/y/z coordinates in the log can be mapped back to earth
    coordinates.
    """

    __slots__ = (
        "_gnss_name",
        "_gnss_id",
        "_gnss_to_imu_se3",
        "_datum_lla",
        "_has_solution_quality",
        "_has_arrival_time",
    )

    def __init__(
        self,
        gnss_name: str,
        gnss_id: Optional[str] = None,
        gnss_to_imu_se3: PoseSE3 = PoseSE3.identity(),
        datum_lla: Optional[Tuple[float, float, float]] = None,
        has_solution_quality: bool = False,
        has_arrival_time: bool = False,
    ):
        """Initialize GNSS metadata.

        :param gnss_name: The name of the GNSS receiver from the dataset.
        :param gnss_id: Optional identifier to distinguish multiple receivers in one rig. None
            (the default) means the log has a single receiver and the modality key is ``gnss``.
        :param gnss_to_imu_se3: The extrinsic pose of the GNSS antenna relative to the IMU frame.
        :param datum_lla: Optional (latitude_deg, longitude_deg, altitude_m) reference point of
            the log's local metric frame.
        :param has_solution_quality: Whether the receiver reports the solution-quality fields
            (satellite count, fix type, reported accuracies, DOP, NED velocity). Logs converted
            before these fields existed leave it False and store no such columns.
        :param has_arrival_time: Whether the log stores the time each measurement was received
            (see :attr:`~py123d.datatypes.BaseModality.arrival_timestamp`) in the column
            ``<modality_key>.arrival_us``. Logs written without it leave it False and have no
            such column.
        """
        self._gnss_name = gnss_name
        self._gnss_id = gnss_id
        self._gnss_to_imu_se3 = gnss_to_imu_se3
        self._datum_lla = datum_lla
        self._has_solution_quality = has_solution_quality
        self._has_arrival_time = has_arrival_time

    @property
    def gnss_name(self) -> str:
        """The name of the GNSS receiver from the dataset."""
        return self._gnss_name

    @property
    def gnss_id(self) -> Optional[str]:
        """Optional identifier to distinguish multiple GNSS receivers in one rig."""
        return self._gnss_id

    @property
    def gnss_to_imu_se3(self) -> PoseSE3:
        """The extrinsic :class:`~py123d.geometry.PoseSE3` of the GNSS antenna, relative to the IMU frame."""
        return self._gnss_to_imu_se3

    @property
    def datum_lla(self) -> Optional[Tuple[float, float, float]]:
        """The (latitude_deg, longitude_deg, altitude_m) datum of the log's local frame, if recorded."""
        return self._datum_lla

    @property
    def has_solution_quality(self) -> bool:
        """Whether the receiver reports the solution-quality fields."""
        return self._has_solution_quality

    @property
    def has_arrival_time(self) -> bool:
        """Whether the log stores the time each measurement was received."""
        return self._has_arrival_time

    @property
    def modality_type(self) -> ModalityType:
        return ModalityType.GNSS

    @property
    def modality_id(self) -> Optional[Union[str, SerialIntEnum]]:
        return self._gnss_id

    @classmethod
    def from_dict(cls, data_dict: dict) -> GnssMetadata:
        """Construct the GNSS metadata from a dictionary.

        :param data_dict: A dictionary containing GNSS metadata.
        :return: An instance of GnssMetadata.
        """
        datum = data_dict.get("datum_lla")
        return GnssMetadata(
            gnss_name=data_dict["gnss_name"],
            gnss_id=data_dict.get("gnss_id"),
            gnss_to_imu_se3=PoseSE3.from_list(data_dict["gnss_to_imu_se3"]),
            datum_lla=tuple(datum) if datum is not None else None,
            has_solution_quality=data_dict.get("has_solution_quality", False),
            has_arrival_time=data_dict.get("has_arrival_time", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the GNSS metadata to a dictionary.

        :return: A dictionary representation of the GNSS metadata.
        """
        data_dict = {
            "gnss_name": self._gnss_name,
            "gnss_id": self._gnss_id,
            "gnss_to_imu_se3": self._gnss_to_imu_se3.tolist(),
            "datum_lla": list(self._datum_lla) if self._datum_lla is not None else None,
            "has_solution_quality": self._has_solution_quality,
        }
        # Only written when set, so the metadata of a log without arrival times is unchanged.
        if self._has_arrival_time:
            data_dict["has_arrival_time"] = True
        return data_dict


class Gnss(BaseModality):
    """Data structure for a single GNSS fix and associated metadata.

    Field semantics follow ``sensor_msgs/msg/NavSatFix``: WGS-84 geodetic coordinates, ENU
    position covariance in m^2, and the NavSatStatus fix status / service constants
    (status: -1 no fix, 0 fix, 1 SBAS fix, 2 GBAS fix).

    The solution-quality fields (:attr:`num_satellites`, :attr:`fix_type`,
    :attr:`horizontal_accuracy`, :attr:`vertical_accuracy`, :attr:`position_dop`,
    :attr:`velocity_ned`) are what a receiver reports about the solution itself rather than
    the position it produced. They are what distinguishes a measured fix from one the
    receiver is extrapolating, which NavSatFix alone cannot express: entering a tunnel the
    status and covariance can stay unchanged for seconds while the satellite count is
    already falling. They are None for receivers or datasets that do not report them.
    """

    __slots__ = (
        "_timestamp",
        "_metadata",
        "_latitude",
        "_longitude",
        "_altitude",
        "_position_covariance",
        "_position_covariance_type",
        "_status",
        "_service",
        "_num_satellites",
        "_fix_type",
        "_horizontal_accuracy",
        "_vertical_accuracy",
        "_position_dop",
        "_velocity_ned",
        "_speed_accuracy",
        "_arrival_timestamp",
    )

    def __init__(
        self,
        timestamp: Timestamp,
        metadata: GnssMetadata,
        latitude: float,
        longitude: float,
        altitude: float,
        position_covariance: Optional[npt.NDArray[np.float64]] = None,
        position_covariance_type: Optional[int] = None,
        status: Optional[int] = None,
        service: Optional[int] = None,
        num_satellites: Optional[int] = None,
        fix_type: Optional[int] = None,
        horizontal_accuracy: Optional[float] = None,
        vertical_accuracy: Optional[float] = None,
        position_dop: Optional[float] = None,
        velocity_ned: Optional[npt.NDArray[np.float64]] = None,
        speed_accuracy: Optional[float] = None,
        arrival_timestamp: Optional[Timestamp] = None,
    ) -> None:
        """Initialize a GNSS fix.

        :param timestamp: The timestamp of the fix.
        :param metadata: The GNSS metadata.
        :param latitude: WGS-84 latitude in degrees.
        :param longitude: WGS-84 longitude in degrees.
        :param altitude: Altitude in meters above the WGS-84 ellipsoid.
        :param position_covariance: Optional row-major 3x3 ENU position covariance in m^2, flattened to (9,).
        :param position_covariance_type: Optional NavSatFix covariance type constant.
        :param status: Optional NavSatStatus fix status constant.
        :param service: Optional NavSatStatus service bitmask.
        :param num_satellites: Optional number of satellites used in the solution.
        :param fix_type: Optional receiver fix type (0 none, 2 2D, 3 3D, higher values are
            receiver specific, e.g. 4 GNSS + dead reckoning).
        :param horizontal_accuracy: Optional reported 1-sigma horizontal accuracy in meters.
        :param vertical_accuracy: Optional reported 1-sigma vertical accuracy in meters.
        :param position_dop: Optional position dilution of precision.
        :param velocity_ned: Optional (north, east, down) velocity in m/s.
        :param speed_accuracy: Optional reported 1-sigma accuracy of that velocity in m/s. The
            velocity's counterpart to :attr:`horizontal_accuracy`, and what a consumer fusing the
            velocity needs in order to weight it.
        :param arrival_timestamp: Optional time the recording system received the measurement
            (see :attr:`~py123d.datatypes.BaseModality.arrival_timestamp`).
        """
        self._timestamp = timestamp
        self._metadata = metadata
        self._latitude = latitude
        self._longitude = longitude
        self._altitude = altitude
        self._position_covariance = position_covariance
        self._position_covariance_type = position_covariance_type
        self._status = status
        self._service = service
        self._num_satellites = num_satellites
        self._fix_type = fix_type
        self._horizontal_accuracy = horizontal_accuracy
        self._vertical_accuracy = vertical_accuracy
        self._position_dop = position_dop
        self._velocity_ned = velocity_ned
        self._speed_accuracy = speed_accuracy
        self._arrival_timestamp = arrival_timestamp

    @property
    def timestamp(self) -> Timestamp:
        """The timestamp associated with this GNSS fix."""
        return self._timestamp

    @property
    def metadata(self) -> GnssMetadata:
        """The :class:`GnssMetadata` associated with this GNSS fix."""
        return self._metadata

    @property
    def arrival_timestamp(self) -> Optional[Timestamp]:
        """The time the recording system received this measurement, if recorded."""
        return self._arrival_timestamp

    @property
    def latitude(self) -> float:
        """WGS-84 latitude in degrees."""
        return self._latitude

    @property
    def longitude(self) -> float:
        """WGS-84 longitude in degrees."""
        return self._longitude

    @property
    def altitude(self) -> float:
        """Altitude in meters above the WGS-84 ellipsoid."""
        return self._altitude

    @property
    def lla(self) -> npt.NDArray[np.float64]:
        """The fix as a (3,) array of (latitude_deg, longitude_deg, altitude_m)."""
        return np.array([self._latitude, self._longitude, self._altitude], dtype=np.float64)

    @property
    def position_covariance(self) -> Optional[npt.NDArray[np.float64]]:
        """Row-major 3x3 ENU position covariance in m^2 flattened to (9,), or None if not provided."""
        return self._position_covariance

    @property
    def position_covariance_type(self) -> Optional[int]:
        """NavSatFix covariance type constant, or None if not provided."""
        return self._position_covariance_type

    @property
    def status(self) -> Optional[int]:
        """NavSatStatus fix status constant (-1 no fix, 0 fix, 1 SBAS, 2 GBAS), or None if not provided."""
        return self._status

    @property
    def service(self) -> Optional[int]:
        """NavSatStatus service bitmask, or None if not provided."""
        return self._service

    @property
    def num_satellites(self) -> Optional[int]:
        """Number of satellites used in the solution, or None if not reported."""
        return self._num_satellites

    @property
    def fix_type(self) -> Optional[int]:
        """Receiver fix type (0 none, 2 2D, 3 3D, higher receiver specific), or None if not reported."""
        return self._fix_type

    @property
    def horizontal_accuracy(self) -> Optional[float]:
        """Reported 1-sigma horizontal accuracy in meters, or None if not reported."""
        return self._horizontal_accuracy

    @property
    def vertical_accuracy(self) -> Optional[float]:
        """Reported 1-sigma vertical accuracy in meters, or None if not reported."""
        return self._vertical_accuracy

    @property
    def position_dop(self) -> Optional[float]:
        """Position dilution of precision, or None if not reported."""
        return self._position_dop

    @property
    def velocity_ned(self) -> Optional[npt.NDArray[np.float64]]:
        """(north, east, down) velocity in m/s as a (3,) array, or None if not reported."""
        return self._velocity_ned

    @property
    def speed_accuracy(self) -> Optional[float]:
        """Reported 1-sigma accuracy of :attr:`velocity_ned` in m/s, if the receiver supplied it."""
        return self._speed_accuracy

    @property
    def ground_speed(self) -> Optional[float]:
        """Horizontal speed over ground in m/s from :attr:`velocity_ned`, or None if not reported."""
        if self._velocity_ned is None:
            return None
        return float(np.hypot(self._velocity_ned[0], self._velocity_ned[1]))
