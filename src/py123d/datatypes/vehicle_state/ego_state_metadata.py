from __future__ import annotations

from py123d.datatypes.modalities.base_modality import BaseModalityMetadata, ModalityType
from py123d.geometry import PoseSE2, PoseSE3
from py123d.geometry.transform import abs_to_rel_se2, abs_to_rel_se3, rel_to_abs_se2, rel_to_abs_se3


class EgoStateSE3Metadata(BaseModalityMetadata):
    """Metadata that describes the physical dimensions of the ego vehicle."""

    __slots__ = ("vehicle_name", "width", "length", "height", "wheel_base", "center_to_imu_se3", "rear_axle_to_imu_se3")

    vehicle_name: str
    """Name of the vehicle model."""

    width: float
    """Width of the vehicle."""

    length: float
    """Length of the vehicle."""

    height: float
    """Height of the vehicle."""

    wheel_base: float
    """Wheel base of the vehicle (longitudinal distance between front and rear axles)."""

    center_to_imu_se3: PoseSE3
    """The center-to-IMU extrinsic :class:`~py123d.geometry.PoseSE3` of the vehicle.

    Maps coordinates from the vehicle center frame to the IMU frame. The translation
    component gives the position of the vehicle center in the IMU frame.
    """

    rear_axle_to_imu_se3: PoseSE3
    """The rear axle-to-IMU extrinsic :class:`~py123d.geometry.PoseSE3` of the vehicle.

    Maps coordinates from the rear axle frame to the IMU frame. Identity for most
    datasets where the IMU is co-located with the rear axle.
    """

    def __init__(
        self,
        vehicle_name: str,
        width: float,
        length: float,
        height: float,
        wheel_base: float,
        center_to_imu_se3: PoseSE3,
        rear_axle_to_imu_se3: PoseSE3,
    ) -> None:
        self.vehicle_name = vehicle_name
        self.width = width
        self.length = length
        self.height = height
        self.wheel_base = wheel_base
        self.center_to_imu_se3 = center_to_imu_se3
        self.rear_axle_to_imu_se3 = rear_axle_to_imu_se3

    @classmethod
    def from_dict(cls, data_dict: dict) -> EgoStateSE3Metadata:
        """Creates a EgoStateSE3Metadata instance from a dictionary.

        :param data_dict: Dictionary containing ego metadata.
        :return: EgoStateSE3Metadata instance.
        """
        data_dict = dict(data_dict)
        data_dict["center_to_imu_se3"] = PoseSE3.from_list(data_dict["center_to_imu_se3"])
        data_dict["rear_axle_to_imu_se3"] = PoseSE3.from_list(data_dict["rear_axle_to_imu_se3"])
        return EgoStateSE3Metadata(**data_dict)

    @property
    def half_width(self) -> float:
        """Half the width of the vehicle."""
        return self.width / 2.0

    @property
    def half_length(self) -> float:
        """Half the length of the vehicle."""
        return self.length / 2.0

    @property
    def half_height(self) -> float:
        """Half the height of the vehicle."""
        return self.height / 2.0

    @property
    def rear_axle_to_center_longitudinal(self) -> float:
        """Longitudinal offset from the rear axle to the vehicle center (along the x-axis)."""
        return self.center_to_imu_se3.x - self.rear_axle_to_imu_se3.x

    @property
    def rear_axle_to_center_vertical(self) -> float:
        """Vertical offset from the rear axle to the vehicle center (along the z-axis)."""
        return self.center_to_imu_se3.z - self.rear_axle_to_imu_se3.z

    @property
    def modality_type(self) -> ModalityType:
        """Returns the name of the modality that this metadata describes."""
        return ModalityType.EGO_STATE_SE3

    def to_dict(self) -> dict:
        """Converts the :class:`EgoStateSE3Metadata` instance to a dictionary.

        :return: Dictionary representation of the ego metadata.
        """
        return {
            "vehicle_name": self.vehicle_name,
            "width": self.width,
            "length": self.length,
            "height": self.height,
            "wheel_base": self.wheel_base,
            "center_to_imu_se3": self.center_to_imu_se3.tolist(),
            "rear_axle_to_imu_se3": self.rear_axle_to_imu_se3.tolist(),
        }


def get_carla_lincoln_mkz_2020_metadata() -> EgoStateSE3Metadata:
    """Helper function to get CARLA Lincoln MKZ 2020 ego metadata."""
    # NOTE: These parameters are taken from the CARLA simulator vehicle model. The rear axles to center transform
    # parameters are calculated based on parameters from the CARLA simulator.
    return EgoStateSE3Metadata(
        vehicle_name="carla_lincoln_mkz_2020",
        width=1.83671,
        length=4.89238,
        height=1.49028,
        wheel_base=2.86048,
        center_to_imu_se3=PoseSE3(x=1.64855, y=0.0, z=0.38579, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
        rear_axle_to_imu_se3=PoseSE3.identity(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# IMU <-> Rear Axle conversions (SE3 / SE2)
# ──────────────────────────────────────────────────────────────────────────────


def imu_se3_to_rear_axle_se3(imu_se3: PoseSE3, metadata: EgoStateSE3Metadata) -> PoseSE3:
    """Converts an IMU world pose to a rear axle world pose in SE3.

    :param imu_se3: The IMU pose in the global frame.
    :param metadata: The ego metadata.
    :return: The rear axle pose in the global frame.
    """
    return rel_to_abs_se3(origin=imu_se3, pose_se3=metadata.rear_axle_to_imu_se3)


def rear_axle_se3_to_imu_se3(rear_axle_se3: PoseSE3, metadata: EgoStateSE3Metadata) -> PoseSE3:
    """Converts a rear axle world pose to an IMU world pose in SE3.

    :param rear_axle_se3: The rear axle pose in the global frame.
    :param metadata: The ego metadata.
    :return: The IMU pose in the global frame.
    """
    imu_in_rear_axle = abs_to_rel_se3(metadata.rear_axle_to_imu_se3, PoseSE3.identity())
    return rel_to_abs_se3(origin=rear_axle_se3, pose_se3=imu_in_rear_axle)


def imu_se2_to_rear_axle_se2(imu_se2: PoseSE2, metadata: EgoStateSE3Metadata) -> PoseSE2:
    """Converts an IMU world pose to a rear axle world pose in SE2.

    :param imu_se2: The IMU pose in the global frame (SE2).
    :param metadata: The ego metadata.
    :return: The rear axle pose in the global frame (SE2).
    """
    return rel_to_abs_se2(origin=imu_se2, pose_se2=metadata.rear_axle_to_imu_se3.pose_se2)


def rear_axle_se2_to_imu_se2(rear_axle_se2: PoseSE2, metadata: EgoStateSE3Metadata) -> PoseSE2:
    """Converts a rear axle world pose to an IMU world pose in SE2.

    :param rear_axle_se2: The rear axle pose in the global frame (SE2).
    :param metadata: The ego metadata.
    :return: The IMU pose in the global frame (SE2).
    """
    imu_in_rear_axle = abs_to_rel_se2(metadata.rear_axle_to_imu_se3.pose_se2, PoseSE2.identity())
    return rel_to_abs_se2(origin=rear_axle_se2, pose_se2=imu_in_rear_axle)


# ──────────────────────────────────────────────────────────────────────────────
# IMU <-> Center conversions (SE3 / SE2)
# ──────────────────────────────────────────────────────────────────────────────


def imu_se3_to_center_se3(imu_se3: PoseSE3, metadata: EgoStateSE3Metadata) -> PoseSE3:
    """Converts an IMU world pose to a vehicle center world pose in SE3.

    :param imu_se3: The IMU pose in the global frame.
    :param metadata: The ego metadata.
    :return: The center pose in the global frame.
    """
    return rel_to_abs_se3(origin=imu_se3, pose_se3=metadata.center_to_imu_se3)


def center_se3_to_imu_se3(center_se3: PoseSE3, metadata: EgoStateSE3Metadata) -> PoseSE3:
    """Converts a vehicle center world pose to an IMU world pose in SE3.

    :param center_se3: The center pose in the global frame.
    :param metadata: The ego metadata.
    :return: The IMU pose in the global frame.
    """
    imu_in_center = abs_to_rel_se3(metadata.center_to_imu_se3, PoseSE3.identity())
    return rel_to_abs_se3(origin=center_se3, pose_se3=imu_in_center)


def imu_se2_to_center_se2(imu_se2: PoseSE2, metadata: EgoStateSE3Metadata) -> PoseSE2:
    """Converts an IMU world pose to a vehicle center world pose in SE2.

    :param imu_se2: The IMU pose in the global frame (SE2).
    :param metadata: The ego metadata.
    :return: The center pose in the global frame (SE2).
    """
    return rel_to_abs_se2(origin=imu_se2, pose_se2=metadata.center_to_imu_se3.pose_se2)


def center_se2_to_imu_se2(center_se2: PoseSE2, metadata: EgoStateSE3Metadata) -> PoseSE2:
    """Converts a vehicle center world pose to an IMU world pose in SE2.

    :param center_se2: The center pose in the global frame (SE2).
    :param metadata: The ego metadata.
    :return: The IMU pose in the global frame (SE2).
    """
    imu_in_center = abs_to_rel_se2(metadata.center_to_imu_se3.pose_se2, PoseSE2.identity())
    return rel_to_abs_se2(origin=center_se2, pose_se2=imu_in_center)
