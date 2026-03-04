from __future__ import annotations

from py123d.datatypes.metadata.abstract_metadata import AbstractMetadata
from py123d.geometry import PoseSE2, PoseSE3
from py123d.geometry.transform import abs_to_rel_se2, abs_to_rel_se3, rel_to_abs_se2, rel_to_abs_se3


class EgoMetadata(AbstractMetadata):
    """Metadata that describes the physical dimensions of the ego vehicle.

    The vehicle geometry is defined through two SE3 extrinsic transforms that relate
    the vehicle's center and rear axle frames to the IMU frame:

    - ``center_to_imu_se3``: maps coordinates from the vehicle center frame to the IMU frame.
    - ``rear_axle_to_imu_se3``: maps coordinates from the rear axle frame to the IMU frame.

    For most datasets the IMU is co-located with the rear axle, so
    ``rear_axle_to_imu_se3`` is the identity. KITTI-360 is the notable exception
    where the IMU has a lateral offset from the rear axle.
    """

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
    def from_dict(cls, data_dict: dict) -> EgoMetadata:
        """Creates a EgoMetadata instance from a dictionary.

        :param data_dict: Dictionary containing vehicle parameters.
        :return: EgoMetadata instance.
        """
        data_dict = dict(data_dict)
        data_dict["center_to_imu_se3"] = PoseSE3.from_list(data_dict["center_to_imu_se3"])
        data_dict["rear_axle_to_imu_se3"] = PoseSE3.from_list(data_dict["rear_axle_to_imu_se3"])
        return EgoMetadata(**data_dict)

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

    def to_dict(self) -> dict:
        """Converts the :class:`EgoMetadata` instance to a dictionary.

        :return: Dictionary representation of the vehicle parameters.
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


def get_nuplan_chrysler_pacifica_parameters() -> EgoMetadata:
    """Helper function to get nuPlan Chrysler Pacifica vehicle parameters."""
    # NOTE: These parameters are mostly available in nuPlan, except for the rear_axle_to_center_vertical.
    # The value is estimated based the Lidar point cloud.
    # [1] https://en.wikipedia.org/wiki/Chrysler_Pacifica_(minivan)

    return EgoMetadata(
        vehicle_name="nuplan_chrysler_pacifica",
        width=2.297,
        length=5.176,
        height=1.777,
        wheel_base=3.089,
        center_to_imu_se3=PoseSE3(x=1.461, y=0.0, z=0.45, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
        rear_axle_to_imu_se3=PoseSE3.identity(),
    )


def get_nuscenes_renault_zoe_parameters() -> EgoMetadata:
    """Helper function to get nuScenes Renault Zoe vehicle parameters."""
    # NOTE: The parameters in nuScenes are estimates, and partially taken from the Renault Zoe model [1].
    # [1] https://en.wikipedia.org/wiki/Renault_Zoe
    return EgoMetadata(
        vehicle_name="nuscenes_renault_zoe",
        width=1.730,
        length=4.084,
        height=1.562,
        wheel_base=2.588,
        center_to_imu_se3=PoseSE3(x=1.385, y=0.0, z=1.562 / 2, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
        rear_axle_to_imu_se3=PoseSE3.identity(),
    )


def get_carla_lincoln_mkz_2020_parameters() -> EgoMetadata:
    """Helper function to get CARLA Lincoln MKZ 2020 vehicle parameters."""
    # NOTE: These parameters are taken from the CARLA simulator vehicle model. The rear axles to center transform
    # parameters are calculated based on parameters from the CARLA simulator.
    return EgoMetadata(
        vehicle_name="carla_lincoln_mkz_2020",
        width=1.83671,
        length=4.89238,
        height=1.49028,
        wheel_base=2.86048,
        center_to_imu_se3=PoseSE3(x=1.64855, y=0.0, z=0.38579, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
        rear_axle_to_imu_se3=PoseSE3.identity(),
    )


def get_wod_perception_chrysler_pacifica_parameters() -> EgoMetadata:
    """Helper function to get Waymo Open Dataset Perception Chrysler Pacifica vehicle parameters."""
    # NOTE: These parameters are estimates based on the vehicle model used in the WOD Perception dataset.
    # The vehicle should be the same (or a similar) vehicle model to nuPlan and PandaSet [1].
    # [1] https://en.wikipedia.org/wiki/Chrysler_Pacifica_(minivan)
    return EgoMetadata(
        vehicle_name="wod-perception_chrysler_pacifica",
        width=2.297,
        length=5.176,
        height=1.777,
        wheel_base=3.089,
        center_to_imu_se3=PoseSE3(x=1.461, y=0.0, z=1.777 / 2, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
        rear_axle_to_imu_se3=PoseSE3.identity(),
    )


def get_wod_motion_chrysler_pacifica_parameters() -> EgoMetadata:
    """Helper function to get Waymo Open Dataset Motion Chrysler Pacifica vehicle parameters."""
    return EgoMetadata(
        vehicle_name="wod-motion_chrysler_pacifica",
        width=2.3320000171661377,
        length=5.285999774932861,
        height=2.3299999237060547,
        wheel_base=3.089,
        center_to_imu_se3=PoseSE3(x=1.461, y=0.0, z=2.3299999237060547 / 2, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
        rear_axle_to_imu_se3=PoseSE3.identity(),
    )


def get_kitti360_vw_passat_parameters() -> EgoMetadata:
    """Helper function to get KITTI-360 VW Passat vehicle parameters."""
    # NOTE: The parameters in KITTI-360 are estimates based on the vehicle model used in the dataset
    # Uses a 2006 VW Passat Variant B6 [1]. Vertical distance is estimated based on the Lidar.
    # KITTI-360 is currently the only dataset where the IMU has a lateral offset to the rear axle [2].
    # The rear axle is at (0.05, -0.32, 0.0) from the IMU in the body frame.
    # [1] https://en.wikipedia.org/wiki/Volkswagen_Passat_(B6)
    # [2] https://www.cvlibs.net/datasets/kitti-360/documentation.php

    rear_axle_to_center_longitudinal = 1.3369
    rear_axle_to_center_vertical = 1.516 / 2 - 0.9
    # Displacement from IMU to rear axle in the body frame
    rear_axle_in_imu_x = 0.05
    rear_axle_in_imu_y = -0.32

    return EgoMetadata(
        vehicle_name="kitti360_vw_passat",
        width=1.820,
        length=4.775,
        height=1.516,
        wheel_base=2.709,
        center_to_imu_se3=PoseSE3(
            x=rear_axle_to_center_longitudinal + rear_axle_in_imu_x,
            y=rear_axle_in_imu_y,
            z=rear_axle_to_center_vertical,
            qw=1.0,
            qx=0.0,
            qy=0.0,
            qz=0.0,
        ),
        rear_axle_to_imu_se3=PoseSE3(
            x=rear_axle_in_imu_x,
            y=rear_axle_in_imu_y,
            z=0.0,
            qw=1.0,
            qx=0.0,
            qy=0.0,
            qz=0.0,
        ),
    )


def get_av2_ford_fusion_hybrid_parameters() -> EgoMetadata:
    """Helper function to get Argoverse 2 Ford Fusion Hybrid vehicle parameters."""
    # NOTE: Parameters are estimated from the vehicle model [1] and Lidar point cloud.
    # [1] https://en.wikipedia.org/wiki/Ford_Fusion_Hybrid#Second_generation
    # https://github.com/argoverse/av2-api/blob/6b22766247eda941cb1953d6a58e8d5631c561da/tests/unit/map/test_map_api.py#L375
    return EgoMetadata(
        vehicle_name="av2_ford_fusion_hybrid",
        width=1.852 + 0.275,  # 0.275 is the estimated width of the side mirrors
        length=4.869,
        height=1.476,
        wheel_base=2.850,
        center_to_imu_se3=PoseSE3(x=1.339, y=0.0, z=0.438, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
        rear_axle_to_imu_se3=PoseSE3.identity(),
    )


def get_pandaset_chrysler_pacifica_parameters() -> EgoMetadata:
    """Helper function to get PandaSet Chrysler Pacifica vehicle parameters."""
    # NOTE: Some parameters are available in PandaSet [1], others are estimated based on the vehicle model [2].
    # [1] https://arxiv.org/pdf/2112.12610 (Figure 3 (a))
    # [2] https://en.wikipedia.org/wiki/Chrysler_Pacifica_(minivan)
    return EgoMetadata(
        vehicle_name="pandaset_chrysler_pacifica",
        width=2.297,
        length=5.176,
        height=1.777,
        wheel_base=3.089,
        center_to_imu_se3=PoseSE3(x=1.461, y=0.0, z=0.45, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
        rear_axle_to_imu_se3=PoseSE3.identity(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# IMU <-> Rear Axle conversions (SE3 / SE2)
# ──────────────────────────────────────────────────────────────────────────────


def imu_se3_to_rear_axle_se3(imu_se3: PoseSE3, vehicle_parameters: EgoMetadata) -> PoseSE3:
    """Converts an IMU world pose to a rear axle world pose in SE3.

    :param imu_se3: The IMU pose in the global frame.
    :param vehicle_parameters: The vehicle parameters.
    :return: The rear axle pose in the global frame.
    """
    return rel_to_abs_se3(origin=imu_se3, pose_se3=vehicle_parameters.rear_axle_to_imu_se3)


def rear_axle_se3_to_imu_se3(rear_axle_se3: PoseSE3, vehicle_parameters: EgoMetadata) -> PoseSE3:
    """Converts a rear axle world pose to an IMU world pose in SE3.

    :param rear_axle_se3: The rear axle pose in the global frame.
    :param vehicle_parameters: The vehicle parameters.
    :return: The IMU pose in the global frame.
    """
    imu_in_rear_axle = abs_to_rel_se3(vehicle_parameters.rear_axle_to_imu_se3, PoseSE3.identity())
    return rel_to_abs_se3(origin=rear_axle_se3, pose_se3=imu_in_rear_axle)


def imu_se2_to_rear_axle_se2(imu_se2: PoseSE2, vehicle_parameters: EgoMetadata) -> PoseSE2:
    """Converts an IMU world pose to a rear axle world pose in SE2.

    :param imu_se2: The IMU pose in the global frame (SE2).
    :param vehicle_parameters: The vehicle parameters.
    :return: The rear axle pose in the global frame (SE2).
    """
    return rel_to_abs_se2(origin=imu_se2, pose_se2=vehicle_parameters.rear_axle_to_imu_se3.pose_se2)


def rear_axle_se2_to_imu_se2(rear_axle_se2: PoseSE2, vehicle_parameters: EgoMetadata) -> PoseSE2:
    """Converts a rear axle world pose to an IMU world pose in SE2.

    :param rear_axle_se2: The rear axle pose in the global frame (SE2).
    :param vehicle_parameters: The vehicle parameters.
    :return: The IMU pose in the global frame (SE2).
    """
    imu_in_rear_axle = abs_to_rel_se2(vehicle_parameters.rear_axle_to_imu_se3.pose_se2, PoseSE2.identity())
    return rel_to_abs_se2(origin=rear_axle_se2, pose_se2=imu_in_rear_axle)


# ──────────────────────────────────────────────────────────────────────────────
# IMU <-> Center conversions (SE3 / SE2)
# ──────────────────────────────────────────────────────────────────────────────


def imu_se3_to_center_se3(imu_se3: PoseSE3, vehicle_parameters: EgoMetadata) -> PoseSE3:
    """Converts an IMU world pose to a vehicle center world pose in SE3.

    :param imu_se3: The IMU pose in the global frame.
    :param vehicle_parameters: The vehicle parameters.
    :return: The center pose in the global frame.
    """
    return rel_to_abs_se3(origin=imu_se3, pose_se3=vehicle_parameters.center_to_imu_se3)


def center_se3_to_imu_se3(center_se3: PoseSE3, vehicle_parameters: EgoMetadata) -> PoseSE3:
    """Converts a vehicle center world pose to an IMU world pose in SE3.

    :param center_se3: The center pose in the global frame.
    :param vehicle_parameters: The vehicle parameters.
    :return: The IMU pose in the global frame.
    """
    imu_in_center = abs_to_rel_se3(vehicle_parameters.center_to_imu_se3, PoseSE3.identity())
    return rel_to_abs_se3(origin=center_se3, pose_se3=imu_in_center)


def imu_se2_to_center_se2(imu_se2: PoseSE2, vehicle_parameters: EgoMetadata) -> PoseSE2:
    """Converts an IMU world pose to a vehicle center world pose in SE2.

    :param imu_se2: The IMU pose in the global frame (SE2).
    :param vehicle_parameters: The vehicle parameters.
    :return: The center pose in the global frame (SE2).
    """
    return rel_to_abs_se2(origin=imu_se2, pose_se2=vehicle_parameters.center_to_imu_se3.pose_se2)


def center_se2_to_imu_se2(center_se2: PoseSE2, vehicle_parameters: EgoMetadata) -> PoseSE2:
    """Converts a vehicle center world pose to an IMU world pose in SE2.

    :param center_se2: The center pose in the global frame (SE2).
    :param vehicle_parameters: The vehicle parameters.
    :return: The IMU pose in the global frame (SE2).
    """
    imu_in_center = abs_to_rel_se2(vehicle_parameters.center_to_imu_se3.pose_se2, PoseSE2.identity())
    return rel_to_abs_se2(origin=center_se2, pose_se2=imu_in_center)
