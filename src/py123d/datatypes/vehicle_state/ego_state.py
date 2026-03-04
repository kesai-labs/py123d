from __future__ import annotations

from typing import Final, Optional

from py123d.datatypes.detections.box_detection_label import DefaultBoxDetectionLabel
from py123d.datatypes.detections.box_detections import BoxDetectionAttributes, BoxDetectionSE2, BoxDetectionSE3
from py123d.datatypes.time.timestamp import Timestamp
from py123d.datatypes.vehicle_state.dynamic_state import DynamicStateSE2, DynamicStateSE3
from py123d.datatypes.vehicle_state.ego_metadata import (
    EgoMetadata,
    center_se2_to_imu_se2,
    center_se3_to_imu_se3,
    imu_se2_to_center_se2,
    imu_se2_to_rear_axle_se2,
    imu_se3_to_center_se3,
    imu_se3_to_rear_axle_se3,
    rear_axle_se2_to_imu_se2,
    rear_axle_se3_to_imu_se3,
)
from py123d.geometry import BoundingBoxSE2, BoundingBoxSE3, PoseSE2, PoseSE3

EGO_TRACK_TOKEN: Final[str] = "ego_vehicle"


class EgoStateSE3:
    """The EgoStateSE3 represents the state of the ego vehicle in SE3 (3D space).

    The IMU pose is the primary internal representation. Rear axle, center,
    and SE2 poses are computed on demand via the vehicle parameters.
    """

    __slots__ = (
        "_imu_se3",
        "_vehicle_parameters",
        "_timestamp",
        "_dynamic_state_se3",
        "_tire_steering_angle",
    )

    _imu_se3: PoseSE3
    _vehicle_parameters: EgoMetadata
    _timestamp: Timestamp
    _dynamic_state_se3: Optional[DynamicStateSE3]
    _tire_steering_angle: Optional[float]

    @classmethod
    def from_imu(
        cls,
        imu_se3: PoseSE3,
        vehicle_parameters: EgoMetadata,
        timestamp: Timestamp,
        dynamic_state_se3: Optional[DynamicStateSE3] = None,
        tire_steering_angle: float = 0.0,
    ) -> EgoStateSE3:
        """Create an :class:`EgoStateSE3` from the IMU pose.

        This is the canonical factory method. The IMU pose is stored directly.

        :param imu_se3: The pose of the IMU in the global frame.
        :param vehicle_parameters: The parameters of the vehicle.
        :param timestamp: The timestamp of the state.
        :param dynamic_state_se3: The dynamic state of the vehicle, defaults to None.
        :param tire_steering_angle: The tire steering angle, defaults to 0.0.
        :return: An :class:`EgoStateSE3` instance.
        """
        instance = object.__new__(cls)
        instance._imu_se3 = imu_se3
        instance._vehicle_parameters = vehicle_parameters
        instance._timestamp = timestamp
        instance._dynamic_state_se3 = dynamic_state_se3
        instance._tire_steering_angle = tire_steering_angle
        return instance

    @classmethod
    def from_rear_axle(
        cls,
        rear_axle_se3: PoseSE3,
        vehicle_parameters: EgoMetadata,
        timestamp: Timestamp,
        dynamic_state_se3: Optional[DynamicStateSE3] = None,
        tire_steering_angle: float = 0.0,
    ) -> EgoStateSE3:
        """Create an :class:`EgoStateSE3` from the rear axle pose.

        Converts the rear axle pose to the IMU pose using the vehicle's
        ``rear_axle_to_imu_se3`` extrinsic calibration.

        :param rear_axle_se3: The pose of the rear axle in SE3.
        :param vehicle_parameters: The parameters of the vehicle.
        :param timestamp: The timestamp of the state
        :param dynamic_state_se3: The dynamic state of the vehicle, defaults to None.
        :param tire_steering_angle: The tire steering angle, defaults to 0.0.
        :return: An :class:`EgoStateSE3` instance.
        """
        imu_se3 = rear_axle_se3_to_imu_se3(
            rear_axle_se3=rear_axle_se3,
            vehicle_parameters=vehicle_parameters,
        )
        return EgoStateSE3.from_imu(
            imu_se3=imu_se3,
            vehicle_parameters=vehicle_parameters,
            timestamp=timestamp,
            dynamic_state_se3=dynamic_state_se3,
            tire_steering_angle=tire_steering_angle,
        )

    @classmethod
    def from_center(
        cls,
        center_se3: PoseSE3,
        vehicle_parameters: EgoMetadata,
        timestamp: Timestamp,
        dynamic_state_se3: Optional[DynamicStateSE3] = None,
        tire_steering_angle: float = 0.0,
    ) -> EgoStateSE3:
        """Create an :class:`EgoStateSE3` from the center pose.

        Converts the center pose to the IMU pose using the vehicle's
        ``center_to_imu_se3`` extrinsic calibration.

        :param center_se3: The center pose in SE3.
        :param vehicle_parameters: The parameters of the vehicle.
        :param timestamp: The timestamp of the state.
        :param dynamic_state_se3: The dynamic state of the vehicle, defaults to None.
        :param tire_steering_angle: The tire steering angle, defaults to 0.0.
        :return: An :class:`EgoStateSE3` instance.
        """
        imu_se3 = center_se3_to_imu_se3(
            center_se3=center_se3,
            vehicle_parameters=vehicle_parameters,
        )
        return EgoStateSE3.from_imu(
            imu_se3=imu_se3,
            vehicle_parameters=vehicle_parameters,
            timestamp=timestamp,
            dynamic_state_se3=dynamic_state_se3,
            tire_steering_angle=tire_steering_angle,
        )

    @property
    def imu_se3(self) -> PoseSE3:
        """The :class:`~py123d.geometry.PoseSE3` of the IMU in SE3."""
        return self._imu_se3

    @property
    def imu_se2(self) -> PoseSE2:
        """The :class:`~py123d.geometry.PoseSE2` of the IMU in SE2."""
        return self._imu_se3.pose_se2

    @property
    def rear_axle_se3(self) -> PoseSE3:
        """The :class:`~py123d.geometry.PoseSE3` of the rear axle in SE3."""
        return imu_se3_to_rear_axle_se3(imu_se3=self._imu_se3, vehicle_parameters=self._vehicle_parameters)

    @property
    def rear_axle_se2(self) -> PoseSE2:
        """The :class:`~py123d.geometry.PoseSE2` of the rear axle in SE2."""
        return self.rear_axle_se3.pose_se2

    @property
    def vehicle_parameters(self) -> EgoMetadata:
        """The :class:`~py123d.datatypes.vehicle_state.EgoMetadata` of the vehicle."""
        return self._vehicle_parameters

    @property
    def dynamic_state_se3(self) -> Optional[DynamicStateSE3]:
        """The :class:`~py123d.datatypes.vehicle_state.DynamicStateSE3` of the vehicle."""
        return self._dynamic_state_se3

    @property
    def timestamp(self) -> Timestamp:
        """The :class:`~py123d.datatypes.time.Timestamp` of the ego state."""
        return self._timestamp

    @property
    def tire_steering_angle(self) -> Optional[float]:
        """The tire steering angle of the ego state, if available."""
        return self._tire_steering_angle

    @property
    def center_se3(self) -> PoseSE3:
        """The :class:`~py123d.geometry.PoseSE3` of the vehicle center in SE3."""
        return imu_se3_to_center_se3(imu_se3=self._imu_se3, vehicle_parameters=self._vehicle_parameters)

    @property
    def center_se2(self) -> PoseSE2:
        """The :class:`~py123d.geometry.PoseSE2` of the vehicle center in SE2."""
        return self.center_se3.pose_se2

    @property
    def bounding_box_se3(self) -> BoundingBoxSE3:
        """The :class:`~py123d.geometry.BoundingBoxSE3` of the ego vehicle."""
        return BoundingBoxSE3(
            center_se3=self.center_se3,
            length=self.vehicle_parameters.length,
            width=self.vehicle_parameters.width,
            height=self.vehicle_parameters.height,
        )

    @property
    def bounding_box_se2(self) -> BoundingBoxSE2:
        """The :class:`~py123d.geometry.BoundingBoxSE2` of the ego vehicle."""
        return self.bounding_box_se3.bounding_box_se2

    @property
    def box_detection_se3(self) -> BoxDetectionSE3:
        """The :class:`~py123d.datatypes.detections.BoxDetectionSE3` projection of the ego vehicle."""
        return BoxDetectionSE3(
            metadata=BoxDetectionAttributes(
                label=DefaultBoxDetectionLabel.EGO,
                track_token=EGO_TRACK_TOKEN,
            ),
            bounding_box_se3=self.bounding_box_se3,
            velocity_3d=self.dynamic_state_se3.velocity_3d if self.dynamic_state_se3 else None,
        )

    @property
    def box_detection_se2(self) -> BoxDetectionSE2:
        """The :class:`~py123d.datatypes.detections.BoxDetectionSE2` projection of the ego vehicle."""
        return self.box_detection_se3.box_detection_se2

    @property
    def ego_state_se2(self) -> EgoStateSE2:
        """The :class:`EgoStateSE2` projection of this SE3 ego state."""
        return EgoStateSE2.from_imu(
            imu_se2=self.imu_se2,
            vehicle_parameters=self.vehicle_parameters,
            dynamic_state_se2=self.dynamic_state_se3.dynamic_state_se2 if self.dynamic_state_se3 else None,
            timestamp=self.timestamp,
            tire_steering_angle=self.tire_steering_angle if self.tire_steering_angle is not None else 0.0,
        )


class EgoStateSE2:
    """The EgoStateSE2 represents the state of the ego vehicle in SE2 (2D space).

    The IMU pose is the primary internal representation. Rear axle and center
    poses are computed on demand via the vehicle parameters.
    """

    __slots__ = ("_imu_se2", "_vehicle_parameters", "_timestamp", "_dynamic_state_se2", "_tire_steering_angle")

    _imu_se2: PoseSE2
    _vehicle_parameters: EgoMetadata
    _timestamp: Timestamp
    _dynamic_state_se2: Optional[DynamicStateSE2]
    _tire_steering_angle: Optional[float]

    @classmethod
    def from_imu(
        cls,
        imu_se2: PoseSE2,
        vehicle_parameters: EgoMetadata,
        timestamp: Timestamp,
        dynamic_state_se2: Optional[DynamicStateSE2] = None,
        tire_steering_angle: float = 0.0,
    ) -> EgoStateSE2:
        """Create an :class:`EgoStateSE2` from the IMU pose.

        This is the canonical factory method. The IMU pose is stored directly.

        :param imu_se2: The pose of the IMU in the global frame (SE2).
        :param vehicle_parameters: The parameters of the vehicle.
        :param timestamp: The timestamp of the state.
        :param dynamic_state_se2: The dynamic state of the vehicle in SE2, defaults to None.
        :param tire_steering_angle: The tire steering angle, defaults to 0.0.
        :return: An instance of :class:`EgoStateSE2`.
        """
        instance = object.__new__(cls)
        instance._imu_se2 = imu_se2
        instance._vehicle_parameters = vehicle_parameters
        instance._dynamic_state_se2 = dynamic_state_se2
        instance._timestamp = timestamp
        instance._tire_steering_angle = tire_steering_angle
        return instance

    @classmethod
    def from_rear_axle(
        cls,
        rear_axle_se2: PoseSE2,
        vehicle_parameters: EgoMetadata,
        timestamp: Timestamp,
        dynamic_state_se2: Optional[DynamicStateSE2] = None,
        tire_steering_angle: float = 0.0,
    ) -> EgoStateSE2:
        """Create an :class:`EgoStateSE2` from the rear axle pose.

        Converts the rear axle pose to the IMU pose using the vehicle's
        ``rear_axle_to_imu_se3`` extrinsic calibration.

        :param rear_axle_se2: The pose of the rear axle in SE2.
        :param vehicle_parameters: The parameters of the vehicle.
        :param timestamp: The timestamp of the state.
        :param dynamic_state_se2: The dynamic state of the vehicle in SE2, defaults to None.
        :param tire_steering_angle: The tire steering angle, defaults to 0.0.
        :return: An instance of :class:`EgoStateSE2`.
        """
        imu_se2 = rear_axle_se2_to_imu_se2(
            rear_axle_se2=rear_axle_se2,
            vehicle_parameters=vehicle_parameters,
        )
        return EgoStateSE2.from_imu(
            imu_se2=imu_se2,
            vehicle_parameters=vehicle_parameters,
            dynamic_state_se2=dynamic_state_se2,
            timestamp=timestamp,
            tire_steering_angle=tire_steering_angle,
        )

    @classmethod
    def from_center(
        cls,
        center_se2: PoseSE2,
        vehicle_parameters: EgoMetadata,
        timestamp: Timestamp,
        dynamic_state_se2: Optional[DynamicStateSE2] = None,
        tire_steering_angle: float = 0.0,
    ) -> EgoStateSE2:
        """Create an :class:`EgoStateSE2` from the center pose.

        Converts the center pose to the IMU pose using the vehicle's
        ``center_to_imu_se3`` extrinsic calibration.

        :param center_se2: The pose of the center in SE2.
        :param vehicle_parameters: The parameters of the vehicle.
        :param timestamp: The timestamp of the state.
        :param dynamic_state_se2: The dynamic state of the vehicle in SE2, defaults to None.
        :param tire_steering_angle: The tire steering angle, defaults to 0.0.
        :return: An instance of :class:`EgoStateSE2`.
        """
        imu_se2 = center_se2_to_imu_se2(
            center_se2=center_se2,
            vehicle_parameters=vehicle_parameters,
        )
        return EgoStateSE2.from_imu(
            imu_se2=imu_se2,
            vehicle_parameters=vehicle_parameters,
            timestamp=timestamp,
            dynamic_state_se2=dynamic_state_se2,
            tire_steering_angle=tire_steering_angle,
        )

    @property
    def imu_se2(self) -> PoseSE2:
        """The :class:`~py123d.geometry.PoseSE2` of the IMU in SE2."""
        return self._imu_se2

    @property
    def rear_axle_se2(self) -> PoseSE2:
        """The :class:`~py123d.geometry.PoseSE2` of the rear axle in SE2."""
        return imu_se2_to_rear_axle_se2(imu_se2=self._imu_se2, vehicle_parameters=self._vehicle_parameters)

    @property
    def vehicle_parameters(self) -> EgoMetadata:
        """The :class:`~py123d.datatypes.vehicle_state.EgoMetadata` of the vehicle."""
        return self._vehicle_parameters

    @property
    def timestamp(self) -> Timestamp:
        """The :class:`~py123d.datatypes.time.Timestamp` of the ego state"""
        return self._timestamp

    @property
    def dynamic_state_se2(self) -> Optional[DynamicStateSE2]:
        """The :class:`~py123d.datatypes.vehicle_state.DynamicStateSE2` of the vehicle."""
        return self._dynamic_state_se2

    @property
    def tire_steering_angle(self) -> Optional[float]:
        """The tire steering angle of the ego state, if available."""
        return self._tire_steering_angle

    @property
    def center_se2(self) -> PoseSE2:
        """The :class:`~py123d.geometry.PoseSE2` of the center in SE2."""
        return imu_se2_to_center_se2(imu_se2=self._imu_se2, vehicle_parameters=self._vehicle_parameters)

    @property
    def bounding_box_se2(self) -> BoundingBoxSE2:
        """The :class:`~py123d.geometry.BoundingBoxSE2` of the ego vehicle."""
        return BoundingBoxSE2(
            center_se2=self.center_se2,
            length=self.vehicle_parameters.length,
            width=self.vehicle_parameters.width,
        )

    @property
    def box_detection_se2(self) -> BoxDetectionSE2:
        """The :class:`~py123d.datatypes.detections.BoxDetectionSE2` projection of the ego vehicle."""
        return BoxDetectionSE2(
            metadata=BoxDetectionAttributes(
                label=DefaultBoxDetectionLabel.EGO,
                track_token=EGO_TRACK_TOKEN,
            ),
            bounding_box_se2=self.bounding_box_se2,
            velocity_2d=self.dynamic_state_se2.velocity_2d if self.dynamic_state_se2 else None,
        )
