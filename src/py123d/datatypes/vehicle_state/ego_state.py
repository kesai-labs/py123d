from __future__ import annotations

from typing import Final, Optional

from py123d.datatypes.detections.box_detection_label import DefaultBoxDetectionLabel
from py123d.datatypes.detections.box_detections import BoxDetectionAttributes, BoxDetectionSE2, BoxDetectionSE3
from py123d.datatypes.modalities.base_modality import BaseModality
from py123d.datatypes.time.timestamp import Timestamp
from py123d.datatypes.vehicle_state.dynamic_state import DynamicStateSE2, DynamicStateSE3
from py123d.datatypes.vehicle_state.ego_state_metadata import (
    EgoStateSE3Metadata,
    center_se2_to_imu_se2,
    center_se3_to_imu_se3,
    imu_se2_to_center_se2,
    imu_se2_to_rear_axle_se2,
    imu_se3_to_center_se3,
    imu_se3_to_rear_axle_se3,
    rear_axle_se2_to_imu_se2,
    rear_axle_se3_to_imu_se3,
)
from py123d.geometry import BoundingBoxSE2, BoundingBoxSE3, PoseSE2, PoseSE3, Vector2D, Vector3D

EGO_TRACK_TOKEN: Final[str] = "ego_vehicle"


class EgoStateSE3(BaseModality):
    """The EgoStateSE3 represents the state of the ego vehicle in SE3 (3D space).

    The IMU pose is the primary internal representation. Rear axle, center,
    and SE2 poses are computed on demand via the ego metadata.
    """

    __slots__ = (
        "_imu_se3",
        "_metadata",
        "_timestamp",
        "_dynamic_state_se3",
        "_tire_steering_angle",
    )

    _imu_se3: PoseSE3
    _metadata: EgoStateSE3Metadata
    _timestamp: Timestamp
    _dynamic_state_se3: Optional[DynamicStateSE3]
    _tire_steering_angle: Optional[float]

    @classmethod
    def from_imu(
        cls,
        imu_se3: PoseSE3,
        metadata: EgoStateSE3Metadata,
        timestamp: Timestamp,
        dynamic_state_se3: Optional[DynamicStateSE3] = None,
        tire_steering_angle: float = 0.0,
    ) -> EgoStateSE3:
        """Create an :class:`EgoStateSE3` from the IMU pose.

        This is the canonical factory method. The IMU pose is stored directly.

        :param imu_se3: The pose of the IMU in the global frame.
        :param metadata: The ego metadata of the vehicle.
        :param timestamp: The timestamp of the state.
        :param dynamic_state_se3: The dynamic state of the vehicle, defaults to None.
        :param tire_steering_angle: The tire steering angle, defaults to 0.0.
        :return: An :class:`EgoStateSE3` instance.
        """
        instance = object.__new__(cls)
        instance._imu_se3 = imu_se3
        instance._metadata = metadata
        instance._timestamp = timestamp
        instance._dynamic_state_se3 = dynamic_state_se3
        instance._tire_steering_angle = tire_steering_angle
        return instance

    @classmethod
    def from_rear_axle(
        cls,
        rear_axle_se3: PoseSE3,
        metadata: EgoStateSE3Metadata,
        timestamp: Timestamp,
        dynamic_state_se3: Optional[DynamicStateSE3] = None,
        tire_steering_angle: float = 0.0,
    ) -> EgoStateSE3:
        """Create an :class:`EgoStateSE3` from the rear axle pose.

        Converts the rear axle pose to the IMU pose using the vehicle's
        ``rear_axle_to_imu_se3`` extrinsic calibration.

        :param rear_axle_se3: The pose of the rear axle in SE3.
        :param metadata: The ego metadata of the vehicle.
        :param timestamp: The timestamp of the state
        :param dynamic_state_se3: The dynamic state of the vehicle in ego frame, defaults to None.
        :param tire_steering_angle: The tire steering angle, defaults to 0.0.
        :return: An :class:`EgoStateSE3` instance.
        """
        imu_se3 = rear_axle_se3_to_imu_se3(
            rear_axle_se3=rear_axle_se3,
            metadata=metadata,
        )
        return EgoStateSE3.from_imu(
            imu_se3=imu_se3,
            metadata=metadata,
            timestamp=timestamp,
            dynamic_state_se3=dynamic_state_se3,
            tire_steering_angle=tire_steering_angle,
        )

    @classmethod
    def from_center(
        cls,
        center_se3: PoseSE3,
        metadata: EgoStateSE3Metadata,
        timestamp: Timestamp,
        dynamic_state_se3: Optional[DynamicStateSE3] = None,
        tire_steering_angle: float = 0.0,
    ) -> EgoStateSE3:
        """Create an :class:`EgoStateSE3` from the center pose.

        Converts the center pose to the IMU pose using the vehicle's
        ``center_to_imu_se3`` extrinsic calibration.

        :param center_se3: The center pose in SE3.
        :param metadata: The ego metadata of the vehicle.
        :param timestamp: The timestamp of the state.
        :param dynamic_state_se3: The dynamic state of the vehicle in ego frame, defaults to None.
        :param tire_steering_angle: The tire steering angle, defaults to 0.0.
        :return: An :class:`EgoStateSE3` instance.
        """
        imu_se3 = center_se3_to_imu_se3(
            center_se3=center_se3,
            metadata=metadata,
        )
        return EgoStateSE3.from_imu(
            imu_se3=imu_se3,
            metadata=metadata,
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
        return imu_se3_to_rear_axle_se3(imu_se3=self._imu_se3, metadata=self._metadata)

    @property
    def rear_axle_se2(self) -> PoseSE2:
        """The :class:`~py123d.geometry.PoseSE2` of the rear axle in SE2."""
        return self.rear_axle_se3.pose_se2

    @property
    def metadata(self) -> EgoStateSE3Metadata:
        """The :class:`~py123d.datatypes.vehicle_state.EgoStateSE3Metadata` of the vehicle."""
        return self._metadata

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
        return imu_se3_to_center_se3(imu_se3=self._imu_se3, metadata=self._metadata)

    @property
    def center_se2(self) -> PoseSE2:
        """The :class:`~py123d.geometry.PoseSE2` of the vehicle center in SE2."""
        return self.center_se3.pose_se2

    @property
    def bounding_box_se3(self) -> BoundingBoxSE3:
        """The :class:`~py123d.geometry.BoundingBoxSE3` of the ego vehicle."""
        return BoundingBoxSE3(
            center_se3=self.center_se3,
            length=self.metadata.length,
            width=self.metadata.width,
            height=self.metadata.height,
        )

    @property
    def bounding_box_se2(self) -> BoundingBoxSE2:
        """The :class:`~py123d.geometry.BoundingBoxSE2` of the ego vehicle."""
        return self.bounding_box_se3.bounding_box_se2

    @property
    def box_detection_se3(self) -> BoxDetectionSE3:
        """The :class:`~py123d.datatypes.detections.BoxDetectionSE3` projection of the ego vehicle."""

        # NOTE @DanielDauner: In contrast to box detections, the ego dynamic state is in ego frame,
        # thus we need to rotate the velocity vector to global frame.
        velocity_3d_global: Optional[Vector3D] = None
        if self.dynamic_state_se3 is not None:
            v_body = self.dynamic_state_se3.velocity_3d.array
            v_global = self.imu_se3.rotation_matrix @ v_body
            velocity_3d_global = Vector3D(float(v_global[0]), float(v_global[1]), float(v_global[2]))

        return BoxDetectionSE3(
            attributes=BoxDetectionAttributes(
                label=DefaultBoxDetectionLabel.EGO,
                track_token=EGO_TRACK_TOKEN,
            ),
            bounding_box_se3=self.bounding_box_se3,
            velocity_3d=velocity_3d_global,
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
            metadata=self.metadata,
            dynamic_state_se2=self.dynamic_state_se3.dynamic_state_se2 if self.dynamic_state_se3 else None,
            timestamp=self.timestamp,
            tire_steering_angle=self.tire_steering_angle if self.tire_steering_angle is not None else 0.0,
        )


class EgoStateSE2:
    """The EgoStateSE2 represents the state of the ego vehicle in SE2 (2D space).

    The IMU pose is the primary internal representation. Rear axle and center
    poses are computed on demand via the ego metadata.
    """

    __slots__ = ("_imu_se2", "_metadata", "_timestamp", "_dynamic_state_se2", "_tire_steering_angle")

    _imu_se2: PoseSE2
    _metadata: EgoStateSE3Metadata
    _timestamp: Timestamp
    _dynamic_state_se2: Optional[DynamicStateSE2]
    _tire_steering_angle: Optional[float]

    @classmethod
    def from_imu(
        cls,
        imu_se2: PoseSE2,
        metadata: EgoStateSE3Metadata,
        timestamp: Timestamp,
        dynamic_state_se2: Optional[DynamicStateSE2] = None,
        tire_steering_angle: float = 0.0,
    ) -> EgoStateSE2:
        """Create an :class:`EgoStateSE2` from the IMU pose.

        This is the canonical factory method. The IMU pose is stored directly.

        :param imu_se2: The pose of the IMU in the global frame (SE2).
        :param metadata: The ego metadata of the vehicle.
        :param timestamp: The timestamp of the state.
        :param dynamic_state_se2: The dynamic state of the vehicle in SE2, defaults to None.
        :param tire_steering_angle: The tire steering angle, defaults to 0.0.
        :return: An instance of :class:`EgoStateSE2`.
        """
        instance = object.__new__(cls)
        instance._imu_se2 = imu_se2
        instance._metadata = metadata
        instance._dynamic_state_se2 = dynamic_state_se2
        instance._timestamp = timestamp
        instance._tire_steering_angle = tire_steering_angle
        return instance

    @classmethod
    def from_rear_axle(
        cls,
        rear_axle_se2: PoseSE2,
        metadata: EgoStateSE3Metadata,
        timestamp: Timestamp,
        dynamic_state_se2: Optional[DynamicStateSE2] = None,
        tire_steering_angle: float = 0.0,
    ) -> EgoStateSE2:
        """Create an :class:`EgoStateSE2` from the rear axle pose.

        Converts the rear axle pose to the IMU pose using the vehicle's
        ``rear_axle_to_imu_se3`` extrinsic calibration.

        :param rear_axle_se2: The pose of the rear axle in SE2.
        :param metadata: The ego metadata of the vehicle.
        :param timestamp: The timestamp of the state.
        :param dynamic_state_se2: The dynamic state of the vehicle in SE2, defaults to None.
        :param tire_steering_angle: The tire steering angle, defaults to 0.0.
        :return: An instance of :class:`EgoStateSE2`.
        """
        imu_se2 = rear_axle_se2_to_imu_se2(
            rear_axle_se2=rear_axle_se2,
            metadata=metadata,
        )
        return EgoStateSE2.from_imu(
            imu_se2=imu_se2,
            metadata=metadata,
            dynamic_state_se2=dynamic_state_se2,
            timestamp=timestamp,
            tire_steering_angle=tire_steering_angle,
        )

    @classmethod
    def from_center(
        cls,
        center_se2: PoseSE2,
        metadata: EgoStateSE3Metadata,
        timestamp: Timestamp,
        dynamic_state_se2: Optional[DynamicStateSE2] = None,
        tire_steering_angle: float = 0.0,
    ) -> EgoStateSE2:
        """Create an :class:`EgoStateSE2` from the center pose.

        Converts the center pose to the IMU pose using the vehicle's
        ``center_to_imu_se3`` extrinsic calibration.

        :param center_se2: The pose of the center in SE2.
        :param metadata: The ego metadata of the vehicle.
        :param timestamp: The timestamp of the state.
        :param dynamic_state_se2: The dynamic state of the vehicle in SE2, defaults to None.
        :param tire_steering_angle: The tire steering angle, defaults to 0.0.
        :return: An instance of :class:`EgoStateSE2`.
        """
        imu_se2 = center_se2_to_imu_se2(
            center_se2=center_se2,
            metadata=metadata,
        )
        return EgoStateSE2.from_imu(
            imu_se2=imu_se2,
            metadata=metadata,
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
        return imu_se2_to_rear_axle_se2(imu_se2=self._imu_se2, metadata=self._metadata)

    @property
    def metadata(self) -> EgoStateSE3Metadata:
        """The :class:`~py123d.datatypes.vehicle_state.EgoStateSE3Metadata` of the vehicle."""
        return self._metadata

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
        return imu_se2_to_center_se2(imu_se2=self._imu_se2, metadata=self._metadata)

    @property
    def bounding_box_se2(self) -> BoundingBoxSE2:
        """The :class:`~py123d.geometry.BoundingBoxSE2` of the ego vehicle."""
        return BoundingBoxSE2(
            center_se2=self.center_se2,
            length=self.metadata.length,
            width=self.metadata.width,
        )

    @property
    def box_detection_se2(self) -> BoxDetectionSE2:
        """The :class:`~py123d.datatypes.detections.BoxDetectionSE2` projection of the ego vehicle."""

        # NOTE @DanielDauner: In contrast to box detections, the ego dynamic state is in ego frame,
        # thus we need to rotate the velocity vector to global frame.
        velocity_2d_global: Optional[Vector2D] = None
        if self.dynamic_state_se2 is not None:
            v_body = self.dynamic_state_se2.velocity_2d.array
            v_global = self.imu_se2.rotation_matrix @ v_body
            velocity_2d_global = Vector2D(float(v_global[0]), float(v_global[1]))

        return BoxDetectionSE2(
            attributes=BoxDetectionAttributes(
                label=DefaultBoxDetectionLabel.EGO,
                track_token=EGO_TRACK_TOKEN,
            ),
            bounding_box_se2=self.bounding_box_se2,
            velocity_2d=velocity_2d_global,
        )
