import numpy as np
import pytest

from py123d.datatypes import (
    DynamicStateSE2,
    DynamicStateSE3,
    EgoStateSE2,
    EgoStateSE3,
    EgoStateSE3Metadata,
    Timestamp,
)
from py123d.datatypes.detections.box_detection_label import DefaultBoxDetectionLabel
from py123d.datatypes.vehicle_state.ego_state import EGO_TRACK_TOKEN
from py123d.geometry import BoundingBoxSE2, PoseSE2, PoseSE3, Vector2D, Vector3D


class TestEgoStateSE2:
    def setup_method(self):
        """Set up test fixtures for EgoStateSE2 tests."""
        self.rear_axle_pose = PoseSE2(x=0.0, y=0.0, yaw=0.0)
        self.metadata = EgoStateSE3Metadata(
            vehicle_name="test_vehicle",
            length=4.5,
            width=2.0,
            height=1.5,
            wheel_base=2.7,
            center_to_imu_se3=PoseSE3(x=1.35, y=0.0, z=0.5, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
            rear_axle_to_imu_se3=PoseSE3.identity(),
        )
        self.dynamic_state = DynamicStateSE2(
            velocity=Vector2D(1.0, 0.0),
            acceleration=Vector2D(0.1, 0.0),
            angular_velocity=0.1,
        )
        self.timestamp = Timestamp.from_us(1000000)
        self.tire_steering_angle = 0.2

    def test_from_rear_axle(self):
        """Test creating EgoStateSE2 from rear axle."""
        ego_state = EgoStateSE2.from_rear_axle(
            rear_axle_se2=self.rear_axle_pose,
            dynamic_state_se2=self.dynamic_state,
            metadata=self.metadata,
            timestamp=self.timestamp,
            tire_steering_angle=self.tire_steering_angle,
        )

        assert ego_state.rear_axle_se2 == self.rear_axle_pose
        assert ego_state.metadata == self.metadata
        assert ego_state.dynamic_state_se2 == self.dynamic_state
        assert ego_state.timestamp == self.timestamp
        assert ego_state.tire_steering_angle == self.tire_steering_angle

    def test_from_center(self):
        """Test creating EgoStateSE2 from center pose."""
        center_pose = PoseSE2(x=1.35, y=0.0, yaw=0.0)
        ego_state = EgoStateSE2.from_center(
            center_se2=center_pose,
            dynamic_state_se2=self.dynamic_state,
            metadata=self.metadata,
            timestamp=self.timestamp,
            tire_steering_angle=self.tire_steering_angle,
        )

        assert ego_state.rear_axle_se2 is not None
        assert ego_state.metadata == self.metadata

    def test_rear_axle_property(self):
        """Test rear_axle property."""
        ego_state = EgoStateSE2.from_rear_axle(
            rear_axle_se2=self.rear_axle_pose,
            metadata=self.metadata,
            timestamp=self.timestamp,
        )
        assert ego_state.rear_axle_se2 == self.rear_axle_pose

    def test_center_property(self):
        """Test center property calculation."""
        ego_state = EgoStateSE2.from_rear_axle(
            rear_axle_se2=self.rear_axle_pose,
            metadata=self.metadata,
            timestamp=self.timestamp,
        )

        center = ego_state.center_se2
        assert center is not None
        assert center.x == pytest.approx(self.metadata.rear_axle_to_center_longitudinal)
        assert center.y == pytest.approx(0.0)
        assert center.yaw == pytest.approx(0.0)

    def test_bounding_box_property(self):
        """Test bounding box properties."""
        ego_state = EgoStateSE2.from_rear_axle(
            rear_axle_se2=self.rear_axle_pose,
            metadata=self.metadata,
            timestamp=self.timestamp,
        )

        bbox = ego_state.bounding_box_se2
        bbox_center = BoundingBoxSE2(ego_state.center_se2, self.metadata.length, self.metadata.width)
        assert bbox is not None
        assert bbox.length == self.metadata.length
        assert bbox.width == self.metadata.width
        assert ego_state.bounding_box_se2 == bbox_center

    def test_box_detection_property(self):
        """Test box detection properties."""
        ego_state = EgoStateSE2.from_rear_axle(
            rear_axle_se2=self.rear_axle_pose,
            metadata=self.metadata,
            dynamic_state_se2=self.dynamic_state,
            timestamp=self.timestamp,
        )

        box_det = ego_state.box_detection_se2
        assert box_det is not None
        assert box_det.attributes.label == DefaultBoxDetectionLabel.EGO
        assert box_det.attributes.track_token == EGO_TRACK_TOKEN

    def test_optional_parameters_default(self):
        """Test EgoStateSE2 with default optional parameters."""
        ego_state = EgoStateSE2.from_rear_axle(
            rear_axle_se2=self.rear_axle_pose,
            metadata=self.metadata,
            timestamp=self.timestamp,
        )

        assert ego_state.dynamic_state_se2 is None
        assert ego_state.timestamp == self.timestamp
        assert ego_state.tire_steering_angle == 0.0

    def test_no_init(self):
        """Test that EgoStateSE2 cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EgoStateSE2(
                rear_axle_se2=self.rear_axle_pose,
                metadata=self.metadata,
                timestamp=self.timestamp,
            )


class TestEgoStateSE3:
    def setup_method(self):
        """Set up test fixtures for EgoStateSE3 tests."""

        self.rear_axle_pose = PoseSE3(x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        self.metadata = EgoStateSE3Metadata(
            vehicle_name="test_vehicle",
            length=4.5,
            width=2.0,
            height=1.5,
            wheel_base=2.7,
            center_to_imu_se3=PoseSE3(x=1.35, y=0.0, z=0.5, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
            rear_axle_to_imu_se3=PoseSE3.identity(),
        )
        self.dynamic_state = DynamicStateSE3(
            velocity=Vector3D(1.0, 0.0, 0.0),
            acceleration=Vector3D(0.1, 0.0, 0.0),
            angular_velocity=Vector3D(0.0, 0.0, 0.1),
        )
        self.timestamp = Timestamp.from_us(1000000)
        self.tire_steering_angle = 0.2

    def test_from_rear_axle(self):
        """Test creating EgoStateSE3 from rear axle."""
        ego_state = EgoStateSE3.from_rear_axle(
            rear_axle_se3=self.rear_axle_pose,
            metadata=self.metadata,
            dynamic_state_se3=self.dynamic_state,
            timestamp=self.timestamp,
            tire_steering_angle=self.tire_steering_angle,
        )

        assert ego_state.rear_axle_se3 == self.rear_axle_pose
        assert ego_state.metadata == self.metadata
        assert ego_state.dynamic_state_se3 == self.dynamic_state
        assert ego_state.timestamp == self.timestamp
        assert ego_state.tire_steering_angle == self.tire_steering_angle

    def test_from_center(self):
        """Test creating EgoStateSE3 from center pose."""
        center_pose = PoseSE3(x=1.35, y=0.0, z=0.5, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        ego_state = EgoStateSE3.from_center(
            center_se3=center_pose,
            metadata=self.metadata,
            dynamic_state_se3=self.dynamic_state,
            timestamp=self.timestamp,
            tire_steering_angle=self.tire_steering_angle,
        )

        assert ego_state.rear_axle_se3 is not None
        assert ego_state.metadata == self.metadata

    def test_from_imu_identity(self):
        """Test creating EgoStateSE3 from IMU pose with identity rear_axle_to_imu."""
        imu_pose = PoseSE3(x=5.0, y=3.0, z=1.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        ego_state = EgoStateSE3.from_imu(
            imu_se3=imu_pose,
            metadata=self.metadata,
            dynamic_state_se3=self.dynamic_state,
            timestamp=self.timestamp,
            tire_steering_angle=self.tire_steering_angle,
        )

        # When rear_axle_to_imu is identity, IMU and rear axle coincide
        assert ego_state.rear_axle_se3.x == pytest.approx(imu_pose.x)
        assert ego_state.rear_axle_se3.y == pytest.approx(imu_pose.y)
        assert ego_state.rear_axle_se3.z == pytest.approx(imu_pose.z)
        assert ego_state.metadata == self.metadata
        assert ego_state.dynamic_state_se3 == self.dynamic_state
        assert ego_state.timestamp == self.timestamp
        assert ego_state.tire_steering_angle == self.tire_steering_angle

    def test_from_imu_with_offset(self):
        """Test creating EgoStateSE3 from IMU pose with a lateral offset (e.g. KITTI-360)."""
        ego_metadata_with_offset = EgoStateSE3Metadata(
            vehicle_name="test_vehicle_offset",
            length=4.5,
            width=2.0,
            height=1.5,
            wheel_base=2.7,
            center_to_imu_se3=PoseSE3(x=1.4, y=-0.32, z=0.5, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
            rear_axle_to_imu_se3=PoseSE3(x=0.05, y=-0.32, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
        )

        imu_pose = PoseSE3(x=10.0, y=5.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        ego_state = EgoStateSE3.from_imu(
            imu_se3=imu_pose,
            metadata=ego_metadata_with_offset,
            timestamp=self.timestamp,
        )

        # Rear axle should be offset from IMU: imu_pos + rear_axle_to_imu translation
        assert ego_state.rear_axle_se3.x == pytest.approx(10.0 + 0.05)
        assert ego_state.rear_axle_se3.y == pytest.approx(5.0 + (-0.32))
        assert ego_state.rear_axle_se3.z == pytest.approx(0.0)

    def test_from_imu_with_rotation(self):
        """Test creating EgoStateSE3 from IMU pose when IMU has a non-identity orientation."""
        # IMU rotated 90 degrees around Z axis
        imu_pose = PoseSE3(x=0.0, y=0.0, z=0.0, qw=np.cos(np.pi / 4), qx=0.0, qy=0.0, qz=np.sin(np.pi / 4))

        ego_metadata_with_offset = EgoStateSE3Metadata(
            vehicle_name="test_vehicle_offset",
            length=4.5,
            width=2.0,
            height=1.5,
            wheel_base=2.7,
            center_to_imu_se3=PoseSE3(x=1.4, y=-0.32, z=0.5, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
            rear_axle_to_imu_se3=PoseSE3(x=1.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
        )

        ego_state = EgoStateSE3.from_imu(
            imu_se3=imu_pose,
            metadata=ego_metadata_with_offset,
            timestamp=self.timestamp,
        )

        # With 90-deg yaw rotation, a +1.0 x-offset in IMU frame becomes +1.0 y-offset in world
        assert ego_state.rear_axle_se3.x == pytest.approx(0.0, abs=1e-10)
        assert ego_state.rear_axle_se3.y == pytest.approx(1.0, abs=1e-10)
        assert ego_state.rear_axle_se3.z == pytest.approx(0.0, abs=1e-10)
        assert ego_state.metadata == ego_metadata_with_offset
        assert ego_state.timestamp == self.timestamp

    def test_rear_axle_properties(self):
        """Test rear_axle properties."""
        ego_state = EgoStateSE3.from_rear_axle(
            rear_axle_se3=self.rear_axle_pose,
            metadata=self.metadata,
            timestamp=self.timestamp,
        )

        assert ego_state.rear_axle_se3 == self.rear_axle_pose
        assert ego_state.rear_axle_se2 is not None
        assert ego_state.timestamp == self.timestamp

    def test_center_properties(self):
        """Test center property calculation."""
        ego_state = EgoStateSE3.from_rear_axle(
            rear_axle_se3=self.rear_axle_pose,
            metadata=self.metadata,
            timestamp=self.timestamp,
        )

        center = ego_state.center_se3
        assert center is not None
        assert center.x == pytest.approx(self.metadata.rear_axle_to_center_longitudinal)
        assert center.y == pytest.approx(0.0)

        center_se2 = ego_state.center_se2
        assert center_se2 is not None

    def test_bounding_box_properties(self):
        """Test bounding box properties."""
        ego_state = EgoStateSE3.from_rear_axle(
            rear_axle_se3=self.rear_axle_pose,
            metadata=self.metadata,
            timestamp=self.timestamp,
        )

        bbox_se3 = ego_state.bounding_box_se3
        assert bbox_se3 is not None
        assert bbox_se3.length == self.metadata.length
        assert bbox_se3.width == self.metadata.width
        assert bbox_se3.height == self.metadata.height

        bbox_se2 = ego_state.bounding_box_se2
        assert bbox_se2 is not None
        assert ego_state.bounding_box_se3 == bbox_se3

    def test_box_detection_properties(self):
        """Test box detection properties."""
        ego_state = EgoStateSE3.from_rear_axle(
            rear_axle_se3=self.rear_axle_pose,
            metadata=self.metadata,
            dynamic_state_se3=self.dynamic_state,
            timestamp=self.timestamp,
        )

        box_det_se3 = ego_state.box_detection_se3
        assert box_det_se3 is not None
        assert box_det_se3.attributes.label == DefaultBoxDetectionLabel.EGO
        assert box_det_se3.attributes.track_token == EGO_TRACK_TOKEN

        box_det_se2 = ego_state.box_detection_se2
        assert box_det_se2 is not None
        assert box_det_se2.attributes.label == DefaultBoxDetectionLabel.EGO
        assert box_det_se2.attributes.track_token == EGO_TRACK_TOKEN

    def test_ego_state_se2_projection(self):
        """Test projection to EgoStateSE2."""
        ego_state = EgoStateSE3.from_rear_axle(
            rear_axle_se3=self.rear_axle_pose,
            metadata=self.metadata,
            dynamic_state_se3=self.dynamic_state,
            timestamp=self.timestamp,
            tire_steering_angle=self.tire_steering_angle,
        )

        ego_state_se2 = ego_state.ego_state_se2
        assert ego_state_se2 is not None
        assert isinstance(ego_state_se2, EgoStateSE2)
        assert ego_state_se2.metadata == self.metadata
        assert ego_state_se2.timestamp == self.timestamp
        assert ego_state_se2.tire_steering_angle == self.tire_steering_angle

    def test_optional_parameters_default(self):
        """Test EgoStateSE3 with default optional parameters."""
        ego_state = EgoStateSE3.from_rear_axle(
            rear_axle_se3=self.rear_axle_pose,
            metadata=self.metadata,
            timestamp=self.timestamp,
        )

        assert ego_state.dynamic_state_se3 is None
        assert ego_state.timestamp == self.timestamp
        assert ego_state.tire_steering_angle == 0.0

    def test_no_init(self):
        """Test that EgoStateSE3 cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EgoStateSE3(
                rear_axle_se3=self.rear_axle_pose,
                metadata=self.metadata,
                timestamp=self.timestamp,
            )
