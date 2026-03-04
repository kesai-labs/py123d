import pytest

from py123d.datatypes.vehicle_state.ego_metadata import EgoMetadata
from py123d.geometry import PoseSE3


class TestEgoMetadata:
    def setup_method(self):
        self.params = EgoMetadata(
            vehicle_name="test_vehicle",
            width=2.0,
            length=5.0,
            height=1.8,
            wheel_base=3.0,
            center_to_imu_se3=PoseSE3(x=1.5, y=0.0, z=0.5, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
            rear_axle_to_imu_se3=PoseSE3.identity(),
        )

    def test_initialization(self):
        """Test that EgoMetadata initializes correctly."""
        assert self.params.vehicle_name == "test_vehicle"
        assert self.params.width == 2.0
        assert self.params.length == 5.0
        assert self.params.height == 1.8
        assert self.params.wheel_base == 3.0
        assert self.params.center_to_imu_se3 == PoseSE3(x=1.5, y=0.0, z=0.5, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        assert self.params.rear_axle_to_imu_se3 == PoseSE3.identity()

    def test_derived_offsets(self):
        """Test that rear_axle_to_center offsets are correctly derived from SE3 transforms."""
        assert self.params.rear_axle_to_center_longitudinal == pytest.approx(1.5)
        assert self.params.rear_axle_to_center_vertical == pytest.approx(0.5)

    def test_derived_offsets_with_imu_offset(self):
        """Test derived offsets when rear_axle_to_imu_se3 is not identity (e.g. KITTI-360)."""
        params = EgoMetadata(
            vehicle_name="test_vehicle",
            width=2.0,
            length=5.0,
            height=1.8,
            wheel_base=3.0,
            center_to_imu_se3=PoseSE3(x=1.55, y=-0.32, z=0.5, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
            rear_axle_to_imu_se3=PoseSE3(x=0.05, y=-0.32, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
        )
        assert params.rear_axle_to_center_longitudinal == pytest.approx(1.5)
        assert params.rear_axle_to_center_vertical == pytest.approx(0.5)

    def test_half_width(self):
        """Test half_width property."""
        assert self.params.half_width == 1.0

    def test_half_length(self):
        """Test half_length property."""
        assert self.params.half_length == 2.5

    def test_half_height(self):
        """Test half_height property."""
        assert self.params.half_height == 0.9

    def test_to_dict(self):
        """Test to_dict method."""
        result = self.params.to_dict()
        expected = {
            "vehicle_name": "test_vehicle",
            "width": 2.0,
            "length": 5.0,
            "height": 1.8,
            "wheel_base": 3.0,
            "center_to_imu_se3": [1.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],
            "rear_axle_to_imu_se3": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        }
        assert result == expected

    def test_from_dict(self):
        """Test from_dict method."""
        data = {
            "vehicle_name": "from_dict_vehicle",
            "width": 1.5,
            "length": 4.0,
            "height": 1.6,
            "wheel_base": 2.5,
            "center_to_imu_se3": [1.2, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
            "rear_axle_to_imu_se3": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        }
        params = EgoMetadata.from_dict(data)
        assert params.vehicle_name == "from_dict_vehicle"
        assert params.width == 1.5
        assert params.length == 4.0
        assert params.height == 1.6
        assert params.wheel_base == 2.5
        assert params.rear_axle_to_center_vertical == pytest.approx(0.4)
        assert params.rear_axle_to_center_longitudinal == pytest.approx(1.2)

    def test_from_dict_to_dict_round_trip(self):
        """Test that from_dict and to_dict are inverses."""
        original_dict = self.params.to_dict()
        recreated_params = EgoMetadata.from_dict(original_dict)
        recreated_dict = recreated_params.to_dict()
        assert original_dict == recreated_dict
