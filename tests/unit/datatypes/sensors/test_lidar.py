import numpy as np

from py123d.datatypes.metadata.base_metadata import BaseMetadata
from py123d.datatypes.sensors.lidar import LIDAR_FEATURE_DTYPES, Lidar, LidarFeature, LidarID, LidarMetadata
from py123d.datatypes.time.timestamp import Timestamp
from py123d.geometry import PoseSE3

DUMMY_TIMESTAMP = Timestamp.from_s(0.0)
DUMMY_TIMESTAMP_END = Timestamp.from_s(0.1)


class TestLidarID:
    """Test LidarID enum functionality."""

    def test_lidar_id_enum_values(self):
        """Test that LidarID enum has correct values."""
        assert LidarID.LIDAR_UNKNOWN.value == 0
        assert LidarID.LIDAR_MERGED.value == 1
        assert LidarID.LIDAR_TOP.value == 2
        assert LidarID.LIDAR_FRONT.value == 3
        assert LidarID.LIDAR_SIDE_LEFT.value == 4
        assert LidarID.LIDAR_SIDE_RIGHT.value == 5
        assert LidarID.LIDAR_BACK.value == 6
        assert LidarID.LIDAR_DOWN.value == 7

    def test_lidar_id_enum_names(self):
        """Test that LidarID enum members have correct names."""
        assert LidarID.LIDAR_UNKNOWN.name == "LIDAR_UNKNOWN"
        assert LidarID.LIDAR_MERGED.name == "LIDAR_MERGED"
        assert LidarID.LIDAR_TOP.name == "LIDAR_TOP"
        assert LidarID.LIDAR_FRONT.name == "LIDAR_FRONT"
        assert LidarID.LIDAR_SIDE_LEFT.name == "LIDAR_SIDE_LEFT"
        assert LidarID.LIDAR_SIDE_RIGHT.name == "LIDAR_SIDE_RIGHT"
        assert LidarID.LIDAR_BACK.name == "LIDAR_BACK"
        assert LidarID.LIDAR_DOWN.name == "LIDAR_DOWN"

    def test_lidar_id_from_value(self):
        """Test that LidarID can be created from integer values."""
        assert LidarID(0) == LidarID.LIDAR_UNKNOWN
        assert LidarID(1) == LidarID.LIDAR_MERGED
        assert LidarID(2) == LidarID.LIDAR_TOP
        assert LidarID(3) == LidarID.LIDAR_FRONT
        assert LidarID(4) == LidarID.LIDAR_SIDE_LEFT
        assert LidarID(5) == LidarID.LIDAR_SIDE_RIGHT
        assert LidarID(6) == LidarID.LIDAR_BACK
        assert LidarID(7) == LidarID.LIDAR_DOWN

    def test_lidar_id_unique_values(self):
        """Test that all LidarID enum values are unique."""
        values = [member.value for member in LidarID]
        assert len(values) == len(set(values))

    def test_lidar_id_count(self):
        """Test that LidarID has expected number of members."""
        assert len(LidarID) == 8


class TestLidarMetadata:
    """Test LidarMetadata functionality."""

    def setup_method(self):
        """Set up test fixtures."""

        self.lidar_name = "TestLidar"

        # Get a lidar index class from registry (assuming at least one exists)
        self.lidar_id = LidarID.LIDAR_TOP
        self.lidar_to_imu_se3 = PoseSE3.from_list([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])

    def test_lidar_metadata_creation_with_lidar_to_imu_se3(self):
        """Test creating LidarMetadata with lidar_to_imu_se3."""
        metadata = LidarMetadata(
            lidar_name=self.lidar_name,
            lidar_id=self.lidar_id,
            lidar_to_imu_se3=self.lidar_to_imu_se3,
        )
        assert metadata.lidar_id == self.lidar_id
        assert metadata.lidar_to_imu_se3 == self.lidar_to_imu_se3

    def test_lidar_metadata_creation_without_lidar_to_imu_se3(self):
        """Test creating LidarMetadata without lidar_to_imu_se3."""
        metadata = LidarMetadata(lidar_name=self.lidar_name, lidar_id=self.lidar_id)
        assert metadata.lidar_id == self.lidar_id
        assert metadata.lidar_to_imu_se3 == PoseSE3.identity()

    def test_lidar_metadata_to_dict_with_lidar_to_imu_se3(self):
        """Test serializing LidarMetadata to dict with lidar_to_imu_se3."""
        metadata = LidarMetadata(
            lidar_name=self.lidar_name, lidar_id=self.lidar_id, lidar_to_imu_se3=self.lidar_to_imu_se3
        )
        data_dict = metadata.to_dict()
        assert data_dict["lidar_id"] == int(self.lidar_id)
        assert isinstance(data_dict["lidar_to_imu_se3"], list)

    def test_lidar_metadata_to_dict_without_lidar_to_imu_se3(self):
        """Test serializing LidarMetadata to dict without lidar_to_imu_se3."""
        metadata = LidarMetadata(lidar_name=self.lidar_name, lidar_id=self.lidar_id)
        data_dict = metadata.to_dict()
        assert data_dict["lidar_id"] == int(self.lidar_id)
        assert data_dict["lidar_to_imu_se3"] == PoseSE3.identity().to_list()

    def test_lidar_metadata_from_dict_with_lidar_to_imu_se3(self):
        """Test deserializing LidarMetadata from dict with lidar_to_imu_se3."""
        data_dict = {
            "lidar_name": self.lidar_name,
            "lidar_id": int(self.lidar_id),
            "lidar_to_imu_se3": [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
        }
        metadata = LidarMetadata.from_dict(data_dict)
        assert metadata.lidar_id == self.lidar_id
        assert metadata.lidar_to_imu_se3 == PoseSE3.from_list(data_dict["lidar_to_imu_se3"])

    def test_lidar_metadata_from_dict_without_lidar_to_imu_se3(self):
        """Test deserializing LidarMetadata from dict without lidar_to_imu_se3."""
        data_dict = {
            "lidar_name": self.lidar_name,
            "lidar_id": int(self.lidar_id),
            "lidar_to_imu_se3": PoseSE3.identity().to_list(),
        }
        metadata = LidarMetadata.from_dict(data_dict)
        assert metadata.lidar_id == self.lidar_id
        assert metadata.lidar_to_imu_se3 == PoseSE3.identity()

    def test_lidar_metadata_roundtrip_with_lidar_to_imu_se3(self):
        """Test roundtrip serialization/deserialization with lidar_to_imu_se3."""
        metadata = LidarMetadata(
            lidar_name=self.lidar_name,
            lidar_id=self.lidar_id,
            lidar_to_imu_se3=self.lidar_to_imu_se3,
        )
        data_dict = metadata.to_dict()
        restored_metadata = LidarMetadata.from_dict(data_dict)
        assert restored_metadata.lidar_id == metadata.lidar_id
        assert restored_metadata.lidar_to_imu_se3 == metadata.lidar_to_imu_se3

    def test_lidar_metadata_roundtrip_without_lidar_to_imu_se3(self):
        """Test roundtrip serialization/deserialization without lidar_to_imu_se3."""
        metadata = LidarMetadata(
            lidar_name=self.lidar_name,
            lidar_id=self.lidar_id,
        )
        data_dict = metadata.to_dict()
        restored_metadata = LidarMetadata.from_dict(data_dict)
        assert restored_metadata.lidar_id == metadata.lidar_id
        assert restored_metadata.lidar_to_imu_se3 == PoseSE3.identity()

    def test_is_instance_of_abstract_metadata(self):
        """LidarMetadata is an instance of BaseMetadata."""
        metadata = LidarMetadata(lidar_name=self.lidar_name, lidar_id=self.lidar_id)
        assert isinstance(metadata, BaseMetadata)


class TestLidar:
    """Test Lidar functionality."""

    NUM_POINTS = 100

    def setup_method(self):
        """Set up test fixtures."""
        self.lidar_to_imu_se3 = PoseSE3.from_list([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self.metadata = LidarMetadata(
            lidar_name="TestLidar",
            lidar_id=LidarID.LIDAR_TOP,
            lidar_to_imu_se3=self.lidar_to_imu_se3,
        )
        self.point_cloud_3d = np.random.rand(self.NUM_POINTS, 3).astype(np.float32)
        self.point_cloud_features = {
            LidarFeature.IDS.serialize(): np.arange(self.NUM_POINTS, dtype=np.uint8),
            LidarFeature.INTENSITY.serialize(): np.random.randint(0, 255, self.NUM_POINTS, dtype=np.uint8),
            LidarFeature.CHANNEL.serialize(): np.random.randint(0, 64, self.NUM_POINTS, dtype=np.uint8),
            LidarFeature.TIMESTAMPS.serialize(): np.arange(self.NUM_POINTS, dtype=np.int64),
            LidarFeature.RANGE.serialize(): np.random.rand(self.NUM_POINTS).astype(np.float32),
            LidarFeature.ELONGATION.serialize(): np.random.rand(self.NUM_POINTS).astype(np.float32),
        }
        self.lidar_with_features = Lidar(
            timestamp=DUMMY_TIMESTAMP,
            timestamp_end=DUMMY_TIMESTAMP_END,
            metadata=self.metadata,
            point_cloud_3d=self.point_cloud_3d,
            point_cloud_features=self.point_cloud_features,
        )
        self.lidar_without_features = Lidar(
            timestamp=DUMMY_TIMESTAMP,
            timestamp_end=DUMMY_TIMESTAMP_END,
            metadata=self.metadata,
            point_cloud_3d=self.point_cloud_3d,
        )

    def test_lidar_metadata_property(self):
        """Test metadata property returns correct metadata."""
        assert self.lidar_with_features.metadata is self.metadata

    def test_lidar_point_cloud_3d_property(self):
        """Test point_cloud_3d property returns the raw array."""
        np.testing.assert_array_equal(self.lidar_with_features.point_cloud_3d, self.point_cloud_3d)

    def test_lidar_xyz_property(self):
        """Test xyz property returns correct shape and values."""
        xyz = self.lidar_with_features.xyz
        assert xyz.shape == (self.NUM_POINTS, 3)
        np.testing.assert_array_equal(xyz, self.point_cloud_3d)

    def test_lidar_xy_property(self):
        """Test xy property returns correct shape and values."""
        xy = self.lidar_with_features.xy
        assert xy.shape == (self.NUM_POINTS, 2)
        np.testing.assert_array_equal(xy, self.point_cloud_3d[:, :2])

    def test_lidar_ids_when_available(self):
        """Test ids property when feature is present."""
        ids = self.lidar_with_features.ids
        assert ids is not None
        assert ids.shape == (self.NUM_POINTS,)
        assert ids.dtype == np.uint8

    def test_lidar_ids_when_not_available(self):
        """Test ids property returns None when features are absent."""
        assert self.lidar_without_features.ids is None

    def test_lidar_intensity_when_available(self):
        """Test intensity property when feature is present."""
        intensity = self.lidar_with_features.intensity
        assert intensity is not None
        assert intensity.shape == (self.NUM_POINTS,)
        assert intensity.dtype == np.uint8

    def test_lidar_intensity_when_not_available(self):
        """Test intensity property returns None when features are absent."""
        assert self.lidar_without_features.intensity is None

    def test_lidar_channel_when_available(self):
        """Test channel property when feature is present."""
        channel = self.lidar_with_features.channel
        assert channel is not None
        assert channel.shape == (self.NUM_POINTS,)
        assert channel.dtype == np.uint8

    def test_lidar_channel_when_not_available(self):
        """Test channel property returns None when features are absent."""
        assert self.lidar_without_features.channel is None

    def test_lidar_timestamp_when_available(self):
        """Test timestamps property when feature is present."""
        timestamps = self.lidar_with_features.timestamps
        assert timestamps is not None
        assert timestamps.shape == (self.NUM_POINTS,)
        assert timestamps.dtype == np.int64

    def test_lidar_timestamp_when_not_available(self):
        """Test timestamps property returns None when features are absent."""
        assert self.lidar_without_features.timestamps is None

    def test_lidar_range_when_available(self):
        """Test range property when feature is present."""
        range_values = self.lidar_with_features.range
        assert range_values is not None
        assert range_values.shape == (self.NUM_POINTS,)
        assert range_values.dtype == np.float32

    def test_lidar_range_when_not_available(self):
        """Test range property returns None when features are absent."""
        assert self.lidar_without_features.range is None

    def test_lidar_elongation_when_available(self):
        """Test elongation property when feature is present."""
        elongation = self.lidar_with_features.elongation
        assert elongation is not None
        assert elongation.shape == (self.NUM_POINTS,)
        assert elongation.dtype == np.float32

    def test_lidar_elongation_when_not_available(self):
        """Test elongation property returns None when features are absent."""
        assert self.lidar_without_features.elongation is None

    def test_lidar_point_cloud_features_property(self):
        """Test point_cloud_features property returns the features dict."""
        assert self.lidar_with_features.point_cloud_features is self.point_cloud_features
        assert self.lidar_without_features.point_cloud_features is None

    def test_lidar_with_empty_point_cloud(self):
        """Test Lidar with empty point cloud."""
        empty_point_cloud = np.empty((0, 3), dtype=np.float32)
        lidar = Lidar(
            timestamp=DUMMY_TIMESTAMP,
            timestamp_end=DUMMY_TIMESTAMP_END,
            metadata=self.metadata,
            point_cloud_3d=empty_point_cloud,
        )
        assert lidar.xyz.shape == (0, 3)
        assert lidar.xy.shape == (0, 2)

    def test_lidar_with_single_point(self):
        """Test Lidar with single point."""
        single_point_cloud = np.random.rand(1, 3).astype(np.float32)
        lidar = Lidar(
            timestamp=DUMMY_TIMESTAMP,
            timestamp_end=DUMMY_TIMESTAMP_END,
            metadata=self.metadata,
            point_cloud_3d=single_point_cloud,
        )
        assert lidar.xyz.shape == (1, 3)
        assert lidar.xy.shape == (1, 2)

    def test_lidar_point_cloud_dtype(self):
        """Test that point cloud maintains float32 dtype."""
        assert self.lidar_with_features.point_cloud_3d.dtype == np.float32
        assert self.lidar_with_features.xyz.dtype == np.float32
        assert self.lidar_with_features.xy.dtype == np.float32

    def test_lidar_feature_dtypes(self):
        """Test that feature properties return correct dtypes as defined in LIDAR_FEATURE_DTYPES."""
        for feature, expected_dtype in LIDAR_FEATURE_DTYPES.items():
            key = feature.serialize()
            if key in self.point_cloud_features:
                prop_name = feature.name.lower()
                if prop_name == "timestamp":
                    prop_name = "timestamps"
                value = getattr(self.lidar_with_features, prop_name)
                assert value is not None, f"Feature {prop_name} should not be None"
                assert value.dtype == expected_dtype, f"Feature {prop_name} dtype mismatch"

    def test_lidar_with_partial_features(self):
        """Test Lidar with only some features present."""
        partial_features = {
            LidarFeature.INTENSITY.serialize(): np.random.randint(0, 255, self.NUM_POINTS, dtype=np.uint8),
        }
        lidar = Lidar(
            timestamp=DUMMY_TIMESTAMP,
            timestamp_end=DUMMY_TIMESTAMP_END,
            metadata=self.metadata,
            point_cloud_3d=self.point_cloud_3d,
            point_cloud_features=partial_features,
        )
        assert lidar.intensity is not None
        assert lidar.range is None
        assert lidar.elongation is None
        assert lidar.channel is None
        assert lidar.ids is None
        assert lidar.timestamps is None
