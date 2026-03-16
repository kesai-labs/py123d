import numpy as np

from py123d.common.io.lidar.laz_lidar_io import (
    encode_point_cloud_as_laz_binary,
    is_laz_binary,
    load_point_cloud_from_laz_binary,
)


class TestIsLazBinary:
    """Test LAZ binary format detection."""

    def test_valid_laz_binary(self):
        """Test that valid LAZ binary is detected correctly."""
        point_cloud = np.random.rand(100, 3).astype(np.float32)
        laz_binary = encode_point_cloud_as_laz_binary(point_cloud)
        assert is_laz_binary(laz_binary) is True

    def test_non_laz_binary(self):
        """Test that non-LAZ binary data is rejected."""
        assert is_laz_binary(b"not a laz file") is False

    def test_draco_binary_rejected(self):
        """Test that Draco binary is not detected as LAZ."""
        assert is_laz_binary(b"DRACO_DATA") is False

    def test_ipc_binary_rejected(self):
        """Test that Arrow IPC binary is not detected as LAZ."""
        assert is_laz_binary(b"ARROW1_DATA") is False


class TestLazRoundtrip:
    """Test LAZ unified encode/decode roundtrip."""

    def test_roundtrip_preserves_shape(self):
        """Test that encode/decode roundtrip preserves point cloud shape."""
        point_cloud = np.random.rand(100, 3).astype(np.float32)
        laz_binary = encode_point_cloud_as_laz_binary(point_cloud)
        decoded_pc, decoded_features = load_point_cloud_from_laz_binary(laz_binary)
        assert decoded_pc.shape == point_cloud.shape
        assert decoded_features is None

    def test_roundtrip_preserves_dtype(self):
        """Test that encode/decode roundtrip preserves float32 dtype."""
        point_cloud = np.random.rand(50, 3).astype(np.float32)
        laz_binary = encode_point_cloud_as_laz_binary(point_cloud)
        decoded_pc, _ = load_point_cloud_from_laz_binary(laz_binary)
        assert decoded_pc.dtype == np.float32

    def test_roundtrip_approximate_values(self):
        """Test that LAZ roundtrip preserves values approximately."""
        point_cloud = np.random.rand(100, 3).astype(np.float32) * 100.0
        laz_binary = encode_point_cloud_as_laz_binary(point_cloud)
        decoded_pc, _ = load_point_cloud_from_laz_binary(laz_binary)
        np.testing.assert_array_almost_equal(decoded_pc, point_cloud, decimal=2)

    def test_roundtrip_single_point(self):
        """Test roundtrip with a single point."""
        point_cloud = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        laz_binary = encode_point_cloud_as_laz_binary(point_cloud)
        decoded_pc, _ = load_point_cloud_from_laz_binary(laz_binary)
        assert decoded_pc.shape == (1, 3)
        np.testing.assert_array_almost_equal(decoded_pc, point_cloud, decimal=2)

    def test_roundtrip_large_point_cloud(self):
        """Test roundtrip with a large point cloud."""
        point_cloud = np.random.rand(10000, 3).astype(np.float32) * 50.0
        laz_binary = encode_point_cloud_as_laz_binary(point_cloud)
        decoded_pc, _ = load_point_cloud_from_laz_binary(laz_binary)
        assert decoded_pc.shape == (10000, 3)

    def test_roundtrip_negative_coordinates(self):
        """Test roundtrip with negative coordinate values."""
        point_cloud = (np.random.rand(50, 3).astype(np.float32) - 0.5) * 200.0
        laz_binary = encode_point_cloud_as_laz_binary(point_cloud)
        decoded_pc, _ = load_point_cloud_from_laz_binary(laz_binary)
        assert decoded_pc.shape == point_cloud.shape
        np.testing.assert_array_almost_equal(decoded_pc, point_cloud, decimal=2)

    def test_roundtrip_zeros(self):
        """Test roundtrip with all-zero point cloud."""
        point_cloud = np.zeros((10, 3), dtype=np.float32)
        laz_binary = encode_point_cloud_as_laz_binary(point_cloud)
        decoded_pc, _ = load_point_cloud_from_laz_binary(laz_binary)
        np.testing.assert_array_almost_equal(decoded_pc, point_cloud, decimal=2)

    def test_encoded_is_compressed(self):
        """Test that LAZ encoding produces smaller data than raw numpy."""
        point_cloud = np.random.rand(1000, 3).astype(np.float32)
        laz_binary = encode_point_cloud_as_laz_binary(point_cloud)
        assert isinstance(laz_binary, bytes)
        assert len(laz_binary) > 0
        assert len(laz_binary) < point_cloud.nbytes


class TestLazWithFeaturesRoundtrip:
    """Test LAZ unified encode/decode roundtrip with features (extra bytes)."""

    def test_roundtrip_single_feature(self):
        """Test roundtrip with a single uint8 feature."""
        point_cloud = np.random.rand(100, 3).astype(np.float32) * 10.0
        features = {"intensity": np.random.randint(0, 255, 100).astype(np.uint8)}
        laz_binary = encode_point_cloud_as_laz_binary(point_cloud, features)
        decoded_pc, decoded_features = load_point_cloud_from_laz_binary(laz_binary)

        np.testing.assert_array_almost_equal(decoded_pc, point_cloud, decimal=2)
        assert decoded_features is not None
        assert "intensity" in decoded_features
        np.testing.assert_array_equal(decoded_features["intensity"], features["intensity"])

    def test_roundtrip_multiple_features(self):
        """Test roundtrip with multiple features of different dtypes."""
        point_cloud = np.random.rand(100, 3).astype(np.float32) * 10.0
        features = {
            "intensity": np.random.randint(0, 255, 100).astype(np.uint8),
            "channel": np.random.randint(0, 64, 100).astype(np.uint8),
            "range_val": np.random.rand(100).astype(np.float32),
        }
        laz_binary = encode_point_cloud_as_laz_binary(point_cloud, features)
        decoded_pc, decoded_features = load_point_cloud_from_laz_binary(laz_binary)

        np.testing.assert_array_almost_equal(decoded_pc, point_cloud, decimal=2)
        assert decoded_features is not None
        assert set(decoded_features.keys()) == set(features.keys())
        for key in features:
            np.testing.assert_array_equal(decoded_features[key], features[key])

    def test_roundtrip_int64_feature(self):
        """Test roundtrip with int64 feature (timestamp)."""
        point_cloud = np.random.rand(100, 3).astype(np.float32) * 10.0
        features = {"timestamp": np.random.randint(0, 10**12, 100).astype(np.int64)}
        laz_binary = encode_point_cloud_as_laz_binary(point_cloud, features)
        decoded_pc, decoded_features = load_point_cloud_from_laz_binary(laz_binary)

        np.testing.assert_array_almost_equal(decoded_pc, point_cloud, decimal=2)
        assert decoded_features is not None
        np.testing.assert_array_equal(decoded_features["timestamp"], features["timestamp"])

    def test_no_features_returns_none(self):
        """Test that encoding without features returns None for features on decode."""
        point_cloud = np.random.rand(50, 3).astype(np.float32) * 10.0
        laz_binary = encode_point_cloud_as_laz_binary(point_cloud)
        _, decoded_features = load_point_cloud_from_laz_binary(laz_binary)
        assert decoded_features is None
