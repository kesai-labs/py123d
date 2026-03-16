import numpy as np
import pytest

from py123d.common.io.lidar.draco_lidar_io import (
    DRACO_QUANTIZATION_BITS,
    encode_point_cloud_as_draco_binary,
    is_draco_binary,
    load_point_cloud_from_draco_binary,
)


class TestIsDracoBinary:
    """Test Draco binary format detection."""

    def test_valid_draco_binary(self):
        """Test that valid Draco binary is detected correctly."""
        point_cloud = np.random.rand(100, 3).astype(np.float32)
        draco_binary = encode_point_cloud_as_draco_binary(point_cloud)
        assert is_draco_binary(draco_binary) is True

    def test_non_draco_binary(self):
        """Test that non-Draco binary data is rejected."""
        assert is_draco_binary(b"not a draco file") is False

    def test_laz_binary_rejected(self):
        """Test that LAZ binary is not detected as Draco."""
        assert is_draco_binary(b"LASF_DATA") is False

    def test_ipc_binary_rejected(self):
        """Test that Arrow IPC binary is not detected as Draco."""
        assert is_draco_binary(b"ARROW1_DATA") is False

    def test_empty_bytes(self):
        """Test that empty bytes are rejected."""
        assert is_draco_binary(b"") is False


class TestDracoRoundtrip:
    """Test Draco unified encode/decode roundtrip."""

    def test_roundtrip_preserves_shape(self):
        """Test that encode/decode roundtrip preserves point cloud shape."""
        point_cloud = np.random.rand(100, 3).astype(np.float32)
        draco_binary = encode_point_cloud_as_draco_binary(point_cloud)
        decoded_pc, decoded_features = load_point_cloud_from_draco_binary(draco_binary)
        assert decoded_pc.shape == point_cloud.shape
        assert decoded_features is None

    def test_roundtrip_preserves_dtype(self):
        """Test that encode/decode roundtrip preserves float32 dtype."""
        point_cloud = np.random.rand(50, 3).astype(np.float32)
        draco_binary = encode_point_cloud_as_draco_binary(point_cloud)
        decoded_pc, _ = load_point_cloud_from_draco_binary(draco_binary)
        assert decoded_pc.dtype == np.float32

    def test_roundtrip_approximate_values(self):
        """Test that Draco roundtrip preserves point values within quantization tolerance."""
        point_cloud = np.random.rand(100, 3).astype(np.float32)
        draco_binary = encode_point_cloud_as_draco_binary(point_cloud)
        decoded_pc, _ = load_point_cloud_from_draco_binary(draco_binary)
        # PRESERVE_ORDER=True, so direct comparison is valid.
        # 16-bit quantization gives sub-mm precision for typical ranges.
        np.testing.assert_array_almost_equal(decoded_pc, point_cloud, decimal=4)

    def test_roundtrip_preserves_order(self):
        """Test that point order is preserved with PRESERVE_ORDER=True."""
        point_cloud = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        draco_binary = encode_point_cloud_as_draco_binary(point_cloud)
        decoded_pc, _ = load_point_cloud_from_draco_binary(draco_binary)
        # Each point should remain closest to its original position.
        for i in range(len(point_cloud)):
            np.testing.assert_array_almost_equal(decoded_pc[i], point_cloud[i], decimal=3)

    def test_roundtrip_single_point(self):
        """Test roundtrip with a single point."""
        point_cloud = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        draco_binary = encode_point_cloud_as_draco_binary(point_cloud)
        decoded_pc, _ = load_point_cloud_from_draco_binary(draco_binary)
        assert decoded_pc.shape == (1, 3)
        np.testing.assert_array_almost_equal(decoded_pc, point_cloud, decimal=4)

    def test_roundtrip_large_point_cloud(self):
        """Test roundtrip with a large point cloud."""
        point_cloud = np.random.rand(10000, 3).astype(np.float32) * 50.0
        draco_binary = encode_point_cloud_as_draco_binary(point_cloud)
        decoded_pc, _ = load_point_cloud_from_draco_binary(draco_binary)
        assert decoded_pc.shape == (10000, 3)
        np.testing.assert_array_almost_equal(decoded_pc, point_cloud, decimal=2)

    def test_roundtrip_negative_coordinates(self):
        """Test roundtrip with negative coordinate values."""
        point_cloud = (np.random.rand(50, 3).astype(np.float32) - 0.5) * 200.0
        draco_binary = encode_point_cloud_as_draco_binary(point_cloud)
        decoded_pc, _ = load_point_cloud_from_draco_binary(draco_binary)
        assert decoded_pc.shape == point_cloud.shape
        np.testing.assert_array_almost_equal(decoded_pc, point_cloud, decimal=1)

    def test_encoded_is_compressed(self):
        """Test that Draco encoding produces smaller data than raw numpy."""
        point_cloud = np.random.rand(1000, 3).astype(np.float32)
        draco_binary = encode_point_cloud_as_draco_binary(point_cloud)
        raw_size = point_cloud.nbytes
        assert len(draco_binary) < raw_size

    def test_invalid_shape_2d_wrong_columns(self):
        """Test that a 2D array with wrong number of columns raises assertion."""
        point_cloud = np.random.rand(100, 4).astype(np.float32)
        with pytest.raises(AssertionError):
            encode_point_cloud_as_draco_binary(point_cloud)

    def test_invalid_shape_1d(self):
        """Test that a 1D array raises assertion."""
        point_cloud = np.random.rand(100).astype(np.float32)
        with pytest.raises(AssertionError):
            encode_point_cloud_as_draco_binary(point_cloud)

    def test_quantization_bits_constant(self):
        """Test that quantization bits constant has expected value."""
        assert DRACO_QUANTIZATION_BITS == 16


class TestDracoWithFeaturesRoundtrip:
    """Test Draco unified encode/decode roundtrip with features (generic_attributes)."""

    def test_roundtrip_float32_feature(self):
        """Test roundtrip with a float32 feature."""
        point_cloud = np.random.rand(100, 3).astype(np.float32)
        features = {"range_val": np.random.rand(100).astype(np.float32)}
        draco_binary = encode_point_cloud_as_draco_binary(point_cloud, features)
        decoded_pc, decoded_features = load_point_cloud_from_draco_binary(draco_binary)

        assert decoded_pc.shape == point_cloud.shape
        assert decoded_features is not None
        assert "range_val" in decoded_features

    def test_roundtrip_uint8_feature(self):
        """Test roundtrip with a uint8 feature (native Draco type)."""
        point_cloud = np.random.rand(100, 3).astype(np.float32)
        features = {"intensity": np.random.randint(0, 255, 100).astype(np.uint8)}
        draco_binary = encode_point_cloud_as_draco_binary(point_cloud, features)
        decoded_pc, decoded_features = load_point_cloud_from_draco_binary(draco_binary)

        assert decoded_features is not None
        assert "intensity" in decoded_features
        # PRESERVE_ORDER=True, so direct comparison is valid.
        np.testing.assert_array_equal(decoded_features["intensity"], features["intensity"])

    def test_roundtrip_int64_feature_via_viewcast(self):
        """Test roundtrip with int64 feature (view-casted to uint8 for Draco)."""
        point_cloud = np.random.rand(100, 3).astype(np.float32)
        features = {"timestamp": np.random.randint(0, 10**12, 100).astype(np.int64)}
        draco_binary = encode_point_cloud_as_draco_binary(point_cloud, features)
        decoded_pc, decoded_features = load_point_cloud_from_draco_binary(draco_binary)

        assert decoded_features is not None
        assert "timestamp" in decoded_features
        assert decoded_features["timestamp"].dtype == np.int64
        # PRESERVE_ORDER=True, so direct comparison is valid.
        np.testing.assert_array_equal(decoded_features["timestamp"], features["timestamp"])

    def test_roundtrip_multiple_features(self):
        """Test roundtrip with multiple features of different dtypes."""
        point_cloud = np.random.rand(100, 3).astype(np.float32)
        features = {
            "intensity": np.random.randint(0, 255, 100).astype(np.uint8),
            "range_val": np.random.rand(100).astype(np.float32),
        }
        draco_binary = encode_point_cloud_as_draco_binary(point_cloud, features)
        _, decoded_features = load_point_cloud_from_draco_binary(draco_binary)

        assert decoded_features is not None
        assert set(decoded_features.keys()) == set(features.keys())
        np.testing.assert_array_equal(decoded_features["intensity"], features["intensity"])

    def test_roundtrip_features_order_matches_points(self):
        """Test that feature values stay associated with the correct points after roundtrip."""
        point_cloud = np.array(
            [[10.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 30.0]],
            dtype=np.float32,
        )
        features = {"label": np.array([1, 2, 3], dtype=np.uint8)}
        draco_binary = encode_point_cloud_as_draco_binary(point_cloud, features)
        decoded_pc, decoded_features = load_point_cloud_from_draco_binary(draco_binary)

        assert decoded_features is not None
        # Each label must stay with the correct point.
        np.testing.assert_array_equal(decoded_features["label"], features["label"])

    def test_no_features_returns_none(self):
        """Test that encoding without features returns None for features on decode."""
        point_cloud = np.random.rand(50, 3).astype(np.float32)
        draco_binary = encode_point_cloud_as_draco_binary(point_cloud)
        _, decoded_features = load_point_cloud_from_draco_binary(draco_binary)
        assert decoded_features is None
