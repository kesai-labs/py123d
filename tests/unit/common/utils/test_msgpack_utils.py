import numpy as np
import numpy.testing as npt

from py123d.common.utils.msgpack_utils import msgpack_decode_with_numpy, msgpack_encode_with_numpy


class TestMsgpackRoundtrip:
    """Tests for msgpack encode/decode with numpy array support."""

    def test_single_float64_array(self):
        """Test roundtrip of a single float64 numpy array."""
        data = {"values": np.array([1.0, 2.0, 3.0], dtype=np.float64)}
        result = msgpack_decode_with_numpy(msgpack_encode_with_numpy(data))
        npt.assert_array_equal(result["values"], data["values"])
        assert result["values"].dtype == np.float64

    def test_single_float32_array(self):
        """Test roundtrip of a float32 numpy array preserves dtype."""
        data = {"values": np.array([1.0, 2.0], dtype=np.float32)}
        result = msgpack_decode_with_numpy(msgpack_encode_with_numpy(data))
        npt.assert_array_equal(result["values"], data["values"])
        assert result["values"].dtype == np.float32

    def test_single_int32_array(self):
        """Test roundtrip of an int32 numpy array preserves dtype."""
        data = {"indices": np.array([0, 1, 2, 3], dtype=np.int32)}
        result = msgpack_decode_with_numpy(msgpack_encode_with_numpy(data))
        npt.assert_array_equal(result["indices"], data["indices"])
        assert result["indices"].dtype == np.int32

    def test_2d_array(self):
        """Test roundtrip of a 2D numpy array preserves shape."""
        data = {"matrix": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)}
        result = msgpack_decode_with_numpy(msgpack_encode_with_numpy(data))
        npt.assert_array_equal(result["matrix"], data["matrix"])
        assert result["matrix"].shape == (2, 2)

    def test_empty_array(self):
        """Test roundtrip of an empty numpy array."""
        data = {"empty": np.array([], dtype=np.float64)}
        result = msgpack_decode_with_numpy(msgpack_encode_with_numpy(data))
        npt.assert_array_equal(result["empty"], data["empty"])
        assert result["empty"].dtype == np.float64

    def test_multiple_arrays(self):
        """Test roundtrip of a dict with multiple numpy arrays."""
        data = {
            "positions": np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            "labels": np.array([0, 1, 1], dtype=np.int32),
        }
        result = msgpack_decode_with_numpy(msgpack_encode_with_numpy(data))
        npt.assert_array_equal(result["positions"], data["positions"])
        npt.assert_array_equal(result["labels"], data["labels"])

    def test_mixed_python_and_numpy(self):
        """Test roundtrip of a dict with both numpy arrays and plain Python types."""
        data = {
            "name": "test_log",
            "count": 42,
            "values": np.array([1.0, 2.0], dtype=np.float64),
        }
        result = msgpack_decode_with_numpy(msgpack_encode_with_numpy(data))
        assert result["name"] == "test_log"
        assert result["count"] == 42
        npt.assert_array_equal(result["values"], data["values"])

    def test_plain_dict_without_numpy(self):
        """Test roundtrip of a plain dict with no numpy arrays."""
        data = {"key": "value", "number": 123}
        result = msgpack_decode_with_numpy(msgpack_encode_with_numpy(data))
        assert result == data

    def test_3d_array_shape_preserved(self):
        """Test roundtrip of a 3D numpy array preserves shape."""
        data = {"volume": np.ones((2, 3, 4), dtype=np.float64)}
        result = msgpack_decode_with_numpy(msgpack_encode_with_numpy(data))
        assert result["volume"].shape == (2, 3, 4)
        npt.assert_array_equal(result["volume"], data["volume"])
