from typing import Dict, Final, Optional, Tuple

import DracoPy
import numpy as np
import numpy.typing as npt

# TODO: add to config
DRACO_QUANTIZATION_BITS: Final[int] = 16
DRACO_COMPRESSION_LEVEL: Final[int] = 7  # Range: 0 (fastest) to 10 (slowest, best compression)
DRACO_QUANTIZATION_RANGE: Final[int] = -1  # Use default range
DRACO_PRESERVE_ORDER: Final[bool] = True

# DracoPy only supports: float32, uint8, uint16, uint32.
# For unsupported dtypes we view-cast to uint8 bytes and decode back on load.
_DRACO_NATIVE_DTYPES = {np.dtype("float32"), np.dtype("uint8"), np.dtype("uint16"), np.dtype("uint32")}

# Metadata key prefix for storing original dtype info for view-casted attributes.
_DTYPE_META_PREFIX = "__dtype__"


def is_draco_binary(draco_binary: bytes) -> bool:
    """Check if the given binary data represents a Draco compressed point cloud.

    :param draco_binary: The binary data to check.
    :return: True if the binary data is a Draco compressed point cloud, False otherwise.
    """
    DRACO_MAGIC_NUMBER = b"DRACO"
    return draco_binary.startswith(DRACO_MAGIC_NUMBER)


def encode_point_cloud_as_draco_binary(
    point_cloud_3d: npt.NDArray[np.float32],
    point_cloud_features: Optional[Dict[str, npt.NDArray]] = None,
) -> bytes:
    """Encode a point cloud (xyz + optional features) into a single Draco binary blob.

    Features with dtypes not natively supported by Draco (e.g. int64, float64) are
    view-casted to uint8 arrays and reconstructed on decode.

    :param point_cloud_3d: The Lidar point cloud data, as numpy array of shape (N, 3).
    :param point_cloud_features: Optional dictionary of per-point features.
    :return: The compressed Draco binary data.
    """
    assert point_cloud_3d.ndim == 2, "Lidar point cloud must be a 2D array of shape (N, 3) for Draco compression."
    assert point_cloud_3d.shape[-1] == 3, "Lidar point cloud must have 3 attributes (x, y, z) for Draco compression."

    generic_attributes = None
    if point_cloud_features:
        generic_attributes = {}
        for name, arr in point_cloud_features.items():
            if arr.dtype in _DRACO_NATIVE_DTYPES:
                generic_attributes[name] = arr.reshape(len(arr), -1) if arr.ndim == 1 else arr
            else:
                # View-cast to uint8 bytes so Draco can store it losslessly.
                n_bytes = arr.dtype.itemsize
                uint8_view = arr.view(np.uint8).reshape(len(arr), n_bytes)
                generic_attributes[name] = uint8_view
                # Store original dtype so we can reconstruct on decode.
                # Encode dtype as a single-element float32 array containing the hash — but actually
                # we use a simpler approach: store dtype string as a small marker attribute.
                generic_attributes[f"{_DTYPE_META_PREFIX}{name}"] = (
                    np.frombuffer(arr.dtype.str.encode("ascii").ljust(8, b"\x00"), dtype=np.uint8)
                    .reshape(1, -1)
                    .repeat(len(arr), axis=0)
                )

    return DracoPy.encode(
        point_cloud_3d,
        quantization_bits=DRACO_QUANTIZATION_BITS,
        compression_level=DRACO_COMPRESSION_LEVEL,
        quantization_range=DRACO_QUANTIZATION_RANGE,
        quantization_origin=None,
        create_metadata=False,
        preserve_order=DRACO_PRESERVE_ORDER,
        generic_attributes=generic_attributes,
    )


def load_point_cloud_from_draco_binary(
    draco_binary: bytes,
) -> Tuple[npt.NDArray[np.float32], Optional[Dict[str, npt.NDArray]]]:
    """Decode a Draco binary blob back into a point cloud (xyz) and optional features.

    :param draco_binary: The compressed Draco binary data.
    :return: Tuple of (point_cloud_3d as Nx3 float32 array, features dict or None).
    """
    mesh = DracoPy.decode(draco_binary)
    lidar_pc = np.array(mesh.points, dtype=np.float32)
    assert lidar_pc.ndim == 2 and lidar_pc.shape[-1] == 3, (
        "Lidar point cloud must be a 2D array of shape (N, 3) for Draco decompression."
    )

    # Filter to only named generic attributes (position attribute has name=None).
    named_attrs = [attr for attr in (mesh.attributes or []) if attr.get("name") is not None]

    point_cloud_features: Optional[Dict[str, npt.NDArray]] = None
    if named_attrs:
        point_cloud_features = {}
        # First pass: collect dtype metadata.
        dtype_map: Dict[str, np.dtype] = {}
        for attr in named_attrs:
            name = attr["name"]
            if name.startswith(_DTYPE_META_PREFIX):
                original_name = name[len(_DTYPE_META_PREFIX) :]
                dtype_bytes = np.array(attr["data"], dtype=np.uint8)[0].tobytes().rstrip(b"\x00")
                dtype_map[original_name] = np.dtype(dtype_bytes.decode("ascii"))

        # Second pass: reconstruct feature arrays.
        for attr in named_attrs:
            name = attr["name"]
            if name.startswith(_DTYPE_META_PREFIX):
                continue
            data = np.array(attr["data"])
            if name in dtype_map:
                # Reconstruct from uint8 view-cast.
                original_dtype = dtype_map[name]
                data = data.view(np.uint8).reshape(len(data), -1)[:, : original_dtype.itemsize]
                data = np.ascontiguousarray(data).view(original_dtype).reshape(-1)
            else:
                data = data.squeeze() if data.ndim > 1 and data.shape[1] == 1 else data
            point_cloud_features[name] = data

    return lidar_pc, point_cloud_features
