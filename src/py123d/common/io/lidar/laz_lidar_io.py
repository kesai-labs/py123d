import io
from typing import Dict, Optional, Tuple

import laspy
import numpy as np
import numpy.typing as npt

# Prefix added to feature names that collide with standard LAS dimension names (e.g. "intensity").
_EXTRA_PREFIX = "extra_"


def is_laz_binary(laz_binary: bytes) -> bool:
    """Check if the given binary data represents a LAZ compressed point cloud.

    :param laz_binary: The binary data to check.
    :return: True if the binary data is a LAZ compressed point cloud, False otherwise.
    """
    LAS_MAGIC_NUMBER = b"LASF"
    return laz_binary[0:4] == LAS_MAGIC_NUMBER


def encode_point_cloud_as_laz_binary(
    point_cloud_3d: npt.NDArray[np.float32],
    point_cloud_features: Optional[Dict[str, npt.NDArray]] = None,
) -> bytes:
    """Encode a point cloud (xyz + optional features) into a single LAZ binary blob.

    Features are stored using the LAS "Extra Bytes" mechanism, which preserves original dtypes.
    Feature names that collide with standard LAS dimension names are prefixed with ``extra_``.

    :param point_cloud_3d: The Lidar point cloud data, as a numpy array of shape (N, 3).
    :param point_cloud_features: Optional dictionary of per-point features.
    :return: The compressed LAZ binary data.
    """
    las = laspy.create(point_format=3, file_version="1.4")
    las.x = point_cloud_3d[:, 0]
    las.y = point_cloud_3d[:, 1]
    las.z = point_cloud_3d[:, 2]

    if point_cloud_features:
        standard_dims = set(las.point_format.standard_dimension_names)
        renamed: Dict[str, str] = {}  # original_name -> las_name
        for name in point_cloud_features:
            renamed[name] = f"{_EXTRA_PREFIX}{name}" if name in standard_dims else name

        extra_dims = [
            laspy.ExtraBytesParams(name=renamed[name], type=arr.dtype) for name, arr in point_cloud_features.items()
        ]
        las.add_extra_dims(extra_dims)
        for name, arr in point_cloud_features.items():
            setattr(las, renamed[name], arr)

    buffer = io.BytesIO()
    las.write(buffer, do_compress=True)
    return buffer.getvalue()


def load_point_cloud_from_laz_binary(
    laz_binary: bytes,
) -> Tuple[npt.NDArray[np.float32], Optional[Dict[str, npt.NDArray]]]:
    """Decode a LAZ binary blob back into a point cloud (xyz) and optional features.

    :param laz_binary: The compressed LAZ binary data.
    :return: Tuple of (point_cloud_3d as Nx3 float32 array, features dict or None).
    """
    buffer = io.BytesIO(laz_binary)
    las = laspy.read(buffer)

    lidar_pc = np.array(las.xyz, dtype=np.float32)
    assert lidar_pc.ndim == 2 and lidar_pc.shape[-1] == 3, (
        "Lidar point cloud must be a 2D array of shape (N, 3) for LAZ decompression."
    )

    # Extract extra bytes dimensions as features.
    standard_dims = set(las.point_format.standard_dimension_names)
    extra_dim_names = [dim.name for dim in las.point_format.dimensions if dim.name not in standard_dims]

    point_cloud_features: Optional[Dict[str, npt.NDArray]] = None
    if extra_dim_names:
        point_cloud_features = {}
        for las_name in extra_dim_names:
            # Reverse the prefix applied during encoding.
            original_name = las_name[len(_EXTRA_PREFIX) :] if las_name.startswith(_EXTRA_PREFIX) else las_name
            point_cloud_features[original_name] = np.array(las[las_name])

    return lidar_pc, point_cloud_features
