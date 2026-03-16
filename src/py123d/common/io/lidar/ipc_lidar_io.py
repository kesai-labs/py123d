from typing import Dict, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from py123d.geometry.geometry_index import Point3DIndex

# Column name prefix used to distinguish xyz columns from feature columns in unified encoding.
_XYZ_COLUMNS = ("x", "y", "z")


def is_ipc_binary(blob: bytes) -> bool:
    """Check if the given binary data represents an Arrow IPC stream.

    :param blob: The binary data to check.
    :return: True if the binary data is an Arrow IPC stream, False otherwise.
    """
    IPC_STREAM_CONTINUATION = b"\xff\xff\xff\xff"
    return blob.startswith(IPC_STREAM_CONTINUATION)


def encode_point_cloud_as_ipc_binary(
    point_cloud_3d: npt.NDArray[np.float32],
    point_cloud_features: Optional[Dict[str, npt.NDArray]] = None,
    codec: Optional[Literal["zstd", "lz4"]] = "zstd",
) -> bytes:
    """Encode a point cloud (xyz + optional features) into a single Arrow IPC binary blob.

    :param point_cloud_3d: The Lidar point cloud data, as a numpy array of shape (N, 3).
    :param point_cloud_features: Optional dictionary of per-point features.
    :param codec: The compression codec to use, either "zstd" or "lz4", defaults to "zstd".
    :return: The compressed Arrow IPC binary data.
    """
    assert point_cloud_3d.ndim == 2 and point_cloud_3d.shape[1] == len(Point3DIndex), (
        "Lidar point cloud must be a 2-dim array of shape (N, 3)."
    )
    data: Dict[str, npt.NDArray] = {
        "x": point_cloud_3d[:, Point3DIndex.X],
        "y": point_cloud_3d[:, Point3DIndex.Y],
        "z": point_cloud_3d[:, Point3DIndex.Z],
    }
    if point_cloud_features:
        data.update(point_cloud_features)
    return _encode_dict_as_ipc_binary(data, codec=codec)


def load_point_cloud_from_ipc_binary(blob: bytes) -> Tuple[npt.NDArray[np.float32], Optional[Dict[str, npt.NDArray]]]:
    """Decode an Arrow IPC binary blob back into a point cloud (xyz) and optional features.

    :param blob: The compressed Arrow IPC binary data.
    :return: Tuple of (point_cloud_3d as Nx3 array, features dict or None).
    """
    all_columns = _load_dict_from_ipc_binary(blob)
    point_cloud_3d = np.stack((all_columns.pop("x"), all_columns.pop("y"), all_columns.pop("z")), axis=-1)
    assert point_cloud_3d.ndim == 2 and point_cloud_3d.shape[1] == len(Point3DIndex), (
        f"Decoded Lidar point cloud must be a 2-dim array of shape (N, 3). Got shape {point_cloud_3d.shape}"
    )
    point_cloud_features = all_columns if all_columns else None
    return point_cloud_3d, point_cloud_features


# ------------------------------------------------------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------------------------------------------------------


def _encode_dict_as_ipc_binary(data: dict[str, np.ndarray], codec: Optional[Literal["zstd", "lz4"]] = "zstd") -> bytes:
    """Encode a dictionary of numpy arrays into an Arrow IPC binary blob."""
    batch = pa.RecordBatch.from_pydict(data)
    sink = pa.BufferOutputStream()
    options = pa.ipc.IpcWriteOptions(compression=codec)
    with pa.ipc.new_stream(sink, batch.schema, options=options) as writer:
        writer.write_batch(batch)
    return sink.getvalue().to_pybytes()


def _load_dict_from_ipc_binary(blob: bytes) -> dict[str, np.ndarray]:
    """Decode an Arrow IPC binary blob into a dictionary of numpy arrays."""
    buffer = pa.BufferReader(blob)
    with pa.ipc.open_stream(buffer) as reader:
        batch = reader.read_next_batch()
    return {col: batch.column(i).to_numpy(zero_copy_only=False) for i, col in enumerate(batch.schema.names)}
