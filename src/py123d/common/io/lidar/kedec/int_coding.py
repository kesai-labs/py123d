"""Lossless coding of integer channels and of raw bytes for the kedec point-cloud codec.

An integer channel (n values of a NumPy integer or bool dtype) is stored in one of two forms:

* raw: the values minus their minimum, ``u = v - min(v)``;
* differences: the zigzag-mapped differences of ``u`` between neighboring values (the first value against 0).

Both forms are non-negative integers. They are written as byte planes (all first bytes, then all second bytes,
...) of the smallest width, 0 to 8 bytes, in which their largest value fits, and compressed with Zstandard. A channel
whose values are all equal has width 0 in raw form and stores no bytes. The arithmetic is modulo 2**64 on the
order-preserving unsigned image of the values, so every int64 and uint64 value is decoded exactly.

The form is chosen per channel: ``"estimate"`` compresses both forms at Zstandard level 1 and stores the smaller one
at the final level (one slow compression per channel), ``"exact"`` compresses both forms at the final level, and
``"raw"`` or ``"diff"`` fix the form.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import numpy.typing as npt

from py123d.common.io.lidar.kedec import zstd_parts
from py123d.common.io.lidar.point_cloud_codec_config import KEDEC_INT_CHOICES as INT_CHOICES

_SIGN_BIT = np.uint64(1 << 63)
_ESTIMATE_LEVEL = 1


def is_integer_dtype(dtype: np.dtype) -> bool:
    """Whether the dtype is stored by the integer coding (all NumPy integer dtypes and bool)."""
    return np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_)


def _to_ordered_u64(values: npt.NDArray) -> npt.NDArray[np.uint64]:
    """The order-preserving uint64 image of integer or bool values."""
    if np.issubdtype(values.dtype, np.bool_):
        return values.astype(np.uint64)
    if np.issubdtype(values.dtype, np.signedinteger):
        return values.astype(np.int64).view(np.uint64) ^ _SIGN_BIT
    return values.astype(np.uint64)


def _from_ordered_u64(u: npt.NDArray[np.uint64], dtype: np.dtype) -> npt.NDArray:
    """The inverse of :func:`_to_ordered_u64`, cast to ``dtype``."""
    if np.issubdtype(dtype, np.bool_):
        return u.astype(np.bool_)
    if np.issubdtype(dtype, np.signedinteger):
        return (u ^ _SIGN_BIT).view(np.int64).astype(dtype)
    return u.astype(dtype)


def _width(u: npt.NDArray[np.uint64]) -> int:
    """The number of bytes (0 to 8) in which the largest value fits."""
    if len(u) == 0:
        return 0
    return (int(u.max()).bit_length() + 7) // 8


def planes(u: npt.NDArray[np.uint64], width: int) -> bytes:
    """The first ``width`` little-endian bytes of each value, as byte planes."""
    if width == 0 or len(u) == 0:
        return b""
    as_bytes = np.ascontiguousarray(u, dtype="<u8").view(np.uint8).reshape(len(u), 8)
    return np.ascontiguousarray(as_bytes[:, :width].T).tobytes()


def unplanes(data: bytes, n: int, width: int) -> npt.NDArray[np.uint64]:
    """The inverse of :func:`planes`: n uint64 values from ``width`` byte planes."""
    out = np.zeros((n, 8), dtype=np.uint8)
    if width > 0 and n > 0:
        out[:, :width] = np.frombuffer(data, dtype=np.uint8).reshape(width, n).T
    return out.view("<u8").reshape(n).astype(np.uint64, copy=False)


def _zigzag_differences(u: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint64]:
    """Zigzag of the differences of u modulo 2**64 (the first value against 0)."""
    d = np.empty_like(u)
    if len(u):
        d[0] = u[0]
        np.subtract(u[1:], u[:-1], out=d[1:])
    s = d.view(np.int64)
    return ((s << 1) ^ (s >> 63)).view(np.uint64)


def _undo_zigzag_differences(z: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint64]:
    """The inverse of :func:`_zigzag_differences` (a cumulative sum modulo 2**64)."""
    s = (z >> np.uint64(1)).view(np.int64) ^ -(z & np.uint64(1)).view(np.int64)
    return np.cumsum(s.view(np.uint64), dtype=np.uint64)


def encode_int_channel(values: npt.NDArray, level: int, choice: str = "estimate") -> Tuple[Dict[str, Any], bytes]:
    """Encode one integer channel.

    :param values: A 1-D array of an integer or bool dtype.
    :param level: The Zstandard level of the stored bytes.
    :param choice: How to choose between raw values and differences, one of :data:`INT_CHOICES`.
    :return: The descriptor (``m`` the minimum as uint64 image, ``w`` the byte width, ``d`` 1 for differences, ``u``
        the size of the uncompressed planes) and the compressed bytes.
    """
    if choice not in INT_CHOICES:
        raise ValueError(f"kedec: unknown integer coding choice {choice!r}, expected one of {INT_CHOICES}.")
    values = np.asarray(values)
    if values.ndim != 1 or not is_integer_dtype(values.dtype):
        raise ValueError(f"kedec: an integer channel must be a 1-D integer array, got {values.dtype} {values.shape}.")
    ordered = _to_ordered_u64(values)
    vmin = np.uint64(ordered.min()) if len(ordered) else np.uint64(0)
    raw = ordered - vmin
    candidates = []
    if choice != "diff":
        width = _width(raw)
        candidates.append((0, width, planes(raw, width)))
    if choice != "raw":
        diff = _zigzag_differences(raw)
        width = _width(diff)
        candidates.append((1, width, planes(diff, width)))
    if len(candidates) == 1:
        use_diff, width, data = candidates[0]
        compressed = zstd_parts.compress(data, level)
    elif choice == "exact":
        options = [(zstd_parts.compress(data, level), d, w, data) for d, w, data in candidates]
        compressed, use_diff, width, data = min(options, key=lambda option: (len(option[0]), option[1]))
    else:
        sizes = [len(zstd_parts.compress(data, _ESTIMATE_LEVEL)) for _, _, data in candidates]
        use_diff, width, data = candidates[int(np.argmin(sizes))]
        compressed = zstd_parts.compress(data, level)
    return {"m": int(vmin), "w": int(width), "d": int(use_diff), "u": len(data)}, compressed


def decode_int_channel(descriptor: Dict[str, Any], compressed: bytes, n: int, dtype: np.dtype) -> npt.NDArray:
    """Decode one integer channel written by :func:`encode_int_channel`.

    :param descriptor: The descriptor returned by the encoder.
    :param compressed: The compressed bytes.
    :param n: The number of values.
    :param dtype: The dtype of the decoded values.
    :return: The decoded 1-D array.
    """
    width = int(descriptor["w"])
    if not 0 <= width <= 8 or int(descriptor["u"]) != width * n:
        raise ValueError(f"kedec: an integer channel of {n} values has an invalid descriptor {descriptor}.")
    data = zstd_parts.decompress(compressed, int(descriptor["u"]))
    stored = unplanes(data, n, width)
    raw = _undo_zigzag_differences(stored) if int(descriptor["d"]) else stored
    return _from_ordered_u64(raw + np.uint64(descriptor["m"]), np.dtype(dtype))


def encode_bytes_channel(values: npt.NDArray, level: int) -> Tuple[Dict[str, Any], bytes]:
    """Encode a channel of any fixed-size dtype without loss, as byte planes of each point's bytes.

    :param values: An array whose first axis runs over the points.
    :param level: The Zstandard level.
    :return: The descriptor (``w`` the bytes per point, ``u`` the uncompressed size) and the compressed bytes.
    """
    values = np.ascontiguousarray(values)
    if values.dtype.hasobject:
        raise ValueError(f"kedec: features of dtype {values.dtype} cannot be stored.")
    n = values.shape[0]
    width = values.dtype.itemsize * int(np.prod(values.shape[1:], dtype=np.int64))
    per_point = values.view(np.uint8).reshape(n, width) if n else np.zeros((0, width), np.uint8)
    data = np.ascontiguousarray(per_point.T).tobytes()
    return {"w": int(per_point.shape[1]), "u": len(data)}, zstd_parts.compress(data, level)


def decode_bytes_channel(
    descriptor: Dict[str, Any], compressed: bytes, n: int, dtype: np.dtype, tail: Tuple[int, ...]
) -> npt.NDArray:
    """Decode a channel written by :func:`encode_bytes_channel`.

    :param descriptor: The descriptor returned by the encoder.
    :param compressed: The compressed bytes.
    :param n: The number of points.
    :param dtype: The dtype of the values.
    :param tail: The shape of the values of one point (``()`` for one value per point).
    :return: The decoded array of shape ``(n, *tail)``.
    """
    dtype = np.dtype(dtype)
    width = int(descriptor["w"])
    if width != dtype.itemsize * int(np.prod(tail, dtype=np.int64)) or int(descriptor["u"]) != width * n:
        raise ValueError(f"kedec: a byte channel of {n} points has an invalid descriptor {descriptor}.")
    data = zstd_parts.decompress(compressed, int(descriptor["u"]))
    per_point = np.ascontiguousarray(np.frombuffer(data, dtype=np.uint8).reshape(width, n).T)
    return per_point.view(dtype).reshape((n,) + tuple(tail))
