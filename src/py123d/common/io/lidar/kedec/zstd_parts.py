"""Zstandard compression of the parts of a kedec blob, through the Zstandard codec of PyArrow.

PyArrow's codec releases the GIL during compression and decompression, so several threads can encode scans in
parallel. One codec object per compression level is cached; the underlying one-shot calls create their own
context per call and are thread-safe.
"""

from __future__ import annotations

import threading
from typing import Dict, Union

import pyarrow as pa

_CODECS: Dict[int, pa.Codec] = {}
_LOCK = threading.Lock()

BytesLike = Union[bytes, bytearray, memoryview]


def _codec(level: int) -> pa.Codec:
    codec = _CODECS.get(level)
    if codec is None:
        with _LOCK:
            codec = _CODECS.get(level)
            if codec is None:
                codec = pa.Codec("zstd", compression_level=level)
                _CODECS[level] = codec
    return codec


def compress(data: BytesLike, level: int) -> bytes:
    """Compress bytes with Zstandard at the given level (an empty input gives an empty output).

    :param data: The bytes to compress.
    :param level: The Zstandard compression level (1 to 22).
    :return: One Zstandard frame, or ``b""`` for empty input.
    """
    if len(data) == 0:
        return b""
    return _codec(level).compress(data, asbytes=True)


def decompress(data: BytesLike, raw_size: int) -> bytes:
    """Decompress one Zstandard frame of known decompressed size.

    :param data: The compressed bytes (``b""`` for an empty part).
    :param raw_size: The size of the decompressed bytes, stored in the blob head.
    :return: The decompressed bytes.
    :raises ValueError: If the frame is corrupt or its decompressed size differs from ``raw_size``.
    """
    if raw_size == 0:
        if len(data) != 0:
            raise ValueError("kedec: a part of size 0 carries compressed bytes.")
        return b""
    try:
        out = _codec(3).decompress(data, decompressed_size=raw_size, asbytes=True)
    except OSError as error:  # PyArrow reports corrupt frames and wrong sizes as OSError
        raise ValueError(f"kedec: a part is corrupt or truncated ({error}).") from error
    if len(out) != raw_size:
        raise ValueError(f"kedec: a part decompressed to {len(out)} bytes instead of {raw_size}.")
    return out
