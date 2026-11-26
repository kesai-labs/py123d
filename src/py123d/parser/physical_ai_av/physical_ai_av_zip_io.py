"""Reads Physical AI AV data directly from the chunk archives of the Hugging Face dataset, without extraction.

The dataset stores its sensor and label features as ``<feature>.chunk_NNNN.zip`` archives of about 100 clips each;
every member is STORED (uncompressed), so each member is one contiguous byte range of the archive. Names::

    <group>/<feature>/<feature>.chunk_NNNN.zip        with members <clip_id>.<feature>.<ext>

for example ``camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_0012.zip`` with the members
``<clip_id>.camera_front_wide_120fov.mp4`` and ``<clip_id>.camera_front_wide_120fov.timestamps.parquet``, and
``lidar/lidar_top_360fov/lidar_top_360fov.chunk_0012.zip`` with ``<clip_id>.lidar_top_360fov.parquet``.
Calibration is stored as plain per-chunk parquet files.
"""

from __future__ import annotations

import io
import os
import struct
import zipfile
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

_LOCAL_FILE_HEADER_LEN = 30
_LOCAL_FILE_HEADER_MAGIC = b"PK\x03\x04"


def pai_chunk_zip_path(data_root: Path, group: str, feature: str, chunk: int) -> Path:
    """The chunk archive of one feature.

    :param group: The top-level folder: "camera", "lidar", "radar" or "labels".
    :param feature: The feature, e.g. "camera_front_wide_120fov", "lidar_top_360fov", "egomotion".
    :param chunk: The chunk number (0 to 3145).
    """
    return Path(data_root) / group / feature / f"{feature}.chunk_{chunk:04d}.zip"


def pai_member_name(clip_id: str, feature: str, ext: str) -> str:
    """The member name of one clip's file in its chunk archive, e.g. "<clip_id>.<feature>.mp4"."""
    return f"{clip_id}.{feature}.{ext}"


class ZipChunkReader:
    """Reads members of the chunk archives; caches the central directory and the payload offsets per archive.

    About 100 clips share a chunk archive, and a conversion worker usually reads several clips of one chunk, so the
    central directory is read once per archive. One reader per worker process (not picklable, not thread-safe).
    """

    def __init__(self) -> None:
        self._names: Dict[Path, Optional[Set[str]]] = {}
        self._zips: Dict[Path, zipfile.ZipFile] = {}
        self._ranges: Dict[Tuple[Path, str], Tuple[int, int]] = {}

    def _zip(self, zip_path: Path) -> zipfile.ZipFile:
        zip_path = Path(zip_path)
        if zip_path not in self._zips:
            self._zips[zip_path] = zipfile.ZipFile(zip_path)
        return self._zips[zip_path]

    def members(self, zip_path: Path) -> Optional[Set[str]]:
        """The member names of an archive, or None when the archive does not exist or is not a zip file."""
        zip_path = Path(zip_path)
        if zip_path not in self._names:
            try:
                self._names[zip_path] = set(self._zip(zip_path).namelist())
            except (FileNotFoundError, zipfile.BadZipFile):
                self._names[zip_path] = None
        return self._names[zip_path]

    def contains(self, zip_path: Path, member: str) -> bool:
        """Whether the archive exists and contains the member (central directory only)."""
        names = self.members(zip_path)
        return names is not None and member in names

    def member_range(self, zip_path: Path, member: str) -> Tuple[int, int]:
        """(offset, size) of the payload of a STORED member in its archive."""
        key = (Path(zip_path), member)
        if key not in self._ranges:
            info = self._zip(zip_path).getinfo(member)
            assert info.compress_type == zipfile.ZIP_STORED, (
                f"Member {member} of {zip_path} is compressed ({info.compress_type}); the chunk archives store "
                "their members uncompressed."
            )
            # The central directory does not give the length of the local header's extra field; the local header
            # gives where the payload starts.
            with open(zip_path, "rb", buffering=0) as f:
                header = os.pread(f.fileno(), _LOCAL_FILE_HEADER_LEN, info.header_offset)
            assert header[:4] == _LOCAL_FILE_HEADER_MAGIC, f"Bad local file header of {member} in {zip_path}"
            name_len, extra_len = struct.unpack("<HH", header[26:30])
            self._ranges[key] = (info.header_offset + _LOCAL_FILE_HEADER_LEN + name_len + extra_len, info.file_size)
        return self._ranges[key]

    def read_bytes(self, zip_path: Path, member: str) -> bytes:
        """The whole payload of a STORED member, read with one ``pread``."""
        offset, size = self.member_range(zip_path, member)
        with open(zip_path, "rb", buffering=0) as f:
            data = os.pread(f.fileno(), size, offset)
        if len(data) != size:
            raise IOError(f"Short read of {member} from {zip_path}: {len(data)} of {size} bytes")
        return data

    def open_bytes(self, zip_path: Path, member: str) -> io.BytesIO:
        """The payload of a member as an in-memory file object (for parquet and video readers)."""
        return io.BytesIO(self.read_bytes(zip_path, member))

    def close(self) -> None:
        for zf in self._zips.values():
            zf.close()
        self._zips.clear()
        self._names.clear()
        self._ranges.clear()
