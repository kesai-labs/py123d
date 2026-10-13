"""Reads the files of a Physical AI AV clip from the chunk archives the dataset is published in.

The Hugging Face Hub stores the per-clip files of every sensor and label in archives of about 100 clips::

    <group>/<sensor>/<sensor>.chunk_NNNN.zip      with members      <clip_id>.<sensor>.<extension>

e.g. ``camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_0012.zip``. The members are stored
without compression, so each one is a byte range of its archive and is read in place.

A file of a clip is addressed by one path, whichever way it is stored: the extracted file
``<group>/<sensor>/<clip_id>.<sensor>.<extension>``, or the member of its archive, written as
``<group>/<sensor>/<sensor>.chunk_NNNN.zip/<clip_id>.<sensor>.<extension>``.
"""

from __future__ import annotations

import contextlib
import io
import shutil
import struct
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Union

import pandas as pd

_LOCAL_FILE_HEADER_SIZE = 30


def resolve_clip_file(
    data_root: Union[Path, str], group: str, sensor: str, clip_id: str, extension: str, chunk: int
) -> Path:
    """Path of one file of a clip: the extracted file if it exists, otherwise the member of its chunk archive.

    :param data_root: Root directory of the dataset.
    :param group: Top-level directory, e.g. "camera", "lidar", "radar" or "labels".
    :param sensor: Name of the sensor or label, e.g. "camera_front_wide_120fov" or "egomotion".
    :param clip_id: Name of the clip.
    :param extension: File extension without the leading dot, e.g. "parquet" or "timestamps.parquet".
    :param chunk: Number of the chunk the clip belongs to.
    :return: The path, which the other functions of this module accept.
    """
    sensor_dir = Path(data_root) / group / sensor
    file_name = f"{clip_id}.{sensor}.{extension}"
    if (sensor_dir / file_name).exists():
        return sensor_dir / file_name
    return sensor_dir / f"{sensor}.chunk_{chunk:04d}.zip" / file_name


def is_archive_member(path: Union[Path, str]) -> bool:
    """Whether the path points into a chunk archive instead of at an extracted file."""
    archive_path = Path(path).parent
    return archive_path.suffix == ".zip" and archive_path.is_file()


def path_exists(path: Union[Path, str]) -> bool:
    """Whether the file exists, extracted or as a member of its chunk archive."""
    return Path(path).exists() or _member_info(Path(path)) is not None


def open_file(path: Union[Path, str]) -> BinaryIO:
    """Opens a file for reading, extracted or as a member of its chunk archive.

    :param path: Path of the file, see :func:`resolve_clip_file`.
    :return: A seekable binary file object.
    """
    path = Path(path)
    if path.exists():
        return open(path, "rb")
    member_info = _member_info(path)
    if member_info is None:
        raise FileNotFoundError(f"No such file or archive member: {path}")
    if member_info.compress_type != zipfile.ZIP_STORED:  # not how the dataset is published, but still readable
        with zipfile.ZipFile(path.parent) as archive:
            return io.BytesIO(archive.read(member_info))
    # The local header in front of the data repeats the file name and has its own extra field.
    with open(path.parent, "rb") as archive_file:
        archive_file.seek(member_info.header_offset)
        header = archive_file.read(_LOCAL_FILE_HEADER_SIZE)
    name_size, extra_size = struct.unpack("<HH", header[26:30])
    data_offset = member_info.header_offset + _LOCAL_FILE_HEADER_SIZE + name_size + extra_size
    return io.BufferedReader(_ByteRangeFile(path.parent, data_offset, member_info.file_size))  # type: ignore[arg-type]


def read_parquet(path: Union[Path, str], columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Reads a parquet file, extracted or as a member of its chunk archive.

    From an archive only the requested columns are read, not the whole member.

    :param path: Path of the file, see :func:`resolve_clip_file`.
    :param columns: Columns to read, or None for all.
    :return: The table as a data frame.
    """
    if Path(path).exists():
        return pd.read_parquet(path, columns=columns)
    with open_file(path) as file:
        # Single-threaded: pyarrow's IO threads may still hold the Python file object when the interpreter exits.
        return pd.read_parquet(file, columns=columns, use_threads=False)


def extract_file(path: Union[Path, str], destination_dir: Path) -> Path:
    """Copies a member of a chunk archive into a directory, for libraries that need a file on disk.

    :param path: Path of the archive member, see :func:`resolve_clip_file`.
    :param destination_dir: Directory to copy into; created if missing.
    :return: Path of the copy.
    """
    # Another process may remove the empty parent folders between their creation and the creation of this one.
    for _ in range(10):
        with contextlib.suppress(FileNotFoundError):
            destination_dir.mkdir(parents=True, exist_ok=True)
            break
    destination_path = destination_dir / Path(path).name
    with open_file(path) as source, open(destination_path, "wb") as destination:
        shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
    return destination_path


def remove_extracted_files(destination_dir: Path) -> None:
    """Removes a directory filled by :func:`extract_file`, and the two folders above it once they are empty."""
    shutil.rmtree(destination_dir, ignore_errors=True)
    for parent_dir in list(destination_dir.parents)[:2]:
        with contextlib.suppress(OSError):  # not empty: other clips still have their files in it
            parent_dir.rmdir()


@lru_cache(maxsize=64)
def _archive_members(archive_path: str) -> Dict[str, zipfile.ZipInfo]:
    """Members of a chunk archive by name. Cached, as the clips of a chunk are converted one after another."""
    with zipfile.ZipFile(archive_path) as archive:
        return {member_info.filename: member_info for member_info in archive.infolist()}


def _member_info(path: Path) -> Optional[zipfile.ZipInfo]:
    """The archive member a path points at, or None if the path does not point into an archive."""
    if not is_archive_member(path):
        return None
    return _archive_members(str(path.parent)).get(path.name)


class _ByteRangeFile(io.RawIOBase):
    """Read-only, seekable file object over a byte range of a file."""

    def __init__(self, file_path: Path, offset: int, size: int) -> None:
        """
        :param file_path: The file to read from.
        :param offset: First byte of the range.
        :param size: Number of bytes in the range.
        """
        super().__init__()
        self._file = open(file_path, "rb")
        self._offset = offset
        self._size = size
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, position: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_CUR:
            position += self._position
        elif whence == io.SEEK_END:
            position += self._size
        elif whence != io.SEEK_SET:
            raise ValueError(f"Invalid whence: {whence}")
        if position < 0:
            raise ValueError(f"Negative seek position: {position}")
        self._position = position
        return self._position

    def readinto(self, buffer) -> int:  # type: ignore[override]
        size = min(len(buffer), max(self._size - self._position, 0))
        if size == 0:
            return 0
        self._file.seek(self._offset + self._position)
        data = self._file.read(size)
        buffer[: len(data)] = data
        self._position += len(data)
        return len(data)

    def close(self) -> None:
        if not self.closed:
            self._file.close()
        super().close()
