"""Index of the nuPlan sensor archives, to read camera images and lidar sweeps without extracting them.

The train, val and test sensor shards of nuPlan are uncompressed tar archives (with a ``.zip`` name), so
a member is one byte range of its archive. :func:`build_tar_index` scans an archive once, headers only,
and writes two files next to it::

    <archive>.index.parquet    one row per file: log_name, relative_path, offset, size
    <archive>.logs.txt         the names of the logs in the archive, one per line

``relative_path`` is the path of the file below the archive's wrapper folder, e.g.
``<log_name>/CAM_L0/<token>.jpg``, which is how the log databases of nuPlan name the file.

:class:`NuplanTarIndex` reads the ``.logs.txt`` files of a directory and tells which archives hold a log;
:func:`load_log_members` then reads only that log's rows of their indexes.

Build the indexes once, before converting with ``dataset=nuplan-tar``::

    python -m py123d.parser.nuplan.nuplan_tar_index <archive> [<archive> ...]

A tar archive is read front to back, so one archive is indexed by one process; index several at once
by starting several commands.
"""

from __future__ import annotations

import argparse
import logging
import os
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

INDEX_SUFFIX = ".index.parquet"
LOGS_SUFFIX = ".logs.txt"

# Where a file is stored: (archive path, offset of its first byte, size in bytes).
TarMember = Tuple[Path, int, int]


def build_tar_index(tar_path: Path, force: bool = False) -> Path:
    """Scans an uncompressed tar archive and writes its index files next to it.

    :param tar_path: The archive to index.
    :param force: Rebuilds an index that is newer than its archive, instead of keeping it.
    :return: Path of the ``.index.parquet`` file.
    """
    tar_path = Path(tar_path)
    index_path = tar_path.with_name(tar_path.name + INDEX_SUFFIX)
    logs_path = tar_path.with_name(tar_path.name + LOGS_SUFFIX)

    if not force and index_path.exists() and logs_path.exists():
        if index_path.stat().st_mtime >= tar_path.stat().st_mtime:
            logger.info("Index is up to date: %s", index_path.name)
            return index_path

    log_names: List[str] = []
    relative_paths: List[str] = []
    offsets: List[int] = []
    sizes: List[int] = []

    with tarfile.open(tar_path, "r:") as archive:
        member = archive.next()
        while member is not None:
            # <wrapper folder>/<log_name>/<sensor folder>/<file>
            parts = member.name.split("/")
            if member.isfile() and len(parts) >= 3:
                log_names.append(parts[1])
                relative_paths.append("/".join(parts[1:]))
                offsets.append(member.offset_data)
                sizes.append(member.size)
            archive.members = []  # tarfile keeps every member it has read, drop them to bound the memory
            member = archive.next()

    table = pa.table(
        {
            "log_name": pa.array(log_names, type=pa.string()),
            "relative_path": pa.array(relative_paths, type=pa.string()),
            "offset": pa.array(offsets, type=pa.int64()),
            "size": pa.array(sizes, type=pa.int64()),
        }
    )
    # Sorted by log, a log's rows sit in few row groups and are read without the rest.
    table = table.sort_by("log_name")

    temporary_index_path = index_path.with_suffix(".tmp")
    pq.write_table(table, temporary_index_path, row_group_size=64_000)
    temporary_index_path.replace(index_path)

    distinct_log_names = sorted(set(log_names))
    temporary_logs_path = logs_path.with_suffix(".tmp")
    temporary_logs_path.write_text("".join(f"{log_name}\n" for log_name in distinct_log_names))
    temporary_logs_path.replace(logs_path)

    logger.info("Indexed %s: %d files of %d logs", tar_path.name, len(relative_paths), len(distinct_log_names))
    return index_path


class NuplanTarIndex:
    """Tells which of the indexed sensor archives in a directory hold the files of a log."""

    def __init__(self, archive_root: Path) -> None:
        """
        :param archive_root: Directory with the sensor archives and their index files.
        """
        self._archive_root = Path(archive_root)
        self._log_to_archives: Dict[str, List[Path]] = {}

        archives = sorted(self._archive_root.glob("*_camera_*")) + sorted(self._archive_root.glob("*_lidar_*"))
        archives_without_index = [
            path.name
            for path in archives
            if not path.name.endswith((INDEX_SUFFIX, LOGS_SUFFIX, ".tmp"))
            and not path.with_name(path.name + INDEX_SUFFIX).exists()
        ]
        if archives_without_index:
            logger.warning(
                "%d sensor archives in %s have no index and are not read (see py123d.parser.nuplan.nuplan_tar_index): %s",
                len(archives_without_index),
                self._archive_root,
                ", ".join(archives_without_index[:5]) + (", ..." if len(archives_without_index) > 5 else ""),
            )

        for index_path in sorted(self._archive_root.glob(f"*{INDEX_SUFFIX}")):
            tar_path = index_path.with_name(index_path.name[: -len(INDEX_SUFFIX)])
            logs_path = tar_path.with_name(tar_path.name + LOGS_SUFFIX)
            if logs_path.exists():
                log_names: Iterable[str] = logs_path.read_text().splitlines()
            else:
                log_names = pq.read_table(index_path, columns=["log_name"])["log_name"].unique().to_pylist()
            for log_name in log_names:
                if log_name:
                    self._log_to_archives.setdefault(log_name, []).append(tar_path)

    @property
    def log_names(self) -> Set[str]:
        """Names of the logs that have files in the indexed archives."""
        return set(self._log_to_archives)

    def archives_for_log(self, log_name: str) -> List[Path]:
        """The archives that hold files of a log, empty if there are none."""
        return list(self._log_to_archives.get(log_name, []))


def load_log_members(log_name: str, archives: Iterable[Path]) -> Dict[str, TarMember]:
    """Reads where the files of a log are stored.

    :param log_name: Name of the log.
    :param archives: The indexed archives that hold files of the log.
    :return: The location of every file, by its relative path ``<log_name>/<sensor folder>/<file>``.
    """
    members: Dict[str, TarMember] = {}
    for tar_path in archives:
        index_path = tar_path.with_name(tar_path.name + INDEX_SUFFIX)
        table = pq.read_table(index_path, filters=[("log_name", "==", log_name)])
        for relative_path, offset, size in zip(
            table["relative_path"].to_pylist(), table["offset"].to_pylist(), table["size"].to_pylist()
        ):
            members[relative_path] = (tar_path, offset, size)
    return members


class TarMemberReader:
    """Reads files out of uncompressed tar archives, keeping the archives open between reads."""

    def __init__(self) -> None:
        self._file_descriptors: Dict[Path, int] = {}

    def read(self, tar_path: Path, offset: int, size: int) -> bytes:
        """Reads one file.

        :param tar_path: The archive the file is in.
        :param offset: Offset of the file's first byte in the archive.
        :param size: Size of the file in bytes.
        :return: The content of the file.
        """
        file_descriptor = self._file_descriptors.get(tar_path)
        if file_descriptor is None:
            file_descriptor = os.open(tar_path, os.O_RDONLY)
            self._file_descriptors[tar_path] = file_descriptor
        data = os.pread(file_descriptor, size, offset)
        if len(data) != size:
            raise IOError(f"Short read from {tar_path} at offset {offset}: got {len(data)} of {size} bytes")
        return data

    def close(self) -> None:
        """Closes the archives."""
        for file_descriptor in self._file_descriptors.values():
            try:
                os.close(file_descriptor)
            except OSError:
                pass
        self._file_descriptors.clear()

    def __del__(self) -> None:
        self.close()


def main() -> int:
    """Builds the index of the archives given on the command line."""
    parser = argparse.ArgumentParser(description="Index nuPlan sensor archives for conversion with dataset=nuplan-tar.")
    parser.add_argument("archives", nargs="+", type=Path, help="The sensor archives (tar files with a .zip name).")
    parser.add_argument("--force", action="store_true", help="Rebuild indexes that are up to date.")
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    num_failed = 0
    for tar_path in arguments.archives:
        try:
            build_tar_index(tar_path, force=arguments.force)
        except Exception:
            logger.exception("Failed to index %s", tar_path)
            num_failed += 1
    return 1 if num_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
