"""Byte-offset index over the nuPlan sensor archives.

The nuPlan sensor archives of the full splits (``nuplan-v1.1_<split>_{camera,lidar}_<n>.zip``) are uncompressed
tar archives with a zip name. Each member can therefore be read with one ``pread`` at its recorded offset: no
extraction, no mount, and one index file per archive instead of millions of extracted files.

:func:`build_tar_index` reads the headers of an archive once (the payloads are skipped) and writes two files next
to it::

    <archive>.index.parquet   columns: log_name, relative_path, offset, size
    <archive>.logs.txt        the distinct log names, one per line

``relative_path`` omits the archive's wrapper folder, so it equals the ``filename`` / ``filename_jpg`` columns of the
nuPlan log databases (for example ``<log_name>/CAM_L0/<token>.jpg``).

:class:`NuplanTarIndex` collects the ``.logs.txt`` files of one or more archive folders into a catalog from log
name to archives; :func:`load_log_members` then reads only the rows of one log from the index files (predicate
pushdown), so the memory of a worker is bounded by one log.

Index archives from the shell (one process per archive; the headers of one tar archive form a sequential chain)::

    python -m py123d.parser.nuplan.nuplan_tar_index <archive> [<archive> ...]
"""

from __future__ import annotations

import argparse
import logging
import os
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple, Union

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

INDEX_SUFFIX = ".index.parquet"
LOGS_SUFFIX = ".logs.txt"

# One member: (archive path, byte offset, byte size).
TarMember = Tuple[Path, int, int]


def build_tar_index(tar_path: Union[Path, str], force: bool = False) -> Path:
    """Read the headers of one uncompressed tar archive and write its index files.

    :param tar_path: The archive.
    :param force: Rebuild an index that is newer than the archive.
    :return: The path of the ``.index.parquet`` file.
    """
    tar_path = Path(tar_path)
    index_path = tar_path.with_name(tar_path.name + INDEX_SUFFIX)
    logs_path = tar_path.with_name(tar_path.name + LOGS_SUFFIX)

    if not force and index_path.exists() and logs_path.exists():
        if index_path.stat().st_mtime >= tar_path.stat().st_mtime:
            logger.info("Index up to date: %s", index_path.name)
            return index_path

    log_names: List[str] = []
    relative_paths: List[str] = []
    offsets: List[int] = []
    sizes: List[int] = []

    # Unbuffered reads without read-ahead: a buffered file object reads st_blksize bytes per call (4 MiB on CephFS),
    # so reading the 512-byte headers, which lie about 1.4 MB apart, would read almost the whole archive.
    with open(tar_path, "rb", buffering=0) as raw, tarfile.open(fileobj=raw, mode="r:") as tf:
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(raw.fileno(), 0, 0, os.POSIX_FADV_RANDOM)
        member = tf.next()
        while member is not None:
            if member.isfile():
                # Without the wrapper folder (e.g. nuplan-v1.1_test_camera_0/), the path equals the database
                # filename columns: <log>/<sensor folder>/<file>.
                parts = member.name.split("/")
                if len(parts) >= 3:
                    log_names.append(parts[1])
                    relative_paths.append("/".join(parts[1:]))
                    offsets.append(member.offset_data)
                    sizes.append(member.size)
            # tarfile keeps every TarInfo in tf.members; clearing the list keeps the memory constant (an archive
            # has up to millions of members).
            tf.members = []  # type: ignore[assignment]
            member = tf.next()

    table = pa.table(
        {
            "log_name": pa.array(log_names, type=pa.string()),
            "relative_path": pa.array(relative_paths, type=pa.string()),
            "offset": pa.array(offsets, type=pa.int64()),
            "size": pa.array(sizes, type=pa.int64()),
        }
    )
    # Sorted by log, the rows of one log form contiguous row groups, which makes per-log reads cheap.
    table = table.sort_by("log_name")

    tmp_index = index_path.with_suffix(".tmp")
    pq.write_table(table, tmp_index, row_group_size=64_000)
    tmp_index.replace(index_path)

    distinct_logs = sorted(set(log_names))
    tmp_logs = logs_path.with_suffix(".tmp")
    tmp_logs.write_text("\n".join(distinct_logs) + ("\n" if distinct_logs else ""))
    tmp_logs.replace(logs_path)

    logger.info("Indexed %s: %d members, %d logs", tar_path.name, len(relative_paths), len(distinct_logs))
    return index_path


class TarMemberReader:
    """Reads members of uncompressed tar archives with ``pread`` on cached file descriptors.

    Threads may share one reader (``pread`` does not move the file position); each worker process creates its own
    reader (it is not picklable).
    """

    def __init__(self) -> None:
        self._fds: Dict[Path, int] = {}

    def _fd(self, tar_path: Path) -> int:
        fd = self._fds.get(tar_path)
        if fd is None:
            fd = os.open(tar_path, os.O_RDONLY)
            self._fds[tar_path] = fd
        return fd

    def prefetch(self, tar_path: Path, offset: int, size: int) -> None:
        """Ask the kernel to read a member into the page cache in the background (``POSIX_FADV_WILLNEED``), so a
        later :meth:`read` does not wait for the storage. Announcing the members of the next steps keeps several
        requests in flight per process instead of one synchronous read at a time."""
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(self._fd(tar_path), offset, size, os.POSIX_FADV_WILLNEED)

    def read(self, tar_path: Path, offset: int, size: int) -> bytes:
        fd = self._fd(tar_path)
        data = os.pread(fd, size, offset)
        if len(data) != size:
            raise IOError(f"Short read from {tar_path} at offset {offset}: {len(data)} of {size} bytes")
        return data

    def close(self) -> None:
        for fd in self._fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds.clear()

    def __del__(self) -> None:
        self.close()


class NuplanTarIndex:
    """Catalog of the indexed sensor archives in one or more folders."""

    def __init__(self, archive_roots: Union[Path, str, Sequence[Union[Path, str]]]) -> None:
        roots = [archive_roots] if isinstance(archive_roots, (str, Path)) else list(archive_roots)
        self._archive_roots = [Path(root) for root in roots]
        self._log_to_archives: Dict[str, List[Path]] = {}

        for root in self._archive_roots:
            unindexed = [
                p.name
                for p in sorted(root.glob("*_camera_*")) + sorted(root.glob("*_lidar_*"))
                if not p.name.endswith((INDEX_SUFFIX, LOGS_SUFFIX, ".tmp", ".part", ".part.done"))
                and not p.with_name(p.name + INDEX_SUFFIX).exists()
            ]
            if unindexed:
                logger.info(
                    "%d sensor archives in %s have no index file and are not used: %s",
                    len(unindexed),
                    root,
                    ", ".join(unindexed[:5]) + ("..." if len(unindexed) > 5 else ""),
                )
            for index_path in sorted(root.glob(f"*{INDEX_SUFFIX}")):
                tar_path = index_path.with_name(index_path.name[: -len(INDEX_SUFFIX)])
                logs_path = index_path.with_name(tar_path.name + LOGS_SUFFIX)
                if logs_path.exists():
                    log_names: Iterable[str] = logs_path.read_text().splitlines()
                else:  # the log_name column of the index
                    log_names = pq.read_table(index_path, columns=["log_name"])["log_name"].unique().to_pylist()
                for log_name in log_names:
                    if log_name:
                        archives = self._log_to_archives.setdefault(log_name, [])
                        if all(a.name != tar_path.name for a in archives):  # one archive indexed in two folders
                            archives.append(tar_path)

    @property
    def log_names(self) -> Set[str]:
        return set(self._log_to_archives)

    def archives_for_log(self, log_name: str) -> List[Path]:
        return list(self._log_to_archives.get(log_name, []))


def load_log_members(log_name: str, archives: Iterable[Path]) -> Dict[str, TarMember]:
    """The members of one log in the index files of the given archives.

    :return: relative_path -> (archive path, offset, size)
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Build byte-offset indexes of nuPlan sensor tar archives.")
    parser.add_argument("archives", nargs="+", type=Path, help="tar archives (named .zip in the nuPlan bucket)")
    parser.add_argument("--force", action="store_true", help="rebuild an index that is newer than its archive")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    failed = 0
    for tar_path in args.archives:
        try:
            build_tar_index(tar_path, force=args.force)
        except Exception:
            logger.exception("Failed to index %s", tar_path)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
