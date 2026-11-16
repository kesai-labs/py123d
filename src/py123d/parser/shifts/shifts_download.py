"""Download utilities for the Shifts vehicle motion prediction dataset.

The canonical Shifts partition is published as plain (uncompressed) ``.tar`` files on Yandex Object
Storage, each wrapping one gzipped tar per split (``train.tar.gz``, ``development.tar.gz``,
``evaluation.tar.gz``) next to pre-rendered feature maps that py123d does not use. The storage
answers HTTP range requests, so a split is fetched by streaming only its member out of the outer
archive, and because that member is a sequential gzip stream of one protobuf file per scene, the
download can stop after the first ``num_scenes`` scenes.
"""

from __future__ import annotations

import logging
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, List, Optional, Sequence, Tuple

import requests

from py123d.parser.base_downloader import BaseDownloader
from py123d.parser.shifts.shifts_constants import (
    SHIFTS_AVAILABLE_SPLITS,
    SHIFTS_BASE_URL,
    SHIFTS_SPLIT_TO_ARCHIVE,
    SHIFTS_SPLIT_TO_DIRECTORY,
    ShiftsArchiveMember,
)

logger = logging.getLogger(__name__)

TAR_BLOCK_SIZE = 512
COMPLETE_MARKER = ".complete"
"""Written into a split directory once its whole archive member was extracted."""

_CONNECT_TIMEOUT_S = 30
_READ_TIMEOUT_S = 300


def _http_get(url: str, headers: dict, stream: bool) -> requests.Response:
    """Single seam for HTTP access, so tests can replace it."""
    response = requests.get(url, headers=headers, stream=stream, timeout=(_CONNECT_TIMEOUT_S, _READ_TIMEOUT_S))
    response.raise_for_status()
    return response


def parse_tar_header(block: bytes) -> Optional[Tuple[str, int, str]]:
    """Parse one 512-byte tar header into ``(name, size, typeflag)``.

    Handles the ustar name prefix and the GNU base-256 size encoding that tar uses for members
    larger than 8 GB (the Shifts train archives are). Returns ``None`` for an end-of-archive block.
    """
    if len(block) < TAR_BLOCK_SIZE or block[0] == 0:
        return None
    name = block[0:100].rstrip(b"\0").decode("utf-8", errors="replace")
    prefix = block[345:500].rstrip(b"\0").decode("utf-8", errors="replace")
    if prefix:
        name = f"{prefix}/{name}"
    size_field = block[124:136]
    if size_field[0] & 0x80:
        size = int.from_bytes(size_field[1:], "big")
    else:
        size = int(size_field.rstrip(b"\0 ") or b"0", 8)
    typeflag = block[156:157].decode("ascii", errors="replace")
    return name, size, typeflag


class ShiftsDownloader(BaseDownloader):
    """Streams the canonical Shifts scene archives from Yandex Object Storage."""

    def __init__(
        self,
        output_dir: Optional[Path],
        splits: Optional[Sequence[str]] = None,
        num_scenes: Optional[int] = None,
        overwrite: bool = False,
        max_retries: int = 3,
        dry_run: bool = False,
    ) -> None:
        """Initialize the Shifts downloader.

        :param output_dir: Destination root. Scenes land in ``<output_dir>/<train|development|evaluation>/NNN/*.pb``.
        :param splits: py123d split names to fetch (``shifts_train``, ``shifts_val``, ``shifts_test``).
            Defaults to all three.
        :param num_scenes: Stop after the first ``num_scenes`` scenes of every split. ``None`` streams the
            whole split.
        :param overwrite: Re-download a split directory that already holds the requested scenes.
        :param max_retries: Attempts per split. A stream that breaks is restarted from its beginning,
            because gzip streams cannot be resumed mid-way.
        :param dry_run: When ``True``, log the plan without downloading.
        """
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.dry_run = dry_run
        self._splits: List[str] = list(splits) if splits is not None else list(SHIFTS_AVAILABLE_SPLITS)
        self.num_scenes = num_scenes
        self.overwrite = overwrite
        self.max_retries = max_retries

        for split in self._splits:
            if split not in SHIFTS_SPLIT_TO_ARCHIVE:
                raise ValueError(f"Unknown Shifts split {split!r}. Available: {SHIFTS_AVAILABLE_SPLITS}")
        if num_scenes is not None and num_scenes <= 0:
            raise ValueError("num_scenes must be a positive integer or None.")

    def download(self) -> None:
        """Fetch the selected splits into :attr:`output_dir`."""
        if self.output_dir is None:
            raise ValueError("ShiftsDownloader.output_dir must be set before calling download().")

        for split in self._splits:
            member = SHIFTS_SPLIT_TO_ARCHIVE[split]
            split_dir = self.output_dir / SHIFTS_SPLIT_TO_DIRECTORY[split]
            scenes = "all scenes" if self.num_scenes is None else f"first {self.num_scenes} scenes"
            logger.info(
                "Shifts download plan: %s ← %s%s!%s (%.1f GB compressed, %s) → %s",
                split,
                SHIFTS_BASE_URL,
                member.outer_tar,
                member.member_name,
                member.size / 1e9,
                scenes,
                split_dir,
            )
            if self.dry_run:
                continue
            if self._is_satisfied(split_dir) and not self.overwrite:
                logger.info("[skip] %s already holds the requested scenes", split_dir)
                continue
            self._download_split(member, split_dir)

    def _is_satisfied(self, split_dir: Path) -> bool:
        """Whether ``split_dir`` already holds what this download would produce."""
        if not split_dir.is_dir():
            return False
        if (split_dir / COMPLETE_MARKER).exists():
            return True
        if self.num_scenes is not None:
            return _count_scene_files(split_dir) >= self.num_scenes
        return False

    def _download_split(self, member: ShiftsArchiveMember, split_dir: Path) -> None:
        data_offset, size = self._locate_member(member)
        url = SHIFTS_BASE_URL + member.outer_tar
        headers = {"Range": f"bytes={data_offset}-{data_offset + size - 1}"}

        for attempt in range(1, self.max_retries + 1):
            if split_dir.exists():
                shutil.rmtree(split_dir)
            split_dir.mkdir(parents=True, exist_ok=True)
            try:
                response = _http_get(url, headers=headers, stream=True)
                if response.status_code != 206:
                    raise RuntimeError(f"Expected a partial-content response for {url}, got {response.status_code}")
                try:
                    num_written, completed = self._extract_scene_stream(response.raw, split_dir)
                finally:
                    response.close()
            except (requests.RequestException, tarfile.TarError, EOFError, OSError) as exc:
                logger.warning(
                    "[retry %d/%d] %s!%s: %s", attempt, self.max_retries, member.outer_tar, member.member_name, exc
                )
                if attempt == self.max_retries:
                    raise RuntimeError(
                        f"Failed to stream {member.member_name} from {url} after {self.max_retries} attempts."
                    ) from exc
                continue

            if completed:
                (split_dir / COMPLETE_MARKER).touch()
            logger.info("[done] %d scenes → %s%s", num_written, split_dir, "" if completed else " (truncated)")
            return

    def _extract_scene_stream(self, fileobj: Any, split_dir: Path) -> Tuple[int, bool]:
        """Extract ``*.pb`` members from a gzipped tar stream until it ends or ``num_scenes`` is reached.

        :param fileobj: Binary stream of the gzipped tar, e.g. the raw urllib3 response of a range request.
        :param split_dir: Directory the scene files are written to.

        :return: ``(number of scene files written, whether the whole stream was consumed)``.
        """
        num_written = 0
        with tarfile.open(fileobj=fileobj, mode="r|gz") as archive:
            for info in archive:
                if not info.isfile() or not info.name.endswith(".pb"):
                    continue
                relative = _safe_relative_path(info.name)
                if relative is None:
                    logger.warning("[skip] unexpected member path in archive: %s", info.name)
                    continue
                source = archive.extractfile(info)
                if source is None:
                    continue
                destination = split_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
                num_written += 1
                if self.num_scenes is not None and num_written >= self.num_scenes:
                    return num_written, False
        return num_written, True

    def _locate_member(self, member: ShiftsArchiveMember) -> Tuple[int, int]:
        """Return ``(data offset, size)`` of ``member`` in its outer tar, verified against the live archive.

        One range request reads the member's header at the expected offset. If the archive was
        re-uploaded with a different layout, the outer tar's headers are walked from the start.
        """
        url = SHIFTS_BASE_URL + member.outer_tar
        header_offset = member.data_offset - TAR_BLOCK_SIZE
        block = _http_get(
            url, headers={"Range": f"bytes={header_offset}-{header_offset + TAR_BLOCK_SIZE - 1}"}, stream=False
        ).content
        parsed = parse_tar_header(block)
        if parsed is not None and parsed[0] == member.member_name and parsed[1] == member.size:
            return member.data_offset, member.size

        logger.warning(
            "Archive layout of %s differs from the expected one (%r); walking its headers.", member.outer_tar, parsed
        )
        for name, size, data_offset in _walk_outer_tar(url):
            if name == member.member_name:
                return data_offset, size
        raise RuntimeError(f"Member {member.member_name!r} not found in {url}.")


def _walk_outer_tar(url: str) -> Iterator[Tuple[str, int, int]]:
    """Yield ``(name, size, data offset)`` for every member of an uncompressed tar, one range request each."""
    offset = 0
    while True:
        block = _http_get(url, headers={"Range": f"bytes={offset}-{offset + TAR_BLOCK_SIZE - 1}"}, stream=False).content
        parsed = parse_tar_header(block)
        if parsed is None:
            return
        name, size, _ = parsed
        yield name, size, offset + TAR_BLOCK_SIZE
        offset += TAR_BLOCK_SIZE + ((size + TAR_BLOCK_SIZE - 1) // TAR_BLOCK_SIZE) * TAR_BLOCK_SIZE


def _safe_relative_path(member_name: str) -> Optional[Path]:
    """Turn an archive member name such as ``./000/000000.pb`` into a relative path, rejecting escapes."""
    parts = [part for part in PurePosixPath(member_name).parts if part not in ("", ".", "/")]
    if not parts or any(part == ".." for part in parts):
        return None
    return Path(*parts)


def _count_scene_files(split_dir: Path) -> int:
    return sum(1 for _ in split_dir.glob("*/*.pb"))
