"""Downloads the Physical AI AV sensor files of NuRec scenes from the Hugging Face Hub.

Every NuRec scene is a reconstruction of one clip of the Physical AI AV dataset
(https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles), which ships that clip's
cameras and lidar. The Hub packs the files of about 100 clips into one archive per sensor
(``camera/<camera>/<camera>.chunk_NNNN.zip`` and so on), and most of the archive is lidar: a
chunk's lidar archive alone is about 32 GB. Fetching the archives of the NuRec clips would
therefore download tens of terabytes for a few hundred gigabytes of files.

The archives store their members uncompressed, and the Hub serves HTTP range requests, so this
downloader reads an archive's central directory from its end and then copies only the members of
the NuRec clips. The files land in the flat Physical AI AV layout, which the conversion reads:

    <output_dir>/clip_index.parquet
    <output_dir>/calibration/<table>/<table>.chunk_NNNN.parquet
    <output_dir>/camera/<camera>/<clip_id>.<camera>.mp4
    <output_dir>/camera/<camera>/<clip_id>.<camera>.timestamps.parquet
    <output_dir>/lidar/lidar_top_360fov/<clip_id>.lidar_top_360fov.parquet
    <output_dir>/labels/egomotion/<clip_id>.egomotion.parquet

The dataset is gated: the Hugging Face account must have accepted NVIDIA's license, and its token
is read from ``HF_TOKEN`` or passed as ``hf_token``.
"""

from __future__ import annotations

import concurrent.futures
import io
import logging
import shutil
import struct
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from py123d.parser.base_downloader import BaseDownloader
from py123d.parser.nurec.nurec_download import NuRecDownloader, resolve_hf_token
from py123d.parser.physical_ai_av.utils.physical_ai_av_constants import PHYSICAL_AI_AV_CAMERA_ID_MAPPING

logger = logging.getLogger(__name__)

NUREC_PAI_REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
_REPO_TYPE = "dataset"
_LIDAR_NAME = "lidar_top_360fov"
_CALIBRATION_TABLES = ("camera_intrinsics", "sensor_extrinsics", "vehicle_dimensions")
_LOCAL_FILE_HEADER_SIZE = 30
_COPY_CHUNK_SIZE = 8 * 1024 * 1024


class NuRecPAIDownloader(BaseDownloader):
    """Downloads the Physical AI AV files of NuRec scenes, clip by clip.

    The scenes are selected like in :class:`~py123d.parser.nurec.nurec_download.NuRecDownloader`:
    by split, narrowed down by ``sequence_ids`` or ``num_sequences``. Of those, the scenes that are
    Physical AI AV clips are fetched; the others have no sensor data and are reported.
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        hf_repo_id: str = NUREC_PAI_REPO_ID,
        hf_token: Optional[str] = None,
        splits: Optional[List[str]] = None,
        scene_lists: Optional[Dict[str, List[str]]] = None,
        sequence_ids: Optional[List[str]] = None,
        num_sequences: Optional[int] = None,
        sample_random: bool = False,
        seed: int = 0,
        cameras: Optional[List[str]] = None,
        include_lidar: bool = True,
        max_workers: int = 8,
        dry_run: bool = False,
    ) -> None:
        """Initializes the downloader.

        :param output_dir: Destination directory, the Physical AI AV data root of the conversion.
        :param hf_repo_id: Hugging Face repo ID of the Physical AI AV dataset.
        :param hf_token: HF access token, defaults to ``$HF_TOKEN``.
        :param splits: NuRec splits to fetch the clips of, defaults to ``nurec-curated_train`` and
            ``nurec-curated_val``.
        :param scene_lists: Mapping of split name to scene names, each ``"<release>/<scene uuid>"``.
            Defaults to the bundled train/val lists.
        :param sequence_ids: Narrows the splits down to these scenes, each given as a scene uuid or
            as ``"<release>/<scene uuid>"``. Mutually exclusive with ``num_sequences``.
        :param num_sequences: The first N scenes of each split, or N random ones when ``sample_random`` is set.
        :param sample_random: Picks the ``num_sequences`` scenes at random.
        :param seed: Random seed used when ``sample_random`` is set.
        :param cameras: Camera names to fetch, defaults to all seven.
        :param include_lidar: Whether to fetch the lidar, the largest file of a clip.
        :param max_workers: Archives read in parallel.
        :param dry_run: Logs the plan without downloading.
        """
        if cameras is not None:
            unknown = set(cameras) - set(PHYSICAL_AI_AV_CAMERA_ID_MAPPING)
            if unknown:
                raise ValueError(
                    f"Unknown cameras {sorted(unknown)}; valid: {sorted(PHYSICAL_AI_AV_CAMERA_ID_MAPPING)}"
                )
        self.output_dir: Optional[Path] = Path(output_dir) if output_dir is not None else None
        self.dry_run: bool = dry_run
        self.hf_repo_id: str = hf_repo_id
        self.hf_token: Optional[str] = resolve_hf_token(hf_token)
        self.cameras: List[str] = list(cameras) if cameras is not None else list(PHYSICAL_AI_AV_CAMERA_ID_MAPPING)
        self.include_lidar: bool = include_lidar
        self.max_workers: int = max_workers
        self._http_session: Optional[requests.Session] = None
        # Selects the scenes; the NuRec repo is only contacted for a split that covers a whole release.
        self._scenes = NuRecDownloader(
            hf_token=hf_token,
            splits=splits,
            scene_lists=scene_lists,
            sequence_ids=sequence_ids,
            num_sequences=num_sequences,
            sample_random=sample_random,
            seed=seed,
        )
        if self.hf_token is None:
            logger.warning("No HF token configured. Physical AI AV is gated; set $HF_TOKEN if downloads fail.")

    @property
    def splits(self) -> List[str]:
        """The NuRec splits the clips are fetched for."""
        return self._scenes.splits

    def selected_clips(self) -> List[Tuple[str, int]]:
        """The (clip id, chunk) pairs of the selected scenes that are Physical AI AV clips.

        Reads ``clip_index.parquet``, which is downloaded into ``output_dir`` first.
        """
        assert self.output_dir is not None, "NuRecPAIDownloader.output_dir must be set."
        scene_uuids = sorted({uuid for uuids in self._scenes.resolve_release_sequence_ids().values() for uuid in uuids})
        clip_index = pd.read_parquet(self._fetch_repo_file("clip_index.parquet"))
        chunk_by_clip = {str(clip_id): int(chunk) for clip_id, chunk in clip_index["chunk"].items()}
        clips = [(uuid, chunk_by_clip[uuid]) for uuid in scene_uuids if uuid in chunk_by_clip]
        logger.info(
            "NuRec scenes selected: %d, of which %d are Physical AI AV clips in %d chunks",
            len(scene_uuids),
            len(clips),
            len({chunk for _, chunk in clips}),
        )
        return clips

    def download(self) -> None:
        """Inherited, see superclass."""
        assert self.output_dir is not None, "NuRecPAIDownloader.output_dir must be set before download()."
        self.output_dir.mkdir(parents=True, exist_ok=True)
        clips = self.selected_clips()

        # The members to copy, grouped by the archive they are in.
        members_by_archive: Dict[Tuple[str, str, int], List[str]] = defaultdict(list)
        for clip_id, chunk in clips:
            for camera in self.cameras:
                members_by_archive[("camera", camera, chunk)] += [
                    f"{clip_id}.{camera}.mp4",
                    f"{clip_id}.{camera}.timestamps.parquet",
                ]
            if self.include_lidar:
                members_by_archive[("lidar", _LIDAR_NAME, chunk)].append(f"{clip_id}.{_LIDAR_NAME}.parquet")
            members_by_archive[("labels", "egomotion", chunk)].append(f"{clip_id}.egomotion.parquet")
        chunks = sorted({chunk for _, chunk in clips})
        num_members = sum(len(members) for members in members_by_archive.values())

        logger.info("Physical AI AV repo:   %s", self.hf_repo_id)
        logger.info("Target directory:      %s", self.output_dir)
        logger.info("Cameras:               %s, lidar: %s", self.cameras, "yes" if self.include_lidar else "no")
        logger.info(
            "Files to fetch:        %d from %d archives, plus the calibration of %d chunks",
            num_members,
            len(members_by_archive),
            len(chunks),
        )
        if self.dry_run:
            logger.info("dry_run=True, not downloading.")
            return
        if not clips:
            logger.warning("None of the selected NuRec scenes is a Physical AI AV clip, nothing to download.")
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            calibration_futures = [
                pool.submit(self._fetch_repo_file, f"calibration/{table}/{table}.chunk_{chunk:04d}.parquet")
                for chunk in chunks
                for table in _CALIBRATION_TABLES
            ]
            for future in concurrent.futures.as_completed(calibration_futures):
                future.result()
            logger.info("Calibration of %d chunks fetched", len(chunks))

            num_done = 0
            futures = [
                pool.submit(self._download_archive_members, group, sensor, chunk, members)
                for (group, sensor, chunk), members in sorted(members_by_archive.items())
            ]
            for future in concurrent.futures.as_completed(futures):
                num_done += future.result()
                if num_done % 100 < future.result():
                    logger.info("Fetched %d of %d files", num_done, num_members)
        logger.info("Physical AI AV download complete: %d files under %s", num_done, self.output_dir)

    @property
    def _session(self) -> requests.Session:
        """The HTTP session, shared by the threads."""
        if self._http_session is None:
            self._http_session = _make_session()
        return self._http_session

    def _fetch_repo_file(self, filename: str) -> Path:
        """Downloads a plain file of the repo into ``output_dir``, keeping its path, unless it is there already."""
        assert self.output_dir is not None
        destination = self.output_dir / filename
        if destination.exists():
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
        response = self._session.get(self._repo_file_url(filename), headers=headers, stream=True, timeout=60)
        response.raise_for_status()
        partial = destination.with_name(destination.name + ".part")
        with open(partial, "wb") as file:
            for chunk in response.iter_content(chunk_size=_COPY_CHUNK_SIZE):
                file.write(chunk)
        partial.replace(destination)
        return destination

    def _repo_file_url(self, filename: str) -> str:
        """URL of a file in the repo."""
        from huggingface_hub import hf_hub_url

        return hf_hub_url(self.hf_repo_id, filename, repo_type=_REPO_TYPE)

    def _download_archive_members(self, group: str, sensor: str, chunk: int, members: Sequence[str]) -> int:
        """Copies the members of one chunk archive that are not on disk yet. Returns the number of members."""
        assert self.output_dir is not None
        destination_dir = self.output_dir / group / sensor
        url = self._archive_url(group, sensor, chunk)
        archive: Optional[_RemoteArchive] = None
        for member in members:
            destination = destination_dir / member
            if archive is None:
                archive = _RemoteArchive(self._session, url, self.hf_token)
            if destination.exists() and destination.stat().st_size == archive.members[member].file_size:
                continue
            archive.extract(member, destination)
        return len(members)

    def _archive_url(self, group: str, sensor: str, chunk: int) -> str:
        """URL of a chunk archive in the repo."""
        from huggingface_hub import hf_hub_url

        return hf_hub_url(self.hf_repo_id, f"{group}/{sensor}/{sensor}.chunk_{chunk:04d}.zip", repo_type=_REPO_TYPE)


def _make_session() -> requests.Session:
    """A session that retries the Hub's rate limits and transient errors with a backoff."""
    session = requests.Session()
    retry = Retry(total=8, backoff_factor=2.0, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=None)
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


class _RemoteArchive:
    """One zip archive on the Hub, read through HTTP range requests."""

    def __init__(self, session: requests.Session, url: str, token: Optional[str]) -> None:
        """
        :param session: HTTP session.
        :param url: URL of the archive in the repo; it redirects to the storage that serves the bytes.
        :param token: HF access token, sent to the repo URL only.
        """
        self._session = session
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        response = session.head(url, headers=headers, allow_redirects=True, timeout=60)
        response.raise_for_status()
        # The signed storage URL needs no token, and keeps the member reads off the Hub's API.
        self._url = response.url
        self._size = int(response.headers["Content-Length"])
        with zipfile.ZipFile(io.BufferedReader(_RangeFile(self), buffer_size=256 * 1024)) as archive:  # type: ignore[arg-type]
            self.members: Dict[str, zipfile.ZipInfo] = {info.filename: info for info in archive.infolist()}

    def read(self, start: int, size: int) -> bytes:
        """The bytes ``[start, start + size)`` of the archive."""
        response = self._session.get(self._url, headers={"Range": f"bytes={start}-{start + size - 1}"}, timeout=60)
        response.raise_for_status()
        return response.content

    def extract(self, member: str, destination: Path) -> None:
        """Copies a member to ``destination``, through a temporary file so that a partial copy is never mistaken
        for a complete one."""
        info = self.members[member]
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")
        with open(partial, "wb") as file:
            if info.compress_type != zipfile.ZIP_STORED:  # not how the dataset is published
                with zipfile.ZipFile(io.BufferedReader(_RangeFile(self))) as archive:  # type: ignore[arg-type]
                    with archive.open(info) as source:
                        shutil.copyfileobj(source, file, length=_COPY_CHUNK_SIZE)
            elif info.file_size > 0:
                # The local header in front of the data repeats the file name and has its own extra field.
                header = self.read(info.header_offset, _LOCAL_FILE_HEADER_SIZE)
                name_size, extra_size = struct.unpack("<HH", header[26:30])
                data_offset = info.header_offset + _LOCAL_FILE_HEADER_SIZE + name_size + extra_size
                response = self._session.get(
                    self._url,
                    headers={"Range": f"bytes={data_offset}-{data_offset + info.file_size - 1}"},
                    stream=True,
                    timeout=60,
                )
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=_COPY_CHUNK_SIZE):
                    file.write(chunk)
        if partial.stat().st_size != info.file_size:
            partial.unlink()
            raise IOError(f"{member}: received {partial.stat().st_size} of {info.file_size} bytes from {self._url}")
        partial.replace(destination)


class _RangeFile(io.RawIOBase):
    """Read-only, seekable file object over a remote archive, for :mod:`zipfile` to read its directory."""

    def __init__(self, archive: _RemoteArchive) -> None:
        super().__init__()
        self._archive = archive
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, position: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._position = position
        elif whence == io.SEEK_CUR:
            self._position += position
        elif whence == io.SEEK_END:
            self._position = self._archive._size + position
        self._position = max(0, min(self._position, self._archive._size))
        return self._position

    def readinto(self, buffer) -> int:  # type: ignore[override]
        size = min(len(buffer), self._archive._size - self._position)
        if size <= 0:
            return 0
        data = self._archive.read(self._position, size)
        buffer[: len(data)] = data
        self._position += len(data)
        return len(data)
