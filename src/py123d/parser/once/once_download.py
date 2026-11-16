"""Download utilities for the ONCE dataset.

ONCE is distributed through public Google Drive folders (https://once-for-auto-driving.github.io/download.html),
one tar per split and modality::

    train_infos.tar  train_lidar.tar  train_cam01.tar ... train_cam09.tar     (22 GB)
    val_infos.tar    val_lidar.tar    val_cam01.tar   ... val_cam09.tar       (13 GB)
    test_infos.tar   test_lidar.tar   test_cam01.tar  ... test_cam09.tar      (31 GB)

Every archive unpacks into the same ``data/<sequence>/...`` tree, so they are extracted into one root. The files
need no account, but Google Drive limits how much of a popular file can be downloaded per day ("quota exceeded").
Bounded range requests keep being served for a while after plain downloads are refused, so archives are fetched in
256 MB ranges; a cut-off download can be rerun later and resumes from the partial file. The raw (unlabelled) splits are multi-part archives and are not supported here.
"""

from __future__ import annotations

import logging
import re
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import requests

from py123d.parser.base_downloader import BaseDownloader
from py123d.parser.once.utils.once_constants import ONCE_CAMERA_NAMES, ONCE_SEQUENCES_BY_SPLIT, ONCE_SPLITS

logger = logging.getLogger(__name__)

GOOGLE_DRIVE_DOWNLOAD_URL = "https://drive.google.com/uc"
_ARCHIVE_DIR = "_archives"
_CHUNK_BYTES = 1 << 20
_RANGE_BYTES = 256 << 20

# Google Drive file ids of the archives, as listed in the official download folders (checked 2026-10-11).
ONCE_ARCHIVES: Dict[str, Dict[str, str]] = {
    "once_train": {
        "infos": "1Qa6JYbp7SEZtq7wI0Sw7bvhLc-CZdX8I",
        "lidar": "1fy_R1TOKOyvMtCgwR5YwpZtiih0c_oD5",
        "cam01": "1bOVcrja8AZi8fmeZwYS98Y9mfDHU7wYl",
        "cam03": "1K5Y4eV4TQJiFYuBOYglVPoBxLIHjn5rA",
        "cam05": "1P2N3uqbuFQ-ACJJv9bt8Pa15jnncrWyc",
        "cam06": "14ARNQu7-Rscr1vta-pv35z9kfYnPeV0X",
        "cam07": "1n9t3nmy84bTJHnUOF2KUyfFOoWnyGJCi",
        "cam08": "1rz239PKqkFmt9m0r95AK7upjgKm1jVrP",
        "cam09": "10CajrJ02H4YEwWM87N1O1mWzYjdV18P3",
    },
    "once_val": {
        "infos": "1slhW1vAR7Ps1TL-QLL4SHpMV6aN6vUse",
        "lidar": "1BQvbh7pdCayjoXKWPD5-_veYtBgvL_8N",
        "cam01": "1GTwLqPhj_63lC8avPR8Ul915Ct5JRbyt",
        "cam03": "1z73ARwGXSumfnzkzYU1eWVa-j2qJbkn2",
        "cam05": "13o7ubCpYtt92j42KkWK4uDC7ANHs1neO",
        "cam06": "1o5KhArGDXDdE3jQhGiOw8Vp2XsXHetc2",
        "cam07": "1vVOw1dm63JbRWX0HcfmUB03k7yQqHRBQ",
        "cam08": "1wIYm-gCK270Nk9qu99qmToXSJgZCVuNb",
        "cam09": "1Qe1UoDfTW1KaolVMXGBK9reTxgY8Ws3w",
    },
    "once_test": {
        "infos": "19R2MRZxpe-4VWJLuCQbcVTZkS4-LV1N3",
        "lidar": "1-6UwiBMlGjgMtQTo3NIVFC0Il_KU_gWv",
        "cam01": "1ySBHGDVusGHXmZap4M_yIz2L9-ft9t5j",
        "cam03": "15l_mmAgjObYDwc2bI8B8jOm_LC-tOcob",
        "cam05": "1UsyoANvqKEe96oUP3_cFAzH67I7yZc4L",
        "cam06": "1b1RWkWIuw7PERBc-OV9Z4H5BTBGCnyJd",
        "cam07": "1YgDsWLCxNRzfHrIMDex84Qu1oqzZhnNV",
        "cam08": "16T8p0dAOArj4awOGd9udrm-9yvkauZ5U",
        "cam09": "1CLf71E8RgbuMViSOFEK7eGaedr8lpikP",
    },
}

# Folder inside ``data/<sequence>/`` that an extracted archive creates (the infos archive creates the json).
_MODALITY_SUBDIR: Dict[str, str] = {"lidar": "lidar_roof", **{camera: camera for camera in ONCE_CAMERA_NAMES}}


def download_google_drive_file(file_id: str, dest: Path, max_attempts: int = 4) -> Path:
    """Downloads a public Google Drive file to ``dest``, resuming a partial ``dest.part`` if present.

    Large files answer the first request with a "cannot scan for viruses" page; its confirmation form is followed.
    Google Drive occasionally answers with a web page instead of the file (rate limiting, or the daily quota of a
    popular file); the download is retried with a growing pause before giving up.

    :param file_id: The Google Drive file id.
    :param dest: Destination path.
    :param max_attempts: Number of attempts before raising.
    :return: ``dest``.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return _download_google_drive_file_once(file_id, dest)
        except (RuntimeError, requests.RequestException) as error:
            if attempt == max_attempts:
                raise
            pause_s = 15 * attempt
            logger.warning("Google Drive download of %s failed (%s). Retrying in %d s ...", dest.name, error, pause_s)
            time.sleep(pause_s)
    raise AssertionError("unreachable")


def _download_google_drive_file_once(file_id: str, dest: Path) -> Path:
    """One attempt of :func:`download_google_drive_file`."""
    session = requests.Session()
    dest.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest.with_suffix(dest.suffix + ".part")
    offset = part_path.stat().st_size if part_path.exists() else 0

    response = session.get(
        GOOGLE_DRIVE_DOWNLOAD_URL, params={"export": "download", "id": file_id}, stream=True, timeout=60
    )
    response.raise_for_status()
    if "text/html" in response.headers.get("content-type", ""):
        # Large files answer with the "cannot scan for viruses" page; its form points at the actual download.
        url, params = _parse_google_drive_confirm_form(response.text)
    else:
        url, params = response.url, None
    response.close()

    # Google Drive refuses plain downloads of popular files ("quota exceeded") but serves bounded range requests,
    # so the file is fetched in chunks, which also makes an interrupted download resumable.
    with open(part_path, "ab") as fh:
        while True:
            headers = {"Range": f"bytes={offset}-{offset + _RANGE_BYTES - 1}"}
            response = session.get(url, params=params, headers=headers, stream=True, timeout=60)
            response.raise_for_status()
            if "text/html" in response.headers.get("content-type", ""):
                excerpt = _html_excerpt(response.text)
                response.close()
                raise RuntimeError(
                    f"Google Drive did not return file {file_id} but a web page ({excerpt}). The file is probably "
                    "over its download quota for today (retry later) or the download page changed."
                )
            if response.status_code != 206:
                # The range was ignored: the whole file follows.
                fh.seek(0)
                fh.truncate()
                offset = 0
            for chunk in response.iter_content(chunk_size=_CHUNK_BYTES):
                if chunk:
                    fh.write(chunk)
                    offset += len(chunk)
            content_range = response.headers.get("content-range", "")
            response.close()
            total = int(content_range.rsplit("/", 1)[-1]) if "/" in content_range else offset
            if offset >= total:
                break
    part_path.replace(dest)
    return dest


def _html_excerpt(html: str, max_chars: int = 160) -> str:
    """Returns the visible text of a web page, shortened, for error messages."""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())[:max_chars]


def _parse_google_drive_confirm_form(html: str) -> Tuple[str, Dict[str, str]]:
    """Extracts the action URL and hidden fields of the Google Drive download confirmation form."""
    action = re.search(r'<form[^>]*action="([^"]+)"', html)
    fields = re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', html)
    if action is None or not any(name == "id" for name, _ in fields):
        raise RuntimeError(
            f"Google Drive returned a web page without a download form ({_html_excerpt(html)}). The file is probably "
            "over its download quota for today (retry later) or the download page changed."
        )
    return action.group(1), dict(fields)


def extract_once_archive(archive_path: Path, output_dir: Path) -> int:
    """Extracts an ONCE tar (all members live under ``data/``) into ``output_dir`` and returns the member count."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r") as tar:
        members = [member for member in tar.getmembers() if member.isfile() or member.isdir()]
        for member in members:
            if not (member.name == "data" or member.name.startswith("data/")) or ".." in Path(member.name).parts:
                raise RuntimeError(f"Unexpected member {member.name!r} in {archive_path}.")
        try:
            tar.extractall(output_dir, members=members, filter="data")
        except TypeError:  # Python < 3.12 without the backported ``filter`` argument.
            tar.extractall(output_dir, members=members)
    return len(members)


class OnceDownloader(BaseDownloader):
    """Downloader for the labelled ONCE splits from the official Google Drive folders.

    Used by ``py123d-download dataset=once`` and by ``dataset=once-stream``, where the parser sets :attr:`splits`.
    Archives are downloaded to ``<output_dir>/_archives``, extracted into ``output_dir`` and deleted afterwards.
    Already extracted archives are skipped, so an interrupted download can be rerun.
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        splits: Optional[List[str]] = None,
        cameras: Optional[List[str]] = None,
        keep_archives: bool = False,
        max_workers: int = 2,
        dry_run: bool = False,
    ) -> None:
        """Initialize the ONCE downloader.

        :param output_dir: Directory that receives ``data/<sequence>/...``.
        :param splits: Splits to download (``once_train``, ``once_val``, ``once_test``). ``None`` downloads all.
        :param cameras: Camera folders to download (e.g. ``["cam01", "cam03"]``). ``None`` downloads all seven,
            ``[]`` downloads no cameras (lidar and annotations only).
        :param keep_archives: Keep the tar files in ``<output_dir>/_archives`` after extraction.
        :param max_workers: Number of archives downloaded in parallel.
        :param dry_run: Log the plan without downloading.
        """
        self.output_dir: Optional[Path] = Path(output_dir) if output_dir is not None else None
        self.dry_run: bool = dry_run
        self.splits: List[str] = list(splits) if splits is not None else list(ONCE_SPLITS)
        self.cameras: List[str] = list(cameras) if cameras is not None else list(ONCE_CAMERA_NAMES)
        self.keep_archives = keep_archives
        self.max_workers = max_workers

        for split in self.splits:
            assert split in ONCE_SPLITS, f"Split {split} is not available. Available splits: {ONCE_SPLITS}"
        for camera in self.cameras:
            assert camera in ONCE_CAMERA_NAMES, (
                f"Camera {camera} is not available. Available cameras: {ONCE_CAMERA_NAMES}"
            )

    def resolve_archives(self) -> List[Tuple[str, str, str]]:
        """Returns the selected (split, modality, file id) triples."""
        archives: List[Tuple[str, str, str]] = []
        for split in self.splits:
            for modality in ["infos", "lidar", *self.cameras]:
                archives.append((split, modality, ONCE_ARCHIVES[split][modality]))
        return archives

    def _is_extracted(self, split: str, modality: str) -> bool:
        """Checks whether every sequence of ``split`` has the folder (or json) that ``modality`` provides."""
        assert self.output_dir is not None
        for sequence in ONCE_SEQUENCES_BY_SPLIT[split]:
            sequence_dir = self.output_dir / "data" / sequence
            target = (
                sequence_dir / f"{sequence}.json" if modality == "infos" else sequence_dir / _MODALITY_SUBDIR[modality]
            )
            if not target.exists():
                return False
        return True

    def _fetch_and_extract(self, split: str, modality: str, file_id: str) -> None:
        assert self.output_dir is not None
        archive_name = f"{split.removeprefix('once_')}_{modality}.tar"
        archive_path = self.output_dir / _ARCHIVE_DIR / archive_name
        if not archive_path.exists():
            logger.info("ONCE: downloading %s ...", archive_name)
            download_google_drive_file(file_id, archive_path)
        logger.info("ONCE: extracting %s ...", archive_name)
        num_members = extract_once_archive(archive_path, self.output_dir)
        logger.info("ONCE: extracted %d entries from %s", num_members, archive_name)
        if not self.keep_archives:
            archive_path.unlink()

    def download(self) -> None:
        """Inherited, see superclass."""
        archives = self.resolve_archives()
        logger.info(
            "ONCE splits: %s, cameras: %s, archives: %d, target: %s",
            self.splits,
            self.cameras,
            len(archives),
            self.output_dir,
        )
        if self.dry_run:
            for split, modality, file_id in archives:
                logger.info("  %s %s (%s)", split, modality, file_id)
            return

        assert self.output_dir is not None, "OnceDownloader.output_dir must be set before download()."
        self.output_dir.mkdir(parents=True, exist_ok=True)
        pending = [archive for archive in archives if not self._is_extracted(archive[0], archive[1])]
        logger.info("ONCE: %d of %d archives already extracted", len(archives) - len(pending), len(archives))

        with ThreadPoolExecutor(max_workers=max(1, self.max_workers)) as executor:
            for _ in executor.map(lambda archive: self._fetch_and_extract(*archive), pending):
                pass

        archive_dir = self.output_dir / _ARCHIVE_DIR
        if archive_dir.exists() and not any(archive_dir.iterdir()):
            archive_dir.rmdir()
        logger.info("ONCE download complete: %s", self.output_dir)
