"""Download utilities for the nuScenes dataset (Motional).

The nuScenes dataset is hosted behind a gated CloudFront distribution fronted by an
AWS Cognito-authenticated API Gateway. This module automates the auth + per-archive
download flow; users still need a registered nuScenes account that has accepted the
dataset terms at https://www.nuscenes.org/nuscenes.

The Cognito ``USER_PASSWORD_AUTH`` flow, per-archive URL pattern, and catalog of
archive filenames + MD5 checksums used here are adapted from the MIT-licensed
community project ``li-xl/nuscenes-download``:

    Copyright (c) 2025 Xiang-Li Li
    https://github.com/li-xl/nuscenes-download — MIT License

The upstream script is a single-file, single-user download helper; this module
rebuilds the same capability around py123d's :class:`BaseDownloader` contract —
adding Hydra-instantiability, archive selection, parallel downloads with atomic
``.part`` writes, tarfile-safe extraction, and integration with the streaming
parser path so ``dataset=nuscenes-stream`` can materialize a chosen archive subset
into a managed temp directory at parser construction time.

Archive catalog (13 files, total ~700 GB)::

    v1.0-trainval_meta.tgz                (~400 MB, metadata tables for trainval)
    v1.0-trainval01_blobs.tgz ... _10_blobs.tgz  (~70 GB each, 10 splits of sensor data)
    v1.0-test_meta.tgz                    (~200 MB, metadata tables for test)
    v1.0-test_blobs.tgz                   (~30 GB, test sensor data)

This module exposes :class:`NuscenesDownloader` (Hydra-instantiable) which powers
two entry points:

1. ``py123d-download dataset=nuscenes`` — fetches the selected archives into a
   session-scoped ``tempfile.TemporaryDirectory``, extracts them into
   :attr:`output_dir` (typically ``$NUSCENES_DATA_ROOT``), and deletes the
   ``.tgz`` files when ``download()`` returns.
2. The ``NuScenesParser`` streaming path (``dataset=nuscenes-stream``) — materializes
   the selected archives into a managed temp directory at parser construction time
   so the nuScenes devkit can load its metadata tables; the temp directory is
   cleaned up when the parser goes out of scope.

The HD-map expansion add-on (``nuScenes-map-expansion-v1.3.zip``) is **not** in the
archive catalog above and is **not** fetched by this downloader — users who need
maps during streaming should download them separately and pass ``nuscenes_map_root``
on the parser.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import os
import random as _random_mod
import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from py123d.parser.base_downloader import BaseDownloader

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Upstream attribution — see module docstring for license/copyright.
# The Cognito ClientId, API gateway host, and archive catalog below are all verbatim
# from li-xl/nuscenes-download @ 2025-04.
# --------------------------------------------------------------------------------------
NUSCENES_COGNITO_URL = "https://cognito-idp.us-east-1.amazonaws.com/"
NUSCENES_COGNITO_CLIENT_ID = "7fq5jvs5ffs1c50hd3toobb3b9"
NUSCENES_API_GATEWAY = "https://o9k5xn5546.execute-api.us-east-1.amazonaws.com/v1/archives/v1.0"

NUSCENES_REGION_CHOICES: Tuple[str, ...] = ("us", "asia")

# --------------------------------------------------------------------------------------
# Archive catalog
# --------------------------------------------------------------------------------------
# The 13 core archives (trainval + test, metadata + sensor blobs) use MD5 checksums
# copied verbatim from li-xl/nuscenes-download (MIT © Xiang-Li Li, 2025). Extras —
# mini, HD map expansion, CAN bus — are served by the same API gateway but Motional
# does not publish MD5 checksums for them, so verification is skipped for those.


@dataclass(frozen=True)
class _ArchiveSpec:
    """Per-archive metadata consumed by the downloader and extractor dispatcher."""

    filename: str
    md5: Optional[str]  # None when the upstream does not publish a checksum
    extract_format: str  # "tgz" or "zip"
    approx_size_gb: float
    description: str
    category: str  # "core_meta" | "core_blob" | "mini" | "maps" | "canbus"


NUSCENES_ARCHIVE_CATALOG: Tuple[_ArchiveSpec, ...] = (
    # Core metadata (needed by every conversion; cheap)
    _ArchiveSpec(
        "v1.0-trainval_meta.tgz",
        "537d3954ec34e5bcb89a35d4f6fb0d4a",
        "tgz",
        0.4,
        "trainval metadata tables",
        "core_meta",
    ),
    _ArchiveSpec(
        "v1.0-test_meta.tgz", "b0263f5c41b780a5a10ede2da99539eb", "tgz", 0.2, "test metadata tables", "core_meta"
    ),
    # Core sensor blobs (the ~700 GB of sensor data)
    _ArchiveSpec(
        "v1.0-trainval01_blobs.tgz",
        "cbf32d2ea6996fc599b32f724e7ce8f2",
        "tgz",
        70.0,
        "trainval sensor data 1/10",
        "core_blob",
    ),
    _ArchiveSpec(
        "v1.0-trainval02_blobs.tgz",
        "aeecea4878ec3831d316b382bb2f72da",
        "tgz",
        70.0,
        "trainval sensor data 2/10",
        "core_blob",
    ),
    _ArchiveSpec(
        "v1.0-trainval03_blobs.tgz",
        "595c29528351060f94c935e3aaf7b995",
        "tgz",
        70.0,
        "trainval sensor data 3/10",
        "core_blob",
    ),
    _ArchiveSpec(
        "v1.0-trainval04_blobs.tgz",
        "b55eae9b4aa786b478858a3fc92fb72d",
        "tgz",
        70.0,
        "trainval sensor data 4/10",
        "core_blob",
    ),
    _ArchiveSpec(
        "v1.0-trainval05_blobs.tgz",
        "1c815ed607a11be7446dcd4ba0e71ed0",
        "tgz",
        70.0,
        "trainval sensor data 5/10",
        "core_blob",
    ),
    _ArchiveSpec(
        "v1.0-trainval06_blobs.tgz",
        "7273eeea36e712be290472859063a678",
        "tgz",
        70.0,
        "trainval sensor data 6/10",
        "core_blob",
    ),
    _ArchiveSpec(
        "v1.0-trainval07_blobs.tgz",
        "46674d2b2b852b7a857d2c9a87fc755f",
        "tgz",
        70.0,
        "trainval sensor data 7/10",
        "core_blob",
    ),
    _ArchiveSpec(
        "v1.0-trainval08_blobs.tgz",
        "37524bd4edee2ab99678909334313adf",
        "tgz",
        70.0,
        "trainval sensor data 8/10",
        "core_blob",
    ),
    _ArchiveSpec(
        "v1.0-trainval09_blobs.tgz",
        "a7fcd6d9c0934e4052005aa0b84615c0",
        "tgz",
        70.0,
        "trainval sensor data 9/10",
        "core_blob",
    ),
    _ArchiveSpec(
        "v1.0-trainval10_blobs.tgz",
        "31e795f2c13f62533c727119b822d739",
        "tgz",
        70.0,
        "trainval sensor data 10/10",
        "core_blob",
    ),
    _ArchiveSpec(
        "v1.0-test_blobs.tgz", "e065445b6019ecc15c70ad9d99c47b33", "tgz", 30.0, "test sensor data", "core_blob"
    ),
    # Extras — no published MD5. Endpoint must be verified empirically before committing.
    _ArchiveSpec("v1.0-mini.tgz", None, "tgz", 0.4, "mini dataset (smoketest)", "mini"),
    _ArchiveSpec("nuScenes-map-expansion-v1.3.zip", None, "zip", 0.1, "HD map expansion v1.3", "maps"),
    _ArchiveSpec("can_bus.zip", None, "zip", 0.1, "CAN bus expansion", "canbus"),
)

_ARCHIVE_BY_NAME: Dict[str, _ArchiveSpec] = {spec.filename: spec for spec in NUSCENES_ARCHIVE_CATALOG}

# Public views retained for backwards-compat + readability.
NUSCENES_ARCHIVES: Tuple[str, ...] = tuple(_ARCHIVE_BY_NAME.keys())
NUSCENES_ARCHIVES_MD5: Dict[str, Optional[str]] = {name: spec.md5 for name, spec in _ARCHIVE_BY_NAME.items()}

# Named selection groups exposed through the downloader's ``preset`` kwarg.
NUSCENES_PRESETS: Dict[str, Tuple[str, ...]] = {
    # Smallest smoketest — mini sensor data + vector maps + CAN bus. ~600 MB total.
    "mini": (
        "v1.0-mini.tgz",
        "nuScenes-map-expansion-v1.3.zip",
        "can_bus.zip",
    ),
    # Smallest useful trainval slice — metadata + first blob + maps + CAN bus. ~75 GB.
    "trainval_one": (
        "v1.0-trainval_meta.tgz",
        "v1.0-trainval01_blobs.tgz",
        "nuScenes-map-expansion-v1.3.zip",
        "can_bus.zip",
    ),
    # Test split only + maps + CAN bus. ~30 GB.
    "test": (
        "v1.0-test_meta.tgz",
        "v1.0-test_blobs.tgz",
        "nuScenes-map-expansion-v1.3.zip",
        "can_bus.zip",
    ),
    # Full catalog — every archive in NUSCENES_ARCHIVE_CATALOG.
    "full": tuple(_ARCHIVE_BY_NAME.keys()),
}

# Default streaming set mirrors the ``trainval_one`` preset — smallest useful
# trainval slice including HD maps so the parser's map_root picks up automatically.
NUSCENES_DEFAULT_STREAMING_ARCHIVES: Tuple[str, ...] = NUSCENES_PRESETS["trainval_one"]

_CHUNK_BYTES = 1 << 20  # 1 MiB per HTTP chunk

# --------------------------------------------------------------------------------------
# Credential / token resolution
# --------------------------------------------------------------------------------------


def resolve_nuscenes_credentials(
    email: Optional[str] = None, password: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve ``(email, password)`` from explicit args or env vars.

    Precedence: explicit arg → ``$NUSCENES_EMAIL`` / ``$NUSCENES_PASSWORD``.
    Returns ``(None, None)`` when nothing is configured — callers are expected to
    raise a clear error in that case.
    """
    resolved_email = email or os.environ.get("NUSCENES_EMAIL")
    resolved_password = password or os.environ.get("NUSCENES_PASSWORD")
    return resolved_email, resolved_password


# --------------------------------------------------------------------------------------
# Cognito authentication (direct HTTPS; no boto3 dependency)
# --------------------------------------------------------------------------------------


def login_nuscenes(email: str, password: str) -> str:
    """Exchange ``(email, password)`` for a Cognito ``IdToken`` bearer token.

    Uses the ``USER_PASSWORD_AUTH`` flow against the public nuScenes Cognito pool.
    Adapted from li-xl/nuscenes-download (MIT © Xiang-Li Li, 2025).

    :raises RuntimeError: when the response is missing ``AuthenticationResult`` or
        the HTTP status is not 200.
    :return: Bearer token to pass as ``Authorization: Bearer <token>``.
    """
    import requests

    headers = {
        "Content-Type": "application/x-amz-json-1.1",
        "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth",
    }
    payload = json.dumps(
        {
            "AuthFlow": "USER_PASSWORD_AUTH",
            "ClientId": NUSCENES_COGNITO_CLIENT_ID,
            "AuthParameters": {"USERNAME": email, "PASSWORD": password},
            "ClientMetadata": {},
        }
    )
    response = requests.post(NUSCENES_COGNITO_URL, headers=headers, data=payload, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"nuScenes Cognito auth failed (HTTP {response.status_code}): {response.text[:500]}")
    try:
        return response.json()["AuthenticationResult"]["IdToken"]
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f"nuScenes Cognito auth missing AuthenticationResult.IdToken: {exc}") from exc


def get_nuscenes_download_url(filename: str, bearer_token: str, region: str = "us") -> str:
    """Resolve a presigned CloudFront URL for ``filename`` via the API gateway.

    :param filename: Archive name from :data:`NUSCENES_ARCHIVES` (e.g. ``v1.0-trainval_meta.tgz``).
    :param bearer_token: Cognito IdToken from :func:`login_nuscenes`.
    :param region: ``"us"`` or ``"asia"`` — selects the CDN routing.
    :raises RuntimeError: on non-200 responses.
    """
    import requests

    if region not in NUSCENES_REGION_CHOICES:
        raise ValueError(f"region must be one of {NUSCENES_REGION_CHOICES}, got {region!r}")
    api_url = f"{NUSCENES_API_GATEWAY}/{filename}?region={region}&project=nuScenes"
    headers = {"Authorization": f"Bearer {bearer_token}", "Content-Type": "application/json"}
    response = requests.get(api_url, headers=headers, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(
            f"nuScenes URL lookup failed for {filename} (HTTP {response.status_code}): {response.text[:500]}"
        )
    return response.json()["url"]


# --------------------------------------------------------------------------------------
# Archive download + verify + extract
# --------------------------------------------------------------------------------------


def _compute_md5(path: Path, chunk_size: int = _CHUNK_BYTES) -> str:
    md5 = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def _http_stream_to_file(url: str, dest: Path) -> None:
    """GET ``url`` with streaming and atomic rename via ``.part`` suffix."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(tmp_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=_CHUNK_BYTES):
                if chunk:
                    fh.write(chunk)
    tmp_path.replace(dest)


def download_nuscenes_archive(
    archive_name: str,
    output_dir: Path,
    bearer_token: str,
    region: str = "us",
    expected_md5: Optional[str] = None,
    verify_md5: bool = True,
) -> Path:
    """Fetch a single archive into ``output_dir`` and verify its MD5.

    :param archive_name: Archive filename (key of :data:`NUSCENES_ARCHIVES_MD5`).
    :param output_dir: Destination directory. The archive lands as ``output_dir/{archive_name}``.
    :param bearer_token: Cognito IdToken (from :func:`login_nuscenes`).
    :param region: ``"us"`` or ``"asia"``.
    :param expected_md5: MD5 to verify against; falls back to :data:`NUSCENES_ARCHIVES_MD5`.
    :param verify_md5: When ``True`` (default), reject files whose MD5 doesn't match.
    :raises RuntimeError: on MD5 mismatch when ``verify_md5=True``.
    :return: Path to the downloaded archive.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / archive_name
    expected = expected_md5 or NUSCENES_ARCHIVES_MD5.get(archive_name)

    # Skip + re-verify existing files (re-runs shouldn't re-download).
    if dest.exists():
        if verify_md5 and expected is not None:
            actual = _compute_md5(dest)
            if actual == expected:
                logger.debug("Archive already present and MD5-valid: %s", dest)
                return dest
            logger.warning(
                "MD5 mismatch for existing %s (expected %s, got %s) — re-downloading.",
                dest,
                expected,
                actual,
            )
            dest.unlink()
        else:
            return dest

    url = get_nuscenes_download_url(archive_name, bearer_token, region=region)
    logger.info("Downloading %s → %s", archive_name, dest)
    _http_stream_to_file(url, dest)

    if verify_md5 and expected is not None:
        actual = _compute_md5(dest)
        if actual != expected:
            raise RuntimeError(
                f"MD5 mismatch for {archive_name}: expected {expected}, got {actual}. "
                "The download may be corrupt, or the upstream catalog is out of date."
            )
        logger.debug("MD5 OK for %s", dest)
    return dest


def _is_unsafe_zip_member(name: str) -> bool:
    """Reject zip entry paths that would escape the extraction root.

    Mirrors the checks :mod:`tarfile`'s ``filter="data"`` applies on tar members:
    no absolute paths, no Windows drive letters, no ``..`` components after
    normalization. Entries that normalize to an empty/``.`` path are also rejected.
    """
    if not name:
        return True
    normalized = name.replace("\\", "/")
    if normalized.startswith("/"):
        return True
    # Windows drive prefix like "C:" or "C:\\"
    if len(normalized) >= 2 and normalized[1] == ":":
        return True
    parts = normalized.split("/")
    return any(part == ".." for part in parts)


def extract_nuscenes_archive(archive_path: Path, output_dir: Path, extract_format: str) -> None:
    """Safely extract a nuScenes archive into ``output_dir``.

    ``.tgz`` archives use :mod:`tarfile`'s ``data`` filter (Python 3.12+) to reject
    path-traversal entries; ``.zip`` archives run :func:`_is_unsafe_zip_member` on
    each member first. The archive layout mirrors the nuScenes expected tree
    (``samples/``, ``sweeps/``, ``v1.0-*/``, ``maps/``, ``can_bus/``), so extracting
    multiple archives into the same ``output_dir`` produces the standard on-disk
    layout.

    :param extract_format: ``"tgz"`` for gzip tarballs or ``"zip"`` for zip archives.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %s → %s", archive_path, output_dir)

    if extract_format == "tgz":
        with tarfile.open(archive_path, "r:gz") as tar:
            # ``filter="data"`` is available in Python 3.12+; older versions ignore it
            # and rely on the default tarfile behavior. Guard by kwargs detection.
            try:
                tar.extractall(output_dir, filter="data")  # type: ignore[arg-type]
            except TypeError:
                tar.extractall(output_dir)
        return

    if extract_format == "zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            unsafe = [info.filename for info in zf.infolist() if _is_unsafe_zip_member(info.filename)]
            if unsafe:
                raise ValueError(
                    f"Refusing to extract {archive_path.name} — {len(unsafe)} unsafe member(s) "
                    f"(first offender: {unsafe[0]!r}). This archive may be attempting a "
                    "path-traversal extraction."
                )
            zf.extractall(output_dir)
        return

    raise ValueError(f"Unknown extract_format {extract_format!r}; expected 'tgz' or 'zip'.")


# --------------------------------------------------------------------------------------
# Downloader (Hydra-instantiable, shared by py123d-download and NuScenesParser streaming)
# --------------------------------------------------------------------------------------


class NuscenesDownloader(BaseDownloader):
    """Downloader for the nuScenes dataset via the Motional CloudFront API.

    Two entry points, one class:

    * :meth:`download` — bulk flow. Fetches selected archives into a session-scoped
      :class:`tempfile.TemporaryDirectory`, extracts them into :attr:`output_dir`,
      and deletes the ``.tgz`` files when the method returns. Used by
      ``py123d-download dataset=nuscenes``.
    * :meth:`materialize_archives` — streaming flow. Fetches selected archives into
      a caller-provided directory (typically a per-parser
      :class:`tempfile.TemporaryDirectory`), extracts them in place, and removes
      the ``.tgz`` files on successful extraction. Used by :class:`NuScenesParser`
      when ``downloader`` is provided.

    Credentials: reads ``$NUSCENES_EMAIL`` / ``$NUSCENES_PASSWORD`` by default;
    override via the ``email`` / ``password`` constructor args. The nuScenes
    Terms of Use require a registered account at https://www.nuscenes.org/nuscenes
    before downloads succeed.
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        email: Optional[str] = None,
        password: Optional[str] = None,
        region: str = "us",
        preset: Optional[str] = None,
        archives: Optional[List[str]] = None,
        num_archives: Optional[int] = None,
        sample_random: bool = False,
        seed: int = 0,
        max_workers: int = 4,
        verify_md5: bool = True,
        dry_run: bool = False,
    ) -> None:
        """Initialize the nuScenes downloader.

        :param output_dir: Destination directory where archives are extracted.
            Each archive is unpacked in-place into ``output_dir``; the ``.tgz`` /
            ``.zip`` itself does not survive :meth:`download`. Ignored by
            :meth:`materialize_archives` (which takes its own directory arg).
        :param email: nuScenes account email. Defaults to ``$NUSCENES_EMAIL``.
        :param password: nuScenes account password. Defaults to ``$NUSCENES_PASSWORD``.
        :param region: ``"us"`` or ``"asia"`` — CDN routing for faster transfer.
        :param preset: Named selection group from :data:`NUSCENES_PRESETS`
            (``"mini"``, ``"trainval_one"``, ``"test"``, ``"full"``). Mutually
            exclusive with ``archives`` and ``num_archives``. When all three are
            ``None``, the full catalog is selected.
        :param archives: Explicit archive filenames (subset of :data:`NUSCENES_ARCHIVES`).
            Mutually exclusive with ``preset`` and ``num_archives``.
        :param num_archives: Alternative to ``archives``: select the first N from
            the catalog (or N random if ``sample_random=True``). Mutually exclusive
            with ``preset`` and ``archives``.
        :param sample_random: Randomize ``num_archives`` selection.
        :param seed: RNG seed used when ``sample_random=True``.
        :param max_workers: Parallel per-archive download threads.
        :param verify_md5: When ``True`` (default), verify MD5 checksums for any
            archive that has one in :data:`NUSCENES_ARCHIVE_CATALOG`. Archives
            without a published checksum (mini, maps, CAN bus) are never verified.
        :param dry_run: When ``True``, :meth:`download` logs the plan (archives,
            sizes, destination) without authenticating or fetching anything.
        """
        # Mutual exclusivity: preset vs archives vs num_archives.
        selectors_set = sum(x is not None for x in (preset, archives, num_archives))
        if selectors_set > 1:
            raise ValueError("preset, archives, and num_archives are mutually exclusive — set at most one.")
        if num_archives is not None and num_archives <= 0:
            raise ValueError("num_archives must be a positive integer.")
        if preset is not None and preset not in NUSCENES_PRESETS:
            raise ValueError(f"Unknown preset {preset!r}; valid choices: {sorted(NUSCENES_PRESETS.keys())}.")
        if archives is not None:
            unknown = [a for a in archives if a not in _ARCHIVE_BY_NAME]
            if unknown:
                raise ValueError(
                    f"Unknown nuScenes archive(s): {unknown}. "
                    f"Must be in NUSCENES_ARCHIVES ({len(NUSCENES_ARCHIVES)} entries)."
                )
        if region not in NUSCENES_REGION_CHOICES:
            raise ValueError(f"region must be one of {NUSCENES_REGION_CHOICES}, got {region!r}")

        self.output_dir: Optional[Path] = Path(output_dir) if output_dir is not None else None
        self.dry_run: bool = dry_run

        self.region: str = region
        self.max_workers: int = max_workers
        self.verify_md5: bool = verify_md5

        self._preset: Optional[str] = preset
        self._explicit_archives: Optional[List[str]] = list(archives) if archives else None
        self._num_archives: Optional[int] = num_archives
        self._sample_random: bool = sample_random
        self._seed: int = seed

        # Credentials are resolved lazily so constructing a NuscenesDownloader for
        # dry-run or listing doesn't require env vars to be set.
        self._email: Optional[str] = email
        self._password: Optional[str] = password

    # ----- Selection ------------------------------------------------------------------

    def resolve_archives(self) -> List[str]:
        """Return the archive filenames selected by the current configuration.

        Precedence (first match wins): ``archives`` > ``num_archives`` > ``preset``
        > full catalog. Deterministic given the same seed when ``sample_random=True``.
        """
        if self._explicit_archives:
            return list(self._explicit_archives)
        if self._num_archives is not None:
            if self._num_archives >= len(NUSCENES_ARCHIVES):
                return list(NUSCENES_ARCHIVES)
            if self._sample_random:
                rng = _random_mod.Random(self._seed)
                return sorted(rng.sample(list(NUSCENES_ARCHIVES), self._num_archives))
            return list(NUSCENES_ARCHIVES[: self._num_archives])
        if self._preset is not None:
            return list(NUSCENES_PRESETS[self._preset])
        return list(NUSCENES_ARCHIVES)

    # ----- Credential resolution ------------------------------------------------------

    def _resolve_credentials(self) -> Tuple[str, str]:
        email, password = resolve_nuscenes_credentials(self._email, self._password)
        if not email or not password:
            raise RuntimeError(
                "nuScenes credentials are required. Set $NUSCENES_EMAIL and $NUSCENES_PASSWORD, "
                "or pass email/password to NuscenesDownloader. Register at "
                "https://www.nuscenes.org/nuscenes to accept the dataset terms first."
            )
        return email, password

    # ----- Bulk download (py123d-download) --------------------------------------------

    def download(self) -> None:
        """Inherited, see superclass.

        Bulk flow: downloads selected archives into a session-scoped
        :class:`tempfile.TemporaryDirectory`, extracts them into :attr:`output_dir`,
        and deletes the ``.tgz`` files when the method returns. No artifact of the
        archives survives :meth:`download` — only the extracted nuScenes tree.
        """
        archives = self.resolve_archives()
        logger.info("nuScenes source:        %s (region=%s)", NUSCENES_API_GATEWAY, self.region)
        logger.info("nuScenes archives:      %d / %d", len(archives), len(NUSCENES_ARCHIVES))
        logger.info("nuScenes target dir:    %s", self.output_dir)

        if self.dry_run:
            logger.info("dry_run=True — not downloading. Plan covers %d archive(s): %s", len(archives), archives)
            return
        if not archives:
            logger.warning("No archives selected — nothing to download.")
            return

        assert self.output_dir is not None, "NuscenesDownloader.output_dir must be set before download()."
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="py123d-nuscenes-bulk-") as tmp:
            tmp_dir = Path(tmp)
            logger.info("Downloading archives to temp dir (auto-cleaned): %s", tmp_dir)
            self._fetch_and_extract(archives=archives, zip_dir=tmp_dir, extract_dir=self.output_dir)
            logger.info("Bulk download complete: %s", self.output_dir)

    # ----- Per-archive materialization (streaming conversion) -------------------------

    def materialize_archives(self, archives: Sequence[str], output_dir: Union[str, Path]) -> Path:
        """Download and extract a selected archive subset into ``output_dir``.

        Used by :class:`~py123d.parser.nuscenes.nuscenes_parser.NuScenesParser` in
        streaming mode: the parser creates a :class:`tempfile.TemporaryDirectory`,
        asks the downloader to populate it with the chosen archives, and reads the
        resulting nuScenes tree as if it had been downloaded locally. ``.tgz`` files
        are written to the same directory (a sibling ``_zip`` subdir) during the
        fetch and removed after successful extraction.

        :param archives: Archive filenames to fetch. Must be keys of
            :data:`NUSCENES_ARCHIVES_MD5`.
        :param output_dir: Destination directory — populated with the standard
            nuScenes tree (``samples/``, ``sweeps/``, ``v1.0-*/``, ...).
        :return: ``Path(output_dir)``.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        zip_dir = output_dir / "_zip"
        zip_dir.mkdir(parents=True, exist_ok=True)

        unknown = [a for a in archives if a not in NUSCENES_ARCHIVES_MD5]
        if unknown:
            raise ValueError(f"Unknown nuScenes archive(s): {unknown}")

        logger.info("nuScenes streaming: materializing %d archive(s) into %s", len(archives), output_dir)
        self._fetch_and_extract(archives=list(archives), zip_dir=zip_dir, extract_dir=output_dir)

        # Zip directory is empty by design (archives deleted post-extract) — drop it.
        try:
            shutil.rmtree(zip_dir)
        except OSError:
            pass
        return output_dir

    # ----- Shared fetch+extract core --------------------------------------------------

    def _fetch_and_extract(self, archives: List[str], zip_dir: Path, extract_dir: Path) -> None:
        """Parallel download into ``zip_dir``, then sequential extract into ``extract_dir``.

        Archives are removed from ``zip_dir`` after successful extraction.
        """
        email, password = self._resolve_credentials()
        logger.info("Authenticating against nuScenes Cognito as %s...", email)
        bearer_token = login_nuscenes(email, password)

        downloaded: Dict[str, Path] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, self.max_workers)) as pool:
            futures = {
                pool.submit(
                    download_nuscenes_archive,
                    archive_name=name,
                    output_dir=zip_dir,
                    bearer_token=bearer_token,
                    region=self.region,
                    verify_md5=self.verify_md5,
                ): name
                for name in archives
            }
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    downloaded[name] = future.result()
                except Exception as exc:
                    logger.error("Failed to download %s: %s", name, exc)
                    raise

        for name in archives:  # preserve deterministic extract order (metadata first)
            archive_path = downloaded[name]
            spec = _ARCHIVE_BY_NAME[name]
            extract_nuscenes_archive(archive_path, extract_dir, extract_format=spec.extract_format)
            try:
                archive_path.unlink()
            except OSError as exc:
                logger.warning("Could not delete %s after extraction: %s", archive_path, exc)
