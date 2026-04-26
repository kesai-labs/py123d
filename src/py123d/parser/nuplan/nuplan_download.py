"""Download utilities for the nuPlan dataset (Motional).

The nuPlan dataset is hosted on a publicly-readable AWS S3 bucket at
``motional-nuplan.s3-ap-northeast-1.amazonaws.com``. No authentication is required —
fetches are plain HTTPS GETs. Users should still review the upstream license at
``https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE`` before use.

Archive families (sizes approximate — Motional publishes no per-archive sizes or
MD5 checksums, so verification is skipped)::

    nuplan-maps-v1.1.zip                                        (HD maps, ~1 GB)
    nuplan-v1.1_train_{boston,pittsburgh,singapore,vegas_1..6}.zip  (train DBs)
    nuplan-v1.1_val.zip                                         (val DBs)
    nuplan-v1.1_test.zip                                        (test DBs)
    nuplan-v1.1_mini.zip                                        (mini DBs)
    sensor_blobs/train_set/nuplan-v1.1_train_{camera,lidar}_{0..42}.zip
    sensor_blobs/val_set/nuplan-v1.1_val_{camera,lidar}_{0..11}.zip
    sensor_blobs/test_set/nuplan-v1.1_test_{camera,lidar}_{0..11}.zip
    sensor_blobs/mini_set/nuplan-v1.1_mini_{camera,lidar}_{0..8}.zip

On-disk layout produced by the downloader, matching what
:class:`~py123d.parser.nuplan.nuplan_parser.NuplanParser` expects::

    <output_dir>/
    ├── maps/                        (from nuplan-maps-v1.1.zip)
    └── nuplan-v1.1/
        ├── splits/
        │   ├── trainval/            (from train_{boston,pittsburgh,singapore,vegas_*}.zip + val.zip)
        │   ├── test/                (from test.zip)
        │   └── mini/                (from mini.zip)
        └── sensor_blobs/            (from every camera/lidar shard)
            └── <log_name>/{CAM_*, MergedPointCloud}/

This module exposes :class:`NuplanDownloader` (Hydra-instantiable) which powers two
entry points:

1. ``py123d-download dataset=nuplan`` — bulk flow. Fetches archives into a
   session-scoped :class:`tempfile.TemporaryDirectory`, extracts them into
   :attr:`output_dir` (typically ``$NUPLAN_DATA_ROOT``), and deletes the ``.zip``
   files on successful extraction.
2. The :class:`NuplanParser` streaming path (``dataset=nuplan-stream`` /
   ``dataset=nuplan-mini-stream``) — materializes archives into a managed temp
   directory at parser construction time; cleaned up on parser GC.

Archive selection is driven by two orthogonal axes:

* **Splits** (``splits=[nuplan_train, nuplan-mini_test, ...]``) — determines which
  log and sensor families are needed. ``nuplan_train`` pulls the 9 train city
  log zips + ``train_set/`` sensors; ``nuplan_val`` pulls ``nuplan-v1.1_val.zip``
  + ``val_set/`` sensors (both train and val log zips extract into the shared
  ``splits/trainval/`` directory); ``nuplan-mini_*`` all share the single
  ``mini`` log + sensor bundle.
* **Content flags** — ``include_maps`` (default ``True``), ``include_cameras``
  (default ``False``), ``include_lidar`` (default ``False``). Logs are always
  included; the flags only gate sensor + map archives.

The atomic ``.part`` download primitive (:func:`_http_stream_to_file`) is
imported from :mod:`py123d.parser.nuscenes.nuscenes_download` — a pure helper
with no nuScenes coupling. Archive extraction is nuPlan-specific (each archive
has wrapper folders that must be stripped before contents land in the
canonical layout) so it lives here as :func:`_extract_nuplan_archive`. If a
future dataset needs the download primitive too, promote it into
``py123d.parser.utils``.
"""

from __future__ import annotations

import concurrent.futures
import logging
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from py123d.parser.base_downloader import BaseDownloader
from py123d.parser.nuplan.utils.nuplan_constants import NUPLAN_DATA_SPLITS
from py123d.parser.nuscenes.nuscenes_download import _http_stream_to_file

logger = logging.getLogger(__name__)

NUPLAN_BASE_URL = "https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1"
NUPLAN_LICENSE_URL = "https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE"

# Archive categories — gated by include_* flags.
_CATEGORY_MAPS = "maps"
_CATEGORY_LOGS = "logs"  # always fetched
_CATEGORY_CAMERA = "camera"
_CATEGORY_LIDAR = "lidar"

# Route keys — per-split archive selection buckets.
_ROUTE_MAPS = "maps"  # shared across all splits
_ROUTE_LOGS_TRAIN = "logs_train"
_ROUTE_LOGS_VAL = "logs_val"
_ROUTE_LOGS_TEST = "logs_test"
_ROUTE_LOGS_MINI = "logs_mini"
_ROUTE_SENSOR_TRAIN = "sensor_train"
_ROUTE_SENSOR_VAL = "sensor_val"
_ROUTE_SENSOR_TEST = "sensor_test"
_ROUTE_SENSOR_MINI = "sensor_mini"

# Per-route post-extraction layout: target subdir under output_dir + number of
# wrapper folders to strip from the extracted tree before moving contents into
# place. Motional ships archives with internal wrapper directories that must be
# skipped to produce the canonical NuplanParser layout. Train and val log zips
# both land in ``splits/trainval/`` — that's where NuplanParser expects them.
_ROUTE_LAYOUT: Dict[str, Tuple[Path, int]] = {
    _ROUTE_MAPS: (Path("maps"), 1),
    _ROUTE_LOGS_TRAIN: (Path("nuplan-v1.1/splits/trainval"), 3),
    _ROUTE_LOGS_VAL: (Path("nuplan-v1.1/splits/trainval"), 3),
    _ROUTE_LOGS_TEST: (Path("nuplan-v1.1/splits/test"), 3),
    _ROUTE_LOGS_MINI: (Path("nuplan-v1.1/splits/mini"), 3),
    _ROUTE_SENSOR_TRAIN: (Path("nuplan-v1.1/sensor_blobs"), 0),
    _ROUTE_SENSOR_VAL: (Path("nuplan-v1.1/sensor_blobs"), 0),
    _ROUTE_SENSOR_TEST: (Path("nuplan-v1.1/sensor_blobs"), 0),
    _ROUTE_SENSOR_MINI: (Path("nuplan-v1.1/sensor_blobs"), 0),
}


@dataclass(frozen=True)
class _NuplanArchiveSpec:
    """Per-archive metadata consumed by the downloader."""

    filename: str
    url: str
    category: str  # one of _CATEGORY_*
    route_key: str  # one of _ROUTE_*
    approx_size_gb: float
    target_subdir: Path  # destination relative to extract_dir
    skip_levels: int  # wrapper folders to strip before moving contents


def _build_catalog() -> Tuple[_NuplanArchiveSpec, ...]:
    """Build the full nuPlan archive catalog programmatically.

    Totals: 1 maps + 9 train logs + 1 val log + 1 test log + 1 mini log + 86 train
    sensor shards + 24 val sensor shards + 24 test sensor shards + 18 mini sensor
    shards = **165 archives**. Sizes are rough estimates; Motional does not publish them.
    """
    specs: List[_NuplanArchiveSpec] = []

    def _make(filename: str, url: str, category: str, route_key: str, approx_size_gb: float) -> _NuplanArchiveSpec:
        target_subdir, skip_levels = _ROUTE_LAYOUT[route_key]
        return _NuplanArchiveSpec(
            filename=filename,
            url=url,
            category=category,
            route_key=route_key,
            approx_size_gb=approx_size_gb,
            target_subdir=target_subdir,
            skip_levels=skip_levels,
        )

    specs.append(
        _make(
            filename="nuplan-maps-v1.1.zip",
            url=f"{NUPLAN_BASE_URL}/nuplan-maps-v1.1.zip",
            category=_CATEGORY_MAPS,
            route_key=_ROUTE_MAPS,
            approx_size_gb=1.0,
        )
    )

    trainval_log_shards: Tuple[str, ...] = (
        "train_boston",
        "train_pittsburgh",
        "train_singapore",
        "train_vegas_1",
        "train_vegas_2",
        "train_vegas_3",
        "train_vegas_4",
        "train_vegas_5",
        "train_vegas_6",
    )
    for shard in trainval_log_shards:
        filename = f"nuplan-v1.1_{shard}.zip"
        specs.append(
            _make(
                filename=filename,
                url=f"{NUPLAN_BASE_URL}/{filename}",
                category=_CATEGORY_LOGS,
                route_key=_ROUTE_LOGS_TRAIN,
                approx_size_gb=15.0,
            )
        )

    specs.append(
        _make(
            filename="nuplan-v1.1_val.zip",
            url=f"{NUPLAN_BASE_URL}/nuplan-v1.1_val.zip",
            category=_CATEGORY_LOGS,
            route_key=_ROUTE_LOGS_VAL,
            approx_size_gb=5.0,
        )
    )
    specs.append(
        _make(
            filename="nuplan-v1.1_test.zip",
            url=f"{NUPLAN_BASE_URL}/nuplan-v1.1_test.zip",
            category=_CATEGORY_LOGS,
            route_key=_ROUTE_LOGS_TEST,
            approx_size_gb=5.0,
        )
    )
    specs.append(
        _make(
            filename="nuplan-v1.1_mini.zip",
            url=f"{NUPLAN_BASE_URL}/nuplan-v1.1_mini.zip",
            category=_CATEGORY_LOGS,
            route_key=_ROUTE_LOGS_MINI,
            approx_size_gb=10.0,
        )
    )

    # (set_dir, shard_prefix, last_shard_idx_inclusive, route_key)
    sensor_shard_groups: Tuple[Tuple[str, str, int, str], ...] = (
        ("train_set", "train", 42, _ROUTE_SENSOR_TRAIN),
        ("val_set", "val", 11, _ROUTE_SENSOR_VAL),
        ("test_set", "test", 11, _ROUTE_SENSOR_TEST),
        ("mini_set", "mini", 8, _ROUTE_SENSOR_MINI),
    )
    for set_dir, shard_prefix, last_idx, route_key in sensor_shard_groups:
        for shard_idx in range(last_idx + 1):
            for category, approx_size_gb in ((_CATEGORY_CAMERA, 20.0), (_CATEGORY_LIDAR, 10.0)):
                filename = f"nuplan-v1.1_{shard_prefix}_{category}_{shard_idx}.zip"
                specs.append(
                    _make(
                        filename=filename,
                        url=f"{NUPLAN_BASE_URL}/sensor_blobs/{set_dir}/{filename}",
                        category=category,
                        route_key=route_key,
                        approx_size_gb=approx_size_gb,
                    )
                )

    return tuple(specs)


NUPLAN_ARCHIVE_CATALOG: Tuple[_NuplanArchiveSpec, ...] = _build_catalog()

# Per-split routing. Train and val log zips are distinct archives that both
# extract into ``splits/trainval/`` (where NuplanParser expects them) — selecting
# only ``nuplan_val`` skips the ~135 GB of train city zips and pulls just the val
# log + val sensors. ``nuplan-mini_*`` all share the single mini log + sensor bundle.
_SPLIT_TO_ROUTE_KEYS: Dict[str, Tuple[str, ...]] = {
    "nuplan_train": (_ROUTE_LOGS_TRAIN, _ROUTE_SENSOR_TRAIN),
    "nuplan_val": (_ROUTE_LOGS_VAL, _ROUTE_SENSOR_VAL),
    "nuplan_test": (_ROUTE_LOGS_TEST, _ROUTE_SENSOR_TEST),
    "nuplan-mini_train": (_ROUTE_LOGS_MINI, _ROUTE_SENSOR_MINI),
    "nuplan-mini_val": (_ROUTE_LOGS_MINI, _ROUTE_SENSOR_MINI),
    "nuplan-mini_test": (_ROUTE_LOGS_MINI, _ROUTE_SENSOR_MINI),
}


class NuplanDownloader(BaseDownloader):
    """Downloader for the nuPlan dataset via the public Motional AWS S3 bucket.

    Two entry points, one class:

    * :meth:`download` — bulk flow. Fetches the selected archives into a
      session-scoped :class:`tempfile.TemporaryDirectory`, extracts them into
      :attr:`output_dir`, and deletes the ``.zip`` files when extraction succeeds.
      Used by ``py123d-download dataset=nuplan``.
    * :meth:`materialize_archives` — streaming flow. Fetches the selected archives
      into a caller-provided directory (typically a per-parser
      :class:`tempfile.TemporaryDirectory`), extracts them in place, and removes
      the ``.zip`` files on successful extraction. Used by :class:`NuplanParser`
      when ``downloader`` is provided.

    No credentials are required — the bucket is public. Users should still review
    the upstream license at :data:`NUPLAN_LICENSE_URL` before use.
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        splits: Optional[List[str]] = None,
        include_maps: bool = True,
        include_cameras: bool = False,
        include_lidar: bool = False,
        max_workers: int = 4,
        dry_run: bool = False,
    ) -> None:
        """Initialize the nuPlan downloader.

        :param output_dir: Destination directory. Archives are extracted in-place —
            the ``.zip`` files do not survive :meth:`download`. When ``None``, a
            streaming caller (:class:`NuplanParser`) provides an output directory
            directly to :meth:`materialize_archives`. In bulk mode, should point at
            ``$NUPLAN_DATA_ROOT`` so the resulting ``maps/`` and ``nuplan-v1.1/``
            subdirs land in their canonical location.
        :param splits: 123D split names (subset of :data:`NUPLAN_DATA_SPLITS`).
            ``None`` expands to every available split. Mixing regular and mini
            splits is allowed; the resolver unions archive sets.
        :param include_maps: Whether to fetch ``nuplan-maps-v1.1.zip``.
            Required when converting with the ``map_writer`` block active.
        :param include_cameras: Whether to fetch the 8-camera sensor shards for
            every requested split. Off by default (hundreds of GB per split).
        :param include_lidar: Whether to fetch the merged-point-cloud sensor shards
            for every requested split. Off by default.
        :param max_workers: Parallel per-archive download threads.
        :param dry_run: When ``True``, log the plan (archive list + size total)
            without hitting the network.
        """
        resolved_splits: List[str] = list(splits) if splits else sorted(NUPLAN_DATA_SPLITS)
        unknown = [s for s in resolved_splits if s not in NUPLAN_DATA_SPLITS]
        if unknown:
            raise ValueError(f"Unknown nuPlan split(s): {unknown}. Must be in {sorted(NUPLAN_DATA_SPLITS)}.")

        self.output_dir: Optional[Path] = Path(output_dir) if output_dir is not None else None
        self.dry_run: bool = dry_run

        self._splits: List[str] = resolved_splits
        self._include_maps: bool = include_maps
        self._include_cameras: bool = include_cameras
        self._include_lidar: bool = include_lidar
        self._max_workers: int = max_workers

    # ----- Selection ------------------------------------------------------------------

    def resolve_archives(self) -> List[_NuplanArchiveSpec]:
        """Return the archives needed to satisfy the current ``(splits, flags)``.

        Logs are always included for every requested split. Maps / cameras / lidar
        are included only when their corresponding ``include_*`` flag is set.
        Order: maps first, then log zips (deterministic — matches
        :data:`NUPLAN_ARCHIVE_CATALOG`), then sensor shards. This order is also the
        extraction order, so the smaller metadata-like archives land first.
        """
        needed_route_keys: Set[str] = set()
        for split in self._splits:
            for route_key in _SPLIT_TO_ROUTE_KEYS[split]:
                needed_route_keys.add(route_key)

        selected: List[_NuplanArchiveSpec] = []
        for spec in NUPLAN_ARCHIVE_CATALOG:
            if spec.category == _CATEGORY_MAPS:
                if self._include_maps:
                    selected.append(spec)
            elif spec.category == _CATEGORY_LOGS:
                if spec.route_key in needed_route_keys:
                    selected.append(spec)
            elif spec.category == _CATEGORY_CAMERA:
                if self._include_cameras and spec.route_key in needed_route_keys:
                    selected.append(spec)
            elif spec.category == _CATEGORY_LIDAR:
                if self._include_lidar and spec.route_key in needed_route_keys:
                    selected.append(spec)
        return selected

    # ----- Bulk download (py123d-download) --------------------------------------------

    def download(self) -> None:
        """Inherited, see superclass.

        Bulk flow: downloads selected archives into a session-scoped
        :class:`tempfile.TemporaryDirectory`, extracts them into :attr:`output_dir`,
        and deletes the ``.zip`` files on successful extraction. No archive file
        survives :meth:`download` — only the extracted nuPlan tree.
        """
        archives = self.resolve_archives()
        total_gb = sum(spec.approx_size_gb for spec in archives)
        logger.info("nuPlan source:         %s", NUPLAN_BASE_URL)
        logger.info("nuPlan splits:         %s", self._splits)
        logger.info(
            "nuPlan content:        maps=%s cameras=%s lidar=%s",
            self._include_maps,
            self._include_cameras,
            self._include_lidar,
        )
        logger.info("nuPlan archives:       %d (≈ %.1f GB)", len(archives), total_gb)
        logger.info("nuPlan target dir:     %s", self.output_dir)
        logger.info("nuPlan license:        %s", NUPLAN_LICENSE_URL)

        if self.dry_run:
            logger.info("dry_run=True — not downloading. Plan covers %d archive(s).", len(archives))
            for spec in archives:
                logger.info("  %s (≈ %.1f GB)", spec.filename, spec.approx_size_gb)
            return
        if not archives:
            logger.warning("No archives selected — nothing to download.")
            return

        assert self.output_dir is not None, "NuplanDownloader.output_dir must be set before download()."
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="py123d-nuplan-bulk-") as tmp:
            tmp_dir = Path(tmp)
            logger.info("Downloading archives to temp dir (auto-cleaned): %s", tmp_dir)
            self._fetch_and_extract(archives=archives, zip_dir=tmp_dir, extract_dir=self.output_dir)
            logger.info("Bulk download complete: %s", self.output_dir)

    # ----- Per-archive materialization (streaming conversion) -------------------------

    def materialize_archives(self, output_dir: Union[str, Path]) -> Path:
        """Download and extract the configured archive subset into ``output_dir``.

        Used by :class:`~py123d.parser.nuplan.nuplan_parser.NuplanParser` in
        streaming mode: the parser creates a :class:`tempfile.TemporaryDirectory`,
        asks the downloader to populate it with the archives resolved from
        ``splits`` + ``include_*`` flags, and reads the resulting nuPlan tree as
        if it had been downloaded locally. ``.zip`` files land in a sibling
        ``_zip`` subdirectory during the fetch and are removed after successful
        extraction.

        :param output_dir: Destination directory — populated with the canonical
            nuPlan tree (``maps/``, ``nuplan-v1.1/splits/*``, ``nuplan-v1.1/sensor_blobs/``).
        :return: ``Path(output_dir)``.
        """
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        zip_dir = output_dir_path / "_zip"
        zip_dir.mkdir(parents=True, exist_ok=True)

        archives = self.resolve_archives()
        total_gb = sum(spec.approx_size_gb for spec in archives)
        logger.info(
            "nuPlan streaming: materializing %d archive(s) (≈ %.1f GB) into %s",
            len(archives),
            total_gb,
            output_dir_path,
        )
        if not archives:
            logger.warning("No archives selected — nothing to materialize.")
            return output_dir_path

        self._fetch_and_extract(archives=archives, zip_dir=zip_dir, extract_dir=output_dir_path)

        try:
            shutil.rmtree(zip_dir)
        except OSError:
            pass
        return output_dir_path

    # ----- Shared fetch + extract core ------------------------------------------------

    def _fetch_and_extract(
        self,
        archives: List[_NuplanArchiveSpec],
        zip_dir: Path,
        extract_dir: Path,
    ) -> None:
        """Parallel download into ``zip_dir``, then sequential extract into ``extract_dir``.

        Each archive is removed from ``zip_dir`` after successful extraction so
        peak disk usage scales with the largest single archive rather than the
        whole selection. Extraction uses a sibling ``_unpack/`` staging dir so the
        wrapper-folder strip happens off-path before contents land in their
        canonical location.
        """
        downloaded: Dict[str, Path] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, self._max_workers)) as pool:
            futures = {pool.submit(_download_archive, spec=spec, output_dir=zip_dir): spec for spec in archives}
            for future in concurrent.futures.as_completed(futures):
                spec = futures[future]
                try:
                    downloaded[spec.filename] = future.result()
                except Exception as exc:
                    logger.error("Failed to download %s: %s", spec.filename, exc)
                    raise

        staging_root = extract_dir / "_unpack"
        staging_root.mkdir(parents=True, exist_ok=True)
        try:
            for spec in archives:  # deterministic extract order: maps → logs → sensors
                archive_path = downloaded[spec.filename]
                _extract_nuplan_archive(
                    archive_path=archive_path,
                    extract_dir=extract_dir,
                    target_subdir=spec.target_subdir,
                    skip_levels=spec.skip_levels,
                    staging_root=staging_root,
                )
                try:
                    archive_path.unlink()
                except OSError as exc:
                    logger.warning("Could not delete %s after extraction: %s", archive_path, exc)
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)


def _download_archive(spec: _NuplanArchiveSpec, output_dir: Path) -> Path:
    """Fetch a single archive into ``output_dir`` using atomic ``.part`` rename.

    Re-runs are cheap: an existing file is kept as-is (no checksum verification
    since Motional does not publish MD5s — partial downloads are rejected by the
    ``.part`` rename pattern).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / spec.filename
    if dest.exists():
        logger.debug("Archive already present, skipping download: %s", dest)
        return dest
    logger.info("Downloading %s (≈ %.1f GB) → %s", spec.filename, spec.approx_size_gb, dest)
    _http_stream_to_file(spec.url, dest)
    return dest


# ----- Archive extraction -------------------------------------------------------------


def _is_unsafe_zip_member(name: str) -> bool:
    """Reject zip entry paths that would escape the extraction root.

    Mirrors the checks :mod:`tarfile`'s ``filter="data"`` applies on tar members:
    no absolute paths, no Windows drive letters, no ``..`` components after
    normalization.
    """
    if not name:
        return True
    normalized = name.replace("\\", "/")
    if normalized.startswith("/"):
        return True
    if len(normalized) >= 2 and normalized[1] == ":":
        return True
    return any(part == ".." for part in normalized.split("/"))


def _jump_k_folder_forward(path: Path, k: int) -> Path:
    """Descend ``k`` directory levels into ``path``, picking the first subdir at each step.

    Stops early if a level has no subdirectories. Used to strip Motional's
    archive-internal wrapper folders before moving contents into their canonical
    location.
    """
    current = path
    for _ in range(k):
        subdirs = [d for d in current.iterdir() if d.is_dir()]
        if not subdirs:
            break
        current = subdirs[0]
    return current


def _move_folder_contents(src: Path, dst: Path) -> None:
    """Move every entry in ``src`` into ``dst``. ``dst`` must exist."""
    for item in src.iterdir():
        shutil.move(str(item), str(dst / item.name))


def _extract_nuplan_archive(
    archive_path: Path,
    extract_dir: Path,
    target_subdir: Path,
    skip_levels: int,
    staging_root: Path,
) -> None:
    """Extract a nuPlan zip into ``extract_dir / target_subdir`` with wrapper-folder strip.

    Motional's archives ship with internal wrapper directories that need to be
    stripped before contents land in the canonical NuplanParser layout. This
    function extracts to a per-archive staging dir, descends ``skip_levels``
    folders, and moves the contents into the canonical destination.

    :param archive_path: Path to the ``.zip`` to extract.
    :param extract_dir: Final root (typically ``output_dir``).
    :param target_subdir: Destination relative to ``extract_dir`` (e.g. ``maps/``).
    :param skip_levels: How many wrapper folders to strip from the extracted tree.
    :param staging_root: Directory under which the per-archive staging subdir is created.
    """
    staging_dir = staging_root / archive_path.stem
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            unsafe = [info.filename for info in zf.infolist() if _is_unsafe_zip_member(info.filename)]
            if unsafe:
                preview = ", ".join(unsafe[:3]) + (" …" if len(unsafe) > 3 else "")
                raise ValueError(f"Refusing to extract {archive_path.name}: {len(unsafe)} unsafe member(s): {preview}")
            zf.extractall(staging_dir)

        target_dir = extract_dir / target_subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        source_dir = _jump_k_folder_forward(staging_dir, skip_levels)
        _move_folder_contents(source_dir, target_dir)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
