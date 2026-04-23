from __future__ import annotations

import concurrent.futures
import logging
import random as _random_mod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple, Union

from py123d.parser.base_downloader import BaseDownloader

if TYPE_CHECKING:
    from google.cloud import storage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _DatasetSpec:
    """Per-dataset download configuration consumed by :func:`download_shards`.

    Folds bucket identity and blob → local-path mapping into one handle so the
    parallel download loop stays dataset-agnostic.
    """

    bucket_name: str
    destination_for_blob: Callable[[str, Path], Path]


# ======================================================================================
# Auth / client (generic)
# ======================================================================================


def _require_gcs():
    """Lazy import — ``google-cloud-storage`` is optional until this CLI or the
    ``downloader`` parser mode is used."""
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise SystemExit(
            "google-cloud-storage is required for WOD downloads. Install it with:\n"
            "  pip install py123d[waymo]\n"
            "or directly:\n"
            "  pip install google-cloud-storage\n"
        ) from exc
    return storage


def resolve_gcs_client(credentials_file: Optional[Path] = None) -> "storage.Client":
    """Build a ``google.cloud.storage.Client`` using the standard auth fallback chain.

    Order:

    1. Explicit ``credentials_file`` (service-account JSON) → ``Client.from_service_account_json``.
    2. ``$GOOGLE_APPLICATION_CREDENTIALS`` or Application Default Credentials → ``Client()``.
    3. Anonymous client fallback. Works for the motion bucket; the perception bucket
       requires an authenticated client (anonymous listing returns 403).

    :param credentials_file: Optional path to a service-account JSON key.
    :return: An initialized GCS client.
    """
    storage = _require_gcs()

    if credentials_file is not None:
        credentials_path = Path(credentials_file).expanduser()
        if not credentials_path.exists():
            raise FileNotFoundError(f"Service account JSON not found: {credentials_path}")
        logger.debug("Using service-account credentials from %s", credentials_path)
        return storage.Client.from_service_account_json(str(credentials_path))

    # Use google.auth.default() directly so we can handle the common case where the user
    # ran `gcloud auth application-default login` but never set a quota project —
    # storage.Client() would raise in that case even though the credentials are valid.
    # WOD bucket reads don't need a real project; we pass a placeholder.
    try:
        import google.auth

        credentials, project = google.auth.default()
        logger.debug("Using Application Default Credentials (project=%s)", project)
        return storage.Client(credentials=credentials, project=project or "py123d-wod")
    except Exception as exc:  # DefaultCredentialsError, etc.
        logger.warning(
            "Could not create authenticated GCS client (%s); falling back to anonymous client. "
            "If you expected authenticated access, run: gcloud auth application-default login",
            exc,
        )
        return storage.Client.create_anonymous_client()


# ======================================================================================
# Selection (generic)
# ======================================================================================


def select_shards(
    shard_blob_names: Sequence[str],
    shard_indices: Optional[Sequence[int]] = None,
    num_shards: Optional[int] = None,
    sample_random: bool = False,
    seed: int = 0,
) -> List[str]:
    """Filter a full shard list down to the requested subset.

    Precedence: ``shard_indices`` > ``num_shards`` > full list.

    :param shard_blob_names: All shards for a split (sorted).
    :param shard_indices: Exact indices into the sorted shard list to keep.
    :param num_shards: If set, keep the first ``num_shards`` (or a random sample if
        ``sample_random=True``).
    :param sample_random: Randomize the ``num_shards`` selection.
    :param seed: RNG seed when ``sample_random=True``.
    :return: Selected subset of ``shard_blob_names``.
    """
    total = len(shard_blob_names)
    if shard_indices is not None:
        selected: List[str] = []
        for idx in shard_indices:
            if idx < 0 or idx >= total:
                raise IndexError(f"shard index {idx} out of range [0, {total}).")
            selected.append(shard_blob_names[idx])
        return selected

    if num_shards is None or num_shards >= total:
        return list(shard_blob_names)

    if sample_random:
        rng = _random_mod.Random(seed)
        return sorted(rng.sample(list(shard_blob_names), num_shards))
    return list(shard_blob_names[:num_shards])


# ======================================================================================
# Download (generic)
# ======================================================================================


def _download_one_blob(
    client: "storage.Client",
    bucket_name: str,
    blob_name: str,
    dest_path: Path,
    overwrite: bool,
) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and not overwrite:
        logger.debug("Skip existing file: %s", dest_path)
        return dest_path
    logger.info("Downloading gs://%s/%s → %s", bucket_name, blob_name, dest_path)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    blob.download_to_filename(str(tmp_path))
    tmp_path.replace(dest_path)
    return dest_path


def download_shards(
    spec: _DatasetSpec,
    client: "storage.Client",
    blob_names: Sequence[str],
    output_dir: Path,
    max_workers: int = 8,
    overwrite: bool = False,
) -> List[Path]:
    """Download a list of shards (by blob name) into ``output_dir`` in parallel.

    The destination path for each blob is derived via ``spec.destination_for_blob``,
    which encodes the dataset-specific key → local-path convention.

    :param spec: Dataset spec (bucket name + destination mapper).
    :param client: GCS client.
    :param blob_names: Blob keys to download.
    :param output_dir: Local root directory.
    :param max_workers: Parallel download threads.
    :param overwrite: If ``False``, skip files that already exist locally.
    :return: Local paths of downloaded (or pre-existing) files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not blob_names:
        return []

    downloaded: List[Path] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = {
            pool.submit(
                _download_one_blob,
                client,
                spec.bucket_name,
                blob_name,
                spec.destination_for_blob(blob_name, output_dir),
                overwrite,
            ): blob_name
            for blob_name in blob_names
        }
        for future in concurrent.futures.as_completed(futures):
            blob_name = futures[future]
            try:
                downloaded.append(future.result())
            except Exception as exc:
                logger.error("Failed to download %s: %s", blob_name, exc)
                raise
    return downloaded


# ======================================================================================
# WOD Motion (WOMD)
# ======================================================================================

WOMD_BUCKET_PREFIX = "waymo_open_dataset_motion_v_"
# Canonical form uses dots ("1.3.0") so Hydra CLI overrides don't get reparsed as the
# numeric literal 130 (Python treats ``1_3_0`` as an int). Internally we normalize
# dots to underscores when building the bucket name.
WOMD_DEFAULT_VERSION = "1.3.0"

MOTION_SCENARIO_SPLITS: Tuple[str, ...] = (
    "training",
    "validation",
    "testing",
    "training_20s",
    "validation_interactive",
    "testing_interactive",
)
MOTION_LIDAR_SPLITS: Tuple[str, ...] = ("training", "validation", "testing")
MOTION_SECTION_CHOICES: Tuple[str, ...] = ("scenario", "lidar", "both")


def motion_bucket_name(version: str) -> str:
    """GCS bucket name for WOMD ``version`` (accepts ``"1.3.0"`` or ``"1_3_0"``)."""
    normalized = str(version).replace(".", "_")
    return f"{WOMD_BUCKET_PREFIX}{normalized}"


def _motion_split_prefix(section: str, split: str) -> str:
    """GCS key prefix for a given motion section+split, without bucket name."""
    if section == "scenario":
        return f"uncompressed/scenario/{split}/"
    if section == "lidar":
        return f"uncompressed/lidar/{split}/"
    raise ValueError(f"Unknown motion section {section!r}; expected one of 'scenario' | 'lidar'.")


def _motion_local_split_dir(output_dir: Path, section: str, split: str) -> Path:
    """Where a motion section+split's tfrecords should land on disk."""
    if section == "scenario":
        return output_dir / split
    if section == "lidar":
        return output_dir / "lidar" / split
    raise ValueError(f"Unknown motion section {section!r}; expected one of 'scenario' | 'lidar'.")


def _motion_destination_for_blob(blob_name: str, output_dir: Path) -> Path:
    """Map a motion GCS blob name back to its on-disk destination.

    ``uncompressed/scenario/{split}/file.tfrecord-*`` → ``{output_dir}/{split}/file.tfrecord-*``
    ``uncompressed/lidar/{split}/file.tfrecord-*`` → ``{output_dir}/lidar/{split}/file.tfrecord-*``
    """
    parts = blob_name.split("/")
    # Expected shape: ["uncompressed", "scenario"|"lidar", "<split>", "<filename>"]
    if len(parts) < 4 or parts[0] != "uncompressed":
        raise ValueError(f"Unexpected motion blob layout: {blob_name!r}")
    section = parts[1]
    split = parts[2]
    filename = "/".join(parts[3:])
    return _motion_local_split_dir(output_dir, section, split) / filename


def list_motion_split_shards(
    client: "storage.Client",
    section: str,
    split: str,
    version: str = WOMD_DEFAULT_VERSION,
) -> List[str]:
    """List all tfrecord blob names for a given WOMD section+split.

    :param client: GCS client.
    :param section: ``"scenario"`` or ``"lidar"``.
    :param split: GCS folder name (e.g. ``"training"``).
    :param version: WOMD version string (accepts ``"1.3.0"`` or ``"1_3_0"``).
    :return: Sorted list of blob names (keys within the bucket).
    """
    bucket = motion_bucket_name(version)
    prefix = _motion_split_prefix(section, split)
    blobs = client.list_blobs(bucket, prefix=prefix)
    blob_names: List[str] = [b.name for b in blobs if ".tfrecord" in b.name]
    blob_names.sort()
    return blob_names


def motion_spec(version: str = WOMD_DEFAULT_VERSION) -> _DatasetSpec:
    """Build the :class:`_DatasetSpec` for WOMD at ``version``."""
    return _DatasetSpec(
        bucket_name=motion_bucket_name(version),
        destination_for_blob=_motion_destination_for_blob,
    )


def download_motion_single_shard(
    client: "storage.Client",
    section: str,
    split: str,
    shard_idx: int,
    output_dir: Path,
    version: str = WOMD_DEFAULT_VERSION,
    overwrite: bool = False,
) -> Path:
    """Download exactly one motion shard identified by its index within the sorted shard list."""
    all_shards = list_motion_split_shards(client, section=section, split=split, version=version)
    if not all_shards:
        raise FileNotFoundError(
            f"No tfrecords found at gs://{motion_bucket_name(version)}/{_motion_split_prefix(section, split)}"
        )
    if shard_idx < 0 or shard_idx >= len(all_shards):
        raise IndexError(f"Shard index {shard_idx} out of range for {section}/{split} (have {len(all_shards)} shards).")
    blob_name = all_shards[shard_idx]
    dest = _motion_destination_for_blob(blob_name, Path(output_dir))
    return _download_one_blob(client, motion_bucket_name(version), blob_name, dest, overwrite=overwrite)


def _motion_split_allowed_for_section(section: str, gcs_split: str) -> bool:
    if section == "scenario":
        return gcs_split in MOTION_SCENARIO_SPLITS
    if section == "lidar":
        return gcs_split in MOTION_LIDAR_SPLITS
    return False


# ======================================================================================
# WOD Perception
# ======================================================================================

WOD_PERCEPTION_BUCKET_PREFIX = "waymo_open_dataset_v_"
WOD_PERCEPTION_DEFAULT_VERSION = "1.4.3"
# Key prefix inside the perception bucket — the "primary" dataset lives under
# ``individual_files/``; ``archived_files/`` (tars) and ``individual_files/domain_adaptation/``
# are intentionally out of scope.
WOD_PERCEPTION_KEY_ROOT = "individual_files"
PERCEPTION_SPLITS: Tuple[str, ...] = ("training", "validation", "testing")
WOD_PERCEPTION_SUPPORTED_VERSIONS: Tuple[str, ...] = ("1.4.3",)


def perception_bucket_name(version: str) -> str:
    """GCS bucket name for WOD Perception ``version`` (accepts ``"1.4.3"`` or ``"1_4_3"``)."""
    normalized = str(version).replace(".", "_")
    return f"{WOD_PERCEPTION_BUCKET_PREFIX}{normalized}"


def _perception_split_prefix(split: str) -> str:
    """GCS key prefix for a given perception split, without bucket name."""
    if split not in PERCEPTION_SPLITS:
        raise ValueError(f"Unknown perception split {split!r}; expected one of {PERCEPTION_SPLITS}.")
    return f"{WOD_PERCEPTION_KEY_ROOT}/{split}/"


def _perception_local_split_dir(output_dir: Path, split: str) -> Path:
    """Where a perception split's tfrecords should land on disk."""
    return output_dir / split


def _perception_destination_for_blob(blob_name: str, output_dir: Path) -> Path:
    """Map a perception GCS blob name back to its on-disk destination.

    ``individual_files/{split}/segment-*.tfrecord`` → ``{output_dir}/{split}/segment-*.tfrecord``
    """
    parts = blob_name.split("/")
    # Expected shape: ["individual_files", "<split>", "<filename>"]
    if len(parts) < 3 or parts[0] != WOD_PERCEPTION_KEY_ROOT or parts[1] not in PERCEPTION_SPLITS:
        raise ValueError(f"Unexpected perception blob layout: {blob_name!r}")
    split = parts[1]
    filename = "/".join(parts[2:])
    return _perception_local_split_dir(output_dir, split) / filename


def list_perception_split_shards(
    client: "storage.Client",
    split: str,
    version: str = WOD_PERCEPTION_DEFAULT_VERSION,
) -> List[str]:
    """List all tfrecord blob names for a given WOD Perception split.

    :param client: GCS client (must be authenticated — the perception bucket is not
        anonymously readable).
    :param split: GCS folder name (``"training"``, ``"validation"``, or ``"testing"``).
    :param version: Perception version string (accepts ``"1.4.3"`` or ``"1_4_3"``).
    :return: Sorted list of blob names (keys within the bucket).
    """
    bucket = perception_bucket_name(version)
    prefix = _perception_split_prefix(split)
    blobs = client.list_blobs(bucket, prefix=prefix)
    blob_names: List[str] = [b.name for b in blobs if ".tfrecord" in b.name]
    blob_names.sort()
    return blob_names


def perception_spec(version: str = WOD_PERCEPTION_DEFAULT_VERSION) -> _DatasetSpec:
    """Build the :class:`_DatasetSpec` for WOD Perception at ``version``."""
    return _DatasetSpec(
        bucket_name=perception_bucket_name(version),
        destination_for_blob=_perception_destination_for_blob,
    )


# ======================================================================================
# Downloaders (Hydra-instantiable, shared by py123d-download and streaming parsers)
# ======================================================================================


class WODMotionDownloader(BaseDownloader):
    """Downloader for the Waymo Open Motion Dataset (WOMD).

    Fetches selected scenario and/or lidar shards from the anonymously-readable
    ``waymo_open_dataset_motion_v_*`` GCS bucket into :attr:`output_dir`, preserving
    the on-disk layout :class:`~py123d.parser.wod.wod_motion_parser.WODMotionParser`
    expects in local mode.
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        splits: Optional[Sequence[str]] = None,
        version: str = WOMD_DEFAULT_VERSION,
        section: str = "scenario",
        num_shards: Optional[int] = None,
        shard_indices: Optional[Dict[str, List[int]]] = None,
        sample_random: bool = False,
        seed: int = 0,
        credentials_file: Optional[Union[str, Path]] = None,
        max_workers: int = 8,
        overwrite: bool = False,
        dry_run: bool = False,
    ) -> None:
        """Initialize the WOMD downloader.

        :param output_dir: Destination directory. When ``None`` the caller (a streaming
            parser) must assign one before invoking :meth:`download`.
        :param splits: 123D WOMD split names to fetch, e.g. ``["wod-motion_train",
            "wod-motion_val"]``. Defaults to all splits registered in
            :data:`WOD_MOTION_SPLIT_TO_GCS_FOLDER`. The 20s and interactive variants
            are only available for a subset of splits — see the module docstring.
        :param version: WOMD version string (e.g. ``"1.3.0"``), mapped to bucket
            ``waymo_open_dataset_motion_v_<version>`` with dots normalized to
            underscores. Use dot-notation so Hydra CLI overrides aren't reparsed as
            numeric literals (``1_3_0`` is the int ``130`` in Python).
        :param section: Which WOMD section to fetch. ``"scenario"`` = motion
            tfrecords (the default and only section parsers currently consume).
            ``"lidar"`` = the separate first-1s lidar archive. ``"both"`` = both.
        :param num_shards: If set, download the first N shards per split (or N random
            shards when ``sample_random=True``). Applied to any split not covered by
            ``shard_indices``.
        :param shard_indices: Per-split exact shard indices, keyed by 123D split name,
            e.g. ``{"wod-motion_train": [0, 1, 2], "wod-motion_val": [0]}``. Takes
            precedence over ``num_shards`` for any split it covers.
        :param sample_random: Randomize ``num_shards`` selection.
        :param seed: RNG seed used when ``sample_random=True``.
        :param credentials_file: Optional service-account JSON for GCS auth. Defaults
            to Application Default Credentials, falling back to an anonymous client.
        :param max_workers: Parallel GCS download threads.
        :param overwrite: If ``False``, skip shards whose local file already exists.
        :param dry_run: If ``True``, log the plan without downloading.
        """
        from py123d.parser.wod.utils.wod_constants import (
            WOD_MOTION_AVAILABLE_SPLITS,
            WOD_MOTION_SPLIT_TO_GCS_FOLDER,
        )

        resolved_splits: List[str] = list(splits) if splits else list(WOD_MOTION_AVAILABLE_SPLITS)
        for split in resolved_splits:
            assert split in WOD_MOTION_AVAILABLE_SPLITS, (
                f"Split {split!r} is not available. Available splits: {WOD_MOTION_AVAILABLE_SPLITS}"
            )
        assert section in MOTION_SECTION_CHOICES, f"section {section!r} must be one of {MOTION_SECTION_CHOICES}"

        self.output_dir: Optional[Path] = Path(output_dir) if output_dir is not None else None
        self.dry_run: bool = dry_run
        self._splits: List[str] = resolved_splits
        self._version: str = version
        self._section: str = section
        self._num_shards: Optional[int] = num_shards
        self._shard_indices: Optional[Dict[str, List[int]]] = (
            {k: list(v) for k, v in shard_indices.items()} if shard_indices else None
        )
        self._sample_random: bool = sample_random
        self._seed: int = seed
        self._credentials_file: Optional[Path] = Path(credentials_file) if credentials_file is not None else None
        self._max_workers: int = max_workers
        self._overwrite: bool = overwrite
        self._split_to_gcs: Dict[str, str] = dict(WOD_MOTION_SPLIT_TO_GCS_FOLDER)

    def download(self) -> None:
        """Inherited, see superclass."""
        assert self.output_dir is not None, "WODMotionDownloader.output_dir must be set before download()."
        sections = ["scenario", "lidar"] if self._section == "both" else [self._section]
        client = resolve_gcs_client(self._credentials_file)
        bucket = motion_bucket_name(self._version)

        blob_names: List[str] = []
        for section in sections:
            for split in self._splits:
                gcs_split = self._split_to_gcs[split]
                if not _motion_split_allowed_for_section(section, gcs_split):
                    logger.debug("Skip split %s (%s) — not available in section %s.", split, gcs_split, section)
                    continue
                per_split_indices = self._shard_indices.get(split) if self._shard_indices else None
                all_shards = list_motion_split_shards(client, section=section, split=gcs_split, version=self._version)
                selected = select_shards(
                    all_shards,
                    shard_indices=per_split_indices,
                    num_shards=self._num_shards if per_split_indices is None else None,
                    sample_random=self._sample_random,
                    seed=self._seed,
                )
                logger.info(
                    "Selected %d / %d shards for section=%s split=%s",
                    len(selected),
                    len(all_shards),
                    section,
                    split,
                )
                blob_names.extend(selected)

        logger.info("WOMD target directory: %s", self.output_dir)
        logger.info("WOMD bucket:           gs://%s/", bucket)
        logger.info("WOMD shards selected:  %d", len(blob_names))

        if self.dry_run:
            logger.info("dry_run=True — not downloading. Plan covers %d blob(s).", len(blob_names))
            for blob_name in blob_names[: min(10, len(blob_names))]:
                logger.info("  gs://%s/%s", bucket, blob_name)
            return

        if not blob_names:
            logger.warning("No shards selected — nothing to download.")
            return

        download_shards(
            spec=motion_spec(self._version),
            client=client,
            blob_names=blob_names,
            output_dir=self.output_dir,
            max_workers=self._max_workers,
            overwrite=self._overwrite,
        )
        logger.info("WOMD download complete: %s", self.output_dir)


class WODPerceptionDownloader(BaseDownloader):
    """Downloader for the Waymo Open Dataset Perception subset.

    Fetches selected segments from the authenticated-only ``waymo_open_dataset_v_*``
    GCS bucket into :attr:`output_dir`, preserving the on-disk layout
    :class:`~py123d.parser.wod.wod_perception_parser.WODPerceptionParser` expects in
    local mode.

    .. warning::
       Perception segments are ~1 GB each — even small values of ``num_shards``
       imply multiple GB of download traffic. The bucket is **not** anonymously
       readable; supply ``credentials_file`` or run ``gcloud auth
       application-default login`` first.
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        splits: Optional[Sequence[str]] = None,
        version: str = WOD_PERCEPTION_DEFAULT_VERSION,
        num_shards: Optional[int] = None,
        shard_indices: Optional[Dict[str, List[int]]] = None,
        sample_random: bool = False,
        seed: int = 0,
        credentials_file: Optional[Union[str, Path]] = None,
        max_workers: int = 8,
        overwrite: bool = False,
        dry_run: bool = False,
    ) -> None:
        """Initialize the WOD Perception downloader.

        :param output_dir: Destination directory. When ``None`` the caller (a streaming
            parser) must assign one before invoking :meth:`download`.
        :param splits: 123D WOD Perception split names to fetch, e.g.
            ``["wod-perception_train", "wod-perception_val"]``. Defaults to all three
            registered splits.
        :param version: WOD Perception version string (e.g. ``"1.4.3"``), mapped to
            bucket ``waymo_open_dataset_v_<version>`` with dots normalized to
            underscores. Only ``"1.4.3"`` is currently supported.
        :param num_shards: If set, download the first N segments per split (or N
            random segments when ``sample_random=True``).
        :param shard_indices: Per-split exact segment indices, keyed by 123D split
            name, e.g. ``{"wod-perception_val": [0, 1, 2]}``.
        :param sample_random: Randomize ``num_shards`` selection.
        :param seed: RNG seed used when ``sample_random=True``.
        :param credentials_file: Optional service-account JSON for GCS auth. Required
            in practice since the perception bucket is not anonymously readable.
        :param max_workers: Parallel GCS download threads.
        :param overwrite: If ``False``, skip segments whose local file already exists.
        :param dry_run: If ``True``, log the plan without downloading.
        """
        from py123d.parser.wod.utils.wod_constants import (
            WOD_PERCEPTION_AVAILABLE_SPLITS,
            WOD_PERCEPTION_SPLIT_TO_GCS_FOLDER,
        )

        resolved_splits: List[str] = list(splits) if splits else list(WOD_PERCEPTION_AVAILABLE_SPLITS)
        for split in resolved_splits:
            assert split in WOD_PERCEPTION_AVAILABLE_SPLITS, (
                f"Split {split!r} is not available. Available splits: {WOD_PERCEPTION_AVAILABLE_SPLITS}"
            )
        assert version in WOD_PERCEPTION_SUPPORTED_VERSIONS, (
            f"version {version!r} not supported; expected one of {WOD_PERCEPTION_SUPPORTED_VERSIONS}"
        )

        self.output_dir: Optional[Path] = Path(output_dir) if output_dir is not None else None
        self.dry_run: bool = dry_run
        self._splits: List[str] = resolved_splits
        self._version: str = version
        self._num_shards: Optional[int] = num_shards
        self._shard_indices: Optional[Dict[str, List[int]]] = (
            {k: list(v) for k, v in shard_indices.items()} if shard_indices else None
        )
        self._sample_random: bool = sample_random
        self._seed: int = seed
        self._credentials_file: Optional[Path] = Path(credentials_file) if credentials_file is not None else None
        self._max_workers: int = max_workers
        self._overwrite: bool = overwrite
        self._split_to_gcs: Dict[str, str] = dict(WOD_PERCEPTION_SPLIT_TO_GCS_FOLDER)

    def download(self) -> None:
        """Inherited, see superclass."""
        assert self.output_dir is not None, "WODPerceptionDownloader.output_dir must be set before download()."
        client = resolve_gcs_client(self._credentials_file)
        bucket = perception_bucket_name(self._version)

        blob_names: List[str] = []
        for split in self._splits:
            gcs_split = self._split_to_gcs[split]
            per_split_indices = self._shard_indices.get(split) if self._shard_indices else None
            all_shards = list_perception_split_shards(client, split=gcs_split, version=self._version)
            selected = select_shards(
                all_shards,
                shard_indices=per_split_indices,
                num_shards=self._num_shards if per_split_indices is None else None,
                sample_random=self._sample_random,
                seed=self._seed,
            )
            logger.info(
                "Selected %d / %d segments for split=%s",
                len(selected),
                len(all_shards),
                split,
            )
            blob_names.extend(selected)

        logger.info("WOD Perception target directory: %s", self.output_dir)
        logger.info("WOD Perception bucket:           gs://%s/", bucket)
        logger.info("WOD Perception segments:         %d", len(blob_names))

        if self.dry_run:
            logger.info("dry_run=True — not downloading. Plan covers %d segment(s).", len(blob_names))
            for blob_name in blob_names[: min(10, len(blob_names))]:
                logger.info("  gs://%s/%s", bucket, blob_name)
            return

        if not blob_names:
            logger.warning("No segments selected — nothing to download.")
            return

        download_shards(
            spec=perception_spec(self._version),
            client=client,
            blob_names=blob_names,
            output_dir=self.output_dir,
            max_workers=self._max_workers,
            overwrite=self._overwrite,
        )
        logger.info("WOD Perception download complete: %s", self.output_dir)
