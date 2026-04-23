"""Download utilities for the Argoverse 2 (AV2) dataset family.

The AV2 datasets — Sensor, Lidar, Motion-Forecasting, and TbV — all live under a
public, anonymously-readable AWS S3 bucket, one subfolder per variant::

    s3://argoverse/datasets/av2/sensor/...
    s3://argoverse/datasets/av2/lidar/...              (not yet supported)
    s3://argoverse/datasets/av2/motion-forecasting/... (not yet supported)
    s3://argoverse/datasets/av2/tbv/...                (not yet supported)

Only the Sensor variant is currently implemented; the class is named
:class:`Av2Downloader` so future variants can be added without introducing a second
class.

Sensor variant
--------------
Bucket layout::

    s3://argoverse/datasets/av2/sensor/
        train/<log_uuid>/
            annotations.feather
            city_SE3_egovehicle.feather
            calibration/{egovehicle_SE3_sensor.feather, intrinsics.feather}
            map/...
            sensors/{cameras/<ring_*>/*.jpg, lidar/*.feather}
        val/<log_uuid>/...
        test/<log_uuid>/...

On-disk output layout under the downloader's ``output_dir``::

    <output_dir>/
        sensor/
            train/<log_uuid>/...
            val/<log_uuid>/...
            test/<log_uuid>/...

This matches what :class:`~py123d.parser.av2.av2_sensor_parser.Av2SensorParser`
expects in local mode: ``av2_data_root`` should point at ``<output_dir>``, and the
parser walks ``<av2_data_root>/sensor/<split_type>/<log_uuid>/`` internally.

This module backs both ``py123d-download dataset=av2-sensor`` and the
``av2-sensor-stream`` conversion config.
"""

from __future__ import annotations

import concurrent.futures
import logging
import random as _random_mod
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

from py123d.parser.base_downloader import BaseDownloader

if TYPE_CHECKING:
    import botocore.client

logger = logging.getLogger(__name__)

AV2_BUCKET_NAME = "argoverse"
# All AV2 objects share this key prefix inside the bucket. Local paths strip the
# ``datasets/av2/`` portion and preserve ``<variant>/...`` so the parser's
# relative-path conventions still resolve.
AV2_DATASET_KEY_ROOT = "datasets/av2"

# Supported variants. Expand as new variants (lidar, motion-forecasting, tbv) are added.
AV2_SENSOR_VARIANT = "sensor"
AV2_SUPPORTED_VARIANTS: Tuple[str, ...] = (AV2_SENSOR_VARIANT,)

# Per-variant: 123D split name → S3 split folder name.
AV2_VARIANT_SPLIT_TO_S3: Dict[str, Dict[str, str]] = {
    AV2_SENSOR_VARIANT: {
        "av2-sensor_train": "train",
        "av2-sensor_val": "val",
        "av2-sensor_test": "test",
    },
}


# ======================================================================================
# S3 client / listing (anonymous)
# ======================================================================================


def _require_boto3():
    """Lazy import — ``boto3`` is optional until this downloader is instantiated."""
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError as exc:
        raise SystemExit(
            "boto3 is required for AV2 downloads. Install it with:\n"
            "  pip install py123d[av2]\n"
            "or directly:\n"
            "  pip install boto3\n"
        ) from exc
    return boto3, UNSIGNED, Config


def resolve_s3_client() -> "botocore.client.BaseClient":
    """Build an anonymous boto3 S3 client for the public ``argoverse`` bucket.

    Uses ``botocore.UNSIGNED`` so no AWS credentials are needed even when the user's
    environment has stale/incorrect ``AWS_*`` vars set.
    """
    boto3, UNSIGNED, Config = _require_boto3()
    # ``max_pool_connections`` matches the thread-pool depth we use when downloading.
    return boto3.client("s3", config=Config(signature_version=UNSIGNED, max_pool_connections=32))


def _variant_key_prefix(variant: str) -> str:
    """Bucket-relative key prefix for a given AV2 variant."""
    if variant not in AV2_SUPPORTED_VARIANTS:
        raise ValueError(f"AV2 variant {variant!r} is not supported. Available: {AV2_SUPPORTED_VARIANTS}")
    return f"{AV2_DATASET_KEY_ROOT}/{variant}"


def list_log_ids(client: "botocore.client.BaseClient", variant: str, s3_split: str) -> List[str]:
    """List all log UUIDs present under ``<variant>/<s3_split>/`` in the bucket.

    :param client: boto3 S3 client.
    :param variant: AV2 variant (currently only ``"sensor"``).
    :param s3_split: S3 folder name (``"train"`` / ``"val"`` / ``"test"``).
    :return: Sorted list of log UUIDs.
    """
    prefix = f"{_variant_key_prefix(variant)}/{s3_split}/"
    paginator = client.get_paginator("list_objects_v2")
    log_ids: set = set()
    for page in paginator.paginate(Bucket=AV2_BUCKET_NAME, Prefix=prefix, Delimiter="/"):
        for common_prefix in page.get("CommonPrefixes") or []:
            # common_prefix["Prefix"] looks like "datasets/av2/<variant>/<s3_split>/<uuid>/"
            rel = common_prefix["Prefix"][len(prefix) :].rstrip("/")
            if rel:
                log_ids.add(rel)
    return sorted(log_ids)


def list_log_object_keys(client: "botocore.client.BaseClient", variant: str, s3_split: str, log_id: str) -> List[str]:
    """List all S3 keys under one log (``<variant>/<s3_split>/<log_id>/``)."""
    prefix = f"{_variant_key_prefix(variant)}/{s3_split}/{log_id}/"
    paginator = client.get_paginator("list_objects_v2")
    keys: List[str] = []
    for page in paginator.paginate(Bucket=AV2_BUCKET_NAME, Prefix=prefix):
        for obj in page.get("Contents") or []:
            keys.append(obj["Key"])
    return keys


# ======================================================================================
# Selection
# ======================================================================================


def select_log_ids(
    all_log_ids: Sequence[str],
    log_ids: Optional[Sequence[str]] = None,
    num_logs: Optional[int] = None,
    sample_random: bool = False,
    seed: int = 0,
) -> List[str]:
    """Filter a full log-id list down to the requested subset.

    Precedence: ``log_ids`` > ``num_logs`` > full list.

    :param all_log_ids: All log UUIDs for a split (sorted).
    :param log_ids: Exact UUIDs to keep; must all exist in ``all_log_ids``.
    :param num_logs: If set, keep the first N (or N random if ``sample_random=True``).
    :param sample_random: Randomize the ``num_logs`` selection.
    :param seed: RNG seed when ``sample_random=True``.
    :return: Selected subset of log UUIDs.
    """
    if log_ids is not None:
        known = set(all_log_ids)
        unknown = [lid for lid in log_ids if lid not in known]
        if unknown:
            raise ValueError(f"Log IDs not found in bucket: {unknown}")
        return list(log_ids)

    total = len(all_log_ids)
    if num_logs is None or num_logs >= total:
        return list(all_log_ids)

    if sample_random:
        rng = _random_mod.Random(seed)
        return sorted(rng.sample(list(all_log_ids), num_logs))
    return list(all_log_ids[:num_logs])


# ======================================================================================
# Download
# ======================================================================================


def _key_to_local_path(key: str, output_dir: Path) -> Path:
    """Map an S3 key back to its on-disk destination.

    ``datasets/av2/<variant>/<split>/<log>/foo/bar.ext`` →
    ``<output_dir>/<variant>/<split>/<log>/foo/bar.ext``
    """
    prefix = f"{AV2_DATASET_KEY_ROOT}/"
    if not key.startswith(prefix):
        raise ValueError(f"Unexpected AV2 blob layout: {key!r}")
    rel = key[len(prefix) :]
    return output_dir / rel


def _download_one_object(
    client: "botocore.client.BaseClient",
    key: str,
    dest_path: Path,
    overwrite: bool,
) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and not overwrite:
        logger.debug("Skip existing file: %s", dest_path)
        return dest_path
    logger.debug("Downloading s3://%s/%s → %s", AV2_BUCKET_NAME, key, dest_path)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    client.download_file(AV2_BUCKET_NAME, key, str(tmp_path))
    tmp_path.replace(dest_path)
    return dest_path


# ======================================================================================
# Downloader (Hydra-instantiable, shared by py123d-download and streaming parsers)
# ======================================================================================


class Av2Downloader(BaseDownloader):
    """Downloader for the Argoverse 2 dataset family.

    Currently implements the ``sensor`` variant only — future variants (``lidar``,
    ``motion-forecasting``, ``tbv``) can slot in by extending
    :data:`AV2_SUPPORTED_VARIANTS` and :data:`AV2_VARIANT_SPLIT_TO_S3` without
    introducing a new class.

    Fetches selected per-log directories from the anonymously-readable ``argoverse``
    S3 bucket into :attr:`output_dir`. The Sensor variant covers ~250 GB across 1000
    logs; provide ``num_logs`` or ``log_ids`` to stage a subset — small subsets make
    this usable for per-conversion streaming without staging the full archive first.
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        variant: str = AV2_SENSOR_VARIANT,
        splits: Optional[Sequence[str]] = None,
        num_logs: Optional[int] = None,
        log_ids: Optional[Dict[str, List[str]]] = None,
        sample_random: bool = False,
        seed: int = 0,
        max_workers: int = 16,
        overwrite: bool = False,
        dry_run: bool = False,
    ) -> None:
        """Initialize the AV2 downloader.

        :param output_dir: Destination directory. When ``None`` the caller (a streaming
            parser) must assign one before invoking :meth:`download`. Files land under
            ``<output_dir>/<variant>/<split>/<log_uuid>/...``.
        :param variant: AV2 variant. Only ``"sensor"`` is currently supported; see
            :data:`AV2_SUPPORTED_VARIANTS`.
        :param splits: 123D split names to fetch. Defaults to every split registered
            for the chosen ``variant`` (for sensor: train/val/test).
        :param num_logs: If set, download the first N logs per split (or N random logs
            when ``sample_random=True``). Applied to any split not covered by
            ``log_ids``.
        :param log_ids: Per-split exact log UUIDs, keyed by 123D split name, e.g.
            ``{"av2-sensor_val": ["00a6ffc1-6ce9-3bc3-a060-6006e9893a1a"]}``. Takes
            precedence over ``num_logs`` for any split it covers.
        :param sample_random: Randomize ``num_logs`` selection.
        :param seed: RNG seed used when ``sample_random=True``.
        :param max_workers: Parallel S3 download threads.
        :param overwrite: If ``False``, skip objects whose local file already exists.
        :param dry_run: If ``True``, log the plan without downloading.
        """
        if variant not in AV2_SUPPORTED_VARIANTS:
            raise ValueError(f"AV2 variant {variant!r} is not supported. Available: {AV2_SUPPORTED_VARIANTS}")

        split_map = AV2_VARIANT_SPLIT_TO_S3[variant]
        available_splits = list(split_map)
        resolved_splits: List[str] = list(splits) if splits else available_splits
        for split in resolved_splits:
            assert split in split_map, (
                f"Split {split!r} is not available for variant={variant!r}. Available: {available_splits}"
            )

        self.output_dir: Optional[Path] = Path(output_dir) if output_dir is not None else None
        self.dry_run: bool = dry_run
        self._variant: str = variant
        self._split_to_s3: Dict[str, str] = dict(split_map)
        self._splits: List[str] = resolved_splits
        self._num_logs: Optional[int] = num_logs
        self._log_ids: Optional[Dict[str, List[str]]] = {k: list(v) for k, v in log_ids.items()} if log_ids else None
        self._sample_random: bool = sample_random
        self._seed: int = seed
        self._max_workers: int = max_workers
        self._overwrite: bool = overwrite

    def download(self) -> None:
        """Inherited, see superclass."""
        assert self.output_dir is not None, "Av2Downloader.output_dir must be set before download()."
        client = resolve_s3_client()
        key_prefix = _variant_key_prefix(self._variant)

        all_keys: List[str] = []
        total_logs_selected = 0
        for split in self._splits:
            s3_split = self._split_to_s3[split]
            per_split_log_ids = self._log_ids.get(split) if self._log_ids else None
            all_log_ids = list_log_ids(client, self._variant, s3_split)
            selected = select_log_ids(
                all_log_ids,
                log_ids=per_split_log_ids,
                num_logs=self._num_logs if per_split_log_ids is None else None,
                sample_random=self._sample_random,
                seed=self._seed,
            )
            logger.info(
                "Selected %d / %d logs for split=%s",
                len(selected),
                len(all_log_ids),
                split,
            )
            for log_id in selected:
                log_keys = list_log_object_keys(client, self._variant, s3_split, log_id)
                logger.debug("  log %s: %d objects", log_id, len(log_keys))
                all_keys.extend(log_keys)
            total_logs_selected += len(selected)

        logger.info("AV2 %s target directory: %s", self._variant, self.output_dir)
        logger.info("AV2 %s bucket:           s3://%s/%s/", self._variant, AV2_BUCKET_NAME, key_prefix)
        logger.info("AV2 %s logs selected:    %d", self._variant, total_logs_selected)
        logger.info("AV2 %s objects:          %d", self._variant, len(all_keys))

        if self.dry_run:
            logger.info("dry_run=True — not downloading. Plan covers %d object(s).", len(all_keys))
            for key in all_keys[: min(10, len(all_keys))]:
                logger.info("  s3://%s/%s", AV2_BUCKET_NAME, key)
            return

        if not all_keys:
            logger.warning("No objects selected — nothing to download.")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, self._max_workers)) as pool:
            futures = {
                pool.submit(
                    _download_one_object,
                    client,
                    key,
                    _key_to_local_path(key, self.output_dir),
                    self._overwrite,
                ): key
                for key in all_keys
            }
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                key = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    logger.error("Failed to download %s: %s", key, exc)
                    raise
                completed += 1
                if completed % 500 == 0 or completed == len(futures):
                    logger.info("  downloaded %d / %d objects", completed, len(futures))
        logger.info("AV2 %s download complete: %s", self._variant, self.output_dir)
