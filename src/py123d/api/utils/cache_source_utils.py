"""Build and check the :class:`CacheSourceInfo` of derived modalities."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pyarrow as pa

from py123d.datatypes.metadata.cache_source import CacheSourceInfo, CacheSourceModality
from py123d.datatypes.modalities.base_modality import BaseModalityMetadata

_HASH_DIGEST_SIZE = 16


class StaleModalityError(ValueError):
    """Raised when a source modality changed after the derived modality was written."""


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string, e.g. ``2026-08-31T09:14:22Z``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_cache_source_info(modality_metadata: BaseModalityMetadata) -> Optional[CacheSourceInfo]:
    """The :class:`CacheSourceInfo` of a modality metadata, or None for raw modalities."""
    cache_source_info = getattr(modality_metadata, "cache_source_info", None)
    return cache_source_info if isinstance(cache_source_info, CacheSourceInfo) else None


def _column_bytes(column: pa.ChunkedArray, qualified_name: str) -> bytes:
    """Value bytes of a primitive or fixed-size-list-of-primitive column."""
    array = column.combine_chunks()
    if pa.types.is_fixed_size_list(array.type):
        array = array.flatten()
    if not (pa.types.is_primitive(array.type) or pa.types.is_boolean(array.type)):
        raise TypeError(
            f"Cannot hash column '{qualified_name}' of type {array.type}: "
            "only primitive and fixed-size-list-of-primitive columns are supported."
        )
    return array.to_numpy(zero_copy_only=False).tobytes()


def hash_modality_columns(log_dir: Path, modality_key: str, columns: Sequence[str]) -> str:
    """Hash the given columns of one modality file in a log directory.

    :param log_dir: The log directory holding ``{modality_key}.arrow``.
    :param modality_key: The modality key, e.g. ``"ego_state_se3"``.
    :param columns: Column names without the modality prefix, e.g. ``["imu_se3"]``.
    :return: A hex digest over the columns' values, in the given order.
    :raises FileNotFoundError: If the modality file does not exist.
    """
    from py123d.api.utils.arrow_helper import open_arrow_table

    path = Path(log_dir) / f"{modality_key}.arrow"
    if not path.exists():
        raise FileNotFoundError(f"Cannot hash '{modality_key}': {path} does not exist.")

    table = open_arrow_table(path)
    digest = hashlib.blake2b(digest_size=_HASH_DIGEST_SIZE)
    for column in columns:
        qualified_name = f"{modality_key}.{column}"
        digest.update(qualified_name.encode())
        digest.update(_column_bytes(table.column(qualified_name), qualified_name))
    return digest.hexdigest()


def build_cache_source_info(
    log_dir: Path,
    computed_by: str,
    source_columns: Dict[str, Sequence[str]],
    external_sources: Optional[List[str]] = None,
) -> CacheSourceInfo:
    """Hash the given source modalities and build the :class:`CacheSourceInfo` of a derived modality.

    :param log_dir: The log directory the source modalities are read from.
    :param computed_by: ``"<namespace>:<name>@<version>"``, e.g. ``"garage:route_backfill@3"``.
    :param source_columns: Mapping of source modality key to the columns the computation read.
    :param external_sources: Descriptions of sources outside the log.
    """
    source_modalities = [
        CacheSourceModality(
            modality_key=modality_key,
            columns=list(columns),
            hash=hash_modality_columns(log_dir, modality_key, columns),
        )
        for modality_key, columns in sorted(source_columns.items())
    ]
    return CacheSourceInfo(
        computed_by=computed_by,
        computed_at=utc_now_iso(),
        source_modalities=source_modalities,
        external_sources=external_sources,
    )


@lru_cache(maxsize=10_000)
def check_cache_source_modalities(log_dir: Path) -> None:
    """Re-hash the source modalities of every derived modality in a log directory.

    Cached per process, so a log is checked once. Logs that cannot be read, or whose
    source modality is missing, are skipped here and left to the caller's error handling.

    :param log_dir: The log directory to check.
    :raises StaleModalityError: If a source modality no longer matches its recorded hash.
    """
    from py123d.api.utils.arrow_metadata_utils import parse_log_directory_metadata

    log_dir = Path(log_dir)
    try:
        modality_metadatas = parse_log_directory_metadata(log_dir).modality_metadatas
    except Exception:
        return

    for modality_key, modality_metadata in modality_metadatas.items():
        cache_source_info = get_cache_source_info(modality_metadata)
        if cache_source_info is None:
            continue
        for source in cache_source_info.source_modalities:
            try:
                actual_hash = hash_modality_columns(log_dir, source.modality_key, source.columns)
            except (FileNotFoundError, KeyError, TypeError):
                continue
            if actual_hash != source.hash:
                raise StaleModalityError(
                    f"'{modality_key}' in {log_dir} is outdated: source '{source.modality_key}' "
                    f"(columns {', '.join(source.columns)}) changed since '{cache_source_info.computed_by}' "
                    f"wrote it at {cache_source_info.computed_at}. Recompute '{modality_key}'."
                )
