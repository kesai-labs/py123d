"""Download utilities for the NVIDIA PhysicalAI-Autonomous-Vehicles-NuRec dataset.

The NuRec dataset is gated on Hugging Face. Access requires a HF account that has
accepted the NVIDIA AV dataset license agreement, plus a token supplied via the
``HF_TOKEN`` environment variable or the ``hf_token`` downloader argument.

Dataset: https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec

Per-sequence on-disk layout after download (mirroring the HF repo structure)::

    sample_set/26.04_release/{sequence_id}/
    ├── {sequence_id}.usdz              (USDZ package: ego, boxes, map, neural volume)
    ├── camera_front_wide_120fov.mp4    (sidecar camera — ~150 MB each, up to 7 cameras)
    ├── camera_front_tele_30fov.mp4
    ├── camera_cross_left_120fov.mp4
    ├── camera_cross_right_120fov.mp4
    ├── camera_rear_left_70fov.mp4
    ├── camera_rear_right_70fov.mp4
    └── camera_rear_tele_30fov.mp4

This module exposes :class:`NuRecDownloader` (Hydra-instantiable) and a handful of
reusable library functions. :class:`NuRecDownloader` powers two paths:

1. The ``py123d-download dataset=nurec`` CLI — bulk-fetches all selected sequences
   into :attr:`NuRecDownloader.output_dir`.

2. The ``NuRecParser`` streaming path — calls
   :meth:`NuRecDownloader.download_single_sequence` for each sequence into a per-sequence
   temp directory, converts it, and deletes the temp dir before moving on.
"""

from __future__ import annotations

import logging
import random as _random_mod
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from py123d.parser.base_downloader import BaseDownloader

logger = logging.getLogger(__name__)

NUREC_REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"
NUREC_REPO_TYPE = "dataset"

# Default prefix for sequence directories inside the HF repo.
# Mirrors the local on-disk layout: {data_root}/sample_set/26.04_release/{sequence_id}/
NUREC_DEFAULT_HF_SEQUENCES_PREFIX = "sample_set/26.04_release"

# Sidecar camera MP4 file names (one per camera, lives alongside the USDZ).
NUREC_CAMERA_NAMES = (
    "camera_front_wide_120fov",
    "camera_front_tele_30fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_rear_left_70fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
)


def _require_hf_hub():
    """Lazy import — the dependency is optional until a downloader is instantiated."""
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for NuRec downloads. Install it with:\n  pip install py123d[nurec]\n"
        ) from exc
    return HfApi, snapshot_download


def resolve_hf_token(cli_token: Optional[str] = None) -> Optional[str]:
    """Resolve the HuggingFace token from (in order): explicit arg, ``HF_TOKEN``, ``HUGGINGFACE_HUB_TOKEN``."""
    import os

    return cli_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def list_all_sequence_ids(
    token: Optional[str] = None,
    revision: str = "main",
    hf_repo_id: str = NUREC_REPO_ID,
    hf_sequences_prefix: str = NUREC_DEFAULT_HF_SEQUENCES_PREFIX,
) -> List[str]:
    """List all sequence UUIDs present under ``{hf_sequences_prefix}/`` in the repo.

    :param token: HuggingFace access token.
    :param revision: Dataset branch/tag/commit.
    :param hf_repo_id: HuggingFace repo ID.
    :param hf_sequences_prefix: Path prefix for sequences inside the HF repo.
    :return: Sorted list of sequence UUIDs.
    """
    HfApi, _ = _require_hf_hub()
    api = HfApi(token=token)
    entries = api.list_repo_tree(
        repo_id=hf_repo_id,
        repo_type=NUREC_REPO_TYPE,
        path_in_repo=hf_sequences_prefix,
        revision=revision,
        recursive=False,
    )
    prefix = hf_sequences_prefix.rstrip("/") + "/"
    return sorted(Path(e.path).name for e in entries if e.path.startswith(prefix))


def build_sequence_allow_patterns(
    sequence_ids: Sequence[str],
    hf_sequences_prefix: str = NUREC_DEFAULT_HF_SEQUENCES_PREFIX,
    cameras: Optional[Sequence[str]] = None,
    include_usdz: bool = True,
    include_sidecars: bool = True,
) -> List[str]:
    """Build ``allow_patterns`` for ``snapshot_download`` covering the given sequences.

    :param sequence_ids: Sequence UUIDs to include.
    :param hf_sequences_prefix: Path prefix for sequences in the HF repo.
    :param cameras: Sidecar camera names to include. Defaults to all 7 when ``None``.
    :param include_usdz: Include the USDZ package file.
    :param include_sidecars: Include sidecar camera MP4 files.
    :return: ``allow_patterns`` list for ``snapshot_download``.
    """
    patterns: List[str] = []
    prefix = hf_sequences_prefix.rstrip("/")
    target_cameras = cameras if cameras is not None else NUREC_CAMERA_NAMES
    for seq_id in sequence_ids:
        base = f"{prefix}/{seq_id}"
        if include_usdz:
            patterns.append(f"{base}/{seq_id}.usdz")
        if include_sidecars:
            for cam in target_cameras:
                patterns.append(f"{base}/{cam}.mp4")
    return patterns


def download_sequence(
    sequence_id: str,
    output_dir: Path,
    hf_repo_id: str = NUREC_REPO_ID,
    hf_sequences_prefix: str = NUREC_DEFAULT_HF_SEQUENCES_PREFIX,
    cameras: Optional[Sequence[str]] = None,
    include_usdz: bool = True,
    include_sidecars: bool = True,
    hf_token: Optional[str] = None,
    revision: str = "main",
    max_workers: int = 4,
) -> Path:
    """Download a single sequence into ``output_dir`` and return the sequence root path.

    The sequence lands at ``{output_dir}/{hf_sequences_prefix}/{sequence_id}/``.
    Only files for this specific sequence are fetched.

    :param sequence_id: Sequence UUID.
    :param output_dir: Destination root (typically a ``tempfile.TemporaryDirectory``).
    :param hf_repo_id: HuggingFace repo ID.
    :param hf_sequences_prefix: Path prefix for sequences in the HF repo.
    :param cameras: Sidecar camera names to include. ``None`` = all cameras.
    :param include_usdz: Download the USDZ package.
    :param include_sidecars: Download sidecar camera MP4 files.
    :param hf_token: HuggingFace access token.
    :param revision: HF dataset revision.
    :param max_workers: Parallel download workers.
    :return: Path to the downloaded sequence root directory.
    """
    _, snapshot_download = _require_hf_hub()
    allow_patterns = build_sequence_allow_patterns(
        sequence_ids=[sequence_id],
        hf_sequences_prefix=hf_sequences_prefix,
        cameras=cameras,
        include_usdz=include_usdz,
        include_sidecars=include_sidecars,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=hf_repo_id,
        repo_type=NUREC_REPO_TYPE,
        revision=revision,
        local_dir=str(output_dir),
        allow_patterns=allow_patterns,
        token=hf_token,
        max_workers=max_workers,
    )

    sequence_root = output_dir / hf_sequences_prefix.rstrip("/") / sequence_id
    usdz_path = sequence_root / f"{sequence_id}.usdz"
    if include_usdz and not usdz_path.exists():
        raise RuntimeError(
            f"Sequence {sequence_id} download completed but USDZ {usdz_path} is missing. "
            "The sequence may not exist at the requested revision, or the HF token lacks access."
        )
    return sequence_root


# ======================================================================================
# Downloader (Hydra-instantiable, shared by py123d-download and the NuRec streaming parser)
# ======================================================================================


class NuRecDownloader(BaseDownloader):
    """Downloader for the NVIDIA PhysicalAI-Autonomous-Vehicles-NuRec dataset.

    Operates in two modes:

    * :meth:`download` — bulk-fetch all selected sequences into :attr:`output_dir` in one
      ``snapshot_download`` call. Used by ``py123d-download dataset=nurec``.
    * :meth:`download_single_sequence` — fetch one sequence to a caller-provided directory.
      Used by :class:`~py123d.parser.nurec.nurec_parser.NuRecParser` in streaming mode to
      drop each sequence into a per-sequence temp directory.

    The instance is picklable (simple attrs only) so it can be embedded in log-parser
    objects shipped across a Ray process pool.
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        hf_repo_id: str = NUREC_REPO_ID,
        hf_sequences_prefix: str = NUREC_DEFAULT_HF_SEQUENCES_PREFIX,
        revision: str = "main",
        hf_token: Optional[str] = None,
        sequence_ids: Optional[List[str]] = None,
        num_sequences: Optional[int] = None,
        sample_random: bool = False,
        seed: int = 0,
        cameras: Optional[List[str]] = None,
        include_usdz: bool = True,
        include_sidecars: bool = True,
        max_workers: int = 4,
        dry_run: bool = False,
    ) -> None:
        """Initialize the NuRec downloader.

        :param output_dir: Destination directory for :meth:`download`. Ignored by
            :meth:`download_single_sequence` and by streaming parsers.
        :param hf_repo_id: HuggingFace repo ID for the NuRec dataset.
        :param hf_sequences_prefix: Path prefix for sequence directories inside the HF repo.
        :param revision: HuggingFace dataset branch, tag, or commit.
        :param hf_token: HF access token. Resolves through :func:`resolve_hf_token`.
        :param sequence_ids: Explicit sequence UUIDs. Mutually exclusive with ``num_sequences``.
        :param num_sequences: Select the first N sequences (or N random sequences when
            ``sample_random=True``) from the full catalog.
        :param sample_random: Randomize ``num_sequences`` selection.
        :param seed: RNG seed used when ``sample_random=True``.
        :param cameras: Sidecar camera names to include. Defaults to all 7 cameras when ``None``.
        :param include_usdz: Whether to download the USDZ package for each sequence.
        :param include_sidecars: Whether to download sidecar camera MP4 files.
        :param max_workers: Parallel HF download workers.
        :param dry_run: If ``True``, :meth:`download` logs the plan without fetching.
        """
        if sequence_ids and num_sequences is not None:
            raise ValueError("sequence_ids and num_sequences are mutually exclusive.")
        if num_sequences is not None and num_sequences <= 0:
            raise ValueError("num_sequences must be a positive integer.")
        if cameras is not None:
            for cam in cameras:
                if cam not in NUREC_CAMERA_NAMES:
                    raise ValueError(f"camera {cam!r} is not valid; must be one of {NUREC_CAMERA_NAMES}")

        self.output_dir: Optional[Path] = Path(output_dir) if output_dir is not None else None
        self.dry_run: bool = dry_run

        self.hf_repo_id: str = hf_repo_id
        self.hf_sequences_prefix: str = hf_sequences_prefix
        self.revision: str = revision
        self.hf_token: Optional[str] = resolve_hf_token(hf_token)
        self.cameras: Optional[Tuple[str, ...]] = tuple(cameras) if cameras else None
        self.include_usdz: bool = include_usdz
        self.include_sidecars: bool = include_sidecars
        self.max_workers: int = max_workers

        self._explicit_sequence_ids: Optional[List[str]] = list(sequence_ids) if sequence_ids else None
        self._num_sequences: Optional[int] = num_sequences
        self._sample_random: bool = sample_random
        self._seed: int = seed

        if self.hf_token is None:
            logger.warning(
                "No HF token configured for NuRecDownloader. NuRec is gated — set $HF_TOKEN "
                "or pass hf_token if downloads fail with 401/403."
            )

    def resolve_sequence_ids(self) -> List[str]:
        """Return the sequence UUIDs selected by the current configuration.

        Fetches the full catalog from HuggingFace when ``sequence_ids`` was not explicitly
        configured. Used by :meth:`download` and by the streaming parser to enumerate
        work across Ray workers.
        """
        if self._explicit_sequence_ids:
            return list(self._explicit_sequence_ids)

        all_ids = list_all_sequence_ids(
            token=self.hf_token,
            revision=self.revision,
            hf_repo_id=self.hf_repo_id,
            hf_sequences_prefix=self.hf_sequences_prefix,
        )
        logger.info("NuRec catalog: %d sequences at %s@%s", len(all_ids), self.hf_repo_id, self.revision)

        if self._num_sequences is None or self._num_sequences >= len(all_ids):
            return all_ids
        if self._sample_random:
            rng = _random_mod.Random(self._seed)
            return sorted(rng.sample(all_ids, self._num_sequences))
        return all_ids[: self._num_sequences]

    def download(self) -> None:
        """Inherited, see superclass. Bulk-fetches all selected sequences into ``output_dir``."""
        assert self.output_dir is not None, "NuRecDownloader.output_dir must be set before download()."
        sequence_ids = self.resolve_sequence_ids()

        allow_patterns = build_sequence_allow_patterns(
            sequence_ids=sequence_ids,
            hf_sequences_prefix=self.hf_sequences_prefix,
            cameras=list(self.cameras) if self.cameras else None,
            include_usdz=self.include_usdz,
            include_sidecars=self.include_sidecars,
        )

        logger.info("NuRec target directory:   %s", self.output_dir)
        logger.info("NuRec repo:               %s@%s", self.hf_repo_id, self.revision)
        logger.info(
            "NuRec sidecars:           %s%s",
            "enabled" if self.include_sidecars else "disabled",
            f" (cameras={list(self.cameras)})" if self.cameras else "",
        )
        logger.info("NuRec sequences selected: %d", len(sequence_ids))
        for pat in allow_patterns[:10]:
            logger.debug("  allow_pattern: %s", pat)

        if self.dry_run:
            logger.info("dry_run=True — not downloading. Plan covers %d sequence(s).", len(sequence_ids))
            return

        if not sequence_ids:
            logger.warning("No sequences selected — nothing to download.")
            return

        _, snapshot_download = _require_hf_hub()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=self.hf_repo_id,
            repo_type=NUREC_REPO_TYPE,
            revision=self.revision,
            local_dir=str(self.output_dir),
            allow_patterns=allow_patterns,
            token=self.hf_token,
            max_workers=self.max_workers,
        )
        logger.info("NuRec download complete: %s", self.output_dir)

    def download_single_sequence(self, sequence_id: str, output_dir: Union[str, Path]) -> Path:
        """Fetch one sequence to ``output_dir`` using the configured auth/cameras.

        Intended for the NuRec streaming parser — each sequence lands in its own per-sequence
        temp directory, is converted, and the temp directory is deleted. Returns the path to
        the sequence root directory (where the USDZ and sidecar MP4s live).
        """
        return download_sequence(
            sequence_id=sequence_id,
            output_dir=Path(output_dir),
            hf_repo_id=self.hf_repo_id,
            hf_sequences_prefix=self.hf_sequences_prefix,
            cameras=list(self.cameras) if self.cameras else None,
            include_usdz=self.include_usdz,
            include_sidecars=self.include_sidecars,
            hf_token=self.hf_token,
            revision=self.revision,
            max_workers=self.max_workers,
        )
