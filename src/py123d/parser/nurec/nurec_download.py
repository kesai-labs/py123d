"""Download utilities for the NVIDIA PhysicalAI-Autonomous-Vehicles-NuRec dataset.

The NuRec dataset is gated on Hugging Face. Access requires a HF account that has
accepted the NVIDIA AV dataset license agreement, plus a token supplied via the
``HF_TOKEN`` environment variable or the ``hf_token`` downloader argument.

Dataset: https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec

Per-sequence on-disk layout after download (mirroring the HF repo structure)::

    sample_set/{release}_release/{sequence_id}/
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

1. The ``py123d-download dataset=nurec-<...>`` CLI — fetches all selected sequences
   into :attr:`NuRecDownloader.output_dir`.

2. The ``NuRecParser`` streaming path — fetches all selected sequences into a temporary
   directory, which is converted and deleted afterwards.
"""

from __future__ import annotations

import logging
import random as _random_mod
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, TypeVar, Union

from py123d.parser.base_downloader import BaseDownloader
from py123d.parser.nurec.nurec_splits import (
    NUREC_DEFAULT_SPLITS,
    NUREC_RELEASE_SPLITS,
    NUREC_RELEASES,
    resolve_scene_lists,
    scenes_of_split,
    validate_splits,
)

_T = TypeVar("_T")

logger = logging.getLogger(__name__)

NUREC_REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"
NUREC_REPO_TYPE = "dataset"

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
    hf_sequences_prefix: str,
    revision: str,
    token: Optional[str] = None,
    hf_repo_id: str = NUREC_REPO_ID,
) -> List[str]:
    """List all sequence UUIDs present under ``{hf_sequences_prefix}/`` in the repo.

    :param hf_sequences_prefix: Path prefix for sequences inside the HF repo.
    :param revision: Dataset branch/tag/commit.
    :param token: HuggingFace access token.
    :param hf_repo_id: HuggingFace repo ID.
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
    hf_sequences_prefix: str,
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


def _scene_names(scene: Tuple[str, str]) -> set:
    """The names a (release, sequence UUID) scene can be requested by in ``sequence_ids``."""
    release, sequence_id = scene
    return {sequence_id, f"{release}/{sequence_id}"}


# ======================================================================================
# Downloader (Hydra-instantiable, shared by py123d-download and the NuRec streaming parser)
# ======================================================================================


class NuRecDownloader(BaseDownloader):
    """Downloader for the NVIDIA PhysicalAI-Autonomous-Vehicles-NuRec dataset.

    The sequences to download are selected by split. A split is either a whole release
    (``nurec-2601_train``, ``nurec-2604_train``) or a list of named scenes
    (``nurec-curated_train``, ``nurec-curated_val``, see ``scene_lists``), which can span
    several releases. ``sequence_ids`` or ``num_sequences`` narrow the selection down.

    :meth:`download` fetches all selected sequences into :attr:`output_dir`, with one
    ``snapshot_download`` call per release. It is used by ``py123d-download dataset=nurec-<...>``,
    and by :class:`~py123d.parser.nurec.nurec_parser.NuRecParser` in streaming mode, which
    points :attr:`output_dir` at a temporary directory.
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        hf_repo_id: str = NUREC_REPO_ID,
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
        splits: Optional[List[str]] = None,
        scene_lists: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Initialize the NuRec downloader.

        :param output_dir: Destination directory for :meth:`download`. A streaming parser
            assigns a temporary directory when this is ``None``.
        :param hf_repo_id: HuggingFace repo ID for the NuRec dataset.
        :param hf_token: HF access token. Resolves through :func:`resolve_hf_token`.
        :param sequence_ids: Narrows the splits down to these sequences, each given as a
            sequence UUID or as ``"<release>/<sequence UUID>"``. Mutually exclusive with
            ``num_sequences``.
        :param num_sequences: Select the first N sequences of each split (or N random
            sequences when ``sample_random=True``).
        :param sample_random: Randomize ``num_sequences`` selection.
        :param seed: RNG seed used when ``sample_random=True``.
        :param cameras: Sidecar camera names to include. Defaults to all 7 cameras when ``None``.
        :param include_usdz: Whether to download the USDZ package for each sequence.
        :param include_sidecars: Whether to download sidecar camera MP4 files.
        :param max_workers: Parallel HF download workers.
        :param dry_run: If ``True``, :meth:`download` logs the plan without fetching.
        :param splits: Splits to download, defaults to ``nurec-curated_train`` and
            ``nurec-curated_val``. Each release they draw from is fetched from its own
            HuggingFace branch.
        :param scene_lists: Mapping of split name to scene names, each ``"<release>/<scene uuid>"``.
            Defaults to the bundled train/val lists.
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
        self.hf_token: Optional[str] = resolve_hf_token(hf_token)
        self.cameras: Optional[Tuple[str, ...]] = tuple(cameras) if cameras else None
        self.include_usdz: bool = include_usdz
        self.include_sidecars: bool = include_sidecars
        self.max_workers: int = max_workers

        self.sequence_ids: Optional[List[str]] = list(sequence_ids) if sequence_ids else None
        self.num_sequences: Optional[int] = num_sequences
        self.sample_random: bool = sample_random
        self.seed: int = seed

        self.scene_lists: Dict[str, List[str]] = resolve_scene_lists(scene_lists)
        self.splits: List[str] = validate_splits(
            splits if splits is not None else NUREC_DEFAULT_SPLITS, self.scene_lists
        )

        if self.hf_token is None:
            logger.warning(
                "No HF token configured for NuRecDownloader. NuRec is gated — set $HF_TOKEN "
                "or pass hf_token if downloads fail with 401/403."
            )

    def resolve_release_sequence_ids(self) -> Dict[str, List[str]]:
        """Return the sequence UUIDs selected by the current configuration, grouped by release.

        Fetches a release's catalog from HuggingFace for a split that covers the whole release.
        """
        requested = set(self.sequence_ids) if self.sequence_ids else None
        matched: set = set()
        sequence_ids: Dict[str, set] = {}
        for split in self.splits:
            if split in NUREC_RELEASE_SPLITS:
                release = NUREC_RELEASES[NUREC_RELEASE_SPLITS[split]]
                catalog = list_all_sequence_ids(
                    hf_sequences_prefix=release.hf_sequences_prefix,
                    revision=release.revision,
                    token=self.hf_token,
                    hf_repo_id=self.hf_repo_id,
                )
                scenes = [(release.name, sequence_id) for sequence_id in catalog]
            else:
                scenes = scenes_of_split(split, self.scene_lists)
            if requested is not None:
                scenes = [scene for scene in scenes if requested & _scene_names(scene)]
                for scene in scenes:
                    matched |= requested & _scene_names(scene)
            scenes = self._limit(scenes)
            logger.info("NuRec split %s: %d sequence(s)", split, len(scenes))
            for release_name, sequence_id in scenes:
                sequence_ids.setdefault(release_name, set()).add(sequence_id)
        if requested is not None and requested - matched:
            logger.warning(
                "NuRec: %d requested sequence_ids are in none of the splits %s: %s",
                len(requested - matched),
                self.splits,
                sorted(requested - matched),
            )
        return {release_name: sorted(ids) for release_name, ids in sorted(sequence_ids.items())}

    def _limit(self, candidates: List[_T]) -> List[_T]:
        """Applies ``num_sequences`` and ``sample_random`` to a list of candidates."""
        if self.num_sequences is None or self.num_sequences >= len(candidates):
            return candidates
        if self.sample_random:
            rng = _random_mod.Random(self.seed)
            return sorted(rng.sample(candidates, self.num_sequences))
        return candidates[: self.num_sequences]

    def download(self) -> None:
        """Inherited, see superclass. Bulk-fetches all selected sequences into ``output_dir``."""
        assert self.output_dir is not None, "NuRecDownloader.output_dir must be set before download()."
        # One (hf_sequences_prefix, revision, sequence_ids) entry per release to fetch from.
        targets = [
            (NUREC_RELEASES[release].hf_sequences_prefix, NUREC_RELEASES[release].revision, sequence_ids)
            for release, sequence_ids in self.resolve_release_sequence_ids().items()
        ]
        num_selected = sum(len(sequence_ids) for _, _, sequence_ids in targets)

        logger.info("NuRec target directory:   %s", self.output_dir)
        for hf_sequences_prefix, revision, sequence_ids in targets:
            logger.info(
                "NuRec repo:               %s@%s, %d sequence(s) from %s",
                self.hf_repo_id,
                revision,
                len(sequence_ids),
                hf_sequences_prefix,
            )
        logger.info(
            "NuRec sidecars:           %s%s",
            "enabled" if self.include_sidecars else "disabled",
            f" (cameras={list(self.cameras)})" if self.cameras else "",
        )
        logger.info("NuRec sequences selected: %d", num_selected)

        if self.dry_run:
            logger.info("dry_run=True — not downloading. Plan covers %d sequence(s).", num_selected)
            return

        if num_selected == 0:
            logger.warning("No sequences selected — nothing to download.")
            return

        _, snapshot_download = _require_hf_hub()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for hf_sequences_prefix, revision, sequence_ids in targets:
            allow_patterns = build_sequence_allow_patterns(
                sequence_ids=sequence_ids,
                hf_sequences_prefix=hf_sequences_prefix,
                cameras=list(self.cameras) if self.cameras else None,
                include_usdz=self.include_usdz,
                include_sidecars=self.include_sidecars,
            )
            for pat in allow_patterns[:10]:
                logger.debug("  allow_pattern: %s", pat)
            snapshot_download(
                repo_id=self.hf_repo_id,
                repo_type=NUREC_REPO_TYPE,
                revision=revision,
                local_dir=str(self.output_dir),
                allow_patterns=allow_patterns,
                token=self.hf_token,
                max_workers=self.max_workers,
            )
        logger.info("NuRec download complete: %s", self.output_dir)
