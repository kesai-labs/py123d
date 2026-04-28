"""Download utilities for the NVIDIA PhysicalAI-Autonomous-Vehicles-NCore dataset.

The NCore dataset is gated on Hugging Face. Access requires a HF account that has
accepted the NVIDIA AV dataset license agreement, plus a token supplied via the
``HF_TOKEN`` environment variable or the ``hf_token`` downloader argument.

Dataset: https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore
Devkit:  https://github.com/NVIDIA/ncore

Per-clip on-disk layout (one UUID-named subdirectory under ``clips/``)::

    clips/{clip_id}/
    ├── pai_{clip_id}.json                                        (sequence manifest)
    ├── pai_{clip_id}.ncore4.zarr.itar                            (poses, intrinsics, cuboids)
    ├── pai_{clip_id}.ncore4-lidar_top_360fov.zarr.itar           (~1.0 GB)
    └── pai_{clip_id}.ncore4-camera_{name}.zarr.itar              (~150 MB x 7 cameras)

This module exposes :class:`NCoreDownloader` (Hydra-instantiable) and a handful of
reusable library functions. :class:`NCoreDownloader` powers two paths:

1. The ``py123d-download dataset=ncore`` CLI — bulk-fetches all selected clips into
   :attr:`NCoreDownloader.output_dir`.

2. The ``NCoreParser`` streaming path — calls
   :meth:`NCoreDownloader.download_single_clip` for each clip into a per-clip temp
   directory, converts it, and deletes the temp dir before moving on.
"""

from __future__ import annotations

import logging
import random as _random_mod
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from py123d.parser.base_downloader import BaseDownloader

logger = logging.getLogger(__name__)

NCORE_REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles-NCore"
NCORE_REPO_TYPE = "dataset"

CAMERA_IDS = (
    "camera_front_wide_120fov",
    "camera_front_tele_30fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_rear_left_70fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
)

MODALITY_CHOICES = ("all", "metadata", "lidar", "cameras")

# Top-level files in the repo (outside of clips/). Downloaded by default unless
# ``clips_only=True``.
REPO_META_FILES = ("README.md", "rename_clip_folders.py")


def _require_hf_hub():
    """Lazy import — the dependency is optional until a downloader is instantiated."""
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for this module. Install it with:\n  pip install py123d[ncore]\n"
        ) from exc
    return HfApi, snapshot_download


def resolve_hf_token(cli_token: Optional[str] = None) -> Optional[str]:
    """Resolve the HuggingFace token from (in order): explicit arg, ``HF_TOKEN``, ``HUGGINGFACE_HUB_TOKEN``."""
    import os

    return cli_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def list_all_clip_ids(token: Optional[str] = None, revision: str = "main") -> List[str]:
    """List all clip UUIDs present under ``clips/`` in the repo.

    :param token: HuggingFace access token (optional for public listings, required for gated
        repos — use :func:`resolve_hf_token` for the standard fallback chain).
    :param revision: Dataset branch/tag/commit.
    :return: Sorted list of clip UUIDs.
    """
    HfApi, _ = _require_hf_hub()
    api = HfApi(token=token)
    entries = api.list_repo_tree(
        repo_id=NCORE_REPO_ID,
        repo_type=NCORE_REPO_TYPE,
        path_in_repo="clips",
        revision=revision,
        recursive=False,
    )
    return sorted(Path(e.path).name for e in entries if e.path.startswith("clips/"))


def build_clip_allow_patterns(
    clip_ids: Sequence[str],
    modality: str = "all",
    cameras: Optional[Sequence[str]] = None,
) -> List[str]:
    """Build ``allow_patterns`` for ``snapshot_download`` that cover the given clips+modalities.

    Selection logic:

    - ``metadata``: always include ``pai_{id}.json`` + the default component store
      (poses, intrinsics, cuboids).
    - ``lidar``:    also include the top lidar ``.zarr.itar``.
    - ``cameras``:  also include one ``.zarr.itar`` per requested camera (or all 7 if
      ``cameras`` is ``None``).
    - ``all``:      every file under the clip directory.
    """
    patterns: List[str] = []
    for clip_id in clip_ids:
        base = f"clips/{clip_id}"
        if modality == "all":
            patterns.append(f"{base}/*")
            continue

        # metadata is always included for non-"all" modalities so the sequence remains loadable.
        patterns.append(f"{base}/pai_{clip_id}.json")
        patterns.append(f"{base}/pai_{clip_id}.ncore4.zarr.itar")

        if modality == "lidar":
            patterns.append(f"{base}/pai_{clip_id}.ncore4-lidar_top_360fov.zarr.itar")
        elif modality == "cameras":
            target_cams = cameras if cameras else CAMERA_IDS
            for cam in target_cams:
                patterns.append(f"{base}/pai_{clip_id}.ncore4-{cam}.zarr.itar")

    return patterns


def download_clip(
    clip_id: str,
    output_dir: Path,
    modality: str = "all",
    cameras: Optional[Sequence[str]] = None,
    hf_token: Optional[str] = None,
    revision: str = "main",
    max_workers: int = 4,
) -> Path:
    """Download a single clip into ``output_dir`` and return the path to its sequence manifest.

    The clip is written to ``{output_dir}/clips/{clip_id}/``. Only the per-clip files are
    fetched (no repo-level README etc.) — useful for per-clip streaming during conversion.

    :param clip_id: Clip UUID.
    :param output_dir: Destination directory (typically a ``tempfile.TemporaryDirectory``).
    :param modality: Which modalities to pull for this clip. See :func:`build_clip_allow_patterns`.
    :param cameras: Camera IDs to pull when ``modality="cameras"``.
    :param hf_token: HuggingFace access token.
    :param revision: HF dataset revision.
    :param max_workers: Parallel download workers (per clip).
    :return: Path to ``{output_dir}/clips/{clip_id}/pai_{clip_id}.json``.
    """
    _, snapshot_download = _require_hf_hub()
    allow_patterns = build_clip_allow_patterns([clip_id], modality=modality, cameras=cameras)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=NCORE_REPO_ID,
        repo_type=NCORE_REPO_TYPE,
        revision=revision,
        local_dir=str(output_dir),
        allow_patterns=allow_patterns,
        token=hf_token,
        max_workers=max_workers,
    )
    manifest_path = output_dir / "clips" / clip_id / f"pai_{clip_id}.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"Clip {clip_id} download completed but manifest {manifest_path} is missing. "
            "The clip may not exist on the requested revision, or the HF token lacks access."
        )
    return manifest_path


# ======================================================================================
# Downloader (Hydra-instantiable, shared by py123d-download and the NCore streaming parser)
# ======================================================================================


class NCoreDownloader(BaseDownloader):
    """Downloader for the NVIDIA PhysicalAI-Autonomous-Vehicles-NCore dataset.

    Operates in two modes:

    * :meth:`download` — bulk-fetch all selected clips into :attr:`output_dir` in one
      ``snapshot_download`` call. Used by ``py123d-download dataset=ncore``.
    * :meth:`download_single_clip` — fetch one clip to a caller-provided directory.
      Used by :class:`~py123d.parser.ncore.ncore_parser.NCoreParser` in streaming
      mode to drop each clip into a per-clip temp directory.

    The instance is picklable (simple attrs only) so it can be embedded in log-parser
    objects shipped across a Ray process pool.
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        revision: str = "main",
        hf_token: Optional[str] = None,
        clip_ids: Optional[List[str]] = None,
        num_clips: Optional[int] = None,
        sample_random: bool = False,
        seed: int = 0,
        modality: str = "all",
        cameras: Optional[List[str]] = None,
        max_workers: int = 8,
        clips_only: bool = False,
        dry_run: bool = False,
    ) -> None:
        """Initialize the NCore downloader.

        :param output_dir: Destination directory for :meth:`download`. Ignored by
            :meth:`download_single_clip` (which takes its own ``output_dir`` arg) and
            by streaming parsers (which manage per-clip temp directories).
        :param revision: HuggingFace dataset branch, tag, or commit.
        :param hf_token: HF access token. Resolves through
            :func:`resolve_hf_token` — falls back to ``$HF_TOKEN`` /
            ``$HUGGINGFACE_HUB_TOKEN`` when ``None``.
        :param clip_ids: Explicit clip UUIDs. Mutually exclusive with ``num_clips``.
        :param num_clips: Select the first N clips (or N random clips when
            ``sample_random=True``) from the full catalog.
        :param sample_random: Randomize ``num_clips`` selection.
        :param seed: RNG seed used when ``sample_random=True``.
        :param modality: Which modalities to fetch per clip. One of
            :data:`MODALITY_CHOICES`. Non-``"all"`` choices still include the
            sequence metadata + default components (poses, intrinsics, cuboids).
        :param cameras: When ``modality="cameras"``, restrict to these camera IDs.
            Defaults to all 7 cameras.
        :param max_workers: Parallel HF download workers (per snapshot or per clip).
        :param clips_only: Skip repo-level files (``README.md`` etc.) in
            :meth:`download`. Always effectively true for
            :meth:`download_single_clip`.
        :param dry_run: If ``True``, :meth:`download` logs the plan without fetching.
        """
        if clip_ids and num_clips is not None:
            raise ValueError("clip_ids and num_clips are mutually exclusive.")
        if num_clips is not None and num_clips <= 0:
            raise ValueError("num_clips must be a positive integer.")
        if modality not in MODALITY_CHOICES:
            raise ValueError(f"modality {modality!r} must be one of {MODALITY_CHOICES}")
        if cameras is not None:
            for cam in cameras:
                if cam not in CAMERA_IDS:
                    raise ValueError(f"camera {cam!r} is not valid; must be one of {CAMERA_IDS}")
        if cameras and modality != "cameras":
            logger.warning(
                "cameras=%s provided but modality=%r — cameras is only honored when modality='cameras'.",
                cameras,
                modality,
            )

        self.output_dir: Optional[Path] = Path(output_dir) if output_dir is not None else None
        self.dry_run: bool = dry_run

        # Public config — also read by the streaming parser.
        self.revision: str = revision
        self.hf_token: Optional[str] = resolve_hf_token(hf_token)
        self.modality: str = modality
        self.cameras: Optional[Tuple[str, ...]] = tuple(cameras) if cameras else None
        self.max_workers: int = max_workers

        # Selection knobs — only consumed by :meth:`resolve_clip_ids` and :meth:`download`.
        self._explicit_clip_ids: Optional[List[str]] = list(clip_ids) if clip_ids else None
        self._num_clips: Optional[int] = num_clips
        self._sample_random: bool = sample_random
        self._seed: int = seed
        self._clips_only: bool = clips_only

        if self.hf_token is None:
            logger.warning(
                "No HF token configured for NCoreDownloader. NCore is gated — set $HF_TOKEN "
                "or pass hf_token if downloads fail with 401/403."
            )

    def resolve_clip_ids(self) -> List[str]:
        """Return the clip UUIDs selected by the current configuration.

        Fetches the full catalog from HuggingFace when ``clip_ids`` was not explicitly
        configured. Used by :meth:`download` and by the streaming parser to enumerate
        work across Ray workers.
        """
        if self._explicit_clip_ids:
            resolved = list(self._explicit_clip_ids)
        else:
            all_ids = list_all_clip_ids(token=self.hf_token, revision=self.revision)
            logger.info("NCore catalog: %d clips at %s@%s", len(all_ids), NCORE_REPO_ID, self.revision)
            if self._num_clips is None or self._num_clips >= len(all_ids):
                resolved = all_ids
            elif self._sample_random:
                rng = _random_mod.Random(self._seed)
                resolved = sorted(rng.sample(all_ids, self._num_clips))
            else:
                resolved = all_ids[: self._num_clips]
        return resolved

    def download(self) -> None:
        """Inherited, see superclass."""
        assert self.output_dir is not None, "NCoreDownloader.output_dir must be set before download()."
        clip_ids = self.resolve_clip_ids()

        allow_patterns: List[str] = [] if self._clips_only else list(REPO_META_FILES)
        allow_patterns.extend(
            build_clip_allow_patterns(
                clip_ids=clip_ids,
                modality=self.modality,
                cameras=list(self.cameras) if self.cameras else None,
            )
        )

        logger.info("NCore target directory: %s", self.output_dir)
        logger.info("NCore revision:         %s", self.revision)
        logger.info(
            "NCore modality:         %s%s",
            self.modality,
            f" (cameras={list(self.cameras)})" if self.cameras else "",
        )
        logger.info("NCore clips selected:   %d", len(clip_ids))
        for pat in allow_patterns[: min(10, len(allow_patterns))]:
            logger.debug("  allow_pattern: %s", pat)

        if self.dry_run:
            logger.info("dry_run=True — not downloading. Plan covers %d clip(s).", len(clip_ids))
            return

        if not clip_ids:
            logger.warning("No clips selected — nothing to download.")
            return

        _, snapshot_download = _require_hf_hub()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=NCORE_REPO_ID,
            repo_type=NCORE_REPO_TYPE,
            revision=self.revision,
            local_dir=str(self.output_dir),
            allow_patterns=allow_patterns,
            token=self.hf_token,
            max_workers=self.max_workers,
        )
        logger.info("NCore download complete: %s", self.output_dir)

    def download_single_clip(self, clip_id: str, output_dir: Union[str, Path]) -> Path:
        """Fetch one clip to ``output_dir`` using the configured modality/auth.

        Intended for the NCore streaming parser — each clip lands in its own per-clip
        temp directory, is converted, and the temp directory is deleted. Returns the
        path to the clip's sequence manifest.
        """
        return download_clip(
            clip_id=clip_id,
            output_dir=Path(output_dir),
            modality=self.modality,
            cameras=list(self.cameras) if self.cameras else None,
            hf_token=self.hf_token,
            revision=self.revision,
            max_workers=self.max_workers,
        )
