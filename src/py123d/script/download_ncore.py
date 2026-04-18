"""Download script for the NVIDIA PhysicalAI-Autonomous-Vehicles-NCore dataset.

The NCore dataset is gated on Hugging Face. Access requires a HF account that has
accepted the NVIDIA AV dataset license agreement, plus a token supplied via the
``HF_TOKEN`` environment variable or the ``--hf-token`` flag.

Dataset: https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore
Devkit:  https://github.com/NVIDIA/ncore

Each clip on disk has the following layout (one UUID-named subdirectory per clip)::

    clips/{clip_id}/
    ├── pai_{clip_id}.json                                        (sequence metadata)
    ├── pai_{clip_id}.ncore4.zarr.itar                            (poses, intrinsics, cuboids)
    ├── pai_{clip_id}.ncore4-lidar_top_360fov.zarr.itar           (~1.0 GB)
    └── pai_{clip_id}.ncore4-camera_{name}.zarr.itar              (~150 MB x 7 cameras)
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Sequence

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

# Top-level files in the repo (outside of clips/). Always downloaded unless --clips-only.
REPO_META_FILES = ("README.md", "rename_clip_folders.py")


def _require_hf_hub():
    """Lazy import — the dependency is optional until a user actually runs this script."""
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for this script. Install it with:\n  pip install huggingface_hub\n"
        ) from exc
    return HfApi, snapshot_download


def _list_all_clip_ids(token: Optional[str], revision: str) -> List[str]:
    """Returns sorted list of clip UUIDs present under ``clips/`` in the repo."""
    HfApi, _ = _require_hf_hub()
    api = HfApi(token=token)
    entries = api.list_repo_tree(
        repo_id=NCORE_REPO_ID,
        repo_type=NCORE_REPO_TYPE,
        path_in_repo="clips",
        revision=revision,
        recursive=False,
    )
    clip_ids = sorted(Path(e.path).name for e in entries if e.path.startswith("clips/"))
    return clip_ids


def _resolve_clip_selection(
    requested_ids: Optional[Sequence[str]],
    num_clips: Optional[int],
    sample_random: bool,
    seed: int,
    token: Optional[str],
    revision: str,
) -> List[str]:
    """Decides which clip UUIDs to download.

    Precedence: explicit --clip-ids > --num-clips > full dataset.
    """
    if requested_ids:
        selected = list(requested_ids)
    else:
        all_ids = _list_all_clip_ids(token=token, revision=revision)
        logger.info("Found %d clips in %s@%s", len(all_ids), NCORE_REPO_ID, revision)
        if num_clips is None or num_clips >= len(all_ids):
            selected = all_ids
        elif sample_random:
            rng = random.Random(seed)
            selected = sorted(rng.sample(all_ids, num_clips))
        else:
            selected = all_ids[:num_clips]
    return selected


def _build_allow_patterns(
    clip_ids: Sequence[str],
    modality: str,
    cameras: Optional[Sequence[str]],
    include_repo_meta: bool,
) -> List[str]:
    """Builds fine-grained ``allow_patterns`` for ``snapshot_download``.

    Selection logic:
    - ``metadata``: always include ``pai_{id}.json`` + default component store (poses/intrinsics/cuboids).
    - ``lidar``:    add the top lidar zarr.itar.
    - ``cameras``:  add zarr.itar for each requested camera (or all cameras if unset).
    - ``all``:      every file under the clip dir.
    """
    patterns: List[str] = []
    if include_repo_meta:
        patterns.extend(REPO_META_FILES)

    for clip_id in clip_ids:
        base = f"clips/{clip_id}"
        if modality == "all":
            patterns.append(f"{base}/*")
            continue

        # `metadata` is always included for any non-"all" modality so the sequence is loadable.
        patterns.append(f"{base}/pai_{clip_id}.json")
        patterns.append(f"{base}/pai_{clip_id}.ncore4.zarr.itar")

        if modality in ("lidar", "cameras"):
            if modality == "lidar":
                patterns.append(f"{base}/pai_{clip_id}.ncore4-lidar_top_360fov.zarr.itar")
            else:
                target_cams = cameras if cameras else CAMERA_IDS
                for cam in target_cams:
                    patterns.append(f"{base}/pai_{clip_id}.ncore4-{cam}.zarr.itar")

    return patterns


def _resolve_token(cli_token: Optional[str]) -> Optional[str]:
    return cli_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _resolve_output_dir(cli_output: Optional[str]) -> Path:
    """Falls back to ``$NCORE_DATA_ROOT`` then ``./data/ncore``."""
    if cli_output:
        resolved = Path(cli_output)
    elif os.environ.get("NCORE_DATA_ROOT"):
        resolved = Path(os.environ["NCORE_DATA_ROOT"])
    else:
        resolved = Path.cwd() / "data" / "ncore"
    return resolved.expanduser().resolve()


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="py123d-ncore-download",
        description="Download the NVIDIA PhysicalAI-Autonomous-Vehicles-NCore dataset (or a subset).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Destination directory. Falls back to $NCORE_DATA_ROOT, then ./data/ncore.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="HuggingFace dataset branch, tag, or commit to download.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HF access token. Defaults to $HF_TOKEN or $HUGGINGFACE_HUB_TOKEN.",
    )

    selection = parser.add_argument_group("clip selection")
    selection.add_argument(
        "--clip-ids",
        nargs="+",
        default=None,
        help="Specific clip UUIDs to download (mutually exclusive with --num-clips).",
    )
    selection.add_argument(
        "--num-clips",
        type=int,
        default=None,
        help="Download only the first N clips (or N random clips if --random is set).",
    )
    selection.add_argument(
        "--random",
        dest="sample_random",
        action="store_true",
        help="Sample --num-clips randomly instead of taking the first N.",
    )
    selection.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when --random is set.",
    )

    modalities = parser.add_argument_group("modality filter")
    modalities.add_argument(
        "--modality",
        choices=MODALITY_CHOICES,
        default="all",
        help=(
            "Which modalities to fetch per clip. Non-'all' choices still include the sequence "
            "metadata + default components (poses, intrinsics, cuboids)."
        ),
    )
    modalities.add_argument(
        "--cameras",
        nargs="+",
        choices=CAMERA_IDS,
        default=None,
        help="When --modality=cameras, restrict to these camera IDs. Defaults to all 7 cameras.",
    )

    misc = parser.add_argument_group("misc")
    misc.add_argument(
        "--list-clips",
        action="store_true",
        help="Print available clip UUIDs in the repo and exit.",
    )
    misc.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan (output dir, clip count, allow_patterns) without downloading.",
    )
    misc.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Parallel download workers passed to huggingface_hub.",
    )
    misc.add_argument(
        "--clips-only",
        action="store_true",
        help="Skip README.md / rename_clip_folders.py at the repo root.",
    )
    misc.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    args = parser.parse_args(argv)

    if args.clip_ids and args.num_clips is not None:
        parser.error("--clip-ids and --num-clips are mutually exclusive")
    if args.num_clips is not None and args.num_clips <= 0:
        parser.error("--num-clips must be a positive integer")
    if args.cameras and args.modality != "cameras":
        parser.error("--cameras only applies when --modality=cameras")

    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    token = _resolve_token(args.hf_token)
    if token is None:
        logger.warning(
            "No HF token provided. The NCore dataset is gated — set $HF_TOKEN or pass --hf-token "
            "if the download fails with 401/403."
        )

    if args.list_clips:
        clip_ids = _list_all_clip_ids(token=token, revision=args.revision)
        logger.info("Listing %d clips in %s@%s", len(clip_ids), NCORE_REPO_ID, args.revision)
        for clip_id in clip_ids:
            sys.stdout.write(f"{clip_id}\n")
        return 0

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_ids = _resolve_clip_selection(
        requested_ids=args.clip_ids,
        num_clips=args.num_clips,
        sample_random=args.sample_random,
        seed=args.seed,
        token=token,
        revision=args.revision,
    )
    if not clip_ids:
        logger.error("No clips selected — nothing to download.")
        return 1

    allow_patterns = _build_allow_patterns(
        clip_ids=clip_ids,
        modality=args.modality,
        cameras=args.cameras,
        include_repo_meta=not args.clips_only,
    )

    logger.info("Target directory: %s", output_dir)
    logger.info("Revision:         %s", args.revision)
    logger.info("Modality:         %s%s", args.modality, f" (cameras={list(args.cameras)})" if args.cameras else "")
    logger.info("Clips selected:   %d", len(clip_ids))
    logger.debug("First clip IDs: %s", clip_ids[: min(5, len(clip_ids))])
    logger.debug("Allow patterns (%d total):", len(allow_patterns))
    for pat in allow_patterns[: min(10, len(allow_patterns))]:
        logger.debug("  %s", pat)

    if args.dry_run:
        logger.info("--dry-run set, not downloading.")
        return 0

    _, snapshot_download = _require_hf_hub()
    snapshot_download(
        repo_id=NCORE_REPO_ID,
        repo_type=NCORE_REPO_TYPE,
        revision=args.revision,
        local_dir=str(output_dir),
        allow_patterns=allow_patterns,
        token=token,
        max_workers=args.max_workers,
    )
    logger.info("Done. Data written to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
