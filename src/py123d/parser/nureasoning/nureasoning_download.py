"""Download utilities for the nuReasoning dataset (Hugging Face ``nureasoning/nuReasoning``).

The dataset is gated: request access on the dataset page and provide a token that has it
(``$HF_TOKEN`` or ``hf auth login``). It stores **each clip as an individual ``.tar``**::

    nureasoning/nuReasoning/
    └── data/
        ├── train/                         (part_1 ... part_10)
        │   └── <part>/<log_name>_<keyframe_token>.tar
        ├── validation/                    (part_1)
        │   └── <part>/<log_name>_<keyframe_token>.tar
        └── test/                          (no part folders)
            └── <log_name>_<keyframe_token>.tar

Each tar unpacks into a per-clip directory::

    <log_name>_<keyframe_token>/
    ├── metadata.json
    ├── map.pkl
    ├── ego_state/<timestamp_us>.pkl
    ├── annotations/<timestamp_us>.pkl
    ├── reasoning/<timestamp_us>.json
    ├── cameras/<camera>/<camera>_<timestamp_us>.jpg
    └── lidar/<timestamp_us>.pcd           (only in some clips)

Test clips ship without ground truth: cameras, ego states up to the key frame, and a
``reasoning_questions.json``, but no map, annotations, lidar or reasoning.

This module exposes :class:`NureasoningDownloader` (Hydra-instantiable) which powers
both entry points:

1. ``py123d-download dataset=nureasoning`` — :meth:`download` fetches every selected
   clip's tar, extracts it into ``output_dir/<split>/[<part>/]<clip>/``, and deletes the
   tar. The repo's leading ``data/`` prefix is stripped so ``output_dir`` *is* the local
   ``data/`` root (== ``nureasoning_data_root``).

2. The :class:`~py123d.parser.nureasoning.nureasoning_parser.NureasoningParser`
   streaming path — the parser points ``output_dir`` at a managed temp directory, calls
   :meth:`download` for the selected clips, converts them, and deletes the temp directory.

Selection is incremental: pick ``splits`` (``nureasoning_train``, ...), ``parts``
(``part_1``, ...), explicit ``log_names``, or the first/random ``num_logs``. The repo tree
is enumerated live via :class:`huggingface_hub.HfApi`. Nothing is written to the
HuggingFace hub cache — tars are always fetched next to their destination.
"""

from __future__ import annotations

import importlib
import logging
import os
import random as _random_mod
import shutil
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Set, Union

from tqdm import tqdm

from py123d.parser.base_downloader import BaseDownloader
from py123d.parser.nureasoning.utils.nureasoning_constants import (
    NUREASONING_DATA_SPLITS,
    NUREASONING_DEFAULT_SPLITS,
    NUREASONING_HF_SPLITS,
    NUREASONING_REPO_DATA_DIR,
    NUREASONING_REPO_ID,
    NUREASONING_REPO_TYPE,
    NUREASONING_SPLIT_SOURCES,
)

logger = logging.getLogger(__name__)

_ARCHIVE_SUFFIX = ".tar"
_METADATA_FILE = "metadata.json"

# huggingface_hub errors that mean "this token cannot read the gated repo". Matched by class name
# to avoid depending on where a specific huggingface_hub version exports them.
_ACCESS_ERROR_NAMES = {"GatedRepoError", "RepositoryNotFoundError", "LocalTokenNotFoundError"}


class NureasoningAccessError(RuntimeError):
    """Raised when Hugging Face denies access to the gated nuReasoning repository."""


def _require_hf_hub():
    """Lazy import — ``huggingface_hub`` is only needed once a download is requested."""
    try:
        return importlib.import_module("huggingface_hub")
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for nuReasoning downloads. Install it with:\n"
            "  pip install py123d[hf]\n"
            "or directly:\n"
            "  pip install huggingface_hub\n"
        ) from exc


def resolve_hf_token(cli_token: Optional[str] = None) -> Optional[str]:
    """Resolve the HF token from (in order): explicit arg, ``$HF_TOKEN``, ``$HUGGINGFACE_HUB_TOKEN``.

    ``None`` lets ``huggingface_hub`` fall back to the token stored by ``hf auth login``.
    """
    return cli_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _is_repo_dir(entry: object) -> bool:
    """Return ``True`` if a ``list_repo_tree`` entry is a folder (``RepoFolder``).

    Detected by class name to avoid depending on a specific ``huggingface_hub`` export.
    """
    return type(entry).__name__ == "RepoFolder"


def _part_number(part: Optional[str]) -> int:
    """Numeric order of a ``part_<k>`` folder, so ``part_10`` sorts after ``part_2``."""
    suffix = part.rsplit("_", 1)[-1] if part else ""
    return int(suffix) if suffix.isdigit() else 0


def extract_nureasoning_clip(archive_path: Path, clip_dir: Path) -> Path:
    """Extract a per-clip ``.tar`` into ``clip_dir`` and return it.

    The clip's files (``metadata.json``, ``ego_state/``, ``annotations/``, ...) end up
    directly under ``clip_dir`` so the result matches what
    :class:`~py123d.parser.nureasoning.nureasoning_parser.NureasoningParser` expects
    (``clip_dir/metadata.json``, ``clip_dir/ego_state/...``).

    The upstream tars wrap everything in a single ``<clip_name>/`` folder, which is
    stripped; a flat (root-level) archive is accepted too. The archive is unpacked next to
    ``clip_dir`` and moved into place once complete, so an interrupted extraction never
    leaves a clip directory that looks finished.

    :param archive_path: Path to the downloaded clip tar.
    :param clip_dir: Destination directory for the clip's contents (its ``.name`` is the
        clip name, used to recognize the ``<clip_name>/`` wrapper).
    :return: ``clip_dir``.
    """
    clip_dir = Path(clip_dir)
    clip_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=clip_dir.parent, prefix=f".{clip_dir.name}-") as tmp:
        staging_dir = Path(tmp) / "extract"
        staging_dir.mkdir()
        with tarfile.open(archive_path) as tar:
            # The ``data`` filter rejects absolute paths and links leaving the target directory (Python >= 3.12,
            # and backported to recent patch releases of older versions).
            if hasattr(tarfile, "data_filter"):
                tar.extractall(staging_dir, filter="data")
            else:
                tar.extractall(staging_dir)

        wrapped_dir = staging_dir / clip_dir.name
        extracted_dir = wrapped_dir if (wrapped_dir / _METADATA_FILE).is_file() else staging_dir
        if not (extracted_dir / _METADATA_FILE).is_file():
            raise RuntimeError(f"Archive {archive_path} contains no {_METADATA_FILE}; it may be corrupt.")

        if clip_dir.exists():
            # Only reached for a clip directory without metadata.json, i.e. an incomplete leftover.
            shutil.rmtree(clip_dir)
        extracted_dir.rename(clip_dir)

    return clip_dir


@dataclass(frozen=True)
class NureasoningClipEntry:
    """Picklable locator for one clip tar in the repo.

    :ivar split: HF split directory (``train`` / ``validation`` / ``test``).
    :ivar part: Part directory (``part_1``, ...), or ``None`` for splits without parts (``test``).
    :ivar clip_name: Clip directory name ``<log_name>_<keyframe_token>`` (no ``.tar``).
    :ivar repo_path: Full path inside the repo, ``data/<split>/[<part>/]<clip>.tar``.
    :ivar size_bytes: Size of the tar, if known from the repo listing.
    """

    split: str
    part: Optional[str]
    clip_name: str
    repo_path: str
    size_bytes: Optional[int] = None

    @property
    def relative_dir(self) -> Path:
        """Clip directory relative to the local data root, ``<split>/[<part>/]<clip>``."""
        parent = Path(self.split) / self.part if self.part else Path(self.split)
        return parent / self.clip_name

    @property
    def sort_key(self) -> tuple:
        split_order = NUREASONING_HF_SPLITS.index(self.split) if self.split in NUREASONING_HF_SPLITS else -1
        return (split_order, _part_number(self.part), self.part or "", self.clip_name)


def _repo_path_for(split: str, part: Optional[str], clip_name: str) -> str:
    """Build the in-repo tar path ``data/<split>/[<part>/]<clip>.tar``."""
    folder = f"{NUREASONING_REPO_DATA_DIR}/{split}/{part}" if part else f"{NUREASONING_REPO_DATA_DIR}/{split}"
    return f"{folder}/{clip_name}{_ARCHIVE_SUFFIX}"


# ======================================================================================
# Downloader (Hydra-instantiable, shared by py123d-download and the streaming parser)
# ======================================================================================


class NureasoningDownloader(BaseDownloader):
    """Downloader for the gated nuReasoning dataset via Hugging Face ``nureasoning/nuReasoning``.

    Operates in two modes:

    * :meth:`download` — bulk-fetch every selected clip, extracting each into
      ``output_dir/<split>/[<part>/]<clip>/``. Used by ``py123d-download dataset=nureasoning``
      and by :class:`~py123d.parser.nureasoning.nureasoning_parser.NureasoningParser` in
      streaming mode, which points ``output_dir`` at a managed temp directory.
    * :meth:`download_single_clip` — fetch one clip into a caller-provided directory.
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        revision: str = "main",
        hf_token: Optional[str] = None,
        splits: Optional[List[str]] = None,
        parts: Optional[List[str]] = None,
        log_names: Optional[List[str]] = None,
        num_logs: Optional[int] = None,
        sample_random: bool = False,
        seed: int = 0,
        max_workers: int = 8,
        keep_archive: bool = False,
        dry_run: bool = False,
    ) -> None:
        """Initialize the nuReasoning downloader.

        :param output_dir: Destination for :meth:`download` — the local ``data/`` root
            (== ``nureasoning_data_root``). Selected clips land at
            ``output_dir/<split>/[<part>/]<clip>/``. Ignored by :meth:`download_single_clip`
            (which takes its own ``output_dir`` arg).
        :param revision: HuggingFace dataset branch, tag, or commit.
        :param hf_token: HF access token with access to the gated repo. Resolves through
            :func:`resolve_hf_token` — falls back to ``$HF_TOKEN`` / ``$HUGGINGFACE_HUB_TOKEN``,
            then to the token stored by ``hf auth login``.
        :param splits: Splits to download: ``nureasoning_train``, ``nureasoning_val``,
            ``nureasoning_test`` or ``nureasoning-mini_train`` (parts 1-3 of train). ``None``
            (default) selects the train, val and test splits.
        :param parts: Restrict to these part directories (``part_1``, ...) within the
            selected splits. ``None`` (default) uses every part of each split. Clips outside
            a part folder (the test split) are dropped when a part restriction is given.
        :param log_names: Explicit clip names ``<log_name>_<keyframe_token>``. Mutually
            exclusive with ``num_logs``. Validated against the (split/part-filtered) repo
            listing.
        :param num_logs: Select the first N clips (or N random clips when
            ``sample_random=True``) from the filtered catalog.
        :param sample_random: Randomize ``num_logs`` selection.
        :param seed: RNG seed used when ``sample_random=True``.
        :param max_workers: Parallel clip download/extract workers.
        :param keep_archive: When ``True``, also keep each downloaded ``.tar`` next to its
            extracted directory. Default ``False`` extracts then discards the tar
            (roughly halves disk use).
        :param dry_run: If ``True``, :meth:`download` logs the plan without fetching.
        """
        if log_names and num_logs is not None:
            raise ValueError("log_names and num_logs are mutually exclusive.")
        if num_logs is not None and num_logs <= 0:
            raise ValueError("num_logs must be a positive integer.")

        self.output_dir: Optional[Path] = Path(output_dir) if output_dir is not None else None
        self.dry_run: bool = dry_run

        # Public config — also read by the streaming parser.
        self.revision: str = revision
        self.hf_token: Optional[str] = resolve_hf_token(hf_token)
        self.max_workers: int = max_workers
        self.keep_archive: bool = keep_archive
        self.splits: List[str] = list(splits) if splits else list(NUREASONING_DEFAULT_SPLITS)
        for split in self.splits:
            assert split in NUREASONING_DATA_SPLITS, (
                f"Split {split} is not available. Available splits: {sorted(NUREASONING_DATA_SPLITS)}"
            )

        # Selection knobs — consumed by :meth:`resolve_clip_entries`.
        self._parts: Optional[List[str]] = list(parts) if parts else None
        self._explicit_log_names: Optional[List[str]] = list(log_names) if log_names else None
        self._num_logs: Optional[int] = num_logs
        self._sample_random: bool = sample_random
        self._seed: int = seed

    # ----- Selection ------------------------------------------------------------------

    def _selected_sources(self) -> Dict[str, Optional[Set[str]]]:
        """Map each selected HF split folder to its selected part folders (``None`` = all parts)."""
        sources: Dict[str, Optional[Set[str]]] = {}
        for split in self.splits:
            hf_split, split_parts = NUREASONING_SPLIT_SOURCES[split]
            if hf_split in sources and sources[hf_split] is None:
                continue  # Already selected in full by another split.
            if split_parts is None:
                sources[hf_split] = None
            else:
                sources[hf_split] = (sources.get(hf_split) or set()) | set(split_parts)
        return sources

    def _list_clip_entries(self) -> List[NureasoningClipEntry]:
        """Enumerate every selected clip tar in the repo via the HuggingFace tree API."""
        api = _require_hf_hub().HfApi(token=self.hf_token)
        part_filter = set(self._parts) if self._parts is not None else None

        entries: List[NureasoningClipEntry] = []
        for hf_split, split_parts in self._selected_sources().items():
            split_path = f"{NUREASONING_REPO_DATA_DIR}/{hf_split}"
            for item in api.list_repo_tree(
                repo_id=NUREASONING_REPO_ID,
                repo_type=NUREASONING_REPO_TYPE,
                path_in_repo=split_path,
                revision=self.revision,
                recursive=True,
            ):
                if _is_repo_dir(item) or not item.path.endswith(_ARCHIVE_SUFFIX):
                    continue
                # Either "<clip>.tar" (no part folders) or "<part>/<clip>.tar".
                relative_path = PurePosixPath(item.path).relative_to(split_path)
                if len(relative_path.parts) > 2:
                    continue
                part = relative_path.parts[0] if len(relative_path.parts) == 2 else None
                if split_parts is not None and part not in split_parts:
                    continue
                if part_filter is not None and part not in part_filter:
                    continue
                entries.append(
                    NureasoningClipEntry(
                        split=hf_split,
                        part=part,
                        clip_name=relative_path.name[: -len(_ARCHIVE_SUFFIX)],
                        repo_path=item.path,
                        size_bytes=getattr(item, "size", None),
                    )
                )
        return entries

    def resolve_clip_entries(self) -> List[NureasoningClipEntry]:
        """Return the clip entries selected by the current configuration (deterministic)."""
        entries: List[NureasoningClipEntry] = []
        seen_clips: Set[tuple] = set()
        for entry in sorted(self._list_clip_entries(), key=lambda e: e.sort_key):
            # A few clips are uploaded to two parts of the same split. They convert to the same log,
            # so only the first copy is kept.
            if (entry.split, entry.clip_name) in seen_clips:
                logger.info("nuReasoning: skipping duplicate upload %s", entry.repo_path)
                continue
            seen_clips.add((entry.split, entry.clip_name))
            entries.append(entry)

        if self._explicit_log_names:
            wanted = set(self._explicit_log_names)
            found = {e.clip_name for e in entries}
            unknown = wanted - found
            if unknown:
                raise ValueError(
                    f"Unknown nuReasoning clip name(s): {sorted(unknown)}. Not found in the "
                    f"selected splits/parts of {NUREASONING_REPO_ID}@{self.revision}."
                )
            resolved = [e for e in entries if e.clip_name in wanted]
        elif self._num_logs is None or self._num_logs >= len(entries):
            resolved = entries
        elif self._sample_random:
            rng = _random_mod.Random(self._seed)
            resolved = sorted(rng.sample(entries, self._num_logs), key=lambda e: e.sort_key)
        else:
            resolved = entries[: self._num_logs]
        return resolved

    # ----- Bulk download (py123d-download) --------------------------------------------

    def download(self) -> None:
        """Inherited, see superclass.

        Bulk flow: every selected clip tar is downloaded next to its destination, extracted
        into ``output_dir/<split>/[<part>/]<clip>/``, and (unless ``keep_archive``) the tar
        is discarded. Already-extracted clips are skipped, so re-running resumes.
        """
        hf_hub = _require_hf_hub()
        if self.hf_token is None and hf_hub.get_token() is None:
            logger.warning(
                "No HF token configured for NureasoningDownloader. nuReasoning is gated — request access at "
                "https://huggingface.co/datasets/%s, then set $HF_TOKEN or run `hf auth login`.",
                NUREASONING_REPO_ID,
            )

        entries = self.resolve_clip_entries()

        n_folders = len({(e.split, e.part) for e in entries})
        total_gb = sum(e.size_bytes or 0 for e in entries) / 1e9
        logger.info("nuReasoning source:    %s@%s", NUREASONING_REPO_ID, self.revision)
        logger.info(
            "nuReasoning selected:  %d clip(s) in %d split/part folder(s), %.1f GB", len(entries), n_folders, total_gb
        )
        logger.info("nuReasoning target:    %s", self.output_dir)

        # dry_run previews the plan without writing, so it does not require output_dir.
        if self.dry_run:
            logger.info("dry_run=True — not downloading. Plan covers %d clip(s).", len(entries))
            return

        if not entries:
            logger.warning("No clips selected — nothing to download.")
            return

        assert self.output_dir is not None, "NureasoningDownloader.output_dir must be set before download()."
        self.output_dir.mkdir(parents=True, exist_ok=True)

        failures: List[str] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_entry = {
                executor.submit(self._fetch_and_extract, entry, self.output_dir / entry.relative_dir): entry
                for entry in entries
            }
            for future in tqdm(as_completed(future_to_entry), total=len(future_to_entry), desc="nuReasoning clips"):
                entry = future_to_entry[future]
                try:
                    future.result()
                except NureasoningAccessError:
                    # Every other clip would fail the same way.
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                except Exception as exc:  # noqa: BLE001 — collect and report all failures.
                    logger.error("Failed to download/extract clip %s: %s", entry.clip_name, exc)
                    failures.append(entry.clip_name)

        if failures:
            raise RuntimeError(
                f"{len(failures)} / {len(entries)} nuReasoning clip(s) failed to download: {failures[:10]}"
                + (" ..." if len(failures) > 10 else "")
            )
        logger.info("nuReasoning download complete: %s", self.output_dir)

    # ----- Per-clip fetch -------------------------------------------------------------

    def download_single_clip(
        self, split: str, part: Optional[str], clip_name: str, output_dir: Union[str, Path]
    ) -> Path:
        """Fetch and extract one clip into ``output_dir/<clip_name>/`` and return that path.

        A convenience for materializing a single clip on demand (idempotent).

        :param split: HF split directory (``train`` / ``validation`` / ``test``).
        :param part: Part directory (``part_1``, ...), or ``None`` for the test split.
        :param clip_name: Clip name ``<log_name>_<keyframe_token>``.
        :param output_dir: Directory that receives the ``<clip_name>/`` folder.
        """
        entry = NureasoningClipEntry(
            split=split, part=part, clip_name=clip_name, repo_path=_repo_path_for(split, part, clip_name)
        )
        return self._fetch_and_extract(entry, Path(output_dir) / clip_name)

    def _fetch_and_extract(self, entry: NureasoningClipEntry, clip_dir: Path) -> Path:
        """Download ``entry``'s tar and extract it into ``clip_dir`` (idempotent)."""
        clip_dir = Path(clip_dir)
        if (clip_dir / _METADATA_FILE).is_file():
            logger.debug("Skip already-extracted clip %s at %s", entry.clip_name, clip_dir)
            return clip_dir

        hf_hub = _require_hf_hub()
        clip_dir.parent.mkdir(parents=True, exist_ok=True)
        # The tar is staged next to its destination (not in the system temp dir), so it lands on the
        # same filesystem as the extracted clip.
        with tempfile.TemporaryDirectory(dir=clip_dir.parent, prefix=f".{entry.clip_name}-") as tmp:
            try:
                archive_path = Path(
                    hf_hub.hf_hub_download(
                        repo_id=NUREASONING_REPO_ID,
                        repo_type=NUREASONING_REPO_TYPE,
                        filename=entry.repo_path,
                        revision=self.revision,
                        token=self.hf_token,
                        local_dir=tmp,
                    )
                )
            except Exception as exc:
                if type(exc).__name__ in _ACCESS_ERROR_NAMES:
                    raise NureasoningAccessError(
                        f"Cannot access {NUREASONING_REPO_ID}. Request access at "
                        f"https://huggingface.co/datasets/{NUREASONING_REPO_ID}, then set $HF_TOKEN "
                        "or run `hf auth login` with a token that has it."
                    ) from exc
                raise
            extract_nureasoning_clip(archive_path, clip_dir)
            if self.keep_archive:
                shutil.move(str(archive_path), clip_dir.parent / f"{entry.clip_name}{_ARCHIVE_SUFFIX}")
        return clip_dir
