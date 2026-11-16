"""Download utilities for the KITScenes Multimodal dataset on Hugging Face."""

from __future__ import annotations

import concurrent.futures
import csv
import importlib
import logging
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from py123d.parser.base_downloader import BaseDownloader
from py123d.parser.kitscenes.kitscenes_constants import (
    DATA_SUBDIR,
    HF_KITSCENES_INDEX_FILE,
    HF_KITSCENES_REPO_ID,
    HF_KITSCENES_REPO_TYPE,
    KITSCENES_SPLITS,
)

logger = logging.getLogger(__name__)


def _require_hf_hub():
    """Lazy import — ``huggingface_hub`` is only required when a download is requested."""
    try:
        return importlib.import_module("huggingface_hub")
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for KITScenes downloads. Install it with:\n  pip install py123d[hf]\n"
        ) from exc


@dataclass(frozen=True)
class _SceneArchive:
    scene_id: str
    split: str
    archive_path: str  # path in the Hugging Face repository, e.g. ``data/val/<scene_id>.tar``
    size_bytes: int


class KITScenesDownloader(BaseDownloader):
    """Downloader for the gated KITScenes Multimodal release hosted on Hugging Face.

    Every scene is a single ``.tar`` archive (a few GB). Archives are extracted into the layout the parser reads,
    ``<output_dir>/data/<split>/<scene_id>/``. Access must be requested on Hugging Face beforehand.
    """

    def __init__(
        self,
        output_dir: Optional[Path],
        dry_run: bool = False,
        splits: Optional[Sequence[str]] = None,
        scene_ids: Optional[Sequence[str]] = None,
        max_num_scenes: Optional[int] = None,
        revision: str = "main",
        hf_token: Optional[str] = None,
        max_workers: int = 2,
        extract: bool = True,
        delete_archives: bool = False,
    ) -> None:
        """Initialize the KITScenes downloader.

        :param output_dir: Destination directory, i.e. the KITScenes data root.
        :param dry_run: When ``True``, log the plan without downloading.
        :param splits: KITScenes splits to download. When ``None``, all splits.
        :param scene_ids: Explicit scene UUIDs to download. When ``None``, all scenes of the selected splits.
        :param max_num_scenes: Optional cap on the number of scenes, e.g. to try the dataset.
        :param revision: Hugging Face dataset revision (branch/tag/commit).
        :param hf_token: Hugging Face token. Falls back to ``HF_TOKEN`` or the token stored by ``hf auth login``.
        :param max_workers: Parallel scene downloads.
        :param extract: Extract the scene archives after download.
        :param delete_archives: Delete each archive after it was extracted, to save disk space.
        """
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.dry_run = dry_run
        self.splits = list(splits) if splits is not None else None
        self.scene_ids = set(scene_ids) if scene_ids else None
        self.max_num_scenes = max_num_scenes
        self.revision = revision
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.max_workers = max_workers
        self.extract = extract
        self.delete_archives = delete_archives

        if self.splits is not None:
            unknown_splits = set(self.splits) - set(KITSCENES_SPLITS)
            assert not unknown_splits, f"Unknown KITScenes splits {unknown_splits}, expected {KITSCENES_SPLITS}."

    def download(self) -> None:
        """Fetch the selected KITScenes scenes into :attr:`output_dir`."""
        if self.output_dir is None:
            raise ValueError("KITScenesDownloader.output_dir must be set before calling download().")

        scene_archives = self._select_scene_archives(self._load_scene_index())
        if not scene_archives:
            raise RuntimeError("No KITScenes scenes matched the selected splits/scene IDs.")

        total_gb = sum(archive.size_bytes for archive in scene_archives) / 1e9
        logger.info(
            "KITScenes download plan: %d scenes (%.1f GB) into %s", len(scene_archives), total_gb, self.output_dir
        )
        for archive in scene_archives:
            logger.info("  %s (%.1f GB)", archive.archive_path, archive.size_bytes / 1e9)
        if self.dry_run:
            logger.info("Dry run enabled; skipping download.")
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._download_scene, archive) for archive in scene_archives]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    def _load_scene_index(self) -> List[_SceneArchive]:
        """Download the small CSV index that lists all scenes with their split and archive."""
        hf_hub = _require_hf_hub()
        try:
            index_path = hf_hub.hf_hub_download(
                repo_id=HF_KITSCENES_REPO_ID,
                repo_type=HF_KITSCENES_REPO_TYPE,
                filename=HF_KITSCENES_INDEX_FILE,
                revision=self.revision,
                token=self.hf_token,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to download the KITScenes scene index from Hugging Face. Accept the dataset terms at "
                f"https://huggingface.co/datasets/{HF_KITSCENES_REPO_ID} and log in with `hf auth login`."
            ) from exc

        with open(index_path, "r", encoding="utf-8", newline="") as file:
            return [
                _SceneArchive(
                    scene_id=row["sequence_id"],
                    split=row["split"],
                    archive_path=row["archive_path"],
                    size_bytes=int(row["archive_size_bytes"]),
                )
                for row in csv.DictReader(file)
            ]

    def _select_scene_archives(self, scene_archives: List[_SceneArchive]) -> List[_SceneArchive]:
        selected = [
            archive
            for archive in sorted(scene_archives, key=lambda archive: (archive.split, archive.scene_id))
            if (self.splits is None or archive.split in self.splits)
            and (self.scene_ids is None or archive.scene_id in self.scene_ids)
        ]
        if self.scene_ids is not None:
            missing_scene_ids = self.scene_ids - {archive.scene_id for archive in selected}
            if missing_scene_ids:
                logger.warning("KITScenes scenes not found in the selected splits: %s", sorted(missing_scene_ids))
        if self.max_num_scenes is not None:
            selected = selected[: self.max_num_scenes]
        return selected

    def _download_scene(self, archive: _SceneArchive) -> None:
        assert self.output_dir is not None
        scene_dir = self.output_dir / DATA_SUBDIR / archive.split / archive.scene_id
        if scene_dir.is_dir() and any(scene_dir.iterdir()):
            logger.info("[skip] %s already extracted", archive.scene_id)
            return

        logger.info("[download] %s", archive.archive_path)
        hf_hub = _require_hf_hub()
        try:
            archive_file = Path(
                hf_hub.hf_hub_download(
                    repo_id=HF_KITSCENES_REPO_ID,
                    repo_type=HF_KITSCENES_REPO_TYPE,
                    filename=archive.archive_path,
                    revision=self.revision,
                    token=self.hf_token,
                    local_dir=self.output_dir,
                )
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to download {archive.archive_path} from Hugging Face.") from exc

        if not self.extract:
            return
        logger.info("[extract] %s", archive_file)
        with tarfile.open(archive_file) as tar:
            # The ``data`` filter rejects absolute paths and links leaving the target directory (Python >= 3.12,
            # and backported to recent patch releases of older versions).
            if hasattr(tarfile, "data_filter"):
                tar.extractall(scene_dir.parent, filter="data")
            else:
                tar.extractall(scene_dir.parent)
        if not scene_dir.is_dir():
            raise RuntimeError(f"Archive {archive_file} did not contain the expected folder {archive.scene_id}/.")
        if self.delete_archives:
            archive_file.unlink()
