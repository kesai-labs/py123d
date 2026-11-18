"""Download utilities for the TruckDrive dataset on Hugging Face."""

from __future__ import annotations

import concurrent.futures
import importlib
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from py123d.parser.base_downloader import BaseDownloader
from py123d.parser.truckdrive.truckdrive_constants import (
    DEFAULT_MODALITIES,
    HF_TRUCKDRIVE_REPO_ID,
    HF_TRUCKDRIVE_REPO_TYPE,
    HF_TRUCKDRIVE_ROOT,
    MODALITY_ZIP_FILES,
)

logger = logging.getLogger(__name__)


def _require_hf_hub():
    """Lazy import — ``huggingface_hub`` is only required when download is requested."""
    try:
        hf_hub = importlib.import_module("huggingface_hub")
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for TruckDrive downloads. Install it with:\n  pip install py123d[hf]\n"
        ) from exc
    return hf_hub.HfApi, hf_hub.hf_hub_download, hf_hub.login


def resolve_hf_token(cli_token: Optional[str] = None) -> Optional[str]:
    """Resolve HF token from arg, ``HF_TOKEN``, or ``HUGGINGFACE_HUB_TOKEN``."""
    return cli_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


class TruckDriveDownloader(BaseDownloader):
    """Downloader for the gated TruckDrive release hosted on Hugging Face."""

    def __init__(
        self,
        output_dir: Optional[Path],
        dry_run: bool = False,
        scenes: Optional[Sequence[str]] = None,
        modalities: Optional[Sequence[str]] = None,
        revision: str = "main",
        hf_token: Optional[str] = None,
        hf_login: bool = True,
        max_workers: int = 4,
        unzip: bool = True,
        overwrite: bool = False,
    ) -> None:
        """Initialize the TruckDrive downloader.

        :param output_dir: Destination directory for downloaded scenes.
        :param dry_run: When ``True``, log the plan without downloading.
        :param scenes: Explicit scene names to download. When ``None``, all scenes are listed.
        :param modalities: Modalities to download. Defaults to camera/lidar/poses/calibrations/annotations.
        :param revision: Hugging Face dataset revision (branch/tag/commit).
        :param hf_token: Hugging Face token. Falls back to ``HF_TOKEN`` or ``HUGGINGFACE_HUB_TOKEN``.
        :param hf_login: Attempt ``huggingface_hub.login`` when ``hf_token`` is available.
        :param max_workers: Parallel download workers.
        :param unzip: Extract modality zips into the viewer layout after download.
        :param overwrite: Re-download files even when they already exist locally.
        """
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.dry_run = dry_run
        self.scenes = list(scenes) if scenes is not None else None
        self.modalities = tuple(modalities) if modalities is not None else DEFAULT_MODALITIES
        self.revision = revision
        self.hf_token = resolve_hf_token(hf_token)
        self.hf_login = hf_login
        self.max_workers = max_workers
        self.unzip = unzip
        self.overwrite = overwrite

        if self.hf_token is None:
            logger.warning(
                "No HF token configured for TruckDriveDownloader. TruckDrive is gated; "
                "set $HF_TOKEN (or pass hf_token) and request access on Hugging Face if downloads fail."
            )

    def download(self) -> None:
        """Fetch selected TruckDrive scenes into :attr:`output_dir`."""
        if self.output_dir is None:
            raise ValueError("TruckDriveDownloader.output_dir must be set before calling download().")

        self._authenticate_hf()
        selected_modalities = self._resolve_modalities()
        try:
            scene_names = self.scenes if self.scenes else self._list_scene_names()
        except Exception as exc:
            raise RuntimeError(
                "Failed to list TruckDrive scenes from Hugging Face. "
                "Ensure your account has accepted dataset access and your HF token is valid."
            ) from exc
        if not scene_names:
            raise RuntimeError("No TruckDrive scenes found to download.")

        selected_keys: List[str] = []
        for scene_name in scene_names:
            selected_keys.extend(self._list_keys_for_scene(scene_name, selected_modalities))

        selected_keys = sorted(set(selected_keys))
        if not selected_keys:
            raise RuntimeError("No TruckDrive files matched the selected scenes/modalities.")

        logger.info("TruckDrive download plan: %d files under %s", len(selected_keys), self.output_dir)
        for key in selected_keys:
            logger.info("  %s", key)

        if self.dry_run:
            logger.info("Dry run enabled; skipping download.")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._download_key, key) for key in selected_keys]
            for future in concurrent.futures.as_completed(futures):
                future.result()

        if self.unzip:
            for key in selected_keys:
                self._unzip_modality_archive(self.output_dir / key)

    def _resolve_modalities(self) -> Tuple[str, ...]:
        unknown = [modality for modality in self.modalities if modality not in MODALITY_ZIP_FILES]
        if unknown:
            raise ValueError(f"Unknown TruckDrive modalities: {unknown}")
        return tuple(MODALITY_ZIP_FILES[modality] for modality in self.modalities)

    def _list_scene_names(self) -> List[str]:
        api = self._create_hf_api()
        entries = api.list_repo_tree(
            repo_id=HF_TRUCKDRIVE_REPO_ID,
            repo_type=HF_TRUCKDRIVE_REPO_TYPE,
            path_in_repo=HF_TRUCKDRIVE_ROOT,
            revision=self.revision,
            recursive=False,
        )

        scene_names: List[str] = []
        for entry in entries:
            remainder = entry.path[len(HF_TRUCKDRIVE_ROOT) :].strip("/")
            if remainder.startswith("scene_"):
                scene_names.append(remainder.split("/")[0])

        return sorted(set(scene_names))

    def _list_keys_for_scene(self, scene_name: str, selected_modalities: Tuple[str, ...]) -> List[str]:
        api = self._create_hf_api()
        scene_prefix = f"{HF_TRUCKDRIVE_ROOT}/{scene_name}"
        entries = api.list_repo_tree(
            repo_id=HF_TRUCKDRIVE_REPO_ID,
            repo_type=HF_TRUCKDRIVE_REPO_TYPE,
            path_in_repo=scene_prefix,
            revision=self.revision,
            recursive=False,
        )

        selected = [entry.path for entry in entries if Path(entry.path).name in selected_modalities]
        return sorted(set(selected))

    def _download_key(self, key: str) -> None:
        assert self.output_dir is not None
        destination = self.output_dir / key
        if destination.exists() and not self.overwrite:
            logger.info("[skip] %s", key)
            return

        logger.info("[download] %s", key)
        _, hf_hub_download, _ = _require_hf_hub()
        try:
            # ``local_dir`` writes the file to ``output_dir / key``, without a copy in the Hugging Face cache.
            hf_hub_download(
                repo_id=HF_TRUCKDRIVE_REPO_ID,
                repo_type=HF_TRUCKDRIVE_REPO_TYPE,
                filename=key,
                revision=self.revision,
                token=self.hf_token,
                force_download=self.overwrite,
                local_dir=self.output_dir,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download {key} from Hugging Face. "
                "Ensure dataset access is approved and your HF token has read scope."
            ) from exc

    def _authenticate_hf(self) -> None:
        if not self.hf_login or not self.hf_token:
            return

        _, _, login = _require_hf_hub()
        try:
            login(token=self.hf_token, add_to_git_credential=False)
        except TypeError:
            # Older huggingface_hub versions may not expose add_to_git_credential.
            login(token=self.hf_token)

    def _create_hf_api(self):
        HfApi, _, _ = _require_hf_hub()
        return HfApi(token=self.hf_token)

    def _unzip_modality_archive(self, zip_path: Path) -> None:
        if not zip_path.is_file():
            logger.warning("[unzip] skip missing archive: %s", zip_path)
            return

        scene_dir = zip_path.parent
        modality = zip_path.stem
        scene_name = scene_dir.name
        destination = scene_dir / modality
        if destination.is_dir() and any(destination.iterdir()):
            logger.info("[unzip] skip existing: %s", destination)
            return

        logger.info("[unzip] %s", zip_path)
        # Extracted next to the archive, so moving the result into place stays on one filesystem.
        tmp_path = scene_dir / f"_unzip_{modality}"
        try:
            with zipfile.ZipFile(zip_path) as archive:
                if not archive.namelist():
                    return
                archive.extractall(tmp_path)

            matches = [path for path in tmp_path.rglob(modality) if path.is_dir() and path.name == modality]
            scoped = [path for path in matches if scene_name in path.relative_to(tmp_path).parts]
            if scoped:
                matches = scoped
            if len(matches) == 1:
                self._move_tree(matches[0], destination)
                return

            direct = tmp_path / modality
            if direct.is_dir():
                self._move_tree(direct, destination)
                return

            if any(tmp_path.iterdir()):
                self._merge_tree(tmp_path, destination)
                return
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

        raise RuntimeError(f"Could not determine extracted layout for archive: {zip_path}")

    @staticmethod
    def _move_tree(source: Path, destination: Path) -> None:
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(source), str(destination))

    @staticmethod
    def _merge_tree(source_dir: Path, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        for child in sorted(source_dir.iterdir()):
            target = destination / child.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(child), str(target))
