"""TruckDrive dataset parser."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Union

from py123d.parser.base_dataset_parser import BaseDatasetParser, BaseLogParser, BaseMapParser
from py123d.parser.truckdrive.truckdrive_download import TruckDriveDownloader
from py123d.parser.truckdrive.truckdrive_log_parser import TruckDriveLogParser, resolve_truckdrive_split
from py123d.parser.truckdrive.truckdrive_map_parser import TruckDriveMapParser

logger = logging.getLogger(__name__)


class TruckDriveDatasetParser(BaseDatasetParser):
    """Top-level parser for the public TruckDrive dataset."""

    def __init__(
        self,
        truckdrive_data_root: Optional[Union[Path, str]] = None,
        scene_names: Optional[Sequence[str]] = None,
        split: Optional[str] = None,
        downloader: Optional[TruckDriveDownloader] = None,
    ) -> None:
        """Initialize the TruckDrive dataset parser.

        :param truckdrive_data_root: Root directory containing ``scene_XX_N/`` folders.
        :param scene_names: Explicit scene list. When empty or ``None``, all ``scene_*`` dirs are scanned.
        :param split: Optional split override applied to every scene.
        :param downloader: Optional downloader used for streaming mode.
        """
        self._downloader = downloader
        self._split_override = split

        if downloader is not None:
            downloader.download()
            assert downloader.output_dir is not None
            self._data_root = Path(downloader.output_dir)
        else:
            assert truckdrive_data_root is not None, (
                "`truckdrive_data_root` must be provided when `downloader` is None."
            )
            self._data_root = Path(truckdrive_data_root)
            assert self._data_root.exists(), f"`truckdrive_data_root` path {self._data_root} does not exist."

        nested_root = self._data_root / "TruckDrive"
        if nested_root.is_dir() and not any(self._data_root.glob("scene_*")):
            self._data_root = nested_root

        self._scene_names = self._resolve_scene_names(scene_names)

    def _resolve_scene_names(self, scene_names: Optional[Sequence[str]]) -> List[str]:
        if scene_names:
            return sorted(set(scene_names))
        return sorted(
            path.name for path in self._data_root.iterdir() if path.is_dir() and path.name.startswith("scene_")
        )

    def get_log_parsers(self) -> List[BaseLogParser]:
        """Inherited, see superclass."""
        parsers: List[BaseLogParser] = []
        for scene_name in self._scene_names:
            split = self._split_override or resolve_truckdrive_split(scene_name)
            if split == "truckdrive_test":
                logger.warning(
                    "Skipping log parser for scene %s in split %s, as the test split does not have ground truth trajectory nor annotations.",
                    scene_name,
                    split,
                )
                continue
            parsers.append(
                TruckDriveLogParser(
                    data_root=self._data_root,
                    scene_name=scene_name,
                    split=split,
                )
            )
        return parsers

    def get_map_parsers(self) -> List[BaseMapParser]:
        """Inherited, see superclass."""
        parsers: List[BaseMapParser] = []
        for scene_name in self._scene_names:
            split = self._split_override or resolve_truckdrive_split(scene_name)
            parsers.append(
                TruckDriveMapParser(
                    data_root=self._data_root,
                    scene_name=scene_name,
                    split=split,
                )
            )
        return parsers
