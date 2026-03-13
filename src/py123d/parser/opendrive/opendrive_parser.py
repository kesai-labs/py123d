from pathlib import Path
from typing import List, Optional, Union

from typing_extensions import override

from py123d.parser.base_dataset_parser import BaseDatasetParser, BaseLogParser, BaseMapParser
from py123d.parser.opendrive.opendrive_map_parser import OpenDriveMapParser

_CARLA_MAPS_DIR = Path(__file__).parent / "carla_maps"
_VALID_SUFFIXES = {".xodr", ".gz"}


class OpenDriveParser(BaseDatasetParser):
    """Dataset parser for OpenDRIVE (.xodr) map files.

    This parser only converts maps — no log conversion is needed.
    """

    def __init__(
        self,
        xodr_paths: List[Union[str, Path]],
        location: Optional[str] = None,
        interpolation_step_size: float = 1.0,
        connection_distance_threshold: float = 0.1,
        internal_only: bool = True,
    ) -> None:
        """Initializes the OpenDriveParser.

        :param xodr_paths: List of paths to OpenDRIVE (.xodr or .xodr.gz) files.
            Relative paths are resolved against the bundled ``carla_maps/`` directory.
        :param location: Optional location name for map metadata.
        :param interpolation_step_size: Step size for interpolating polylines, defaults to 1.0
        :param connection_distance_threshold: Distance threshold for connecting road elements, defaults to 0.1
        :param internal_only: If True, only write internal road lines, defaults to True
        """
        self._xodr_paths = [Path(p) if Path(p).is_absolute() else _CARLA_MAPS_DIR / p for p in xodr_paths]
        for p in self._xodr_paths:
            assert p.exists(), f"XODR file not found: {p}"
            assert p.suffix in _VALID_SUFFIXES, f"Expected .xodr or .xodr.gz file, got: {p}"
        self._location = location
        self._interpolation_step_size = interpolation_step_size
        self._connection_distance_threshold = connection_distance_threshold
        self._internal_only = internal_only

    def get_map_parsers(self) -> List[BaseMapParser]:
        """Returns one map parser per XODR file."""
        return [
            OpenDriveMapParser(
                xodr_path=xodr_path,
                location=self._location,
                interpolation_step_size=self._interpolation_step_size,
                connection_distance_threshold=self._connection_distance_threshold,
                internal_only=self._internal_only,
            )
            for xodr_path in self._xodr_paths
        ]

    @override
    def get_log_parsers(self) -> List[BaseLogParser]:
        """No log conversion for OpenDRIVE maps."""
        return []
