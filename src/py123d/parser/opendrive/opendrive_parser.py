from pathlib import Path
from typing import Dict, List, Optional, Union

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
        road_edge_fill_hole_points: Optional[Dict[str, List[List[float]]]] = None,
        road_edge_non_drivable_points: Optional[Dict[str, List[List[float]]]] = None,
        non_drivable_none_lane_min_width: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initializes the OpenDriveParser.

        :param xodr_paths: List of paths to OpenDRIVE (.xodr or .xodr.gz) files.
            Relative paths are resolved against the bundled ``carla_maps/`` directory.
        :param location: Optional location name for map metadata.
        :param interpolation_step_size: Step size for interpolating polylines, defaults to 1.0
        :param connection_distance_threshold: Distance threshold for connecting road elements, defaults to 0.1
        :param internal_only: If True, only write internal road lines, defaults to True
        :param road_edge_fill_hole_points: Per-location (x, y) points marking road-edge holes to fill,
            patching known bugs in the source maps, defaults to None
        :param road_edge_non_drivable_points: Per-location (x, y) points marking shoulder/none-lane
            surfaces that are not drivable in reality, defaults to None
        :param non_drivable_none_lane_min_width: Per-location width threshold above which none
            lanes on non-junction roads count as median strips, defaults to None
        """
        self._xodr_paths = [Path(p) if Path(p).is_absolute() else _CARLA_MAPS_DIR / p for p in xodr_paths]
        for p in self._xodr_paths:
            assert p.exists(), f"XODR file not found: {p}"
            assert p.suffix in _VALID_SUFFIXES, f"Expected .xodr or .xodr.gz file, got: {p}"
        self._location = location
        self._interpolation_step_size = interpolation_step_size
        self._connection_distance_threshold = connection_distance_threshold
        self._internal_only = internal_only
        self._road_edge_fill_hole_points = road_edge_fill_hole_points or {}
        self._road_edge_non_drivable_points = road_edge_non_drivable_points or {}
        self._non_drivable_none_lane_min_width = non_drivable_none_lane_min_width or {}

    def get_map_parsers(self) -> List[BaseMapParser]:
        """Returns one map parser per XODR file."""
        return [
            OpenDriveMapParser(
                xodr_path=xodr_path,
                location=self._location,
                interpolation_step_size=self._interpolation_step_size,
                connection_distance_threshold=self._connection_distance_threshold,
                internal_only=self._internal_only,
                road_edge_fill_hole_points=self._road_edge_fill_hole_points.get(
                    xodr_path.name.removesuffix("".join(xodr_path.suffixes))
                ),
                road_edge_non_drivable_points=self._road_edge_non_drivable_points.get(
                    xodr_path.name.removesuffix("".join(xodr_path.suffixes))
                ),
                non_drivable_none_lane_min_width=self._non_drivable_none_lane_min_width.get(
                    xodr_path.name.removesuffix("".join(xodr_path.suffixes))
                ),
            )
            for xodr_path in self._xodr_paths
        ]

    @override
    def get_log_parsers(self) -> List[BaseLogParser]:
        """No log conversion for OpenDRIVE maps."""
        return []
