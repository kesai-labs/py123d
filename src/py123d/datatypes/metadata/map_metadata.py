from __future__ import annotations

from typing import Any, Dict, Optional

import py123d
from py123d.datatypes.metadata.base_metadata import BaseMetadata


class MapMetadata(BaseMetadata):
    """Class to hold metadata information about a map."""

    __slots__ = ("_dataset", "_split", "_log_name", "_location", "_map_has_z", "_map_is_per_log", "_version")

    def __init__(
        self,
        dataset: str,
        location: Optional[str],
        map_has_z: bool,
        map_is_per_log: bool,
        split: Optional[str] = None,
        log_name: Optional[str] = None,
        version: str = str(py123d.__version__),
    ):
        """Initialize a MapMetadata instance.

        :param dataset: The dataset name in lowercase.
        :param location: The location of the map data.
        :param map_has_z: Indicates if the map includes Z (elevation) data.
        :param map_is_per_log: Indicates if the map is per-log (map for each log) or
            global (map for multiple logs in dataset).
        :param split: Data split name, typically ``{dataset_name}_{train/val/test}``, defaults to None
        :param log_name: Name of the log file, defaults to None
        :param version: Version of the py123d library used to create this map metadata,
            defaults to str(py123d.__version__)
        """
        self._dataset = dataset
        self._split = split
        self._log_name = log_name
        self._location = location
        self._map_has_z = map_has_z
        self._map_is_per_log = map_is_per_log
        self._version = version

        if self._map_is_per_log:
            assert self._split is not None, "For per-log maps, split must be provided."
            assert self._log_name is not None, "For per-log maps, log_name must be provided."

    @property
    def dataset(self) -> str:
        """The dataset name in lowercase."""
        return self._dataset

    @property
    def split(self) -> Optional[str]:
        """Data split name, typically ``{dataset_name}_{train/val/test}``."""
        return self._split

    @property
    def log_name(self) -> Optional[str]:
        """Name of the log file."""
        return self._log_name

    @property
    def location(self) -> Optional[str]:
        """Location of the map data."""
        return self._location

    @property
    def map_has_z(self) -> bool:
        """Indicates if the map includes Z (elevation) data."""
        return self._map_has_z

    @property
    def map_is_per_log(self) -> bool:
        """Indicates if the map is per-log (map for each log) or global (map for multiple logs in dataset)."""
        return self._map_is_per_log

    @property
    def version(self) -> str:
        """Version of the py123d library used to create this map metadata."""
        return self._version

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> MapMetadata:
        """Create a MapMetadata instance from a dictionary.

        :param data_dict: A dictionary representation of a MapMetadata instance.
        :return: A MapMetadata instance.
        """
        return MapMetadata(**data_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the MapMetadata instance to a dictionary.

        :return: A dictionary representation of the MapMetadata instance.
        """
        return {slot.lstrip("_"): getattr(self, slot) for slot in self.__slots__}

    def __repr__(self) -> str:
        return (
            f"MapMetadata("
            f"dataset={self.dataset!r}, "
            f"split={self.split!r}, "
            f"log_name={self.log_name!r}, "
            f"location={self.location!r}, "
            f"map_has_z={self.map_has_z}, "
            f"map_is_per_log={self.map_is_per_log}, "
            f"version={self.version!r}"
            f")"
        )
