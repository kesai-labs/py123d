from __future__ import annotations

from typing import Dict, Optional

import py123d
from py123d.datatypes.metadata.base_metadata import BaseMetadata
from py123d.datatypes.metadata.map_metadata import MapMetadata


class LogMetadata(BaseMetadata):
    """Class to hold metadata information about a log."""

    __slots__ = ("_dataset", "_split", "_log_name", "_location", "_map_metadata", "_version")

    def __init__(
        self,
        dataset: str,
        split: str,
        log_name: str,
        location: Optional[str],
        map_metadata: Optional[MapMetadata] = None,
        version: str = str(py123d.__version__),
    ):
        """Create a :class:`LogMetadata` instance from a dictionary.

        :param dataset: The dataset name in lowercase.
        :param split: Data split name, typically ``{dataset_name}_{train/val/test}``.
        :param log_name: Name of the log file.
        :param location: Location of the log data.
        """

        # Basic log info
        self._dataset = dataset
        self._split = split
        self._log_name = log_name
        self._location = location

        # Map metadata
        self._map_metadata: Optional[MapMetadata] = map_metadata

        # Currently not used, but can be helpful for tracking library version used to create the log metadata
        self._version = version

    @property
    def dataset(self) -> str:
        """The dataset name in lowercase."""
        return self._dataset

    @property
    def split(self) -> str:
        """Data split name, typically ``{dataset_name}_{train/val/test}``."""
        return self._split

    @property
    def log_name(self) -> str:
        """Name of the log file."""
        return self._log_name

    @property
    def location(self) -> Optional[str]:
        """Location of the log data."""
        return self._location

    @property
    def version(self) -> str:
        """Version of the py123d library used to create this log metadata (not used currently)."""
        return self._version

    @property
    def map_metadata(self) -> Optional[MapMetadata]:
        """Map metadata for this log, if available."""
        return self._map_metadata

    @classmethod
    def from_dict(cls, data_dict: Dict) -> LogMetadata:
        """Create a :class:`LogMetadata` instance from a Python dictionary.

        Deserializes both basic log fields and modality metadata (if present).
        Older dictionaries that only contain basic fields are handled gracefully.

        :param data_dict: Dictionary containing log metadata.
        :return: A :class:`LogMetadata` instance.
        """
        # Map metadata
        map_meta_raw = data_dict.get("map_metadata")
        map_metadata = MapMetadata.from_dict(map_meta_raw) if map_meta_raw is not None else None

        return LogMetadata(
            dataset=data_dict["dataset"],
            split=data_dict["split"],
            log_name=data_dict["log_name"],
            location=data_dict.get("location"),
            version=data_dict.get("version", "unknown"),
            map_metadata=map_metadata,
        )

    def to_dict(self) -> Dict:
        """Convert the :class:`LogMetadata` instance to a JSON-serializable dictionary.

        :return: A dictionary representation of the log metadata.
        """
        return {
            "dataset": self._dataset,
            "split": self._split,
            "log_name": self._log_name,
            "location": self._location,
            "version": self._version,
            "map_metadata": self._map_metadata.to_dict() if self._map_metadata is not None else None,
        }

    def __repr__(self) -> str:
        return (
            f"LogMetadata(dataset={self.dataset}, split={self.split}, log_name={self.log_name}, "
            f"location={self.location}, version={self.version})"
        )
