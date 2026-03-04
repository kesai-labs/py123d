from __future__ import annotations

from typing import Dict, Optional

import py123d
from py123d.datatypes.metadata.abstract_metadata import AbstractMetadata


class LogMetadata(AbstractMetadata):
    """Class to hold metadata information about a log."""

    __slots__ = ("_dataset", "_split", "_log_name", "_location", "_timestep_seconds", "_version")

    def __init__(
        self,
        dataset: str,
        split: str,
        log_name: str,
        location: Optional[str],
        timestep_seconds: float,
        version: str = str(py123d.__version__),
    ):
        """Create a :class:`LogMetadata` instance from a dictionary.

        :param dataset: The dataset name in lowercase.
        :param split: Data split name, typically ``{dataset_name}_{train/val/test}``.
        :param log_name: Name of the log file.
        :param location: Location of the log data.
        :param timestep_seconds: The time interval between consecutive frames in seconds.
        """
        self._dataset = dataset
        self._split = split
        self._log_name = log_name
        self._location = location
        self._timestep_seconds = timestep_seconds
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
    def timestep_seconds(self) -> float:
        """The time interval between consecutive frames in seconds."""
        return self._timestep_seconds

    @property
    def version(self) -> str:
        """Version of the py123d library used to create this log metadata (not used currently)."""
        return self._version

    @classmethod
    def from_dict(cls, data_dict: Dict) -> LogMetadata:
        """Create a :class:`LogMetadata` instance from a Python dictionary.

        :param data_dict: Dictionary containing log metadata.
        :raises ValueError: If the dictionary is missing required fields.
        :return: A :class:`LogMetadata` instance.
        """

        return LogMetadata(**data_dict)

    def to_dict(self) -> Dict:
        """Convert the :class:`LogMetadata` instance to a Python dictionary.

        :return: A dictionary representation of the log metadata.
        """
        data_dict = {slot.lstrip("_"): getattr(self, slot) for slot in self.__slots__}
        return data_dict

    def __repr__(self) -> str:
        return (
            f"LogMetadata(dataset={self.dataset}, split={self.split}, log_name={self.log_name}, "
            f"location={self.location}, timestep_seconds={self.timestep_seconds}, "
            f"version={self.version})"
        )
