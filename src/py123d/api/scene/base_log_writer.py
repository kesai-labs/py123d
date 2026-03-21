from __future__ import annotations

import abc

from py123d.datatypes import LogMetadata
from py123d.datatypes.modalities.base_modality import BaseModality
from py123d.parser.base_dataset_parser import ModalitiesSync


class BaseLogWriter(abc.ABC):
    """Base class for log writers.

    A log writer is responsible for specifying the output format of a converted log.
    This includes how data is organized, how it is serialized, and how it is stored.
    """

    @abc.abstractmethod
    def reset(self, log_metadata: LogMetadata) -> bool:
        """Prepare the writer for a new log. Returns True if the log needs writing."""

    @abc.abstractmethod
    def write_sync(self, modalities_sync: ModalitiesSync) -> None:
        """Write one synchronized frame — all modalities plus one sync-table row.

        :param timestamp: The timestamp of the frame.
        :param uuid: Optional UUID for the frame. If None, a deterministic UUID is generated.
        :param kwargs: Modality name -> data pairs to write.
        """

    @abc.abstractmethod
    def write_async(self, modality: BaseModality) -> None:
        """Write a single async modality observation.

        :param timestamp: The timestamp of the observation.
        :param modality_key: The modality name identifying the writer.
        :param data: The modality data to write.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Closes the log writer and finalizes the log io operations."""
