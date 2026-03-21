import abc
from abc import abstractmethod

from py123d.datatypes import BaseMapObject, MapMetadata


class BaseMapWriter(abc.ABC):
    """Base class for map writers."""

    @abstractmethod
    def reset(self, map_metadata: MapMetadata) -> bool:
        """Reset the writer to its initial state."""

    @abstractmethod
    def write_map_object(self, map_object: BaseMapObject) -> None:
        """Writes a map objects."""

    @abstractmethod
    def close(self) -> None:
        """Close the writer and finalize any resources."""
