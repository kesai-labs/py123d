from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Union

from py123d.common.utils.enums import SerialIntEnum
from py123d.datatypes.modalities.base_modality import BaseModality, BaseModalityMetadata, ModalityType
from py123d.datatypes.time.timestamp import Timestamp


class CustomModalityMetadata(BaseModalityMetadata):
    """Metadata for a custom modality."""

    __slots__ = ("_modality_id", "_metadata")

    def __init__(self, modality_id: str, metadata: Dict[str, Any] = {}) -> None:
        """Initializes a CustomModalityMetadata instance.

        :param modality_id: The ID of the custom modality that this metadata describes.
        :param metadata: The metadata dictionary for the custom modality.
        """
        self._modality_id = modality_id
        self._metadata = metadata

    @property
    def modality_type(self) -> ModalityType:
        """Returns the type of the modality that this metadata describes."""
        return ModalityType.CUSTOM

    @property
    def modality_id(self) -> Optional[Union[str, SerialIntEnum]]:
        """Returns the ID of the custom modality that this metadata describes."""
        return self._modality_id

    @property
    def metadata(self) -> Dict[str, Any]:
        """Static, log-scoped metadata dictionary for this custom modality."""
        return self._metadata

    def to_dict(self) -> Dict[str, Any]:
        return {"modality_id": self._modality_id, "metadata": self._metadata}

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> CustomModalityMetadata:
        """Deserializes a CustomModalityMetadata from a dictionary.

        :param data_dict: A dictionary containing the metadata fields.
        :return: A CustomModalityMetadata instance.
        """
        modality_id = data_dict["modality_id"]
        metadata = data_dict.get("metadata", {})
        return cls(modality_id=modality_id, metadata=metadata)


class CustomModality(BaseModality):
    """A custom modality for dataset-specific information.

    This class wraps a dictionary (with string keys) and a corresponding
    :class:`~py123d.datatypes.time.Timestamp`. Values can be Python native
    types (``dict``, ``list``, ``str``, ``int``, ``float``, ``bytes``,
    ``bool``, ``None``) or ``numpy.ndarray``.
    """

    __slots__ = ("_data", "_metadata", "_timestamp")

    def __init__(self, data: Dict[str, Any], metadata: CustomModalityMetadata, timestamp: Timestamp) -> None:
        self._data = data
        self._metadata = metadata
        self._timestamp = timestamp

    @property
    def data(self) -> Dict[str, Any]:
        """The custom data dictionary."""
        return self._data

    @property
    def metadata(self) -> CustomModalityMetadata:
        """The :class:`CustomModalityMetadata` associated with this custom modality."""
        return self._metadata

    @property
    def timestamp(self) -> Timestamp:
        """The :class:`~py123d.datatypes.time.Timestamp` of this custom modality."""
        return self._timestamp

    def keys(self) -> List[str]:
        """Returns the keys of the custom data dictionary."""
        return list(self._data.keys())

    def __getitem__(self, key: str) -> Any:
        """Returns the value for *key*. Raises :class:`KeyError` if the key is not present."""
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        """Returns ``True`` if *key* exists in the custom data."""
        return key in self._data

    def __len__(self) -> int:
        """Returns the number of entries in the custom data."""
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        """Iterates over the keys of the custom data."""
        return iter(self._data)

    def __getattr__(self, name: str) -> Optional[Any]:
        """Provides attribute-style access to data keys. Returns ``None`` if the key is not present."""
        attr: Optional[Any] = self._data[name] if name in self._data.keys() else None
        return attr

    def __repr__(self) -> str:
        """Returns a string representation showing the available keys and timestamp."""
        keys = list(self._data.keys())
        return f"CustomModality(keys={keys}, timestamp={self._timestamp})"
