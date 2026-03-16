from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseMetadata(ABC):
    """Abstract base class for metadata information.

    All metadata classes must implement serialization to and deserialization
    from a plain Python dictionary containing only default Python types.
    """

    __slots__ = ()

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the metadata instance to a plain Python dictionary.

        :return: A dictionary representation using only default Python types.
        """

    @classmethod
    @abstractmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> BaseMetadata:
        """Construct a metadata instance from a plain Python dictionary.

        :param data_dict: A dictionary containing the metadata fields.
        :return: A metadata instance.
        """
