from __future__ import annotations

import enum
import sys
from typing import List, Optional, Sequence, Type, Union

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


class classproperty(object):
    """Decorator for class-level properties."""

    def __init__(self, f):
        """Initialize the classproperty with the given function."""
        self.f = f

    def __get__(self, obj, owner):
        """Get the property value."""
        return self.f(owner)


class SerialIntEnum(enum.Enum):
    """Base class for serializable integer enums."""

    def __int__(self) -> int:
        """Get the integer value of the enum."""
        return self.value

    def serialize(self, lower: bool = True) -> str:
        """Serialize the type when saving."""
        # Allow for lower/upper case letters during serialize
        return self.name.lower() if lower else self.name

    @classmethod
    def deserialize(cls, key: str) -> Self:
        """Deserialize the type when loading from a string."""
        # Allow for lower/upper case letters during deserialize
        return cls.__members__[key.upper()] if key.islower() else cls.__members__[key]

    @classmethod
    def from_int(cls, value: int) -> Self:
        """Get the enum from an int."""
        return cls(value)

    @classmethod
    def from_arbitrary(cls, value: Union[int, str, SerialIntEnum]) -> Self:
        """Get the enum from an int, string, or enum instance."""
        if isinstance(value, cls):
            return value
        elif isinstance(value, int):
            return cls.from_int(value)
        elif isinstance(value, str):
            return cls.deserialize(value)
        else:
            raise ValueError(f"Invalid value for {cls.__name__}: {value}")


def resolve_enum_arguments(
    serial_enum_cls: Type[SerialIntEnum], input: Optional[Sequence[Union[int, str, SerialIntEnum]]]
) -> Optional[List[SerialIntEnum]]:
    """Resolve a list of arbitrary enum representations to proper enum instances."""
    if input is None:
        return None
    if not isinstance(input, (list, tuple)):
        raise TypeError(f"input must be a list of {serial_enum_cls.__name__}, got {type(input)}")
    return [serial_enum_cls.from_arbitrary(value) for value in input]
