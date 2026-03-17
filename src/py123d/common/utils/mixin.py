from __future__ import annotations

from enum import IntEnum

import numpy as np
import numpy.typing as npt
from typing_extensions import Self


class ArrayMixin:
    """Mixin class to provide array-like behavior for classes.

    Example:
    >>> import numpy as np
    >>> from py123d.common.utils.mixin import ArrayMixin
    >>> class MyVector(ArrayMixin):
    ...     def __init__(self, x: float, y: float):
    ...         self._array = np.array([x, y], dtype=np.float64)
    ...     @property
    ...     def array(self) -> npt.NDArray[np.float64]:
    ...         return self._array
    ...     @classmethod
    ...     def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> MyVector:
    ...         if copy:
    ...             array = array.copy()
    ...         return cls(array[0], array[1])
    >>> vec = MyVector(1.0, 2.0)
    >>> print(vec)
    MyVector(array=[1. 2.])
    >>> np.array(vec, dtype=np.float32)
    array([1., 2.], dtype=float32)

    """

    __slots__ = ()

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> Self:
        """Create an instance from a NumPy array."""
        raise NotImplementedError

    @classmethod
    def from_list(cls, values: list) -> Self:
        """Create an instance from a list of values."""
        return cls.from_array(np.asarray(values, dtype=np.float64), copy=False)

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """The array representation of the geometric entity."""
        raise NotImplementedError

    def __array__(self, dtype: npt.DTypeLike = None, copy: bool = False) -> npt.NDArray:  # noqa: PLW3201
        array = self.array
        return array if dtype is None else array.astype(dtype=dtype, copy=copy)

    def __len__(self) -> int:
        """Return the length of the array."""
        return len(self.array)

    def __getitem__(self, key):
        """Allow indexing into the array."""
        return self.array[key]

    def __eq__(self, other) -> bool:
        """Equality comparison based on array values."""
        if isinstance(other, ArrayMixin):
            return np.array_equal(self.array, other.array)
        return False

    @property
    def shape(self) -> tuple:
        """Return the shape of the array."""
        return self.array.shape

    def tolist(self) -> list:
        """Convert the array to a Python list."""
        return self.array.tolist()

    def to_list(self) -> list:
        """Convert the array to a Python list."""
        return self.array.tolist()

    def copy(self) -> ArrayMixin:
        """Return a copy of the object with a copied array."""
        return self.__class__.from_array(self.array, copy=True)

    def __repr__(self) -> str:
        """String representation of the ArrayMixin instance."""
        return f"{self.__class__.__name__}(array={self.array})"

    def __hash__(self):
        """Hash based on the array values."""
        return hash(self.array.tobytes())


def indexed_array_repr(array_mixin: ArrayMixin, indexing: type[IntEnum]) -> str:
    """Generate a string representation of an ArrayMixin instance using an indexing enum.

    :param array_mixin: An instance of ArrayMixin.
    :param indexing: An IntEnum used for indexing the array.
    :return: A string representation of the ArrayMixin instance with named fields.
    """
    args = ", ".join(
        f"{index.name.lower()}={array_mixin.array[index.value]}" for index in indexing.__members__.values()
    )
    return f"{array_mixin.__class__.__name__}({args})"
