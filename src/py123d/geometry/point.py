from __future__ import annotations

from typing import Union

import numpy as np
import numpy.typing as npt
import shapely.geometry as geom

from py123d.common.utils.mixin import ArrayMixin, indexed_array_repr
from py123d.geometry.geometry_index import Point2DIndex, Point3DIndex
from py123d.geometry.vector import Vector2D, Vector3D


class Point2D(ArrayMixin):
    """Class presenting a 2D point.

    Example:
        >>> from py123d.geometry import Point2D
        >>> point_2d = Point2D(1.0, 2.0)
        >>> point_2d.x, point_2d.y
        (1.0, 2.0)
        >>> point_2d.array
        array([1., 2.])
    """

    __slots__ = ("_array",)
    _array: npt.NDArray[np.float64]

    def __init__(self, x: float, y: float):
        """Initializes :class:`Point2D` with x, y coordinates.

        :param x: The x coordinate.
        :param y: The y coordinate.
        """
        array = np.zeros(len(Point2DIndex), dtype=np.float64)
        array[Point2DIndex.X] = x
        array[Point2DIndex.Y] = y
        object.__setattr__(self, "_array", array)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> Point2D:
        """Creates a :class:`Point2D` from a (2,) shaped numpy array, indexed by :class:`~py123d.geometry.Point2DIndex`.

        :param array: A (2,) shaped numpy array representing the point coordinates (x,y).
        :param copy: Whether to copy the input array. Defaults to True.
        :return: A Point2D instance.
        """
        assert array.ndim == 1
        assert array.shape[0] == len(Point2DIndex)
        instance = object.__new__(cls)
        instance._array = array.copy() if copy else array
        return instance

    @property
    def x(self) -> float:
        """The x coordinate of the point."""
        return self._array[Point2DIndex.X]

    @property
    def y(self) -> float:
        """The y coordinate of the point."""
        return self._array[Point2DIndex.Y]

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """The array representation of shape (2,), indexed by :class:`~py123d.geometry.Point2DIndex`."""
        return self._array

    @property
    def shapely_point(self) -> geom.Point:
        """The shapely point representation of the 2D point."""
        return geom.Point(self.x, self.y)

    @property
    def point_2d(self) -> Point2D:
        """Returns the :class:`Point2D` instance itself."""
        return self

    def __add__(self, other: Vector2D) -> Point2D:
        """Translates the point by a vector: Point + Vector = Point.

        :param other: The displacement vector.
        :return: A new :class:`Point2D` at the translated position.
        """
        if not isinstance(other, Vector2D):
            return NotImplemented
        return Point2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Union[Point2D, Vector2D]) -> Union[Vector2D, Point2D]:
        """Point - Point = Vector (displacement), or Point - Vector = Point (translate backwards).

        :param other: Either a :class:`Point2D` or :class:`~py123d.geometry.Vector2D`.
        :return: A :class:`~py123d.geometry.Vector2D` if subtracting a point, \
            or a :class:`Point2D` if subtracting a vector.
        """
        if isinstance(other, Point2D):
            return Vector2D(self.x - other.x, self.y - other.y)
        elif isinstance(other, Vector2D):
            return Point2D(self.x - other.x, self.y - other.y)
        return NotImplemented

    def __repr__(self) -> str:
        """String representation of :class:`Point2D`."""
        return indexed_array_repr(self, Point2DIndex)


class Point3D(ArrayMixin):
    """Class presenting a 3D point.

    Example:
        >>> from py123d.geometry import Point3D
        >>> point_3d = Point3D(1.0, 2.0, 3.0)
        >>> point_3d.x, point_3d.y, point_3d.z
        (1.0, 2.0, 3.0)
        >>> point_3d.array
        array([1., 2., 3.])

    """

    __slots__ = ("_array",)
    _array: npt.NDArray[np.float64]

    def __init__(self, x: float, y: float, z: float):
        """Initializes :class:`Point3D` with x, y, z coordinates.

        :param x: The x coordinate.
        :param y: The y coordinate.
        :param z: The z coordinate.
        """
        array = np.zeros(len(Point3DIndex), dtype=np.float64)
        array[Point3DIndex.X] = x
        array[Point3DIndex.Y] = y
        array[Point3DIndex.Z] = z
        object.__setattr__(self, "_array", array)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> Point3D:
        """Creates a :class:`Point3D` from a (3,) shaped numpy array, indexed by :class:`~py123d.geometry.Point3DIndex`.

        :param array: A (3,) shaped numpy array representing the point coordinates (x,y,z).
        :param copy: Whether to copy the input array. Defaults to True.
        :return: A new :class:`Point3D` instance.
        """
        assert array.ndim == 1
        assert array.shape[0] == len(Point3DIndex)
        instance = object.__new__(cls)
        instance._array = array.copy() if copy else array
        return instance

    @property
    def x(self) -> float:
        """The x coordinate of the point."""
        return self._array[Point3DIndex.X]

    @property
    def y(self) -> float:
        """The y coordinate of the point."""
        return self._array[Point3DIndex.Y]

    @property
    def z(self) -> float:
        """The z coordinate of the point."""
        return self._array[Point3DIndex.Z]

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """The array representation of shape (3,), indexed by :class:`~py123d.geometry.Point3DIndex`."""
        return self._array

    @property
    def point_3d(self) -> Point3D:
        """Returns the :class:`Point3D` instance itself."""
        return self

    @property
    def point_2d(self) -> Point2D:
        """The 2D projection of the 3D point as a :class:`Point2D` instance."""
        return Point2D.from_array(self.array[Point3DIndex.XY], copy=False)

    @property
    def shapely_point(self) -> geom.Point:
        """The shapely point representation of the 3D point."""
        return geom.Point(self.x, self.y, self.z)

    def __add__(self, other: Vector3D) -> Point3D:
        """Translates the point by a vector: Point + Vector = Point.

        :param other: The displacement vector.
        :return: A new :class:`Point3D` at the translated position.
        """
        if not isinstance(other, Vector3D):
            return NotImplemented
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Union[Point3D, Vector3D]) -> Union[Vector3D, Point3D]:
        """Point - Point = Vector (displacement), or Point - Vector = Point (translate backwards).

        :param other: Either a :class:`Point3D` or :class:`~py123d.geometry.Vector3D`.
        :return: A :class:`~py123d.geometry.Vector3D` if subtracting a point, \
            or a :class:`Point3D` if subtracting a vector.
        """
        if isinstance(other, Point3D):
            return Vector3D.from_array(self.array - other.array, copy=False)
        elif isinstance(other, Vector3D):
            return Point3D.from_array(self.array - other.array, copy=False)
        return NotImplemented

    def __repr__(self) -> str:
        """String representation of :class:`Point3D`."""
        return indexed_array_repr(self, Point3DIndex)
