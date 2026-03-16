from __future__ import annotations

from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import shapely.creation as geom_creation
import shapely.geometry as geom
from scipy.interpolate import interp1d

from py123d.common.utils.mixin import ArrayMixin
from py123d.geometry.geometry_index import Point2DIndex, Point3DIndex, PoseSE2Index, PoseSE3Index
from py123d.geometry.point import Point2D, Point3D
from py123d.geometry.pose import PoseSE2, PoseSE3
from py123d.geometry.utils.constants import DEFAULT_Z
from py123d.geometry.utils.polyline_utils import get_linestring_yaws, get_path_progress_2d, get_path_progress_3d
from py123d.geometry.utils.rotation_utils import nlerp_quaternion_arrays, normalize_angle, slerp_quaternion_arrays


class Polyline2D(ArrayMixin):
    """Represents a interpolatable 2D polyline.

    Example:
        >>> import numpy as np
        >>> from py123d.geometry import Polyline2D
        >>> polyline = Polyline2D.from_array(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]))
        >>> polyline.length
        2.8284271247461903
        >>> polyline.interpolate(np.sqrt(2))
        Point2D(array=[1. 1.])

    """

    __slots__ = ("_linestring",)
    _linestring: geom.LineString

    @classmethod
    def from_linestring(cls, linestring: geom.LineString) -> Polyline2D:
        """Creates a :class:`Polyline2D` from a Shapely LineString. If the LineString has Z-coordinates, they are ignored.

        :param linestring: A shapely LineString object.
        :return: A Polyline2D instance.
        """
        if linestring.has_z:
            linestring_ = geom_creation.linestrings(*linestring.xy)  # pyright: ignore[reportUnknownMemberType]
        else:
            linestring_ = linestring

        instance = object.__new__(cls)
        instance._linestring = linestring_
        return instance

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> Polyline2D:
        """Creates a :class:`Polyline2D` from a (N, 2) or (N, 3) shaped numpy array. \
            Assumes [...,:2] slices are XY coordinates.

        :param polyline_array: A numpy array of shape (N, 2) or (N, 3), e.g. indexed by \
            :class:`~py123d.geometry.Point2DIndex` or :class:`~py123d.geometry.Point3DIndex`.
        :raises ValueError: If the input array is not of the expected shape.
        :return: A :class:`Polyline2D` instance.
        """
        assert array.ndim == 2
        linestring_: Optional[geom.LineString] = None
        if array.shape[-1] == len(Point2DIndex):
            linestring_ = geom.LineString(array)
        elif array.shape[-1] == len(Point3DIndex):
            linestring_ = geom.LineString(array[:, Point3DIndex.XY])  # pyright: ignore[reportUnknownMemberType]
        else:
            raise ValueError("Array must have shape (N, 2) or (N, 3) for Point2D or Point3D respectively.")

        instance = object.__new__(cls)
        instance._linestring = linestring_
        return instance

    @property
    def linestring(self) -> geom.LineString:
        """The shapely LineString representation of the polyline."""
        return self._linestring

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """The numpy array representation of shape (N, 2), indexed by :class:`~py123d.geometry.Point2DIndex`."""
        x, y = self.linestring.xy
        array = np.zeros((len(x), len(Point2DIndex)), dtype=np.float64)
        array[:, Point2DIndex.X] = x
        array[:, Point2DIndex.Y] = y
        return array

    @property
    def polyline_se2(self) -> PolylineSE2:
        """The :class:`~py123d.geometry.PolylineSE2` representation of the polyline, with inferred yaw angles."""
        return PolylineSE2.from_linestring(self._linestring)

    @property
    def length(self) -> float:
        """Returns the length of the polyline."""
        return self.linestring.length

    def interpolate(
        self,
        distances: Union[float, npt.NDArray[np.float64]],
        normalized: bool = False,
    ) -> Union[Point2D, npt.NDArray[np.float64]]:
        """Interpolates the :class:`Polyline2D` at the given distances.

        :param distances: Array-like or float distances along the polyline to interpolate.
        :return: The interpolated point(s) on the polyline.
        """

        if isinstance(distances, float) or isinstance(distances, int):
            point = self.linestring.interpolate(distances, normalized=normalized)
            return Point2D(point.x, point.y)
        else:
            points = self.linestring.interpolate(distances, normalized=normalized)
            return np.array([[p.x, p.y] for p in points], dtype=np.float64)

    def project(
        self,
        point: Union[
            geom.Point,
            Point2D,
            PoseSE2,
            npt.NDArray[np.float64],
        ],
        normalized: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Projects a point onto the polyline and returns the distance along the polyline to the closest point.

        :param point: The point to project onto the polyline.
        :param normalized: Whether to return the normalized distance, defaults to False.
        :return: The distance along the polyline to the closest point.
        """
        if isinstance(point, Point2D) or isinstance(point, PoseSE2):
            point_ = point.shapely_point
        elif isinstance(point, geom.Point):
            point_ = point
        else:
            point_ = geom.Point(np.array(point, dtype=np.float64))
        return self._linestring.project(point_, normalized=normalized)  # type: ignore


class PolylineSE2(ArrayMixin):
    """Represents a interpolatable SE2 polyline.

    Example:
        >>> import numpy as np
        >>> from py123d.geometry import PolylineSE2
        >>> polyline_se2 = PolylineSE2.from_array(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, np.pi/4], [2.0, 0.0, 0.0]]))
        >>> polyline_se2.length
        2.8284271247461903
        >>> polyline_se2.interpolate(np.sqrt(2))
        PoseSE2(array=[1.         1.         0.78539816])

    """

    __slots__ = ("_array", "_progress", "_linestring")

    def __init__(
        self,
        array: npt.NDArray[np.float64],
        linestring: Optional[geom.LineString] = None,
    ):
        """Initializes :class:`PolylineSE2` with a numpy array of SE2 states.

        :param array: A numpy array of shape (N, 3) representing SE2 states, indexed by \
            :class:`~py123d.geometry.PoseSE2Index`.
        :param linestring: Optional shapely LineString representing the XY path. If not provided,\
            it will be created from the array.
        """

        self._array = array
        self._array[:, PoseSE2Index.YAW] = np.unwrap(self._array[:, PoseSE2Index.YAW], axis=0)
        self._progress = get_path_progress_2d(self._array)
        self._linestring = geom.LineString(self._array[..., PoseSE2Index.XY]) if linestring is None else linestring

    @classmethod
    def from_linestring(cls, linestring: geom.LineString) -> PolylineSE2:
        """Creates a :class:`PolylineSE2` from a shapely LineString. \
            The yaw angles are inferred from the LineString coordinates.

        :param linestring: The LineString to convert.
        :return: A :class:`PolylineSE2` representing the same path as the LineString.
        """
        points_2d_array = np.array(linestring.coords, dtype=np.float64)[..., PoseSE2Index.XY]
        se2_array = np.zeros((len(points_2d_array), len(PoseSE2Index)), dtype=np.float64)
        se2_array[:, PoseSE2Index.XY] = points_2d_array
        se2_array[:, PoseSE2Index.YAW] = get_linestring_yaws(linestring)
        return PolylineSE2(se2_array, linestring)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> PolylineSE2:
        """Creates a :class:`PolylineSE2` from a numpy array.

        :param polyline_array: The input numpy array representing, either indexed by \
            :class:`~py123d.geometry.Point2DIndex` or :class:`~py123d.geometry.PoseSE2Index`.
        :param copy: Whether to copy the input array or not (ignored).
        :raises ValueError: If the input array is not of the expected shape.
        :return: A :class:`PolylineSE2` representing the same path as the input array.
        """
        assert array.ndim == 2
        if array.shape[-1] == len(Point2DIndex):
            se2_array = np.zeros((len(array), len(PoseSE2Index)), dtype=np.float64)
            se2_array[:, PoseSE2Index.XY] = array
            se2_array[:, PoseSE2Index.YAW] = get_linestring_yaws(geom_creation.linestrings(*array.T))
        elif array.shape[-1] == len(PoseSE2Index):
            se2_array = np.array(array, dtype=np.float64)
        else:
            raise ValueError(f"Invalid polyline array shape, expected (N, 2) or (N, 3), got {array.shape}.")
        return PolylineSE2(se2_array)

    @property
    def linestring(self) -> geom.LineString:
        """The shapely LineString representation of the polyline."""
        return self._linestring

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """The numpy array representation of shape (N, 3), indexed by :class:`~py123d.geometry.PoseSE2Index`."""
        return self._array

    @property
    def length(self) -> float:
        """Returns the length of the polyline."""
        assert self._progress is not None
        return float(self._progress[-1])

    def interpolate(
        self,
        distances: Union[float, npt.NDArray[np.float64]],
        normalized: bool = False,
    ) -> Union[PoseSE2, npt.NDArray[np.float64]]:
        """Interpolates the polyline at the given distances.

        :param distances: The distances along the polyline to interpolate.
        :param normalized: Whether the distances are normalized (0 to 1), defaults to False
        :return: The interpolated StateSE2 or an array of interpolated states, according to
        """
        _interpolator = interp1d(self._progress, self._array, axis=0, bounds_error=False, fill_value=0.0)
        distances_ = distances * self.length if normalized else distances
        clipped_distances = np.clip(distances_, 1e-8, self.length)

        interpolated_se2_array = _interpolator(clipped_distances)
        interpolated_se2_array[..., PoseSE2Index.YAW] = normalize_angle(interpolated_se2_array[..., PoseSE2Index.YAW])

        if clipped_distances.ndim == 0:
            return PoseSE2(*interpolated_se2_array)
        else:
            return interpolated_se2_array

    def project(
        self,
        point: Union[
            geom.Point,
            Point2D,
            PoseSE2,
            npt.NDArray[np.float64],
        ],
        normalized: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Projects a point onto the polyline and returns the distance along the polyline to the closest point.

        :param point: The point to project onto the polyline.
        :param normalized: Whether to return the normalized distance, defaults to False.
        :return: The distance along the polyline to the closest point.
        """
        if isinstance(point, Point2D) or isinstance(point, PoseSE2):
            point_ = point.shapely_point
        elif isinstance(point, geom.Point):
            point_ = point
        else:
            point_ = geom.Point(np.array(point, dtype=np.float64))
        return self.linestring.project(point_, normalized=normalized)  # type: ignore


class PolylineSE3(ArrayMixin):
    """Represents an interpolatable SE3 polyline (3D position + quaternion rotation).

    Supports pluggable rotation interpolation strategies: SLERP (default) and NLERP.

    Example:
        >>> import numpy as np
        >>> from py123d.geometry import PolylineSE3
        >>> poses = np.array([
        ...     [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ...     [2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ... ])
        >>> polyline = PolylineSE3.from_array(poses)
        >>> polyline.length
        2.0
        >>> polyline.interpolate(1.0)
        PoseSE3(array=[1. 0. 0. 1. 0. 0. 0.])

    """

    __slots__ = ("_array", "_progress", "_translation_interpolator", "_rotation_interpolation")

    def __init__(
        self,
        array: npt.NDArray[np.float64],
        rotation_interpolation: str = "slerp",
    ):
        """Initializes :class:`PolylineSE3` with a numpy array of SE3 poses.

        :param array: A numpy array of shape (N, 7) representing SE3 poses, indexed by \
            :class:`~py123d.geometry.PoseSE3Index`.
        :param rotation_interpolation: Rotation interpolation strategy, either ``"slerp"`` or ``"nlerp"``.
        """
        assert array.ndim == 2 and array.shape[1] == len(PoseSE3Index)
        assert rotation_interpolation in ("slerp", "nlerp"), (
            f"Unknown rotation interpolation: {rotation_interpolation!r}. Expected 'slerp' or 'nlerp'."
        )

        self._array = array
        self._rotation_interpolation = rotation_interpolation
        self._progress = get_path_progress_3d(array[:, PoseSE3Index.XYZ])
        self._translation_interpolator = interp1d(
            self._progress,
            array[:, PoseSE3Index.XYZ],
            axis=0,
            bounds_error=False,
            fill_value="extrapolate",  # pyright: ignore[reportArgumentType]
        )

    @classmethod
    def from_array(
        cls,
        array: npt.NDArray[np.float64],
        copy: bool = True,
        rotation_interpolation: str = "slerp",
    ) -> "PolylineSE3":
        """Creates a :class:`PolylineSE3` from a numpy array.

        :param array: A numpy array of shape (N, 7) representing SE3 poses, indexed by \
            :class:`~py123d.geometry.PoseSE3Index`.
        :param copy: Whether to copy the input array. Defaults to True.
        :param rotation_interpolation: Rotation interpolation strategy, either ``"slerp"`` or ``"nlerp"``.
        :return: A :class:`PolylineSE3` instance.
        """
        assert array.ndim == 2 and array.shape[1] == len(PoseSE3Index)
        return cls(array.copy() if copy else array, rotation_interpolation=rotation_interpolation)

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """The numpy array representation of shape (N, 7), indexed by :class:`~py123d.geometry.PoseSE3Index`."""
        return self._array

    @property
    def length(self) -> float:
        """Returns the translational path length of the SE3 polyline."""
        return float(self._progress[-1])

    @property
    def rotation_interpolation(self) -> str:
        """The rotation interpolation strategy used by this polyline."""
        return self._rotation_interpolation

    def interpolate(
        self,
        distances: Union[float, npt.NDArray[np.float64]],
        normalized: bool = False,
    ) -> Union[PoseSE3, npt.NDArray[np.float64]]:
        """Interpolates the SE3 polyline at the given distances.

        Translation is interpolated linearly; rotation is interpolated using the configured strategy.

        :param distances: A float or numpy array of distances along the polyline.
        :param normalized: Whether to interpret the distances as fractions of the length.
        :return: A :class:`PoseSE3` instance for scalar input, or a numpy array of shape (N, 7).
        """
        distances_ = np.asarray(distances * self.length if normalized else distances, dtype=np.float64)
        clipped = np.clip(distances_, 1e-8, self.length)

        # Interpolate translation via scipy
        translations = self._translation_interpolator(clipped)

        # Find surrounding keyframe indices and compute local interpolation parameter t
        indices = np.searchsorted(self._progress, clipped, side="right") - 1
        indices = np.clip(indices, 0, len(self._progress) - 2)
        segment_length = self._progress[indices + 1] - self._progress[indices]
        t = np.where(segment_length > 0, (clipped - self._progress[indices]) / segment_length, 0.0)

        # Interpolate rotation
        q1 = self._array[indices, PoseSE3Index.QUATERNION]
        q2 = self._array[indices + 1, PoseSE3Index.QUATERNION]

        if self._rotation_interpolation == "slerp":
            rotations = slerp_quaternion_arrays(q1, q2, t)
        else:
            rotations = nlerp_quaternion_arrays(q1, q2, t)

        # Combine translation and rotation
        result = np.empty(translations.shape[:-1] + (len(PoseSE3Index),), dtype=np.float64)
        result[..., PoseSE3Index.XYZ] = translations
        result[..., PoseSE3Index.QUATERNION] = rotations

        if clipped.ndim == 0:
            return PoseSE3(*result)
        return result


class Polyline3D(ArrayMixin):
    """Represents a interpolatable 3D polyline.

    Example:
        >>> import numpy as np
        >>> from py123d.geometry import Polyline3D
        >>> polyline_3d = Polyline3D.from_array(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 0.0, 0.0]]))
        >>> polyline_3d.length
        3.4641016151377544
        >>> polyline_3d.interpolate(np.sqrt(3))
        Point3D(array=[1. 1. 1.])

    """

    __slots__ = ("_array", "_progress", "_linestring")

    def __init__(self, array: npt.NDArray[np.float64], linestring: Optional[geom.LineString] = None):
        """Initializes :class:`Polyline3D` with a numpy array of 3D points.

        :param array: A numpy array of shape (N, 3) representing 3D points, e.g. indexed by \
            :class:`~py123d.geometry.Point3DIndex`.
        :param linestring: Optional shapely LineString representing the 3D path. If not provided,\
            it will be created from the array.
        """
        assert len(array.shape) == 2 and array.shape[1] == len(Point3DIndex)
        self._array = array

        # Dynamically computed for faster initialization
        self._progress: Optional[npt.NDArray[np.float64]] = None
        self._linestring: Optional[geom.LineString] = linestring

    @classmethod
    def from_linestring(cls, linestring: geom.LineString) -> Polyline3D:
        """Creates a :class:`Polyline3D` from a shapely LineString. If the LineString does not have Z-coordinates, \
            the coordinate is zero-padded.

        :param linestring: The input LineString.
        :return: A :class:`Polyline3D` instance.
        """
        if linestring.has_z:
            linestring_ = linestring
        else:
            linestring_ = geom_creation.linestrings(*linestring.xy, z=DEFAULT_Z)  # type: ignore
        array = np.array(linestring_.coords, dtype=np.float64)
        return Polyline3D(array, linestring_)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> Polyline3D:
        """Creates a :class:`Polyline3D` from a numpy array.

        :param array: A numpy array of shape (N, 3) representing 3D points, e.g. indexed by \
            :class:`~py123d.geometry.Point3DIndex`.
        :return: A :class:`Polyline3D` instance.
        """
        assert array.ndim == 2, "Array must be 2D with shape (N, 3) or (N, 2)."
        if array.shape[1] == len(Point2DIndex):
            array = np.hstack((array, np.full((array.shape[0], 1), DEFAULT_Z)))
        elif array.shape[1] != len(Point3DIndex):
            raise ValueError("Array must have shape (N, 3) for Point3D.")
        return Polyline3D(array.copy() if copy else array)

    @property
    def linestring(self) -> geom.LineString:
        """The shapely LineString representation of the 3D polyline."""
        if self._linestring is None or not self._linestring.has_z:
            self._linestring = geom_creation.linestrings(*self._array.T)  # type: ignore
        assert self._linestring is not None, "Linestring should have been initialized."
        return self._linestring

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """The numpy array representation of shape (N, 3), indexed by :class:`~py123d.geometry.Point3DIndex`."""
        return np.array(self.linestring.coords, dtype=np.float64)

    @property
    def polyline_2d(self) -> Polyline2D:
        """The :class:`~py123d.geometry.Polyline2D` representation of the 3D polyline."""
        return Polyline2D.from_linestring(geom_creation.linestrings(*self.linestring.xy))  # type: ignore

    @property
    def polyline_se2(self) -> PolylineSE2:
        """The :class:`~py123d.geometry.PolylineSE2` representation of the 3D polyline."""
        return PolylineSE2.from_linestring(self.linestring)  # type: ignore\

    @property
    def length(self) -> float:
        """Returns the length of the 3D polyline."""
        if self._progress is None:
            self._progress = get_path_progress_3d(self._array[:, Point3DIndex.XYZ])

        assert self._progress is not None, "Progress should have been initialized."
        return float(self._progress[-1])

    def interpolate(
        self,
        distances: Union[float, npt.NDArray[np.float64]],
        normalized: bool = False,
    ) -> Union[Point3D, npt.NDArray[np.float64]]:
        """Interpolates the 3D polyline at the given distances.

        :param distances: A float or numpy array of distances along the polyline.
        :param normalized: Whether to interpret the distances as fractions of the length.
        :return: A Point3D instance or a numpy array of shape (N, 3) representing the interpolated points.
        """
        if self._progress is None:
            self._progress = get_path_progress_3d(self._array[:, Point3DIndex.XYZ])
        assert self._progress is not None, "Progress should have been initialized."

        _interpolator = interp1d(
            self._progress,
            self._array,
            axis=0,
            bounds_error=False,
            fill_value="extrapolate",  # pyright: ignore[reportArgumentType]
        )
        distances_ = distances * self.length if normalized else distances
        clipped_distances = np.clip(distances_, 1e-8, self.length)

        interpolated_3d_array = _interpolator(clipped_distances)
        if clipped_distances.ndim == 0:
            return Point3D(*interpolated_3d_array)
        else:
            return interpolated_3d_array

    def project(
        self,
        point: Union[geom.Point, Point2D, Point3D, npt.NDArray[np.float64]],
        normalized: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Projects a point onto the 3D polyline and returns the distance along the polyline to the closest point.

        :param point: The point to project.
        :param normalized: Whether to return normalized distances, defaults to False.
        :return: The distance along the polyline to the closest point.
        """
        if isinstance(point, Point2D) or isinstance(point, PoseSE2) or isinstance(point, Point3D):
            point_ = point.shapely_point
        elif isinstance(point, geom.Point):
            point_ = point
        else:
            point_ = geom.Point(np.array(point, dtype=np.float64))
        return self.linestring.project(point_, normalized=normalized)  # type: ignore
