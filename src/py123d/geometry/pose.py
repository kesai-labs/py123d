from __future__ import annotations

from typing import Union

import numpy as np
import numpy.typing as npt
import shapely.geometry as geom

from py123d.common.utils.mixin import ArrayMixin, indexed_array_repr
from py123d.geometry.geometry_index import (
    EulerAnglesIndex,
    MatrixSE2Index,
    MatrixSE3Index,
    PoseSE2Index,
    PoseSE3Index,
    QuaternionIndex,
)
from py123d.geometry.point import Point2D, Point3D
from py123d.geometry.rotation import EulerAngles, Quaternion
from py123d.geometry.utils.rotation_utils import (
    get_quaternion_array_from_euler_array,
    get_quaternion_array_from_rotation_matrix,
    get_rotation_matrix_from_quaternion_array,
    invert_quaternion_array,
)
from py123d.geometry.vector import Vector2D, Vector3D


class PoseSE2(ArrayMixin):
    """Class to represents a 2D pose as SE2 (x, y, yaw).

    Examples:
        >>> from py123d.geometry import PoseSE2
        >>> pose = PoseSE2(x=1.0, y=2.0, yaw=0.5)
        >>> print(pose.x, pose.y, pose.yaw)
        1.0 2.0 0.5
        >>> print(pose.rotation_matrix)
        [[ 0.87758256 -0.47942554]
         [ 0.47942554  0.87758256]]

    """

    __slots__ = ("_array",)
    _array: npt.NDArray[np.float64]

    def __init__(self, x: float, y: float, yaw: float):
        """Init :class:`PoseSE2` with x, y, yaw coordinates.

        :param x: The x-coordinate.
        :param y: The y-coordinate.
        :param yaw: The yaw angle in radians.
        """
        array = np.zeros(len(PoseSE2Index), dtype=np.float64)
        array[PoseSE2Index.X] = x
        array[PoseSE2Index.Y] = y
        array[PoseSE2Index.YAW] = yaw
        object.__setattr__(self, "_array", array)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> PoseSE2:
        """Constructs a PoseSE2 from a numpy array.

        :param array: Array of shape (3,) representing the state [x, y, yaw], indexed by \
            :class:`~py123d.geometry.geometry_index.PoseSE2Index`.
        :param copy: Whether to copy the input array. Defaults to True.
        :return: A PoseSE2 instance.
        """
        assert array.ndim == 1
        assert array.shape[0] == len(PoseSE2Index)
        instance = object.__new__(cls)
        instance._array = array.copy() if copy else array
        return instance

    @classmethod
    def from_transformation_matrix(cls, transformation_matrix: npt.NDArray[np.float64]) -> PoseSE2:
        """Constructs a PoseSE2 from a 3x3 transformation matrix.

        :param transformation_matrix: A 3x3 numpy array representing the transformation matrix.
        :return: A PoseSE2 instance.
        """
        assert transformation_matrix.ndim == 2
        assert transformation_matrix.shape == (3, 3)
        x, y = transformation_matrix[MatrixSE2Index.TRANSLATION]
        yaw = np.arctan2(transformation_matrix[1, 0], transformation_matrix[0, 0])
        return PoseSE2(x=x, y=y, yaw=yaw)

    @classmethod
    def from_R_t(
        cls,
        rotation: Union[npt.NDArray[np.float64], float],
        translation: Union[npt.NDArray[np.float64], Point2D, Vector2D],
    ) -> PoseSE2:
        """Constructs a :class:`~py123d.geometry.PoseSE2` from a rotation and a translation.

        :param rotation: Any implemented representation of a SO2 rotation: \
            - (2, 2) rotation matrix as numpy array, \
            - (1,) numpy array representing the yaw angle in radians.
        :param translation: The translation component:
            - (2,) numpy array indexed by :class:`~py123d.geometry.geometry_index.Vector2DIndex`, \
            - :class:`~py123d.geometry.Vector2D` instance, \
            - :class:`~py123d.geometry.Point2D` is also accepted for convenience.
        :return: A :class:`~py123d.geometry.PoseSE2` instance.
        """

        array = np.zeros(len(PoseSE2Index), dtype=np.float64)

        # 1. Translation
        if isinstance(translation, np.ndarray):
            assert translation.shape == (2,), (
                "Expected translation to be a numpy array of shape (2,), got shape {}".format(translation.shape)
            )
            array[PoseSE2Index.XY] = translation
        elif isinstance(translation, (Point2D, Vector2D)):
            array[PoseSE2Index.XY] = translation.array
        else:
            raise ValueError(
                "Unsupported translation type for PoseSE2 construction, got type: {}".format(type(translation))
            )

        # 2. Rotation
        if isinstance(rotation, (int, float)):
            array[PoseSE2Index.YAW] = float(rotation)
        elif isinstance(rotation, np.ndarray):
            if rotation.shape == ():
                array[PoseSE2Index.YAW] = float(rotation)
            elif rotation.shape == (1,):
                array[PoseSE2Index.YAW] = float(rotation[0])
            elif rotation.shape == (2, 2):
                array[PoseSE2Index.YAW] = np.arctan2(rotation[1, 0], rotation[0, 0])
            else:
                raise ValueError(
                    "Expected rotation to be a numpy array of shape (2, 2) or (1,), got shape {}".format(rotation.shape)
                )
        else:
            raise ValueError("Unsupported rotation type for PoseSE2 construction, got type: {}".format(type(rotation)))

        return PoseSE2.from_array(array, copy=False)

    @classmethod
    def identity(cls) -> PoseSE2:
        """Constructs an identity PoseSE2.

        :return: An identity PoseSE2 instance.
        """
        return PoseSE2(x=0.0, y=0.0, yaw=0.0)

    @property
    def x(self) -> float:
        """The x-coordinate of the pose."""
        return self._array[PoseSE2Index.X]

    @property
    def y(self) -> float:
        """The y-coordinate of the pose."""
        return self._array[PoseSE2Index.Y]

    @property
    def yaw(self) -> float:
        """The yaw angle of the pose."""
        return self._array[PoseSE2Index.YAW]

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """Pose as numpy array of shape (3,), indexed by :class:`~py123d.geometry.geometry_index.PoseSE2Index`."""
        return self._array

    @property
    def pose_se2(self) -> PoseSE2:
        """Returns self to match interface of other pose classes."""
        return self

    @property
    def point_2d(self) -> Point2D:
        """The :class:`~py123d.geometry.Point2D` of the pose, i.e. the translation part."""
        return Point2D.from_array(self.array[PoseSE2Index.XY])

    @property
    def vector_2d(self) -> Vector2D:
        """The :class:`~py123d.geometry.Vector2D` translation component of the SE2 pose."""
        return Vector2D(self.x, self.y)

    @property
    def rotation_matrix(self) -> npt.NDArray[np.float64]:
        """The 2x2 rotation matrix representation of the pose."""
        cos_yaw = np.cos(self.yaw)
        sin_yaw = np.sin(self.yaw)
        return np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64)

    @property
    def transformation_matrix(self) -> npt.NDArray[np.float64]:
        """The 3x3 transformation matrix representation of the pose."""
        matrix = np.eye(3, dtype=np.float64)
        matrix[MatrixSE2Index.ROTATION] = self.rotation_matrix
        matrix[MatrixSE2Index.TRANSLATION] = self.array[PoseSE2Index.XY]
        return matrix

    @property
    def shapely_point(self) -> geom.Point:
        """The Shapely point representation of the pose."""
        return geom.Point(self.x, self.y)

    def __repr__(self) -> str:
        """String representation of :class:`PoseSE2`."""
        return indexed_array_repr(self, PoseSE2Index)


class PoseSE3(ArrayMixin):
    """Class representing a pose in SE3 space.

    Examples:
        >>> from py123d.geometry import PoseSE3
        >>> pose = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        >>> pose.point_3d
        Point3D(array=[1. 2. 3.])
        >>> pose.transformation_matrix
        array([[1., 0., 0., 1.],
               [0., 1., 0., 2.],
               [0., 0., 1., 3.],
               [0., 0., 0., 1.]])
        >>> PoseSE3.from_transformation_matrix(pose.transformation_matrix) == pose
        True
        >>> print(pose.yaw, pose.pitch, pose.roll)
        0.0 0.0 0.0
    """

    __slots__ = ("_array",)
    _array: npt.NDArray[np.float64]

    def __init__(self, x: float, y: float, z: float, qw: float, qx: float, qy: float, qz: float):
        """Initialize :class:`PoseSE3` with x, y, z, qw, qx, qy, qz coordinates.

        :param x: The x-coordinate.
        :param y: The y-coordinate.
        :param z: The z-coordinate.
        :param qw: The w-coordinate of the quaternion, representing the scalar part.
        :param qx: The x-coordinate of the quaternion, representing the first component of the vector part.
        :param qy: The y-coordinate of the quaternion, representing the second component of the vector part.
        :param qz: The z-coordinate of the quaternion, representing the third component of the vector part.
        """
        array = np.zeros(len(PoseSE3Index), dtype=np.float64)
        array[PoseSE3Index.X] = x
        array[PoseSE3Index.Y] = y
        array[PoseSE3Index.Z] = z
        array[PoseSE3Index.QW] = qw
        array[PoseSE3Index.QX] = qx
        array[PoseSE3Index.QY] = qy
        array[PoseSE3Index.QZ] = qz
        object.__setattr__(self, "_array", array)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> PoseSE3:
        """Constructs a :class:`PoseSE3` from a numpy array of shape (7,), \
            indexed by :class:`~py123d.geometry.geometry_index.PoseSE3Index`.

        :param array: Array of shape (7,) representing the state [x, y, z, qw, qx, qy, qz].
        :param copy: Whether to copy the input array. Defaults to True.
        :return: A :class:`PoseSE3` instance.
        """
        assert array.ndim == 1
        assert array.shape[0] == len(PoseSE3Index)
        instance = object.__new__(cls)
        instance._array = array.copy() if copy else array
        return instance

    @classmethod
    def from_transformation_matrix(cls, transformation_matrix: npt.NDArray[np.float64]) -> PoseSE3:
        """Constructs a :class:`PoseSE3` from a 4x4 transformation matrix.

        :param transformation_matrix: A 4x4 numpy array representing the transformation matrix.
        :return: A :class:`PoseSE3` instance.
        """
        assert transformation_matrix.ndim == 2
        assert transformation_matrix.shape == (4, 4)
        array = np.zeros(len(PoseSE3Index), dtype=np.float64)
        array[PoseSE3Index.XYZ] = transformation_matrix[MatrixSE3Index.TRANSLATION]
        array[PoseSE3Index.QUATERNION] = Quaternion.from_rotation_matrix(transformation_matrix[MatrixSE3Index.ROTATION])
        return PoseSE3.from_array(array, copy=False)

    @classmethod
    def from_R_t(
        cls,
        rotation: Union[npt.NDArray[np.float64], Quaternion, EulerAngles],
        translation: Union[npt.NDArray[np.float64], Point3D, Vector3D],
    ) -> PoseSE3:
        """Constructs a :class:`PoseSE3` from arbitrary rotation and translation representations.

        :param rotation: Any implemented representation of a SO3 rotation: \
            - (3, 3) rotation matrix as numpy array,\
            - (4,) quaternion array indexed by :class:`~py123d.geometry.geometry_index.QuaternionIndex`, \
            - (3,) euler angles indexed by :class:`~py123d.geometry.geometry_index.EulerAnglesIndex`, \
            - :class:`~py123d.geometry.Quaternion` instance,
            - :class:`~py123d.geometry.EulerAngles` instance.
        :param translation: The translation component:, \
            - (3,) numpy array indexed by :class:`~py123d.geometry.geometry_index.Vector3DIndex`, \
            - :class:`~py123d.geometry.Vector3D` instance, \
            - :class:`~py123d.geometry.Point3D` is also accepted for convenience. \
        :return: A :class:`PoseSE3` instance.
        """

        array = np.zeros(len(PoseSE3Index), dtype=np.float64)

        # 1. Translation
        if isinstance(translation, np.ndarray):
            assert translation.shape == (3,)
            array[PoseSE3Index.XYZ] = translation
        elif isinstance(translation, (Point3D, Vector3D)):
            array[PoseSE3Index.XYZ] = translation.array
        else:
            raise ValueError(
                "Unsupported translation type for PoseSE3 construction, got type: {}".format(type(translation))
            )

        # 2. Rotation
        if isinstance(rotation, np.ndarray):
            if rotation.shape == (3, 3):
                array[PoseSE3Index.QUATERNION] = get_quaternion_array_from_rotation_matrix(rotation)
            elif rotation.shape == (len(QuaternionIndex),):
                array[PoseSE3Index.QUATERNION] = rotation
            elif rotation.shape == (len(EulerAnglesIndex),):
                array[PoseSE3Index.QUATERNION] = get_quaternion_array_from_euler_array(rotation)
            else:
                raise ValueError(
                    "Expected rotation to be a numpy array of shape (3, 3), (4,) or (3,), got shape {}".format(
                        rotation.shape
                    )
                )
        elif isinstance(rotation, Quaternion):
            array[PoseSE3Index.QUATERNION] = rotation.array
        elif isinstance(rotation, EulerAngles):
            array[PoseSE3Index.QUATERNION] = rotation.quaternion.array
        else:
            raise ValueError("Unsupported rotation type for PoseSE3 construction, got type: {}".format(type(rotation)))

        return PoseSE3.from_array(array, copy=False)

    @classmethod
    def identity(cls) -> PoseSE3:
        """Constructs an identity :class:`PoseSE3`.

        :return: An identity :class:`PoseSE3` instance.
        """
        array = np.zeros(len(PoseSE3Index), dtype=np.float64)
        array[PoseSE3Index.QW] = 1.0
        return PoseSE3.from_array(array, copy=False)

    @property
    def x(self) -> float:
        """The x-coordinate of the pose."""
        return self._array[PoseSE3Index.X]

    @property
    def y(self) -> float:
        """The y-coordinate of the pose."""
        return self._array[PoseSE3Index.Y]

    @property
    def z(self) -> float:
        """The z-coordinate of the pose."""
        return self._array[PoseSE3Index.Z]

    @property
    def qw(self) -> float:
        """The w-coordinate of the quaternion, representing the scalar part."""
        return self._array[PoseSE3Index.QW]

    @property
    def qx(self) -> float:
        """The x-coordinate of the quaternion, representing the first component of the vector part."""
        return self._array[PoseSE3Index.QX]

    @property
    def qy(self) -> float:
        """The y-coordinate of the quaternion, representing the second component of the vector part."""
        return self._array[PoseSE3Index.QY]

    @property
    def qz(self) -> float:
        """The z-coordinate of the quaternion, representing the third component of the vector part."""
        return self._array[PoseSE3Index.QZ]

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """The numpy array representation of the pose with shape (7,), \
            indexed by :class:`~py123d.geometry.geometry_index.PoseSE3Index`"""
        return self._array

    @property
    def pose_se3(self) -> PoseSE3:
        """The :class:`PoseSE3` itself."""
        return self

    @property
    def pose_se2(self) -> PoseSE2:
        """The :class:`PoseSE2` representation of the SE3 pose."""
        return PoseSE2(self.x, self.y, self.yaw)

    @property
    def point_3d(self) -> Point3D:
        """The :class:`Point3D` representation of the SE3 pose, i.e. the translation part."""
        return Point3D(self.x, self.y, self.z)

    @property
    def point_2d(self) -> Point2D:
        """The :class:`Point2D` representation of the SE3 pose, i.e. the translation part."""
        return Point2D(self.x, self.y)

    @property
    def vector_3d(self) -> Vector3D:
        """The :class:`~py123d.geometry.Vector3D` translation component of the SE3 pose."""
        return Vector3D(self.x, self.y, self.z)

    @property
    def vector_2d(self) -> Vector2D:
        """The :class:`~py123d.geometry.Vector2D` 2D translation component (x, y) of the SE3 pose."""
        return Vector2D(self.x, self.y)

    @property
    def shapely_point(self) -> geom.Point:
        """The Shapely point representation, of the translation part of the SE3 pose."""
        return self.point_3d.shapely_point

    @property
    def quaternion(self) -> Quaternion:
        """The :class:`~py123d.geometry.Quaternion` representation of the state's orientation."""
        return Quaternion.from_array(self.array[PoseSE3Index.QUATERNION])

    @property
    def euler_angles(self) -> EulerAngles:
        """The :class:`~py123d.geometry.EulerAngles` representation of the state's orientation."""
        return self.quaternion.euler_angles

    @property
    def roll(self) -> float:
        """The roll (x-axis rotation) angle in radians."""
        return self.euler_angles.roll

    @property
    def pitch(self) -> float:
        """The pitch (y-axis rotation) angle in radians."""
        return self.euler_angles.pitch

    @property
    def yaw(self) -> float:
        """The yaw (z-axis rotation) angle in radians."""
        return self.euler_angles.yaw

    @property
    def rotation_matrix(self) -> npt.NDArray[np.float64]:
        """Returns the 3x3 rotation matrix representation of the state's orientation."""
        return self.quaternion.rotation_matrix

    @property
    def transformation_matrix(self) -> npt.NDArray[np.float64]:
        """Returns the 4x4 transformation matrix representation of the state."""
        transformation_matrix = np.eye(4, dtype=np.float64)
        transformation_matrix[MatrixSE3Index.ROTATION] = self.rotation_matrix
        transformation_matrix[MatrixSE3Index.TRANSLATION] = self.array[PoseSE3Index.XYZ]
        return transformation_matrix

    @property
    def inverse(self) -> PoseSE3:
        """Returns the inverse of the SE3 pose."""
        inverse_array = np.zeros_like(self.array)
        inv_quat = invert_quaternion_array(self.array[PoseSE3Index.QUATERNION])
        inverse_array[PoseSE3Index.QUATERNION] = inv_quat
        r_inv = get_rotation_matrix_from_quaternion_array(inv_quat)
        inverse_array[PoseSE3Index.XYZ] = -r_inv @ self.array[PoseSE3Index.XYZ]
        return PoseSE3.from_array(inverse_array, copy=False)

    def __repr__(self) -> str:
        """String representation of :class:`PoseSE3`."""
        return indexed_array_repr(self, PoseSE3Index)
