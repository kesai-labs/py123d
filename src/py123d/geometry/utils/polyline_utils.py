import numpy as np
import numpy.typing as npt
from shapely.geometry import LineString

from py123d.geometry.geometry_index import Point2DIndex, Point3DIndex, PoseSE2Index
from py123d.geometry.transform.transform_se2 import translate_2d_along_body_frame


def get_linestring_yaws(linestring: LineString) -> npt.NDArray[np.float64]:
    """Compute the heading of each coordinate to its successor coordinate. The last coordinate \
        will have the same heading as the second last coordinate.

    :param linestring: linestring as a shapely LineString.
    :return: An array of headings associated to each starting coordinate.
    """
    coords: npt.NDArray[np.float64] = np.asarray(linestring.coords, dtype=np.float64)[..., Point2DIndex.XY]
    return get_points_2d_yaws(coords)


def get_points_2d_yaws(points_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute the heading of each 2D point to its successor point. The last point \
        will have the same heading as the second last point.

    :param points_array: Array of shape (..., 2) representing 2D points.
    :return: Array of shape (...,) representing the yaw angles of the points.
    """
    assert points_array.ndim == 2
    assert points_array.shape[-1] == len(Point2DIndex)
    vectors = np.diff(points_array, axis=0)
    yaws = np.arctan2(vectors.T[1], vectors.T[0])
    yaws = np.append(yaws, yaws[-1])  # pad end with duplicate heading
    assert len(yaws) == len(points_array), "Calculated heading must have the same length as input coordinates"
    return yaws


def get_path_progress_2d(points_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute the cumulative path progress along a series of 2D points.

    :param points_array: Array of shape (..., 2) representing 2D points.
    :raises ValueError: If the input points_array is not valid.
    :return: Array of shape (...) representing the cumulative path progress.
    """
    if points_array.shape[-1] == len(Point2DIndex):
        x_diff = np.diff(points_array[..., Point2DIndex.X])
        y_diff = np.diff(points_array[..., Point2DIndex.Y])
    elif points_array.shape[-1] == len(PoseSE2Index):
        x_diff = np.diff(points_array[..., PoseSE2Index.X])
        y_diff = np.diff(points_array[..., PoseSE2Index.Y])
    else:
        raise ValueError(
            f"Invalid points_array shape: {points_array.shape}. Expected last dimension to be {len(Point2DIndex)} or "
            f"{len(PoseSE2Index)}."
        )
    points_diff: npt.NDArray[np.float64] = np.concatenate(([x_diff], [y_diff]), axis=0, dtype=np.float64)
    progress_diff = np.append(0.0, np.linalg.norm(points_diff, axis=0))
    return np.cumsum(progress_diff, dtype=np.float64)


def get_path_progress_3d(points_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute the cumulative path progress along a series of 3D points.

    :param points_array: Array of shape (..., 3) representing 3D points.
    :raises ValueError: If the input points_array is not valid.
    :return: Array of shape (...) representing the cumulative path progress.
    """
    if points_array.shape[-1] == len(Point3DIndex):
        x_diff = np.diff(points_array[..., Point3DIndex.X])
        y_diff = np.diff(points_array[..., Point3DIndex.Y])
        z_diff = np.diff(points_array[..., Point3DIndex.Z])
    else:
        raise ValueError(
            f"Invalid points_array shape: {points_array.shape}. Expected last dimension to be {len(Point3DIndex)}."
        )
    points_diff: npt.NDArray[np.float64] = np.concatenate(([x_diff], [y_diff], [z_diff]), axis=0, dtype=np.float64)
    progress_diff = np.append(0.0, np.linalg.norm(points_diff, axis=0))
    return np.cumsum(progress_diff, dtype=np.float64)


def offset_points_perpendicular(points_array: npt.NDArray[np.float64], offset: float) -> npt.NDArray[np.float64]:
    """Offset 2D points or SE2 poses perpendicularly by a given offset.

    :param points_array: Array points of shape (..., 2) representing 2D points \
        or shape (..., 3) representing SE2 poses.
    :param offset: Offset distance to apply perpendicularly.
    :raises ValueError: If the input points_array is not valid.
    :return: Array of shape (..., 2) representing the offset points.
    """
    if points_array.shape[-1] == len(Point2DIndex):
        xy = points_array[..., Point2DIndex.XY]
        yaws = get_points_2d_yaws(points_array[..., Point2DIndex.XY])
    elif points_array.shape[-1] == len(PoseSE2Index):
        xy = points_array[..., PoseSE2Index.XY]
        yaws = points_array[..., PoseSE2Index.YAW]
    else:
        raise ValueError(
            f"Invalid points_array shape: {points_array.shape}. Expected last dimension to be {len(Point2DIndex)} or "
            f"{len(PoseSE2Index)}."
        )

    return translate_2d_along_body_frame(
        points_2d=xy,
        yaws=yaws,
        y_translate=offset,  # type: ignore
        x_translate=0.0,  # type: ignore
    )
