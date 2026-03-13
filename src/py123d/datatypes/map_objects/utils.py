import numpy as np
import numpy.typing as npt
import trimesh

from py123d.geometry import Polyline3D


def get_trimesh_from_boundaries(
    left_boundary: Polyline3D, right_boundary: Polyline3D, resolution: float = 0.25
) -> trimesh.Trimesh:
    """Helper function to create a trimesh from two lane boundaries.

    :param left_boundary: The left boundary polyline.
    :param right_boundary: The right boundary polyline.
    :param resolution: The resolution for the mesh, defaults to 0.25.
    :return: A trimesh representation of the lane.
    """

    def _interpolate_polyline(polyline_3d: Polyline3D, num_samples: int) -> npt.NDArray[np.float64]:
        """Helper function to interpolate a polyline to a fixed number of samples."""
        num_samples = max(num_samples, 2)
        distances = np.linspace(0, polyline_3d.length, num=num_samples, endpoint=True, dtype=np.float64)
        points = polyline_3d.interpolate(distances)
        assert isinstance(points, np.ndarray) and points.shape == (num_samples, 3)
        return points

    average_length = (left_boundary.length + right_boundary.length) / 2
    num_samples = int(average_length // resolution) + 1
    left_boundary_array = _interpolate_polyline(left_boundary, num_samples)
    right_boundary_array = _interpolate_polyline(right_boundary, num_samples)
    return _create_lane_mesh_from_boundary_arrays(left_boundary_array, right_boundary_array)


def _create_lane_mesh_from_boundary_arrays(
    left_boundary_array: npt.NDArray[np.float64], right_boundary_array: npt.NDArray[np.float64]
) -> trimesh.Trimesh:
    """Helper function to create a trimesh from two boundary arrays.

    :param left_boundary_array: The left boundary array.
    :param right_boundary_array: The right boundary array.
    :raises ValueError: If the boundary arrays do not have the same number of points.
    :return: A trimesh representation of the lane.
    """

    # Ensure both polylines have the same number of points
    if left_boundary_array.shape[0] != right_boundary_array.shape[0]:
        raise ValueError("Both polylines must have the same number of points")

    n_points = left_boundary_array.shape[0]
    vertices = np.vstack([left_boundary_array, right_boundary_array])

    faces = []
    for i in range(n_points - 1):
        faces.append([i, i + n_points, i + 1])
        faces.append([i + 1, i + n_points, i + n_points + 1])

    faces = np.array(faces)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh
