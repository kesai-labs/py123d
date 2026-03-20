import logging
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from py123d.datatypes import Lidar

logger = logging.getLogger(__name__)


def _continuous_colormap(
    values: npt.NDArray,
    cmap_name: str = "viridis",
    vmin: float = None,
    vmax: float = None,
) -> npt.NDArray[np.uint8]:
    """Map continuous values to RGB colors using a matplotlib colormap.

    :param values: 1D array of continuous values.
    :param cmap_name: Name of the matplotlib colormap to use.
    :param vmin: Minimum value for normalization. Defaults to values.min().
    :param vmax: Maximum value for normalization. Defaults to values.max().
    :return: Nx3 array of RGB uint8 values.
    """
    min_val = vmin if vmin is not None else values.min()
    max_val = vmax if vmax is not None else values.max()
    if max_val - min_val < 1e-8:
        normalized = np.zeros_like(values, dtype=np.float64)
    else:
        normalized = np.clip((values - min_val) / (max_val - min_val), 0.0, 1.0)
    colormap = plt.get_cmap(cmap_name)
    colors = colormap(normalized)
    return (colors[:, :3] * 255).astype(np.uint8)


def _discrete_colormap(values: npt.NDArray, cmap_name: str = "tab20") -> npt.NDArray[np.uint8]:
    """Map discrete class values to RGB colors using a qualitative colormap.

    :param values: 1D array of discrete class labels (e.g. uint8 IDs).
    :param cmap_name: Name of the qualitative matplotlib colormap to use.
    :return: Nx3 array of RGB uint8 values.
    """
    unique_classes, inverse_indices = np.unique(values, return_inverse=True)
    n_classes = len(unique_classes)
    colormap = plt.get_cmap(cmap_name, n_classes)
    class_colors = colormap(np.linspace(0, 1, n_classes))[:, :3]
    colors = class_colors[inverse_indices]
    return (colors * 255).astype(np.uint8)


def get_lidar_pc_color(
    lidar: Lidar,
    feature: Literal[
        "none",
        "height",
        "distance",
        "ids",
        "intensity",
        "channel",
        "timestamps",
        "range",
        "elongation",
    ] = "none",
    dark_mode: bool = False,
) -> npt.NDArray[np.uint8]:
    """Compute per-point RGB colors for a lidar point cloud based on a feature.

    :param lidar: Lidar object containing the point cloud and its metadata.
    :param feature: The feature to color the point cloud by.
    :param dark_mode: If True, use white as the default color; otherwise use black.
    :return: Nx3 array of RGB uint8 values.
    """
    point_cloud_3d = lidar.point_cloud_3d
    n_points = len(point_cloud_3d)

    default_value = 255 if dark_mode else 0
    default_color = np.ones((n_points, 3), dtype=np.uint8) * default_value

    if feature == "none":
        return default_color
    elif feature == "height":
        return _continuous_colormap(-point_cloud_3d[:, 2], cmap_name="viridis", vmin=-6.0, vmax=2.0)
    elif feature == "distance":
        distances = -np.linalg.norm(point_cloud_3d, axis=-1)
        return _continuous_colormap(distances)

    # Features that require point_cloud_features to be present
    discrete_features = {"ids", "channel"}
    continuous_features = {"intensity", "timestamps", "range", "elongation"}
    feature_accessor = {
        "ids": lidar.ids,
        "intensity": lidar.intensity,
        "channel": lidar.channel,
        "timestamps": lidar.timestamps,
        "range": lidar.range,
        "elongation": lidar.elongation,
    }

    values = feature_accessor.get(feature)
    if values is None:
        logger.warning(f"LiDAR point cloud does not contain {feature} feature. Falling back to black.")
        return default_color

    if feature in discrete_features:
        return _discrete_colormap(values)
    elif feature in continuous_features:
        if values.dtype == np.uint8:
            values = values.astype(np.float32)
        elif values.dtype == np.int64:
            values = values.astype(np.float64)
        return _continuous_colormap(values)

    raise ValueError(f"Unknown feature: {feature}")
