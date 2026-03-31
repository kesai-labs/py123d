from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as geom

from py123d.api import SceneAPI
from py123d.datatypes.sensors.base_camera import BaseCameraMetadata, CameraID
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.geometry.pose import PoseSE3
from py123d.visualization.color.color import BLACK, TAB_10, Color
from py123d.visualization.color.config import PlotConfig
from py123d.visualization.matplotlib.utils import add_shapely_polygons_to_ax

DEFAULT_FRUSTUM_LENGTH: float = 5.0

EGO_OUTLINE_CONFIG: PlotConfig = PlotConfig(
    fill_color=BLACK,
    fill_color_alpha=0.0,
    line_color=BLACK,
    line_color_alpha=1.0,
    line_width=1.5,
    line_style="-",
    zorder=4,
)

_CAMERA_COLORS: List[Color] = [
    TAB_10[0],  # blue
    TAB_10[1],  # orange
    TAB_10[2],  # green
    TAB_10[3],  # red
    TAB_10[4],  # violet
    TAB_10[5],  # brown
    TAB_10[6],  # pink
    TAB_10[7],  # grey
    TAB_10[8],  # yellow
    TAB_10[9],  # cyan
]


def _get_camera_optical_axis_yaw(camera_to_imu_se3_rotation: np.ndarray) -> float:
    """Compute the yaw of the camera optical axis (Z-forward) projected onto the BEV plane.

    :param camera_to_imu_se3_rotation: The 3x3 rotation matrix of the camera-to-IMU extrinsic.
    :return: The yaw angle in radians of the camera's forward direction in BEV.
    """
    forward_imu = camera_to_imu_se3_rotation[:, 2]
    yaw = np.arctan2(forward_imu[1], forward_imu[0])
    return float(yaw)


def _make_frustum_polygon(
    camera_x: float,
    camera_y: float,
    yaw: float,
    fov_x: float,
    frustum_length: float,
) -> geom.Polygon:
    """Create a 2D triangle polygon representing a camera frustum in BEV.

    :param camera_x: Camera X position in the local (IMU) frame.
    :param camera_y: Camera Y position in the local (IMU) frame.
    :param yaw: Yaw of the camera optical axis in the BEV plane (radians).
    :param fov_x: Horizontal field of view in radians.
    :param frustum_length: Length of the frustum rays in meters.
    :return: A shapely Polygon (triangle) for the frustum.
    """
    half_fov = fov_x / 2.0
    left_angle = yaw + half_fov
    right_angle = yaw - half_fov

    polygon = geom.Polygon(
        [
            (camera_x, camera_y),
            (camera_x + frustum_length * np.cos(left_angle), camera_y + frustum_length * np.sin(left_angle)),
            (camera_x + frustum_length * np.cos(right_angle), camera_y + frustum_length * np.sin(right_angle)),
        ]
    )
    return polygon


def add_camera_frustums_to_ax(
    ax: plt.Axes,
    camera_metadatas: Dict[CameraID, BaseCameraMetadata],
    frustum_length: float = DEFAULT_FRUSTUM_LENGTH,
    color_per_camera: bool = True,
    add_labels: bool = True,
) -> plt.Axes:
    """Add camera FOV frustums to a BEV axes, in IMU-local coordinates.

    Each camera's position and orientation are taken directly from the static
    ``camera_to_imu_se3`` extrinsic stored in the camera metadata.

    :param ax: The matplotlib axes to draw on.
    :param camera_metadatas: Dictionary of camera ID to metadata.
    :param frustum_length: Length of the frustum rays in meters.
    :param color_per_camera: If True, each camera gets a distinct color.
    :param add_labels: If True, add camera name labels near each frustum.
    :return: The axes with frustums drawn.
    """
    sorted_cameras = sorted(camera_metadatas.items(), key=lambda item: int(item[0]))

    for idx, (_, cam_meta) in enumerate(sorted_cameras):
        extrinsic = cam_meta.camera_to_imu_se3
        cam_x = extrinsic.x
        cam_y = extrinsic.y
        optical_yaw = _get_camera_optical_axis_yaw(extrinsic.rotation_matrix)

        fov_x = getattr(cam_meta, "fov_x", None)
        if fov_x is None:
            fov_x = np.deg2rad(60.0)

        frustum_poly = _make_frustum_polygon(cam_x, cam_y, optical_yaw, fov_x, frustum_length)

        color = _CAMERA_COLORS[idx % len(_CAMERA_COLORS)] if color_per_camera else TAB_10[0]

        config = PlotConfig(
            fill_color=color,
            fill_color_alpha=0.15,
            line_color=color,
            line_color_alpha=0.8,
            line_width=1.5,
            line_style="-",
            zorder=5,
        )
        add_shapely_polygons_to_ax(ax, [frustum_poly], config, label=cam_meta.camera_name)

        if add_labels:
            label_x = cam_x + (frustum_length * 0.6) * np.cos(optical_yaw)
            label_y = cam_y + (frustum_length * 0.6) * np.sin(optical_yaw)
            ax.annotate(
                cam_meta.camera_name,
                xy=(label_x, label_y),
                fontsize=10,
                ha="center",
                va="center",
                color=color.hex,
                zorder=6,
            )

    return ax


def _add_ego_outline_to_ax(ax: plt.Axes, ego_bbox_polygon: geom.Polygon) -> None:
    """Draw the ego vehicle as a closed outline (no fill) so it is always visible.

    :param ax: The matplotlib axes.
    :param ego_bbox_polygon: Shapely polygon of the ego bounding box.
    """
    coords = np.array(ego_bbox_polygon.exterior.coords)
    ax.plot(
        coords[:, 0],
        coords[:, 1],
        color=BLACK.hex,
        linewidth=1.5,
        linestyle="-",
        zorder=4,
        label="ego",
    )


def _add_origin_axes_to_ax(ax: plt.Axes, arrow_length: float = 1.5) -> None:
    """Draw X (red) and Y (green) arrows at the origin to indicate the ego frame.

    :param ax: The matplotlib axes.
    :param arrow_length: Length of each arrow in meters.
    """
    arrow_kwargs = dict(
        head_width=0.15,
        head_length=0.15,
        linewidth=1.5,
        zorder=10,
    )
    ax.arrow(0, 0, arrow_length, 0, fc="#d62728", ec="#d62728", **arrow_kwargs)
    ax.arrow(0, 0, 0, arrow_length, fc="#2ca02c", ec="#2ca02c", **arrow_kwargs)
    ax.annotate(
        "X", xy=(arrow_length + 0.25, 0), fontsize=9, fontweight="bold", color="#d62728", va="center", zorder=10
    )
    ax.annotate(
        "Y", xy=(0, arrow_length + 0.25), fontsize=9, fontweight="bold", color="#2ca02c", ha="center", zorder=10
    )


def add_rig_on_ax(
    ax: plt.Axes,
    scene: SceneAPI,
    radius: float = 15.0,
    frustum_length: float = DEFAULT_FRUSTUM_LENGTH,
    color_per_camera: bool = True,
    add_labels: bool = True,
) -> plt.Axes:
    """Render the static sensor rig in ego-local (IMU) coordinates.

    Draws the ego vehicle outline, an X/Y frame indicator at the origin, and all
    camera frustums (with FOV) using only static metadata (vehicle dimensions and
    camera extrinsics). No iteration or dynamic ego pose is needed.

    :param ax: The matplotlib axes to draw on.
    :param scene: The SceneAPI providing ego and camera metadata.
    :param radius: The half-extent of the view in meters, centered on the IMU origin.
    :param frustum_length: Length of the camera frustum rays in meters.
    :param color_per_camera: If True, each camera gets a distinct color.
    :param add_labels: If True, annotate camera names near each frustum.
    :return: The axes with the rig visualization.
    """
    ego_metadata = scene.get_ego_state_se3_metadata()
    assert ego_metadata is not None, "Ego metadata is required to plot the rig."
    camera_metadatas = scene.get_camera_metadatas()

    # Ego bounding box in IMU-local coords
    ego_state_identity = EgoStateSE3.from_imu(
        imu_se3=PoseSE3.identity(),
        metadata=ego_metadata,
        timestamp=None,
    )
    _add_ego_outline_to_ax(ax, ego_state_identity.bounding_box_se2.shapely_polygon)

    # Origin frame indicator
    _add_origin_axes_to_ax(ax)

    # Camera frustums
    add_camera_frustums_to_ax(
        ax,
        camera_metadatas,
        frustum_length=frustum_length,
        color_per_camera=color_per_camera,
        add_labels=add_labels,
    )

    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Sensor Rig (BEV)")

    return ax
