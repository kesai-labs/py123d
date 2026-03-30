from typing import List, Literal, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from py123d.datatypes.detections import BoxDetectionsSE3
from py123d.datatypes.detections.box_detection_label import DefaultBoxDetectionLabel
from py123d.datatypes.sensors import Camera, Lidar
from py123d.datatypes.vehicle_state import EgoStateSE3
from py123d.geometry import BoundingBoxSE3Index
from py123d.geometry.transform import abs_to_rel_points_3d_array, rel_to_abs_points_3d_array
from py123d.geometry.utils.bounding_box_utils import bbse3_array_to_corners_array
from py123d.visualization.color.default import BOX_DETECTION_CONFIG
from py123d.visualization.matplotlib.helper import undistort_camera
from py123d.visualization.matplotlib.lidar import get_lidar_pc_color


def add_camera_ax(ax: plt.Axes, camera: Camera, undistort: bool = False) -> plt.Axes:
    """Add camera image to matplotlib axis

    :param ax: matplotlib axis
    :param camera: camera object
    :param undistort: whether to undistort the camera image before display, defaults to False
    :return: matplotlib axis with image
    """
    if undistort:
        camera = undistort_camera(camera)

    ax.imshow(camera.image)
    return ax


def add_lidar_to_camera_ax(
    ax: plt.Axes,
    camera: Camera,
    lidar: Lidar,
    ego_state_se3: EgoStateSE3,
    undistort: bool = True,
    color_feature: Literal[
        "none",
        "height",
        "distance",
        "ids",
        "intensity",
        "channel",
        "timestamps",
        "range",
        "elongation",
    ] = "distance",
    alpha: float = 1.0,
    point_size: float = 3.0,
) -> plt.Axes:
    """Add lidar point cloud to camera image on matplotlib axis

    :param ax: matplotlib axis
    :param camera: camera object
    :param lidar: lidar object
    :param ego_state_se3: ego state object
    :param undistort: whether to undistort the camera image, defaults to True
    :param color_feature: lidar color feature to use, defaults to "distance"
    :param alpha: transparency of the lidar points, 0.0 (fully transparent) to 1.0 (fully opaque), defaults to 1.0
    :param point_size: size of the lidar points in pixels, defaults to 3.0
    :return: matplotlib axis with lidar points overlaid on camera image
    """

    if undistort:
        camera = undistort_camera(camera)

    lidar_pc_colors = np.array(get_lidar_pc_color(lidar, color_feature=color_feature, dark_mode=False))
    pc_in_cam, pc_in_fov_mask = _transform_pcs_to_images(lidar.xyz.copy(), camera, ego_state_se3)

    ax.imshow(camera.image)
    ax.scatter(
        pc_in_cam[pc_in_fov_mask, 0],
        pc_in_cam[pc_in_fov_mask, 1],
        c=lidar_pc_colors[pc_in_fov_mask] / 255.0,
        s=point_size,
        alpha=alpha,
        edgecolors="none",
    )
    ax.set_xlim(0, camera.metadata.width)
    ax.set_ylim(camera.metadata.height, 0)
    return ax


def add_box_detections_to_camera_ax(
    ax: plt.Axes, camera: Camera, box_detections: BoxDetectionsSE3, undistort: bool = False
) -> plt.Axes:
    """Add box detections to camera image on matplotlib axis

    :param ax: matplotlib axis
    :param camera: camera object
    :param box_detections: box detection wrapper object
    :param undistort: whether to undistort the camera image, defaults to False
    :return: matplotlib axis with box detections overlaid on camera image
    """
    if undistort:
        camera = undistort_camera(camera)

    box_detection_array = np.zeros((len(box_detections.box_detections), len(BoundingBoxSE3Index)), dtype=np.float64)
    default_labels = np.array(
        [detection.attributes.default_label for detection in box_detections.box_detections], dtype=object
    )
    for idx, box_detection in enumerate(box_detections.box_detections):
        box_detection_array[idx] = box_detection.bounding_box_se3.array

    # Compute corners in global frame using full quaternion rotation
    corners_global = bbse3_array_to_corners_array(box_detection_array)  # (N, 8, 3)

    # Transform corners to camera frame and project to image
    corners_cam = abs_to_rel_points_3d_array(camera.camera_to_global_se3, corners_global.reshape(-1, 3))
    pixel_coords, in_fov_mask, _depth = camera.metadata.project_to_image(corners_cam)
    box_corners = pixel_coords.reshape(-1, 8, 2)
    corners_pc_in_fov = in_fov_mask.reshape(-1, 8)
    valid_corners = corners_pc_in_fov.any(-1)

    box_corners, default_labels = box_corners[valid_corners], default_labels[valid_corners]
    image = _plot_rect_3d_on_img(camera.image.copy(), box_corners, default_labels)

    ax.imshow(image)
    return ax


def _plot_rect_3d_on_img(
    image: npt.NDArray[np.uint8],
    box_corners: npt.NDArray[np.float32],
    labels: List[DefaultBoxDetectionLabel],
    thickness: int = 3,
) -> npt.NDArray[np.uint8]:
    """Plot 3D bounding boxes on image

    :param image: The image to plot on
    :param box_corners: The corners of the boxes to plot
    :param labels: The labels of the boxes to plot
    :param thickness: The thickness of the lines, defaults to 3
    :return: The image with 3D bounding boxes plotted
    """

    line_indices = (
        (0, 1),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 5),
        (3, 2),
        (3, 7),
        (4, 5),
        (4, 7),
        (2, 6),
        (5, 6),
        (6, 7),
    )
    for i in range(len(box_corners)):
        color = BOX_DETECTION_CONFIG[labels[i]].fill_color.rgb
        corners = box_corners[i].astype(np.int64)
        for start, end in line_indices:
            cv2.line(
                image,
                (corners[start, 0], corners[start, 1]),
                (corners[end, 0], corners[end, 1]),
                color,
                thickness,
                cv2.LINE_AA,
            )
    return image.astype(np.uint8)


def _transform_pcs_to_images(
    lidar_xyz: npt.NDArray,
    camera: Camera,
    ego_state_se3: EgoStateSE3,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Transforms lidar point cloud to image pixel coordinates.

    :param lidar_xyz: lidar point cloud in ego-relative xyz coordinates.
    :param camera: camera with global pose and projection model.
    :param ego_state_se3: ego state for ego-to-global transformation.
    :return: points in pixel coordinates, mask of values in frame.
    """
    global_pts = rel_to_abs_points_3d_array(ego_state_se3.imu_se3, lidar_xyz)
    pixel_coords, in_fov_mask, _depth = camera.project_points_global(global_pts)
    return pixel_coords, in_fov_mask
