from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pyquaternion import Quaternion

from py123d.datatypes.detections import BoxDetectionsSE3
from py123d.datatypes.detections.box_detection_label import DefaultBoxDetectionLabel
from py123d.datatypes.sensors import Lidar, PinholeCamera, PinholeIntrinsics
from py123d.datatypes.vehicle_state import EgoStateSE3
from py123d.geometry import BoundingBoxSE3Index, Corners3DIndex
from py123d.geometry.transform import abs_to_rel_se3_array
from py123d.visualization.color.default import BOX_DETECTION_CONFIG
from py123d.visualization.matplotlib.helper import undistort_image_from_camera
from py123d.visualization.matplotlib.lidar import get_lidar_pc_color


def add_pinhole_camera_ax(ax: plt.Axes, pinhole_camera: PinholeCamera) -> plt.Axes:
    """Add pinhole camera image to matplotlib axis

    :param ax: matplotlib axis
    :param pinhole_camera: pinhole camera object
    :return: matplotlib axis with image
    """
    ax.imshow(pinhole_camera.image)
    return ax


def add_lidar_to_camera_ax(ax: plt.Axes, camera: PinholeCamera, lidar: Lidar, undistort: bool = True) -> plt.Axes:
    """Add lidar point cloud to camera image on matplotlib axis

    :param ax: matplotlib axis
    :param camera: pinhole camera object
    :param lidar: lidar object
    :return: matplotlib axis with lidar points overlaid on camera image
    """

    image = camera.image.copy()

    if undistort:
        image = undistort_image_from_camera(image, camera.metadata, mode="keep_focal_length")

    # lidar_pc = filter_lidar_pc(lidar_pc)
    lidar_pc_colors = np.array(get_lidar_pc_color(lidar, feature="distance"))
    pc_in_cam, pc_in_fov_mask = _transform_pcs_to_images(lidar.xyz.copy(), camera)

    for (x, y), color in zip(pc_in_cam[pc_in_fov_mask], lidar_pc_colors[pc_in_fov_mask]):
        color = (int(color[0]), int(color[1]), int(color[2]))
        cv2.circle(image, (int(x), int(y)), 5, color, -1)  # type: ignore

    ax.imshow(image)
    return ax


def add_box_detections_to_camera_ax(
    ax: plt.Axes,
    camera: PinholeCamera,
    box_detections: BoxDetectionsSE3,
    ego_state_se3: EgoStateSE3,
) -> plt.Axes:
    """Add box detections to camera image on matplotlib axis

    :param ax: matplotlib axis
    :param camera: pinhole camera object
    :param box_detections: box detection wrapper object
    :param ego_state_se3: ego state object
    :return: matplotlib axis with box detections overlaid on camera image
    """

    box_detection_array = np.zeros((len(box_detections.box_detections), len(BoundingBoxSE3Index)), dtype=np.float64)
    default_labels = np.array(
        [detection.metadata.default_label for detection in box_detections.box_detections], dtype=object
    )
    for idx, box_detection in enumerate(box_detections.box_detections):
        box_detection_array[idx] = box_detection.bounding_box_se3.array

    # FIXME
    box_detection_array[..., BoundingBoxSE3Index.SE3] = abs_to_rel_se3_array(
        ego_state_se3.rear_axle_se3, box_detection_array[..., BoundingBoxSE3Index.SE3]
    )
    # box_detection_array[..., BoundingBoxSE3Index.XYZ] -= ego_state_se3.rear_axle_se3.point_3d.array
    detection_positions, detection_extents, detection_yaws = _transform_annotations_to_camera(
        box_detection_array, camera.extrinsic.transformation_matrix
    )

    corners_norm = np.stack(np.unravel_index(np.arange(len(Corners3DIndex)), [2] * 3), axis=1)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
    corners = detection_extents.reshape([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

    corners = _rotation_3d_in_axis(corners, detection_yaws, axis=1)
    corners += detection_positions.reshape(-1, 1, 3)

    # Then draw project corners to image.
    box_corners, corners_pc_in_fov = _transform_points_to_image(corners.reshape(-1, 3), camera.metadata.intrinsics)
    box_corners = box_corners.reshape(-1, 8, 2)
    corners_pc_in_fov = corners_pc_in_fov.reshape(-1, 8)
    valid_corners = corners_pc_in_fov.any(-1)

    box_corners, default_labels = box_corners[valid_corners], default_labels[valid_corners]
    image = _plot_rect_3d_on_img(camera.image.copy(), box_corners, default_labels)

    ax.imshow(image)
    return ax


def _transform_annotations_to_camera(boxes: npt.NDArray, extrinsic: npt.NDArray) -> npt.NDArray:
    """Transforms the box annotations from sensor frame to camera frame.

    :param boxes: array of bounding box parameters.
    :param extrinsic: The (4x4) transformation matrix from ego to camera frame.
    :return: transformed bounding box parameters in camera frame.
    """
    sensor2lidar_rotation = extrinsic[:3, :3]
    sensor2lidar_translation = extrinsic[:3, 3]

    locs, quaternions = (
        boxes[:, BoundingBoxSE3Index.XYZ],
        boxes[:, BoundingBoxSE3Index.QUATERNION],
    )
    dims_cam = boxes[
        :, [BoundingBoxSE3Index.LENGTH, BoundingBoxSE3Index.HEIGHT, BoundingBoxSE3Index.WIDTH]
    ]  # l, w, h -> l, h, w

    rots_cam = np.zeros_like(quaternions[..., 0])
    for idx, quaternion in enumerate(quaternions):
        rot = Quaternion(array=quaternion)
        rot = Quaternion(matrix=sensor2lidar_rotation).inverse * rot
        rots_cam[idx] = -rot.yaw_pitch_roll[0]

    lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
    lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    locs_cam = np.concatenate([locs, np.ones_like(locs)[:, :1]], -1)  # -1, 4
    locs_cam = lidar2cam_rt.T @ locs_cam.T
    locs_cam = locs_cam.T
    locs_cam = locs_cam[:, :-1]
    return locs_cam, dims_cam, rots_cam


def _rotation_3d_in_axis(points: npt.NDArray[np.float32], angles: npt.NDArray[np.float32], axis: int = 0):
    """Rotate points in 3D along specific axis."""
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack(
            [
                np.stack([rot_cos, zeros, -rot_sin]),
                np.stack([zeros, ones, zeros]),
                np.stack([rot_sin, zeros, rot_cos]),
            ]
        )
    elif axis in [2, -1]:
        rot_mat_T = np.stack(
            [
                np.stack([rot_cos, -rot_sin, zeros]),
                np.stack([rot_sin, rot_cos, zeros]),
                np.stack([zeros, zeros, ones]),
            ]
        )
    elif axis == 0:
        rot_mat_T = np.stack(
            [
                np.stack([zeros, rot_cos, -rot_sin]),
                np.stack([zeros, rot_sin, rot_cos]),
                np.stack([ones, zeros, zeros]),
            ]
        )
    else:
        raise ValueError(f"axis should in range [0, 1, 2], got {axis}")
    return np.einsum("aij,jka->aik", points, rot_mat_T)


def _plot_rect_3d_on_img(
    image: npt.NDArray[np.uint8],
    box_corners: npt.NDArray[np.float32],
    labels: List[DefaultBoxDetectionLabel],
    thickness: int = 3,
) -> npt.NDArray[np.uint8]:
    """Plot 3D bounding boxes on image

    TODO: refactor
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


def _transform_points_to_image(
    points: npt.NDArray[np.float32],
    intrinsics: PinholeIntrinsics,
    image_shape: Optional[Tuple[int, int]] = None,
    eps: float = 1e-3,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Transforms points in camera frame to image pixel coordinates

    TODO: refactor
    :param points: points in camera frame
    :param intrinsic: camera intrinsics
    :param image_shape: shape of image in pixel
    :param eps: lower threshold of points, defaults to 1e-3
    :return: points in pixel coordinates, mask of values in frame
    """
    points = points[:, :3]

    K = intrinsics.camera_matrix

    viewpad = np.eye(4)
    viewpad[: K.shape[0], : K.shape[1]] = K

    pc_img = np.concatenate([points, np.ones_like(points)[:, :1]], -1)
    pc_img = viewpad @ pc_img.T
    pc_img = pc_img.T

    cur_pc_in_fov = pc_img[:, 2] > eps
    pc_img = pc_img[..., 0:2] / np.maximum(pc_img[..., 2:3], np.ones_like(pc_img[..., 2:3]) * eps)
    if image_shape is not None:
        img_h, img_w = image_shape
        cur_pc_in_fov = (
            cur_pc_in_fov
            & (pc_img[:, 0] < (img_w - 1))
            & (pc_img[:, 0] > 0)
            & (pc_img[:, 1] < (img_h - 1))
            & (pc_img[:, 1] > 0)
        )
    return pc_img, cur_pc_in_fov


def _transform_pcs_to_images(
    lidar_xyz: npt.NDArray[np.float32],
    camera: PinholeCamera,
    eps: float = 1e-3,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Transforms lidar point cloud to image pixel coordinates

    TODO: refactor
    :param lidar_xyz: lidar point cloud in xyz coordinates
    :param camera: pinhole camera
    :param eps: lower threshold of points, defaults to 1e-3
    :return: points in pixel coordinates, mask of values in frame
    """

    pc_xyz = lidar_xyz

    lidar2cam_r = np.linalg.inv(camera.extrinsic.rotation_matrix)
    lidar2cam_t = camera.extrinsic.point_3d @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    camera_matrix = camera.metadata.intrinsics.camera_matrix
    viewpad = np.eye(4)
    viewpad[: camera_matrix.shape[0], : camera_matrix.shape[1]] = camera_matrix
    lidar2img_rt = viewpad @ lidar2cam_rt.T
    img_shape = camera.image.shape[:2]

    cur_pc_xyz = np.concatenate([pc_xyz, np.ones_like(pc_xyz)[:, :1]], -1)
    cur_pc_cam = lidar2img_rt @ cur_pc_xyz.T
    cur_pc_cam = cur_pc_cam.T
    cur_pc_in_fov = cur_pc_cam[:, 2] > eps
    cur_pc_cam = cur_pc_cam[..., 0:2] / np.maximum(cur_pc_cam[..., 2:3], np.ones_like(cur_pc_cam[..., 2:3]) * eps)

    if img_shape is not None:
        img_h, img_w = img_shape
        cur_pc_in_fov = (
            cur_pc_in_fov
            & (cur_pc_cam[:, 0] < (img_w - 1))
            & (cur_pc_cam[:, 0] > 0)
            & (cur_pc_cam[:, 1] < (img_h - 1))
            & (cur_pc_cam[:, 1] > 0)
        )
    return cur_pc_cam, cur_pc_in_fov
