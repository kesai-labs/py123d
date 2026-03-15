from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from py123d.common.utils.dependencies import check_dependencies
from py123d.datatypes import CameraID, LidarFeature
from py123d.parser.wod.utils.wod_constants import WOD_PERCEPTION_CAMERA_IDS, WOD_PERCEPTION_LIDAR_IDS
from py123d.parser.wod.waymo_open_dataset.utils.frame_utils import parse_range_image_and_camera_projection

check_dependencies(modules=["tensorflow"], optional_name="waymo")
import tensorflow as tf

from py123d.parser.wod.waymo_open_dataset.protos import dataset_pb2
from py123d.parser.wod.waymo_open_dataset.utils import frame_utils


def _get_frame_at_iteration(filepath: Path, iteration: int) -> Optional[dataset_pb2.Frame]:
    """Helper function to load a Waymo Frame at a specific iteration from a TFRecord file."""
    dataset = tf.data.TFRecordDataset(str(filepath), compression_type="")

    frame: Optional[dataset_pb2.Frame] = None
    for i, data in enumerate(dataset):
        if i == iteration:
            frame = dataset_pb2.Frame()
            frame.ParseFromString(data.numpy())
            break
    return frame


def load_jpeg_binary_from_tf_record_file(
    tf_record_path: Path,
    iteration: int,
    pinhole_camera_type: CameraID,
) -> Optional[bytes]:
    """Loads the JPEG binary of a specific pinhole camera from a Waymo TFRecord file at a given iteration."""
    frame = _get_frame_at_iteration(tf_record_path, iteration)
    assert frame is not None, f"Frame at iteration {iteration} not found in Waymo file: {tf_record_path}"

    jpeg_binary: Optional[bytes] = None
    for image_proto in frame.images:
        camera_type = WOD_PERCEPTION_CAMERA_IDS[image_proto.name]
        if camera_type == pinhole_camera_type:
            jpeg_binary = image_proto.image
            break
    return jpeg_binary


def load_wod_perception_point_cloud_data_from_frame(
    frame: dataset_pb2.Frame,
    keep_polar_features: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Loads Waymo Open Dataset (WOD) - Perception Lidar point clouds from a Waymo Frame object."""

    (range_images, camera_projections, _, range_image_top_pose) = parse_range_image_and_camera_projection(frame)
    points, _ = frame_utils.convert_range_image_to_point_cloud(
        frame=frame,
        range_images=range_images,
        camera_projections=camera_projections,
        range_image_top_pose=range_image_top_pose,
        keep_polar_features=keep_polar_features,
    )
    # NOTE: @DanielDauner
    # keep_polar_features=True: points have shape (N, 6) with features in order [RANGE, INTENSITY, ELONGATION, X, Y, Z]
    # keep_polar_features=False: points have shape (N, 3) with features in order [X, Y, Z]

    # Concat all lidar points.
    all_lidar_data = np.concatenate(points, axis=0)

    # Load features and point cloud
    lidar_ids = np.zeros(all_lidar_data.shape[0], dtype=np.uint8)
    start_idx = 0
    for lidar_idx, frame_lidar in enumerate(frame.lasers):
        lidar_id = WOD_PERCEPTION_LIDAR_IDS[frame_lidar.name]
        num_points = points[lidar_idx].shape[0]
        lidar_ids[start_idx : start_idx + num_points] = int(lidar_id)  # type: ignore
        start_idx += num_points

    # Load point cloud and other features based on whether to keep polar features or not.
    if keep_polar_features:
        point_cloud_3d = all_lidar_data[:, 3:6]  # Extract XYZ from the concatenated Lidar data.
        point_cloud_features = {
            LidarFeature.RANGE.serialize(): all_lidar_data[:, 0].astype(np.float32),
            LidarFeature.INTENSITY.serialize(): (all_lidar_data[:, 1] * 255).astype(np.uint8),
            LidarFeature.ELONGATION.serialize(): all_lidar_data[:, 2].astype(np.float32),
            LidarFeature.IDS.serialize(): lidar_ids,
        }
    else:
        point_cloud_3d = all_lidar_data[:, :3]  # Extract XYZ from the concatenated Lidar data.
        point_cloud_features = {
            LidarFeature.IDS.serialize(): lidar_ids,
        }
    return point_cloud_3d, point_cloud_features


def load_wod_perception_point_cloud_data_from_path(
    tf_record_path: Path,
    index: int,
    keep_polar_features: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Loads Waymo Open Dataset (WOD) - Perception Lidar point clouds from a TFRecord file at a given iteration."""

    frame = _get_frame_at_iteration(tf_record_path, index)
    assert frame is not None, f"Frame at iteration {index} not found in Waymo file: {tf_record_path}"
    return load_wod_perception_point_cloud_data_from_frame(frame, keep_polar_features=keep_polar_features)
