import gzip
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from py123d.geometry import PoseSE3, Vector3D
from py123d.geometry.transform import translate_se3_along_body_frame
from py123d.geometry.transform.transform_se3 import (
    reframe_se3_array,
)


def read_json(json_file: Union[Path, str]) -> Any:
    """Read a JSON file and return the parsed contents.

    :param json_file: Path to the JSON file.
    :return: The parsed JSON data.
    """
    with open(json_file, "r") as f:
        json_data = json.load(f)
    return json_data


def read_pkl_gz(pkl_gz_file: Union[Path, str]) -> Any:
    """Read a gzip-compressed pickle file and return the deserialized contents.

    :param pkl_gz_file: Path to the .pkl.gz file.
    :return: The deserialized Python object.
    """
    with gzip.open(pkl_gz_file, "rb") as f:
        pkl_data = pickle.load(f)
    return pkl_data


def pandaset_pose_dict_to_pose_se3(pose_dict: Dict[str, Dict[str, float]]) -> PoseSE3:
    """Convert a PandaSet pose dictionary to PoseSE3.

    The pose dictionary has the format::

        {"position": {"x": ..., "y": ..., "z": ...},
         "heading":  {"w": ..., "x": ..., "y": ..., "z": ...}}

    :param pose_dict: The input pose dict with ``position`` and ``heading`` keys.
    :return: The converted PoseSE3.
    """
    return PoseSE3(
        x=pose_dict["position"]["x"],
        y=pose_dict["position"]["y"],
        z=pose_dict["position"]["z"],
        qw=pose_dict["heading"]["w"],
        qx=pose_dict["heading"]["x"],
        qy=pose_dict["heading"]["y"],
        qz=pose_dict["heading"]["z"],
    )


def rotate_pandaset_pose_to_iso_coordinates(pose: PoseSE3) -> PoseSE3:
    """Rotate the body frame of a PandaSet pose to ISO 8855 coordinates.

    PandaSet uses (x: right, y: forward, z: up).
    ISO 8855 uses (x: forward, y: left, z: up).

    This right-multiplies the rotation by the coordinate frame change matrix,
    converting the body-frame axes while leaving the global-frame position unchanged.

    Reference: https://arxiv.org/pdf/2112.12610.pdf

    :param pose: The input pose in PandaSet body-frame convention.
    :return: The pose with ISO 8855 body-frame convention.
    """
    F = np.array(
        [
            [0.0, 1.0, 0.0],  # new X = old Y (forward)
            [-1.0, 0.0, 0.0],  # new Y = old -X (left)
            [0.0, 0.0, 1.0],  # new Z = old Z (up)
        ],
        dtype=np.float64,
    ).T
    transformation_matrix = pose.transformation_matrix.copy()
    transformation_matrix[0:3, 0:3] = transformation_matrix[0:3, 0:3] @ F

    return PoseSE3.from_transformation_matrix(transformation_matrix)


def global_main_lidar_to_global_imu(pose: PoseSE3) -> PoseSE3:
    """Convert a global main-lidar pose to a global IMU pose.

    Performs two operations:

    1. Rotates the body frame from PandaSet convention (x: right, y: forward, z: up)
       to ISO 8855 (x: forward, y: left, z: up).
    2. Translates 0.84 m backward along the body x-axis to account for the physical
       offset from the roof-mounted Pandar64 lidar to the IMU at the rear axle.

    :param pose: The global pose of the main lidar in PandaSet coordinates.
    :return: The global IMU pose with ISO body-frame convention.
    """
    F = np.array(
        [
            [0.0, 1.0, 0.0],  # new X = old Y (forward)
            [-1.0, 0.0, 0.0],  # new Y = old -X (left)
            [0.0, 0.0, 1.0],  # new Z = old Z (up)
        ],
        dtype=np.float64,
    ).T
    transformation_matrix = pose.transformation_matrix.copy()
    transformation_matrix[0:3, 0:3] = transformation_matrix[0:3, 0:3] @ F

    rotated_pose = PoseSE3.from_transformation_matrix(transformation_matrix)
    imu_pose = translate_se3_along_body_frame(rotated_pose, translation=Vector3D(x=-0.840, y=0.0, z=0.0))

    return imu_pose


def relative_main_lidar_to_relative_imu(pose: PoseSE3 = PoseSE3.identity()) -> PoseSE3:
    """Compute the relative transform from the main-lidar origin to the IMU origin.

    This is the inverse-direction counterpart of :func:`global_main_lidar_to_global_imu`:
    it produces a static transform that maps coordinates expressed in the main-lidar frame
    to the IMU frame.

    1. Translates 0.84 m along the PandaSet +y axis (forward) to the IMU location.
    2. Applies the inverse coordinate rotation (ISO → PandaSet body axes) so that the
       resulting frame has ISO body-frame convention.

    :param pose: Base pose (default: identity at the main-lidar origin).
    :return: The pose of the IMU origin in the main-lidar frame, with ISO body-frame convention.
    """
    imu_location_pose = translate_se3_along_body_frame(pose, translation=Vector3D(x=0.0, y=-0.840, z=0.0))

    F = np.array(
        [
            [0.0, 1.0, 0.0],  # new X = old Y (forward)
            [-1.0, 0.0, 0.0],  # new Y = old -X (left)
            [0.0, 0.0, 1.0],  # new Z = old Z (up)
        ],
        dtype=np.float64,
    ).T
    transformation_matrix = PoseSE3.identity().transformation_matrix.copy()
    transformation_matrix[0:3, 0:3] = transformation_matrix[0:3, 0:3] @ F
    transformation_matrix[0:3, 3] = imu_location_pose.point_3d.array

    rotated_pose = PoseSE3.from_transformation_matrix(transformation_matrix)
    return rotated_pose


def extrinsic_to_imu(pose: PoseSE3) -> PoseSE3:
    """Convert a sensor extrinsic (in the main-lidar frame) to the IMU frame.

    The PandaSet calibration file provides sensor poses relative to the main lidar.
    This function re-expresses those poses relative to the IMU by:

    1. Inverting the sensor pose (lidar→sensor becomes sensor→lidar).
    2. Reframing from the main-lidar origin to the IMU origin via
       :func:`relative_main_lidar_to_relative_imu`.

    :param pose: Sensor-to-main-lidar extrinsic pose.
    :return: Sensor-to-IMU extrinsic pose.
    """

    def _get_inverse_pose(pose: PoseSE3) -> PoseSE3:
        return PoseSE3.from_transformation_matrix(np.linalg.inv(pose.transformation_matrix))

    invert_pose = _get_inverse_pose(pose)

    main_lidar = PoseSE3.identity()
    imu = relative_main_lidar_to_relative_imu(main_lidar)

    new_pose_array = reframe_se3_array(
        from_origin=main_lidar,
        to_origin=imu,
        pose_se3_array=invert_pose.array,
    )
    return PoseSE3.from_array(new_pose_array)
