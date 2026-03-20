from typing import Tuple

import numpy as np
import numpy.typing as npt

from py123d.api.scene.scene_api import SceneAPI
from py123d.datatypes.sensors.base_camera import Camera
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.geometry import EulerAngles, PoseSE3Index, Vector3D
from py123d.geometry.pose import PoseSE3
from py123d.geometry.rotation import Quaternion
from py123d.geometry.transform.transform_se3 import abs_to_rel_se3_array, translate_se3_along_body_frame
from py123d.parser.utils.sensor_utils.camera_conventions import convert_camera_convention


def decompose_camera_pose(
    camera: Camera, scene_center_pose: PoseSE3
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Decompose a camera's global pose into position and quaternion relative to the scene center."""
    global_camera_se3 = camera.camera_to_global_se3.array
    abs_camera_pose = abs_to_rel_se3_array(origin=scene_center_pose, pose_se3_array=global_camera_se3)
    return abs_camera_pose[PoseSE3Index.XYZ], abs_camera_pose[PoseSE3Index.QUATERNION]


def get_scene_center_pose(scene_center_array: npt.NDArray[np.float64]) -> PoseSE3:
    """Create a PoseSE3 at the scene center with identity rotation."""
    return PoseSE3.from_R_t(rotation=Quaternion.identity(), translation=scene_center_array)


def get_ego_3rd_person_view_position(
    scene: SceneAPI,
    iteration: int,
    initial_ego_state: EgoStateSE3,
) -> PoseSE3:
    """Position camera 15m behind and 15m above ego vehicle with 30 degree pitch."""
    scene_center_array = initial_ego_state.center_se3.point_3d.array
    ego_pose = scene.get_ego_state_se3_at_iteration(iteration).imu_se3.array
    ego_pose[PoseSE3Index.XYZ] -= scene_center_array
    ego_pose_se3 = PoseSE3.from_array(ego_pose)
    ego_pose_se3 = translate_se3_along_body_frame(ego_pose_se3, Vector3D(-15.0, 0.0, 15))
    ego_pose_se3 = _pitch_se3_by_degrees(ego_pose_se3, 30.0)

    return convert_camera_convention(
        ego_pose_se3,
        from_convention="pXpZmY",
        to_convention="pZmYpX",
    )


def get_ego_bev_view_position(
    scene: SceneAPI,
    iteration: int,
    initial_ego_state: EgoStateSE3,
) -> PoseSE3:
    """Position camera 50m directly above ego vehicle looking straight down."""
    scene_center_array = initial_ego_state.center_se3.point_3d.array
    ego_center = scene.get_ego_state_se3_at_iteration(iteration).center_se3.array
    ego_center[PoseSE3Index.XYZ] -= scene_center_array
    ego_center_planar = PoseSE3.from_array(ego_center)

    planar_euler_angles = EulerAngles(0.0, 0.0, ego_center_planar.euler_angles.yaw)
    quaternion = planar_euler_angles.quaternion
    ego_center_planar._array[PoseSE3Index.QUATERNION] = quaternion.array

    ego_center_planar = translate_se3_along_body_frame(ego_center_planar, Vector3D(0.0, 0.0, 50))
    ego_center_planar = _pitch_se3_by_degrees(ego_center_planar, 90.0)

    return convert_camera_convention(
        ego_center_planar,
        from_convention="pXpZmY",
        to_convention="pZmYpX",
    )


def _pitch_se3_by_degrees(pose_se3: PoseSE3, degrees: float) -> PoseSE3:
    quaternion = EulerAngles(0.0, np.deg2rad(degrees), pose_se3.yaw).quaternion

    return PoseSE3(
        x=pose_se3.x,
        y=pose_se3.y,
        z=pose_se3.z,
        qw=quaternion.qw,
        qx=quaternion.qx,
        qy=quaternion.qy,
        qz=quaternion.qz,
    )
