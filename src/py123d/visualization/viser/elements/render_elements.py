import numpy as np

from py123d.api.scene.scene_api import SceneAPI
from py123d.conversion.utils.sensor_utils.camera_conventions import convert_camera_convention
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.geometry import EulerAngles, PoseSE3, PoseSE3Index, Vector3D
from py123d.geometry.transform.transform_se3 import translate_se3_along_body_frame


def get_ego_3rd_person_view_position(
    scene: SceneAPI,
    iteration: int,
    initial_ego_state: EgoStateSE3,
) -> PoseSE3:
    scene_center_array = initial_ego_state.center_se3.point_3d.array
    ego_pose = scene.get_ego_state_at_iteration(iteration).imu_se3.array
    ego_pose[PoseSE3Index.XYZ] -= scene_center_array
    ego_pose_se3 = PoseSE3.from_array(ego_pose)
    ego_pose_se3 = translate_se3_along_body_frame(ego_pose_se3, Vector3D(-15.0, 0.0, 15))

    # adjust the pitch to -10 degrees.
    # euler_angles_array = ego_pose_se3.euler_angles.array
    # euler_angles_array[1] += np.deg2rad(30)
    # new_quaternion = EulerAngles.from_array(euler_angles_array).quaternion

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
    scene_center_array = initial_ego_state.center_se3.point_3d.array
    ego_center = scene.get_ego_state_at_iteration(iteration).center_se3.array
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
