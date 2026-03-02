import concurrent.futures
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import viser

from py123d.api.scene.scene_api import SceneAPI
from py123d.datatypes.sensors import (
    FisheyeMEICamera,
    FisheyeMEICameraID,
    FisheyeMEICameraMetadata,
    LidarID,
    PinholeCamera,
    PinholeCameraID,
)
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.geometry import PoseSE3Index
from py123d.geometry.transform.transform_se3 import rel_to_abs_points_3d_array, rel_to_abs_se3_array
from py123d.visualization.matplotlib.lidar import get_lidar_pc_color
from py123d.visualization.viser.viser_config import ViserConfig


def add_camera_frustums_to_viser_server(
    scene: SceneAPI,
    scene_interation: int,
    initial_ego_state: EgoStateSE3,
    viser_server: viser.ViserServer,
    viser_config: ViserConfig,
    camera_frustum_handles: Dict[PinholeCameraID, viser.CameraFrustumHandle],
) -> None:
    if viser_config.camera_frustum_visible:
        scene_center_array = initial_ego_state.center_se3.point_3d.array
        ego_pose = scene.get_ego_state_at_iteration(scene_interation).imu_se3.array
        ego_pose[PoseSE3Index.XYZ] -= scene_center_array

        def _add_camera_frustums_to_viser_server(camera_type: PinholeCameraID) -> None:
            camera = scene.get_pinhole_camera_at_iteration(scene_interation, camera_type)
            if camera is not None:
                camera_position, camera_quaternion, camera_image = _get_camera_values(
                    camera,
                    ego_pose.copy(),
                    viser_config.camera_frustum_image_scale,
                )
                if camera_type in camera_frustum_handles:
                    camera_frustum_handles[camera_type].position = camera_position
                    camera_frustum_handles[camera_type].wxyz = camera_quaternion
                    camera_frustum_handles[camera_type].image = camera_image
                else:
                    camera_frustum_handles[camera_type] = viser_server.scene.add_camera_frustum(
                        f"camera_frustums/{camera_type.serialize()}",
                        fov=camera.metadata.fov_y,
                        aspect=camera.metadata.aspect_ratio,
                        scale=viser_config.camera_frustum_scale,
                        image=camera_image,
                        position=camera_position,
                        wxyz=camera_quaternion,
                    )

        # NOTE; In order to speed up adding camera frustums, we use multithreading and resize the images.
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(viser_config.camera_frustum_types)) as executor:
            future_to_camera = {
                executor.submit(_add_camera_frustums_to_viser_server, camera_type): camera_type
                for camera_type in viser_config.camera_frustum_types
            }
            for future in concurrent.futures.as_completed(future_to_camera):
                _ = future.result()

        # TODO: Remove serial implementation, if not needed anymore.
        # for camera_type in viser_config.camera_frustum_types:
        #     _add_camera_frustums_to_viser_server(camera_type)

        return None


def add_fisheye_frustums_to_viser_server(
    scene: SceneAPI,
    scene_interation: int,
    initial_ego_state: EgoStateSE3,
    viser_server: viser.ViserServer,
    viser_config: ViserConfig,
    fisheye_frustum_handles: Dict[FisheyeMEICameraID, viser.CameraFrustumHandle],
) -> None:
    if viser_config.fisheye_frustum_visible:
        scene_center_array = initial_ego_state.center_se3.point_3d.array
        ego_pose = scene.get_ego_state_at_iteration(scene_interation).imu_se3.array
        ego_pose[PoseSE3Index.XYZ] -= scene_center_array

        def _add_fisheye_frustums_to_viser_server(fisheye_camera_type: FisheyeMEICameraID) -> None:
            camera = scene.get_fisheye_mei_camera_at_iteration(scene_interation, fisheye_camera_type)
            if camera is not None:
                fcam_position, fcam_quaternion, fcam_image = _get_fisheye_camera_values(
                    camera,
                    ego_pose.copy(),
                    viser_config.fisheye_frustum_image_scale,
                )
                if fisheye_camera_type in fisheye_frustum_handles:
                    fisheye_frustum_handles[fisheye_camera_type].position = fcam_position
                    fisheye_frustum_handles[fisheye_camera_type].wxyz = fcam_quaternion
                    fisheye_frustum_handles[fisheye_camera_type].image = fcam_image
                else:
                    # NOTE @DanielDauner: The FOV is just taking as a static value here.
                    # The function se
                    fisheye_frustum_handles[fisheye_camera_type] = viser_server.scene.add_camera_frustum(
                        f"camera_frustums/{fisheye_camera_type.serialize()}",
                        fov=185,  # vertical fov
                        aspect=camera.metadata.aspect_ratio,
                        scale=viser_config.fisheye_frustum_scale,
                        image=fcam_image,
                        position=fcam_position,
                        wxyz=fcam_quaternion,
                    )

        # NOTE; In order to speed up adding camera frustums, we use multithreading and resize the images.
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(viser_config.fisheye_mei_camera_frustum_types)
        ) as executor:
            future_to_camera = {
                executor.submit(_add_fisheye_frustums_to_viser_server, fcam_type): fcam_type
                for fcam_type in viser_config.fisheye_mei_camera_frustum_types
            }
            for future in concurrent.futures.as_completed(future_to_camera):
                _ = future.result()

        return None


def add_camera_gui_to_viser_server(
    scene: SceneAPI,
    scene_interation: int,
    viser_server: viser.ViserServer,
    viser_config: ViserConfig,
    camera_gui_handles: Dict[PinholeCameraID, viser.GuiImageHandle],
) -> None:
    if viser_config.camera_gui_visible:
        for camera_type in viser_config.camera_gui_types:
            camera = scene.get_pinhole_camera_at_iteration(scene_interation, camera_type)
            if camera is not None:
                if camera_type in camera_gui_handles:
                    camera_gui_handles[camera_type].image = _rescale_image(
                        camera.image, viser_config.camera_gui_image_scale
                    )
                else:
                    with viser_server.gui.add_folder(f"Camera {camera_type.serialize()}"):
                        camera_gui_handles[camera_type] = viser_server.gui.add_image(
                            image=_rescale_image(camera.image, viser_config.camera_gui_image_scale),
                            label=camera_type.serialize(),
                        )


def add_lidar_pc_to_viser_server(
    scene: SceneAPI,
    scene_interation: int,
    initial_ego_state: EgoStateSE3,
    viser_server: viser.ViserServer,
    viser_config: ViserConfig,
    lidar_pc_handles: Dict[LidarID, Optional[viser.PointCloudHandle]],
) -> None:
    active_id = viser_config.lidar_ids[0]
    # Ensure the active ID has an entry in the handles dict.
    if active_id not in lidar_pc_handles:
        lidar_pc_handles[active_id] = None

    if viser_config.lidar_visible:
        scene_center_array = initial_ego_state.center_se3.point_3d.array
        ego_pose = scene.get_ego_state_at_iteration(scene_interation).imu_se3.array
        ego_pose[PoseSE3Index.XYZ] -= scene_center_array

        start = time.time()
        lidar = scene.get_lidar_at_iteration(scene_interation, lidar_id=active_id)
        print(f"Time to get lidar data from scene: {(time.time() - start) * 1000:.2f} ms")
        if lidar is not None:
            points = rel_to_abs_points_3d_array(ego_pose, lidar.xyz)
            colors = get_lidar_pc_color(lidar, feature=viser_config.lidar_point_color)
        else:
            points = np.zeros((0, 3), dtype=np.float32)
            colors = np.zeros((0, 3), dtype=np.uint8)

        if lidar_pc_handles[active_id] is not None:
            lidar_pc_handles[active_id].points = points
            lidar_pc_handles[active_id].colors = colors
        else:
            lidar_pc_handles[active_id] = viser_server.scene.add_point_cloud(
                "lidar_points",
                points=points,
                colors=colors,
                point_size=viser_config.lidar_point_size,
                point_shape=viser_config.lidar_point_shape,
            )
    elif lidar_pc_handles[active_id] is not None:
        lidar_pc_handles[active_id].visible = False


def _get_camera_values(
    camera: PinholeCamera,
    ego_pose: npt.NDArray[np.float64],
    resize_factor: Optional[float] = None,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.uint8]]:
    assert ego_pose.ndim == 1 and len(ego_pose) == len(PoseSE3Index)

    rel_camera_pose = camera.extrinsic.array
    abs_camera_pose = rel_to_abs_se3_array(origin=ego_pose, pose_se3_array=rel_camera_pose)

    camera_position = abs_camera_pose[PoseSE3Index.XYZ]
    camera_rotation = abs_camera_pose[PoseSE3Index.QUATERNION]

    camera_image = _rescale_image(camera.image, resize_factor)
    return camera_position, camera_rotation, camera_image


def _get_fisheye_camera_values(
    camera: FisheyeMEICamera,
    ego_pose: npt.NDArray[np.float64],
    resize_factor: Optional[float] = None,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.uint8]]:
    assert ego_pose.ndim == 1 and len(ego_pose) == len(PoseSE3Index)

    rel_camera_pose = camera.extrinsic.array
    abs_camera_pose = rel_to_abs_se3_array(origin=ego_pose, pose_se3_array=rel_camera_pose)

    camera_position = abs_camera_pose[PoseSE3Index.XYZ]
    camera_rotation = abs_camera_pose[PoseSE3Index.QUATERNION]

    camera_image = _rescale_image(camera.image, resize_factor)
    return camera_position, camera_rotation, camera_image


def _rescale_image(image: npt.NDArray[np.uint8], scale: float) -> npt.NDArray[np.uint8]:
    if scale == 1.0:
        return image
    new_width = int(image.shape[1] * scale)
    new_height = int(image.shape[0] * scale)
    downscaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return downscaled_image


def calculate_fov(metadata: FisheyeMEICameraMetadata) -> tuple[float, float]:
    """
    Calculate horizontal and vertical FOV in degrees.

    Returns:
        (horizontal_fov, vertical_fov) in degrees
    """
    xi = metadata.mirror_parameter
    gamma1 = metadata.projection.gamma1
    gamma2 = metadata.projection.gamma2
    u0 = metadata.projection.u0
    v0 = metadata.projection.v0

    width = metadata.width
    height = metadata.height

    # Calculate corner positions (furthest from principal point)
    corners = np.array([[0, 0], [width, 0], [0, height], [width, height]])

    # Convert to normalized coordinates
    x_norm = (corners[:, 0] - u0) / gamma1
    y_norm = (corners[:, 1] - v0) / gamma2

    # For MEI model, inverse projection (ignoring distortion for FOV estimate):
    # r² = x² + y²
    # θ = arctan(r / (1 - ξ·√(1 + r²)))

    r_squared = x_norm**2 + y_norm**2
    r = np.sqrt(r_squared)

    # Calculate incident angle for each corner
    # From MEI model: r = (X/Z_s) where Z_s = Z + ξ·√(X² + Y² + Z²)
    # This gives: θ = arctan(r·√(1 + (1-ξ²)r²) / (1 - ξ²·r²))
    # Simplified approximation:

    if xi < 1e-6:  # Perspective camera
        theta = np.arctan(r)
    else:
        # For small angles or as approximation
        denominator = 1 - xi * np.sqrt(1 + r_squared)
        theta = np.arctan2(r, denominator)

    np.max(np.abs(theta))

    # Calculate horizontal and vertical FOV separately
    x_max = np.max(np.abs(x_norm))
    y_max = np.max(np.abs(y_norm))

    if xi < 1e-6:
        h_fov = 2 * np.arctan(x_max)
        v_fov = 2 * np.arctan(y_max)
    else:
        denom_h = 1 - xi * np.sqrt(1 + x_max**2)
        denom_v = 1 - xi * np.sqrt(1 + y_max**2)
        h_fov = 2 * np.arctan2(x_max, denom_h)
        v_fov = 2 * np.arctan2(y_max, denom_v)

    return h_fov, v_fov
