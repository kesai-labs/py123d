from typing import Any, Dict, Literal, Optional, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
from shapely.geometry import Polygon

from py123d.datatypes.sensors import PinholeCameraMetadata
from py123d.datatypes.sensors.base_camera import BaseCameraMetadata, Camera, CameraID
from py123d.datatypes.sensors.fisheye_mei_camera import FisheyeMEICameraMetadata
from py123d.datatypes.sensors.ftheta_camera import FThetaCameraMetadata
from py123d.datatypes.sensors.pinhole_camera import PinholeDistortion, PinholeIntrinsics
from py123d.geometry.pose import PoseSE3

_InterpolationMode = Literal["nearest", "linear", "cubic"]
_UndistortMode = Literal["optimal", "keep_focal_length"]


def field_of_view_intrinsic(
    intrinsic: npt.NDArray[np.float64],
    scale: float = 30.0,
    transform_matrix: Optional[npt.NDArray[np.float64]] = None,
) -> Polygon:
    """Compute camera field of view in 2D space from its intrinsic matrix.

    Camera coordinate system: x right, y up, z front.

    :param intrinsic: 3x3 camera intrinsic matrix
    :param scale: field of view scale in meters, defaults to 30.0
    :param transform_matrix: optional 4x4 transformation matrix
    :return: field of view as a shapely Polygon projected onto the xy-plane
    """
    half_wide = intrinsic[0, 2] / intrinsic[0, 0]
    fov = (
        np.array(
            [
                [0, 0, 0],
                [half_wide, 0, 1],
                [-half_wide, 0, 1],
            ]
        )
        * scale
    )

    if transform_matrix is not None:
        fov = np.dot(transform_matrix[:3, :3], fov.T).T + transform_matrix[:3, 3]

    fov_geom = Polygon(fov[:, :2])

    return fov_geom


def get_safe_projs(
    obj_pts_cam: npt.NDArray[np.float64],
    dist_coeffs: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """Determine which projected points are safe from distortion artifacts.

    Checks that homogeneous coordinates are within the monotonic region of the distortion
    model to avoid projection errors. See https://github.com/opencv/opencv/issues/17768.

    :param obj_pts_cam: object points in camera coordinates of shape (N, 3)
    :param dist_coeffs: distortion coefficients of shape (5,)
    :return: boolean mask of shape (N,) indicating safe projections
    """
    # Define a list of booleans to denote if a variable is safe
    obj_pts_safe = np.ones(len(obj_pts_cam), dtype=bool)

    # Define the homogeneous coordinates
    x_homo_vals = (obj_pts_cam[:, 0] / obj_pts_cam[:, 2]).astype(complex)
    y_homo_vals = (obj_pts_cam[:, 1] / obj_pts_cam[:, 2]).astype(complex)

    # Define the distortion terms, and vectorize calculating of powers of x_homo_vals
    #   and y_homo_vals
    k1, k2, p1, p2, k3 = dist_coeffs.tolist()
    y_homo_vals_2 = np.power(y_homo_vals, 2)
    y_homo_vals_4 = np.power(y_homo_vals, 4)
    y_homo_vals_6 = np.power(y_homo_vals, 6)

    # Find the bounds on the x_homo coordinate to ensure it is closer than the
    #   inflection point of x_proj as a function of x_homo
    x_homo_min = np.full(x_homo_vals.shape, np.inf)
    x_homo_max = np.full(x_homo_vals.shape, -np.inf)
    for i in range(len(y_homo_vals)):
        # Expanded projection function polynomial coefficients
        x_proj_coeffs = np.array(
            [
                k3,
                0,
                k2 + 3 * k3 * y_homo_vals_2[i],
                0,
                k1 + 2 * k2 * y_homo_vals_2[i] + 3 * k3 * y_homo_vals_4[i],
                3 * p2,
                1 + k1 * y_homo_vals_2[i] + k2 * y_homo_vals_4[i] + k3 * y_homo_vals_6[i] + 2 * p1 * y_homo_vals[i],
                p2 * y_homo_vals_2[i],
            ]
        )

        # Projection function derivative polynomial coefficients
        x_proj_der_coeffs = np.polyder(x_proj_coeffs)

        # Find the root of the derivative
        roots = np.roots(x_proj_der_coeffs)

        # Get the real roots
        # Approximation of real[np.where(np.isreal(roots))]
        real_roots = np.real(roots[np.where(np.abs(np.imag(roots)) < 1e-10)])

        for real_root in real_roots:
            x_homo_min[i] = np.minimum(x_homo_min[i], real_root)
            x_homo_max[i] = np.maximum(x_homo_max[i], real_root)

    # Check that the x_homo values are within the bounds
    obj_pts_safe *= np.where(x_homo_vals > x_homo_min, True, False)
    obj_pts_safe *= np.where(x_homo_vals < x_homo_max, True, False)

    return obj_pts_safe


def undistort_image_optimal(
    image: npt.NDArray[np.uint8],
    intrinsic: npt.NDArray[np.float64],
    distortion: npt.NDArray[np.float64],
    return_mask: bool = False,
    interpolation: _InterpolationMode = "linear",
) -> Union[
    Tuple[npt.NDArray[np.uint8], npt.NDArray[np.float64], Tuple[int, int, int, int]],
    Tuple[npt.NDArray[np.uint8], npt.NDArray[np.float64], Tuple[int, int, int, int], npt.NDArray[np.uint8]],
]:
    """Undistort an image using the optimal new camera matrix.

    :param image: input image of shape (H, W, C)
    :param intrinsic: 3x3 camera intrinsic matrix
    :param distortion: distortion coefficients
    :param return_mask: whether to return a valid pixel mask, defaults to False
    :param interpolation: interpolation method, defaults to "linear"
    :return: undistorted image, new intrinsic matrix, and ROI; optionally includes a valid pixel mask
    """
    interpolation_map = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}

    height, width = image.shape[:2]
    new_intrinsic, roi = cv2.getOptimalNewCameraMatrix(intrinsic, distortion, (width, height), 1)
    map_undistort_optimal = cv2.initUndistortRectifyMap(
        intrinsic, distortion, None, new_intrinsic, (width, height), cv2.CV_32FC2
    )
    image_optimal = cv2.remap(
        image, map_undistort_optimal[0], map_undistort_optimal[1], interpolation=interpolation_map[interpolation]
    )

    if return_mask:
        valid_mask = np.ones_like(image_optimal, dtype=np.uint8) * 255
        valid_mask = cv2.remap(
            valid_mask, map_undistort_optimal[0], map_undistort_optimal[1], interpolation=cv2.INTER_NEAREST
        )
        return image_optimal, new_intrinsic, roi, valid_mask
    else:
        return image_optimal, new_intrinsic, roi


def undistort_image_keep_focal_length(
    image: npt.NDArray[np.uint8],
    intrinsic: npt.NDArray[np.float64],
    distortion: npt.NDArray[np.float64],
    interpolation: _InterpolationMode = "linear",
) -> npt.NDArray[np.uint8]:
    """Undistort an image while preserving the original focal length.

    :param image: input image of shape (H, W, C)
    :param intrinsic: 3x3 camera intrinsic matrix
    :param distortion: distortion coefficients
    :param interpolation: interpolation method, defaults to "linear"
    :return: undistorted image
    """
    interpolation_map = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}

    height, width = image.shape[:2]
    map_undistort = cv2.initUndistortRectifyMap(intrinsic, distortion, None, intrinsic, (width, height), cv2.CV_32FC2)
    image_undistorted = cv2.remap(
        image, map_undistort[0], map_undistort[1], interpolation=interpolation_map[interpolation]
    )

    return image_undistorted


def undistort_image_with_cam_info(
    image: npt.NDArray[np.uint8],
    cam_info: Dict[str, Any],
    return_mask: bool = False,
    interpolation: _InterpolationMode = "linear",
    mode: _UndistortMode = "optimal",
) -> Union[
    npt.NDArray[np.uint8],
    Tuple[npt.NDArray[np.uint8], npt.NDArray[np.float64], Tuple[int, int, int, int]],
    Tuple[npt.NDArray[np.uint8], npt.NDArray[np.float64], Tuple[int, int, int, int], npt.NDArray[np.uint8]],
]:
    """Undistort an image using camera info dictionary.

    :param image: input image of shape (H, W, C)
    :param cam_info: camera info dictionary containing intrinsic and distortion parameters
    :param return_mask: whether to return a valid pixel mask, defaults to False
    :param interpolation: interpolation method, defaults to "linear"
    :param mode: undistortion mode, defaults to "optimal"
    :return: undistorted image, optionally with new intrinsic matrix, ROI, and valid pixel mask
    :raises ValueError: if an invalid mode is provided
    """
    if "colmap_param" in cam_info:
        intrinsic = cam_info["colmap_param"]["cam_intrinsic"]
        distortion = cam_info["colmap_param"]["distortion"]
        intrinsic = intrinsic.copy()
        intrinsic[0, 2] = intrinsic[0, 2] - 0.5
        intrinsic[1, 2] = intrinsic[1, 2] - 0.5
    else:
        intrinsic = cam_info["cam_intrinsic"]
        distortion = cam_info["distortion"]

    if mode == "optimal":
        return undistort_image_optimal(image, intrinsic, distortion, return_mask, interpolation)
    elif mode == "keep_focal_length":
        return undistort_image_keep_focal_length(image, intrinsic, distortion, interpolation)
    else:
        raise ValueError(f"Invalid mode: {mode}")


def undistort_image_from_camera(
    image: npt.NDArray[np.uint8],
    camera_metadata: PinholeCameraMetadata,
    return_mask: bool = False,
    interpolation: _InterpolationMode = "linear",
    mode: _UndistortMode = "optimal",
) -> Union[
    npt.NDArray[np.uint8],
    Tuple[npt.NDArray[np.uint8], npt.NDArray[np.float64], Tuple[int, int, int, int]],
    Tuple[npt.NDArray[np.uint8], npt.NDArray[np.float64], Tuple[int, int, int, int], npt.NDArray[np.uint8]],
]:
    """Undistort an image using pinhole camera metadata.

    :param image: input image of shape (H, W, C)
    :param camera_metadata: pinhole camera metadata containing intrinsics and distortion
    :param return_mask: whether to return a valid pixel mask, defaults to False
    :param interpolation: interpolation method, defaults to "linear"
    :param mode: undistortion mode, defaults to "optimal"
    :return: undistorted image, optionally with new intrinsic matrix, ROI, and valid pixel mask
    :raises ValueError: if an invalid mode is provided
    :raises AssertionError: if image dimensions do not match camera resolution
    """
    width, height = camera_metadata.width, camera_metadata.height
    intrinsic = camera_metadata.intrinsics.camera_matrix
    distortion = camera_metadata.distortion.array
    assert image.shape[0] == height and image.shape[1] == width, (
        f"Image shape {image.shape} does not match camera resolution {(height, width)}"
    )

    if mode == "optimal":
        return undistort_image_optimal(image, intrinsic, distortion, return_mask, interpolation)
    elif mode == "keep_focal_length":
        return undistort_image_keep_focal_length(image, intrinsic, distortion, interpolation)
    else:
        raise ValueError(f"Invalid mode: {mode}")


def undistort_camera(
    camera: Camera,
    interpolation: _InterpolationMode = "linear",
) -> Camera:
    """Convert any camera to an undistorted pinhole camera.

    For pinhole cameras with distortion, the image is undistorted while preserving the
    original focal length (K matrix). For f-theta and fisheye MEI cameras, the image is
    remapped to a virtual pinhole camera derived from the original camera's field of view.

    If the camera is already an undistorted pinhole, it is returned unchanged.

    :param camera: The input camera (any model).
    :param interpolation: Interpolation method for image remapping, defaults to ``"linear"``.
    :return: A new :class:`Camera` with an undistorted pinhole image and matching metadata.
    """
    metadata = camera.metadata

    if isinstance(metadata, PinholeCameraMetadata) and metadata.is_undistorted:
        result = camera
    elif isinstance(metadata, PinholeCameraMetadata):
        result = _undistort_pinhole_camera(camera, metadata, interpolation)
    elif isinstance(metadata, FThetaCameraMetadata):
        result = _undistort_ftheta_camera(camera, metadata, interpolation)
    elif isinstance(metadata, FisheyeMEICameraMetadata):
        result = _undistort_fisheye_mei_camera(camera, metadata, interpolation)
    else:
        raise NotImplementedError(f"Undistortion not implemented for camera model: {type(metadata).__name__}")

    return result


def _undistort_pinhole_camera(
    camera: Camera,
    metadata: PinholeCameraMetadata,
    interpolation: _InterpolationMode,
) -> Camera:
    """Undistort a distorted pinhole camera, preserving the K matrix."""
    assert metadata.intrinsics is not None
    assert metadata.distortion is not None

    if not metadata.is_undistorted:
        K = metadata.intrinsics.camera_matrix
        dist = metadata.distortion.array
        image_undistorted = undistort_image_keep_focal_length(camera.image, K, dist, interpolation)

        new_metadata = PinholeCameraMetadata(
            camera_name=metadata.camera_name,
            camera_id=metadata.camera_id,
            intrinsics=metadata.intrinsics,
            distortion=metadata.distortion,
            width=metadata.width,
            height=metadata.height,
            camera_to_imu_se3=metadata.camera_to_imu_se3,
            is_undistorted=True,
        )
        result = Camera(
            metadata=new_metadata,
            image=image_undistorted,
            camera_to_global_se3=camera.camera_to_global_se3,
            timestamp=camera.timestamp,
        )
    else:
        result = camera
    return result


def _build_pinhole_from_fov(
    fov_x: float,
    fov_y: float,
    width: int,
    height: int,
    camera_id: CameraID,
    camera_name: str,
    camera_to_imu_se3: PoseSE3,
) -> PinholeCameraMetadata:
    """Build an undistorted pinhole camera metadata from a target field of view."""
    fx = (width / 2.0) / np.tan(fov_x / 2.0)
    fy = (height / 2.0) / np.tan(fov_y / 2.0)
    cx = width / 2.0
    cy = height / 2.0
    intrinsics = PinholeIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)
    distortion = PinholeDistortion(k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0)
    result = PinholeCameraMetadata(
        camera_name=camera_name,
        camera_id=camera_id,
        intrinsics=intrinsics,
        distortion=distortion,
        width=width,
        height=height,
        camera_to_imu_se3=camera_to_imu_se3,
        is_undistorted=True,
    )
    return result


def _remap_to_pinhole(
    image: npt.NDArray[np.uint8],
    source_metadata: BaseCameraMetadata,
    target_metadata: PinholeCameraMetadata,
    interpolation: _InterpolationMode,
) -> npt.NDArray[np.uint8]:
    """Remap an image from an arbitrary camera model to a pinhole camera.

    For each pixel (u, v) in the target pinhole image, computes the 3D ray direction,
    then projects it through the source camera model to find the source pixel. Uses
    OpenCV's ``cv2.remap`` for efficient interpolation.

    :param image: Source image.
    :param source_metadata: Source camera metadata (any model).
    :param target_metadata: Target pinhole camera metadata (undistorted).
    :param interpolation: Interpolation method.
    :return: Remapped image matching the target pinhole camera.
    """
    interpolation_map = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}

    assert target_metadata.intrinsics is not None
    h, w = target_metadata.height, target_metadata.width
    fx = target_metadata.intrinsics.fx
    fy = target_metadata.intrinsics.fy
    cx = target_metadata.intrinsics.cx
    cy = target_metadata.intrinsics.cy

    # Build grid of pixel coordinates in the target pinhole image
    u_grid, v_grid = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))

    # Unproject pinhole pixels to 3D rays: (u - cx) / fx, (v - cy) / fy, 1
    rays_x = (u_grid - cx) / fx
    rays_y = (v_grid - cy) / fy
    rays_z = np.ones_like(rays_x)

    # Stack into (H*W, 3) array and project through the source camera model
    rays = np.column_stack([rays_x.ravel(), rays_y.ravel(), rays_z.ravel()])
    source_pixels, _mask, _depth = source_metadata.project_to_image(rays)

    # Build remap arrays
    map_x = source_pixels[:, 0].reshape(h, w).astype(np.float32)
    map_y = source_pixels[:, 1].reshape(h, w).astype(np.float32)

    image_remapped = cv2.remap(image, map_x, map_y, interpolation=interpolation_map[interpolation])
    return image_remapped


def _undistort_ftheta_camera(
    camera: Camera,
    metadata: FThetaCameraMetadata,
    interpolation: _InterpolationMode,
) -> Camera:
    """Convert an f-theta camera to an undistorted pinhole camera."""
    assert metadata.intrinsics is not None
    fov_x = metadata.fov_x
    fov_y = metadata.fov_y
    assert fov_x is not None and fov_y is not None, "Cannot compute FOV from f-theta intrinsics."

    # Clamp FOV to < 180 degrees (pinhole can't represent >= 180)
    max_fov = np.deg2rad(170.0)
    fov_x = min(fov_x, max_fov)
    fov_y = min(fov_y, max_fov)

    target_metadata = _build_pinhole_from_fov(
        fov_x=fov_x,
        fov_y=fov_y,
        width=metadata.width,
        height=metadata.height,
        camera_id=metadata.camera_id,
        camera_name=metadata.camera_name,
        camera_to_imu_se3=metadata.camera_to_imu_se3,
    )
    image_remapped = _remap_to_pinhole(camera.image, metadata, target_metadata, interpolation)

    result = Camera(
        metadata=target_metadata,
        image=image_remapped,
        camera_to_global_se3=camera.camera_to_global_se3,
        timestamp=camera.timestamp,
    )
    return result


def _undistort_fisheye_mei_camera(
    camera: Camera,
    metadata: FisheyeMEICameraMetadata,
    interpolation: _InterpolationMode,
) -> Camera:
    """Convert a fisheye MEI camera to an undistorted pinhole camera."""
    assert metadata.projection is not None
    assert metadata.mirror_parameter is not None

    # Approximate FOV from projection parameters
    u0 = metadata.projection.u0
    v0 = metadata.projection.v0
    gamma1 = metadata.projection.gamma1
    gamma2 = metadata.projection.gamma2

    half_w = max(u0, metadata.width - u0)
    half_h = max(v0, metadata.height - v0)
    fov_x = 2.0 * np.arctan(half_w / gamma1)
    fov_y = 2.0 * np.arctan(half_h / gamma2)

    max_fov = np.deg2rad(170.0)
    fov_x = min(fov_x, max_fov)
    fov_y = min(fov_y, max_fov)

    target_metadata = _build_pinhole_from_fov(
        fov_x=fov_x,
        fov_y=fov_y,
        width=metadata.width,
        height=metadata.height,
        camera_id=metadata.camera_id,
        camera_name=metadata.camera_name,
        camera_to_imu_se3=metadata.camera_to_imu_se3,
    )
    image_remapped = _remap_to_pinhole(camera.image, metadata, target_metadata, interpolation)

    result = Camera(
        metadata=target_metadata,
        image=image_remapped,
        camera_to_global_se3=camera.camera_to_global_se3,
        timestamp=camera.timestamp,
    )
    return result
