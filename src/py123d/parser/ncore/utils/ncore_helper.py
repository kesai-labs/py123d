"""Helpers for parsing NVIDIA NCore V4 data into py123d primitives."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Tuple

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from py123d.datatypes.sensors.ftheta_camera import FThetaIntrinsics
from py123d.geometry.pose import PoseSE3
from py123d.parser.physical_ai_av.utils.physical_ai_av_helper import (  # noqa: F401  (re-exported)
    find_closest_index,
    quat_scalar_last_to_pose_se3,
)

logger = logging.getLogger(__name__)

# Tolerances for NCore's linear_cde = [c, d, e] (affine sensor→image term).
# Real Hyperion 8 calibration ships c ≈ 1.0016 (sub-percent anisotropic pixel aspect).
# We reject non-trivial off-diagonal d/e, and absorb c into the polynomials as a geometric-mean
# approximation when it's close enough to unity.
_LINEAR_CDE_OFF_DIAG_TOL = 1e-3  # max |d| and |e| we accept before raising
_LINEAR_CDE_C_TOL_WARN = 1e-4  # log a warning when |c - 1| exceeds this

if TYPE_CHECKING:
    from ncore.data import CuboidTrackObservation, FThetaCameraModelParameters


def ftheta_params_to_intrinsics(
    params: "FThetaCameraModelParameters",
) -> Tuple[FThetaIntrinsics, int, int]:
    """Convert NCore's :class:`~ncore.data.FThetaCameraModelParameters` to py123d primitives.

    NCore's FTheta model has an extra affine ``linear_cde = [c, d, e]`` mapping sensor-mm →
    pixel coordinates via ``[[c, d], [e, 1]]``. py123d's :class:`FThetaIntrinsics` is radially
    symmetric and has no direct slot for this term. Two real-world cases:

    - ``d ≈ e ≈ 0`` (the only case seen on Hyperion 8 calibration so far): the linear term is
      a pure anisotropic pixel aspect ratio. We absorb ``c`` into the polynomials as a
      geometric-mean isotropic approximation — ``fw_poly *= sqrt(c)`` and ``bw_poly /= sqrt(c)``.
      Residual anisotropy after this absorption is ``|sqrt(c) - 1|`` on each axis (sub-pixel
      at the image edge for ``c`` within a few percent of 1).
    - ``|d|`` or ``|e|`` non-trivial: we refuse — an off-diagonal shear is not expressible in
      an isotropic FTheta model and silent acceptance would miscalibrate every camera.

    :param params: NCore FTheta camera model parameters.
    :return: Tuple of ``(FThetaIntrinsics, width, height)`` where width/height come from the
        NCore ``resolution`` field.
    """
    linear_cde = np.asarray(params.linear_cde, dtype=np.float64)
    c_scale = float(linear_cde[0])
    d_shear = float(linear_cde[1])
    e_shear = float(linear_cde[2])

    if abs(d_shear) > _LINEAR_CDE_OFF_DIAG_TOL or abs(e_shear) > _LINEAR_CDE_OFF_DIAG_TOL:
        raise ValueError(
            "NCore camera reports an FTheta linear_cde with non-trivial off-diagonal terms "
            f"(c={c_scale!r}, d={d_shear!r}, e={e_shear!r}). py123d's isotropic FTheta model "
            "cannot represent this shear — conversion would be silently miscalibrated."
        )
    if c_scale <= 0:
        raise ValueError(
            f"NCore camera reports a non-positive FTheta linear_cde[0]={c_scale!r}; this is not a valid calibration."
        )

    fw_poly = np.asarray(params.angle_to_pixeldist_poly, dtype=np.float64)
    bw_poly = np.asarray(params.pixeldist_to_angle_poly, dtype=np.float64)

    if abs(c_scale - 1.0) > _LINEAR_CDE_C_TOL_WARN:
        sqrt_c = np.sqrt(c_scale)
        logger.warning(
            "Absorbing FTheta linear_cde[0]=%.6f into polynomials as geometric-mean (sqrt(c)=%.6f) "
            "isotropic approximation. Residual per-axis anisotropy: %.4f%%.",
            c_scale,
            sqrt_c,
            abs(sqrt_c - 1.0) * 100.0,
        )
        fw_poly = fw_poly * sqrt_c
        bw_poly = bw_poly / sqrt_c

    intrinsics = FThetaIntrinsics(
        cx=float(params.principal_point[0]),
        cy=float(params.principal_point[1]),
        fw_poly=fw_poly,
        bw_poly=bw_poly,
    )
    width = int(params.resolution[0])
    height = int(params.resolution[1])
    return intrinsics, width, height


def cuboid_bbox_to_rig_se3_array(
    obs: "CuboidTrackObservation",
    reference_to_rig: PoseSE3,
) -> npt.NDArray[np.float64]:
    """Convert one NCore cuboid observation into py123d's 10-dim SE3 bbox array, in rig frame.

    The NCore :class:`~ncore.data.BBox3` carries centroid (in the observation's
    reference frame), dimensions, and an intrinsic XYZ Euler rotation. We convert to
    py123d's ``[x, y, z, qw, qx, qy, qz, length, width, height]`` representation and
    transform from ``reference_frame`` → ``rig`` via the provided static pose.

    The downstream rig→world transform at ``obs.reference_frame_timestamp_us`` is
    applied in the parser, mirroring the physical_ai_av per-detection pattern.

    :param obs: NCore cuboid track observation.
    :param reference_to_rig: Static pose mapping the observation's reference frame to rig;
        pass :meth:`~py123d.geometry.PoseSE3.identity` when the observation is already in rig.
    :return: A length-10 array indexed by :class:`~py123d.geometry.BoundingBoxSE3Index`.
    """
    from py123d.geometry import BoundingBoxSE3Index
    from py123d.geometry.transform import rel_to_abs_se3_array

    bbox3 = obs.bbox3
    # NCore rotates via intrinsic XYZ Euler. scipy 'XYZ' is intrinsic (uppercase).
    quat_scalar_last = Rotation.from_euler("XYZ", list(bbox3.rot), degrees=False).as_quat()

    pose_rot_ref = np.zeros(len(BoundingBoxSE3Index), dtype=np.float64)
    pose_rot_ref[BoundingBoxSE3Index.XYZ] = bbox3.centroid
    pose_rot_ref[BoundingBoxSE3Index.QUATERNION] = [
        quat_scalar_last[3],  # qw
        quat_scalar_last[0],  # qx
        quat_scalar_last[1],  # qy
        quat_scalar_last[2],  # qz
    ]
    pose_rot_ref[BoundingBoxSE3Index.EXTENT] = bbox3.dim

    # Transform pose portion (xyz+quat) from reference frame into rig.
    pose_rot_ref[BoundingBoxSE3Index.SE3] = rel_to_abs_se3_array(
        origin=reference_to_rig,
        pose_se3_array=pose_rot_ref[BoundingBoxSE3Index.SE3].reshape(1, -1),
    )
    return pose_rot_ref
