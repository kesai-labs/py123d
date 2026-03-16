from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, List, Optional

if TYPE_CHECKING:
    import sqlite3

import numpy as np

from py123d.common.utils.dependencies import check_dependencies
from py123d.datatypes.detections.box_detections import BoxDetectionAttributes, BoxDetectionSE3
from py123d.geometry import BoundingBoxSE3, EulerAngles, PoseSE3, Vector3D
from py123d.geometry.utils.constants import DEFAULT_PITCH, DEFAULT_ROLL
from py123d.geometry.utils.rotation_utils import slerp_quaternion_arrays
from py123d.parser.nuplan.utils.nuplan_constants import NUPLAN_DETECTION_NAME_DICT

check_dependencies(modules=["nuplan"], optional_name="nuplan")
from nuplan.database.nuplan_db.query_session import execute_many, execute_one


def get_box_detections_for_lidarpc_token_from_db(log_file: str, token: str) -> List[BoxDetectionSE3]:
    """Gets the box detections for a given Lidar point cloud token from the NuPlan database."""

    query = """
        SELECT  c.name AS category_name,
                lb.x,
                lb.y,
                lb.z,
                lb.yaw,
                lb.width,
                lb.length,
                lb.height,
                lb.vx,
                lb.vy,
                lb.vz,
                lb.token,
                lb.track_token,
                lp.timestamp
        FROM lidar_box AS lb
        INNER JOIN track AS t
            ON t.token = lb.track_token
        INNER JOIN category AS c
            ON c.token = t.category_token
        INNER JOIN lidar_pc AS lp
            ON lp.token = lb.lidar_pc_token
        WHERE lp.token = ?
    """

    box_detections: List[BoxDetectionSE3] = []

    for row in execute_many(query, (bytearray.fromhex(token),), log_file):
        quaternion = EulerAngles(roll=DEFAULT_ROLL, pitch=DEFAULT_PITCH, yaw=row["yaw"]).quaternion
        bounding_box = BoundingBoxSE3(
            center_se3=PoseSE3(
                x=row["x"],
                y=row["y"],
                z=row["z"],
                qw=quaternion.qw,
                qx=quaternion.qx,
                qy=quaternion.qy,
                qz=quaternion.qz,
            ),
            length=row["length"],
            width=row["width"],
            height=row["height"],
        )
        box_detection = BoxDetectionSE3(
            attributes=BoxDetectionAttributes(
                label=NUPLAN_DETECTION_NAME_DICT[row["category_name"]],
                track_token=row["track_token"].hex(),
            ),
            bounding_box_se3=bounding_box,
            velocity_3d=Vector3D(x=row["vx"], y=row["vy"], z=row["vz"]),
        )
        box_detections.append(box_detection)

    return box_detections


def _row_to_pose_se3(row: sqlite3.Row) -> PoseSE3:
    """Converts a database row with x, y, z, qw, qx, qy, qz columns to a :class:`PoseSE3`."""
    return PoseSE3(x=row["x"], y=row["y"], z=row["z"], qw=row["qw"], qx=row["qx"], qy=row["qy"], qz=row["qz"])


def get_interpolated_ego_pose_from_db(log_file: str, timestamp_us: int) -> PoseSE3:
    """Interpolates the ego pose at an arbitrary timestamp by querying the two bracketing ego poses from the database.

    Uses linear interpolation for position and SLERP for orientation. If the timestamp falls exactly on an ego pose
    or outside the range, the nearest boundary pose is returned.

    :param log_file: Path to the nuPlan ``.db`` log file.
    :param timestamp_us: Target timestamp in microseconds.
    :return: Interpolated ego pose as a :class:`PoseSE3`.
    """
    _EGO_POSE_COLS = "ep.x, ep.y, ep.z, ep.qw, ep.qx, ep.qy, ep.qz, ep.timestamp"

    before_row: Optional[sqlite3.Row] = execute_one(
        f"SELECT {_EGO_POSE_COLS} FROM ego_pose AS ep WHERE ep.timestamp <= ? ORDER BY ep.timestamp DESC LIMIT 1",
        (timestamp_us,),
        log_file,
    )
    after_row: Optional[sqlite3.Row] = execute_one(
        f"SELECT {_EGO_POSE_COLS} FROM ego_pose AS ep WHERE ep.timestamp > ? ORDER BY ep.timestamp ASC LIMIT 1",
        (timestamp_us,),
        log_file,
    )

    # Resolve which row(s) to use: boundary clamp, exact match, or interpolation
    if before_row is None and after_row is None:
        raise ValueError(f"No ego poses found in log file {log_file}")

    if before_row is None:
        assert after_row is not None
        pose = _row_to_pose_se3(after_row)
    elif after_row is None or before_row["timestamp"] == timestamp_us:
        pose = _row_to_pose_se3(before_row)
    else:
        # Interpolate between the two bracketing poses
        t0 = int(before_row["timestamp"])
        t1 = int(after_row["timestamp"])
        alpha = float(timestamp_us - t0) / float(t1 - t0)

        pos_before = np.array([before_row["x"], before_row["y"], before_row["z"]], dtype=np.float64)
        pos_after = np.array([after_row["x"], after_row["y"], after_row["z"]], dtype=np.float64)
        position = (1.0 - alpha) * pos_before + alpha * pos_after

        q_before = np.array([before_row["qw"], before_row["qx"], before_row["qy"], before_row["qz"]], dtype=np.float64)
        q_after = np.array([after_row["qw"], after_row["qx"], after_row["qy"], after_row["qz"]], dtype=np.float64)
        quaternion = slerp_quaternion_arrays(q_before, q_after, np.array(alpha))

        pose = PoseSE3.from_R_t(rotation=quaternion, translation=position)

    return pose


# ------------------------------------------------------------------------------------------------------------------
# Native-rate iterators for async conversion
# ------------------------------------------------------------------------------------------------------------------


def iter_all_ego_poses_from_db(log_file: str) -> Iterator:
    """Yields all ego pose rows sorted by timestamp."""
    query = """
        SELECT ep.x, ep.y, ep.z, ep.qw, ep.qx, ep.qy, ep.qz,
               ep.vx, ep.vy, ep.vz,
               ep.acceleration_x, ep.acceleration_y, ep.acceleration_z,
               ep.angular_rate_x, ep.angular_rate_y, ep.angular_rate_z,
               ep.timestamp
        FROM ego_pose AS ep
        ORDER BY ep.timestamp
    """
    yield from execute_many(query, (), log_file)


def iter_all_lidar_pc_from_db(log_file: str) -> Iterator:
    """Yields all lidar_pc rows (token, filename, timestamp) sorted by timestamp."""
    query = """
        SELECT lpc.token, lpc.filename, lpc.timestamp
        FROM lidar_pc AS lpc
        ORDER BY lpc.timestamp
    """
    yield from execute_many(query, (), log_file)


def iter_all_images_from_db(log_file: str) -> Iterator:
    """Yields all image rows with camera channel, sorted by timestamp."""
    query = """
        SELECT i.filename_jpg, i.timestamp, c.channel
        FROM image AS i
        INNER JOIN camera AS c ON c.token = i.camera_token
        ORDER BY i.timestamp
    """
    yield from execute_many(query, (), log_file)


def iter_all_box_detections_from_db(log_file: str) -> Iterator:
    """Yields all box detection rows with lidar_pc timestamp, sorted by timestamp."""
    query = """
        SELECT c.name AS category_name,
               lb.x, lb.y, lb.z, lb.yaw,
               lb.width, lb.length, lb.height,
               lb.vx, lb.vy, lb.vz,
               lb.token, lb.track_token,
               lp.timestamp
        FROM lidar_box AS lb
        INNER JOIN track AS t
            ON t.token = lb.track_token
        INNER JOIN category AS c
            ON c.token = t.category_token
        INNER JOIN lidar_pc AS lp
            ON lp.token = lb.lidar_pc_token
        ORDER BY lp.timestamp
    """
    yield from execute_many(query, (), log_file)


def iter_all_traffic_lights_from_db(log_file: str) -> Iterator:
    """Yields all traffic light status rows with lidar_pc timestamp, sorted by timestamp."""
    query = """
        SELECT tls.lane_connector_id, tls.status, lpc.timestamp
        FROM traffic_light_status AS tls
        INNER JOIN lidar_pc AS lpc
            ON lpc.token = tls.lidar_pc_token
        ORDER BY lpc.timestamp
    """
    yield from execute_many(query, (), log_file)
