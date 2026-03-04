from typing import List, Tuple

from py123d.common.utils.dependencies import check_dependencies
from py123d.conversion.datasets.nuplan.utils.nuplan_constants import NUPLAN_DETECTION_NAME_DICT
from py123d.datatypes.detections.box_detections import BoxDetectionAttributes, BoxDetectionSE3
from py123d.geometry import BoundingBoxSE3, EulerAngles, PoseSE3, Vector3D
from py123d.geometry.utils.constants import DEFAULT_PITCH, DEFAULT_ROLL

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
            metadata=BoxDetectionAttributes(
                label=NUPLAN_DETECTION_NAME_DICT[row["category_name"]],
                track_token=row["track_token"].hex(),
            ),
            bounding_box_se3=bounding_box,
            velocity_3d=Vector3D(x=row["vx"], y=row["vy"], z=row["vz"]),
        )
        box_detections.append(box_detection)

    return box_detections


def get_ego_pose_for_timestamp_from_db(log_file: str, timestamp: int) -> PoseSE3:
    """Gets the ego pose for a given timestamp from the NuPlan database."""

    query = """
        SELECT  ep.x,
                ep.y,
                ep.z,
                ep.qw,
                ep.qx,
                ep.qy,
                ep.qz,
                ep.timestamp,
                ep.vx,
                ep.vy,
                ep.acceleration_x,
                ep.acceleration_y
        FROM ego_pose AS ep
        ORDER BY ABS(ep.timestamp - ?)
        LIMIT 1
    """

    row = execute_one(query, (timestamp,), log_file)
    assert row is not None, f"No ego pose found for timestamp {timestamp} in log file {log_file}"
    return PoseSE3(x=row["x"], y=row["y"], z=row["z"], qw=row["qw"], qx=row["qx"], qy=row["qy"], qz=row["qz"])


def get_nearest_ego_pose_for_timestamp_from_db(
    log_file: str,
    timestamp: int,
    tokens: List[str],
    lookahead_window_us: int = 50000,
    lookback_window_us: int = 50000,
) -> Tuple[List[PoseSE3], List[int]]:
    """Gets the nearest ego pose for a given timestamp from the NuPlan database within a lookahead and lookback window."""

    query = f"""
        SELECT  ep.x,
                ep.y,
                ep.z,
                ep.qw,
                ep.qx,
                ep.qy,
                ep.qz,
                ep.timestamp
        FROM ego_pose AS ep
            INNER JOIN lidar_pc AS lpc
                ON  ep.timestamp <= lpc.timestamp + ?
                AND ep.timestamp >= lpc.timestamp - ?
            WHERE lpc.token IN ({("?," * len(tokens))[:-1]})
        ORDER BY ABS(ep.timestamp - ?)
        LIMIT 1
    """  # noqa: E226

    args = [lookahead_window_us, lookback_window_us]
    args += [bytearray.fromhex(t) for t in tokens]
    args += [timestamp]

    poses = []
    times = []

    for row in execute_many(query, args, log_file):
        poses.append(
            PoseSE3(x=row["x"], y=row["y"], z=row["z"], qw=row["qw"], qx=row["qx"], qy=row["qy"], qz=row["qz"])
        )
        times.append(abs(row["timestamp"] - timestamp))
    return poses, times
