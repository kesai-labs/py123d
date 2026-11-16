import logging
from typing import Dict, List

import numpy as np

from py123d.datatypes.map_objects import StopZone, StopZoneType, TrafficLight
from py123d.geometry import EulerAngles, PoseSE3, Quaternion
from py123d.parser.opendrive.utils.lane_helper import OpenDriveLaneHelper
from py123d.parser.opendrive.utils.signal_helper import OpenDriveSignalHelper, get_signal_position_3d
from py123d.parser.opendrive.utils.stop_zone_helper import (
    _absorb_adjacent_entry_lanes,
    _collect_lane_entries,
    _signal_type_to_stop_zone_type,
)
from py123d.parser.opendrive.xodr_parser.road import XODRRoad

logger = logging.getLogger(__name__)


def create_traffic_lights_from_signals(
    signal_dict: Dict[int, OpenDriveSignalHelper],
    lane_helper_dict: Dict[str, OpenDriveLaneHelper],
    road_dict: Dict[int, XODRRoad],
    stop_zones: Dict[int, StopZone],
) -> List[TrafficLight]:
    """Create TrafficLight objects from the traffic light signals. One signal = one TrafficLight.

    The signal position is the s/t/zOffset of the ``<signal>`` on its road, i.e. where the map places the signal
    (for CARLA towns the signal post). The housing faces the driving direction of the controlled lanes at their
    stop line; the road heading is used when no controlled lane is known.

    :param signal_dict: Dictionary of signal helpers keyed by signal_id
    :param lane_helper_dict: Dictionary of lane helpers keyed by lane ID
    :param road_dict: Dictionary of roads keyed by road id
    :param stop_zones: Stop zones created from the same signals, keyed by signal_id
    :return: List of TrafficLight objects
    """
    traffic_lights: List[TrafficLight] = []
    lane_entries = _collect_lane_entries(lane_helper_dict)

    for signal_id, signal_helper in signal_dict.items():
        if signal_helper.is_derived or _signal_type_to_stop_zone_type(signal_helper) != StopZoneType.TRAFFIC_LIGHT:
            continue
        road = road_dict.get(signal_helper.road_id) if signal_helper.road_id is not None else None
        if road is None:
            logger.debug(f"Skipping traffic light {signal_id}: defining road not found")
            continue

        signal = signal_helper.xodr_signal
        x, y, z_bottom, road_heading = get_signal_position_3d(signal, road)
        lane_ids = _absorb_adjacent_entry_lanes(list(signal_helper.lane_ids), lane_entries)
        directions = [lane_entries[lane_id][1] for lane_id in lane_ids if lane_id in lane_entries]
        if directions:
            mean_direction = np.sum(np.asarray(directions, dtype=np.float64), axis=0)
            yaw = float(np.arctan2(mean_direction[1], mean_direction[0]))
        else:
            yaw = road_heading if signal.orientation == "+" else road_heading + np.pi

        height = signal.height if signal.height is not None and signal.height > 0.0 else TrafficLight.DEFAULT_HEIGHT
        width = signal.width if signal.width is not None and signal.width > 0.0 else None
        quaternion = Quaternion.from_euler_angles(EulerAngles(0.0, 0.0, yaw))
        # The signal position is the bottom edge of the housing; the pose is its center.
        pose_se3 = PoseSE3(x, y, z_bottom + height / 2, quaternion.qw, quaternion.qx, quaternion.qy, quaternion.qz)
        traffic_lights.append(
            TrafficLight.from_pose(
                object_id=signal_id,
                pose_se3=pose_se3,
                width=width,
                height=height,
                lane_ids=lane_ids,
                stop_zone_ids=[stop_zones[signal_id].object_id] if signal_id in stop_zones else [],
                intersection_id=signal_helper.junction_id,
            )
        )
    return traffic_lights
