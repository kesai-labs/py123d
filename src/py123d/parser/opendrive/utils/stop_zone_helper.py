import hashlib
import logging
from typing import Dict, List, Optional

import numpy as np
from shapely import MultiPolygon, Polygon, union_all

from py123d.datatypes.map_objects import StopZone, StopZoneType
from py123d.geometry.polyline import Polyline3D
from py123d.parser.opendrive.utils.lane_helper import OpenDriveLaneHelper
from py123d.parser.opendrive.utils.signal_helper import OpenDriveSignalHelper

logger = logging.getLogger(__name__)

STOP_ZONE_DEPTH = 0.5

SIGNAL_TYPE_MAP = {
    "1000001": StopZoneType.TRAFFIC_LIGHT,
    "206": StopZoneType.STOP_SIGN,
    "205": StopZoneType.YIELD_SIGN,
}


def _signal_type_to_stop_zone_type(signal: OpenDriveSignalHelper) -> StopZoneType:
    return SIGNAL_TYPE_MAP.get(signal.xodr_signal.type, StopZoneType.UNKNOWN)


def _get_stop_zone_s_range(helper: OpenDriveLaneHelper) -> tuple:
    """Get the (start_s, end_s) range for a stop zone on a lane."""
    travels_in_s = helper.id < 0
    if travels_in_s:
        start_s = helper.s_range[0]
        end_s = start_s + STOP_ZONE_DEPTH
    else:
        end_s = helper.s_range[1]
        start_s = end_s - STOP_ZONE_DEPTH
    start_s = np.clip(start_s, helper.s_range[0], helper.s_range[1])
    end_s = np.clip(end_s, helper.s_range[0], helper.s_range[1])
    return float(start_s), float(end_s)


def _lane_rectangle_2d(helper: OpenDriveLaneHelper) -> Optional[Polygon]:
    """Create a small 2D rectangle at the start of a lane (STOP_ZONE_DEPTH wide)."""
    start_s, end_s = _get_stop_zone_s_range(helper)

    # Batch all 4 interpolation calls: 2 s-values x 2 boundaries
    s_arr = np.array([start_s, end_s], dtype=np.float64) - helper.s_range[0]
    t_arr = np.zeros(2, dtype=np.float64)
    end_mask = np.array([False, False])
    inner_pts = helper.inner_boundary.interpolate_3d_batch(s_arr, t_arr, end_mask)
    outer_pts = helper.outer_boundary.interpolate_3d_batch(s_arr, t_arr, end_mask)

    coords_2d = [
        (inner_pts[0, 0], inner_pts[0, 1]),
        (outer_pts[0, 0], outer_pts[0, 1]),
        (outer_pts[1, 0], outer_pts[1, 1]),
        (inner_pts[1, 0], inner_pts[1, 1]),
    ]
    poly = Polygon(coords_2d)
    if not poly.is_valid or poly.area < 1e-6:
        return None
    return poly


def _create_stop_zone_outline(
    helpers: List[OpenDriveLaneHelper],
) -> Optional[Polyline3D]:
    """Create stop zone outline by merging per-lane rectangles with shapely.

    Each lane produces a small rectangle, union_all merges them.
    If result is MultiPolygon, pick the largest. Average Z across all lane corners.
    """
    polys = [_lane_rectangle_2d(h) for h in helpers]
    polys = [p for p in polys if p is not None]
    if not polys:
        return None

    # Close hairline gaps between adjacent lane rectangles; a MultiPolygon here would
    # otherwise truncate the stop zone to its largest piece.
    merged = union_all([p.buffer(0.25, join_style=2) for p in polys]).buffer(-0.25, join_style=2)

    if isinstance(merged, MultiPolygon):
        merged = max(merged.geoms, key=lambda g: g.area)

    if not isinstance(merged, Polygon) or merged.is_empty:
        return None

    # Collect Z from all lane corners for averaging using batch calls
    all_z = []
    for h in helpers:
        start_s, end_s = _get_stop_zone_s_range(h)
        s_arr = np.array([start_s, end_s], dtype=np.float64) - h.s_range[0]
        t_arr = np.zeros(2, dtype=np.float64)
        end_mask = np.array([False, False])
        inner_pts = h.inner_boundary.interpolate_3d_batch(s_arr, t_arr, end_mask)
        outer_pts = h.outer_boundary.interpolate_3d_batch(s_arr, t_arr, end_mask)
        all_z.extend(inner_pts[:, 2].tolist())
        all_z.extend(outer_pts[:, 2].tolist())
    avg_z = float(np.mean(all_z))

    # Extract exterior coords from merged polygon, add Z
    merged = merged.buffer(0.01).simplify(0.01, preserve_topology=True).buffer(-0.01)
    xy = np.array(merged.exterior.coords)
    z = np.full((xy.shape[0], 1), avg_z)
    corners_3d = np.hstack([xy, z])

    return Polyline3D.from_array(corners_3d)


MAX_ABSORB_ROUNDS = 16
ABSORB_TOUCH_DISTANCE = 0.4
ABSORB_MIN_DIRECTION_DOT = 0.7


def _lane_entry_direction(helper: OpenDriveLaneHelper) -> Optional[np.ndarray]:
    """Unit travel direction of a lane at its stop-zone end."""
    start_s, end_s = _get_stop_zone_s_range(helper)
    s_arr = np.array([start_s, end_s], dtype=np.float64) - helper.s_range[0]
    t_arr = np.zeros(2, dtype=np.float64)
    end_mask = np.array([False, False])
    inner_pts = helper.inner_boundary.interpolate_3d_batch(s_arr, t_arr, end_mask)
    outer_pts = helper.outer_boundary.interpolate_3d_batch(s_arr, t_arr, end_mask)
    centers = (inner_pts[:, :2] + outer_pts[:, :2]) / 2.0
    direction = centers[1] - centers[0]
    if helper.id > 0:
        direction = -direction
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return None
    return direction / norm


def _collect_lane_entries(lane_helper_dict: Dict[str, OpenDriveLaneHelper]) -> Dict[str, tuple]:
    """Maps driving lane id -> (entry rectangle, travel direction) at the stop-zone end."""
    entries: Dict[str, tuple] = {}
    for lane_id, helper in lane_helper_dict.items():
        if helper.type != "driving":
            continue
        rectangle = _lane_rectangle_2d(helper)
        direction = _lane_entry_direction(helper)
        if rectangle is None or direction is None:
            continue
        entries[lane_id] = (rectangle, direction)
    return entries


def _absorb_adjacent_entry_lanes(
    signal_lane_ids: List[str],
    lane_entries: Dict[str, tuple],
) -> List[str]:
    """Extends signal validity lanes with laterally adjacent same-direction driving lanes.
    Signal validity records often cover only part of a junction entry; touching entry rectangles
    with matching travel direction belong to the same stop line.
    """
    selected = [lane_id for lane_id in signal_lane_ids if lane_id in lane_entries]
    if not selected:
        return signal_lane_ids
    zone = union_all([lane_entries[lane_id][0] for lane_id in selected]).buffer(ABSORB_TOUCH_DISTANCE)
    directions = [lane_entries[lane_id][1] for lane_id in selected]
    selected_set = set(selected)

    for _ in range(MAX_ABSORB_ROUNDS):
        changed = False
        for lane_id, (rectangle, direction) in lane_entries.items():
            if lane_id in selected_set or not rectangle.intersects(zone):
                continue
            if max(float(np.dot(direction, ref)) for ref in directions) < ABSORB_MIN_DIRECTION_DOT:
                continue
            selected_set.add(lane_id)
            selected.append(lane_id)
            directions.append(direction)
            zone = zone.union(rectangle.buffer(ABSORB_TOUCH_DISTANCE))
            changed = True
        if not changed:
            break
    return selected


def create_stop_zones_from_signals(
    signal_dict: Dict[int, OpenDriveSignalHelper],
    lane_helper_dict: Dict[str, OpenDriveLaneHelper],
) -> Dict[int, StopZone]:
    """Create StopZone objects from signal helpers. One signal_id = one StopZone.

    :param signal_dict: Dictionary of signal helpers keyed by signal_id
    :param lane_helper_dict: Dictionary of lane helpers keyed by lane ID
    :return: Dictionary of StopZone objects keyed by signal_id
    """
    stop_zones: Dict[int, StopZone] = {}
    lane_entries = _collect_lane_entries(lane_helper_dict)

    for signal_id, signal_helper in signal_dict.items():
        stop_zone_type = _signal_type_to_stop_zone_type(signal_helper)
        if stop_zone_type == StopZoneType.UNKNOWN:
            continue

        if not signal_helper.lane_ids:
            continue

        signal_lane_ids = _absorb_adjacent_entry_lanes(list(signal_helper.lane_ids), lane_entries)
        helpers = [lane_helper_dict[lid] for lid in signal_lane_ids if lid in lane_helper_dict]
        # Filter out lanes with zero-area rectangles. This can happen when a lane has
        # near-zero width at the stop zone position (e.g. very short lanes or lane tapers).
        helpers = [h for h in helpers if _lane_rectangle_2d(h) is not None]
        if not helpers:
            continue

        outline = _create_stop_zone_outline(helpers)
        if outline is None:
            continue

        object_id = int(hashlib.md5(str(signal_id).encode("utf-8")).hexdigest(), 16) & 0x7FFFFFFF

        stop_zones[signal_id] = StopZone(
            object_id=object_id,
            stop_zone_type=stop_zone_type,
            outline=outline,
            lane_ids=[h.lane_id for h in helpers],
            intersection_id=signal_helper.junction_id,
            phase_idx=signal_helper.phase_idx,
        )

    return stop_zones
