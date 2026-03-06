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

    merged = union_all(polys)

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
    xy = np.array(merged.exterior.coords)
    z = np.full((xy.shape[0], 1), avg_z)
    corners_3d = np.hstack([xy, z])

    return Polyline3D.from_array(corners_3d)


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

    for signal_id, signal_helper in signal_dict.items():
        stop_zone_type = _signal_type_to_stop_zone_type(signal_helper)
        if stop_zone_type == StopZoneType.UNKNOWN:
            continue

        if not signal_helper.lane_ids:
            continue

        helpers = [lane_helper_dict[lid] for lid in signal_helper.lane_ids if lid in lane_helper_dict]
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
        )

    return stop_zones
