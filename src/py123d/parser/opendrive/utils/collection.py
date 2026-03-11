import logging
from copy import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import splev, splprep

from py123d.geometry.polyline import Polyline3D, PolylineSE2
from py123d.geometry.utils.polyline_utils import get_points_2d_yaws, offset_points_perpendicular
from py123d.parser.opendrive.utils.id_system import (
    build_lane_id,
    derive_lane_section_id,
    lane_group_id_from_lane_id,
    road_id_from_lane_group_id,
)
from py123d.parser.opendrive.utils.lane_helper import (
    OpenDriveLaneGroupHelper,
    OpenDriveLaneHelper,
    lane_section_to_lane_helpers,
)
from py123d.parser.opendrive.utils.objects_helper import OpenDriveObjectHelper, get_object_helper
from py123d.parser.opendrive.utils.signal_helper import (
    OpenDriveSignalHelper,
    get_signal_reference_helper,
)
from py123d.parser.opendrive.xodr_parser.lane import XODRRoadMark
from py123d.parser.opendrive.xodr_parser.opendrive import XODR, Junction
from py123d.parser.opendrive.xodr_parser.reference import XODRReferenceLine
from py123d.parser.opendrive.xodr_parser.road import XODRRoad

logger = logging.getLogger(__name__)


def collect_element_helpers(
    opendrive: XODR,
    interpolation_step_size: float,
    connection_distance_threshold: float,
) -> Tuple[
    Dict[int, XODRRoad],
    Dict[int, Junction],
    Dict[str, OpenDriveLaneHelper],
    Dict[str, OpenDriveLaneGroupHelper],
    Dict[int, OpenDriveObjectHelper],
    Dict[str, List[XODRRoadMark]],
    Dict[int, OpenDriveSignalHelper],
]:
    # 1. Fill the road and junction dictionaries
    road_dict: Dict[int, XODRRoad] = {road.id: road for road in opendrive.roads}
    junction_dict: Dict[int, Junction] = {junction.id: junction for junction in opendrive.junctions}

    # 2. Create lane helpers from the roads and collect center lane road marks
    lane_helper_dict: Dict[str, OpenDriveLaneHelper] = {}
    center_lane_marks_dict: Dict[str, List[XODRRoadMark]] = {}
    for road in opendrive.roads:
        # Skip roads without predecessor or successor links - (such as car parks in houses)
        if road.link.predecessor is None or road.link.successor is None:
            continue

        reference_line = XODRReferenceLine.from_plan_view(
            road.plan_view,
            road.lanes.lane_offsets,
            road.elevation_profile.elevations,
        )
        lane_section_lengths: List[float] = [ls.s for ls in road.lanes.lane_sections] + [road.length]
        for idx, lane_section in enumerate(road.lanes.lane_sections):
            lane_section_id = derive_lane_section_id(road.id, idx)
            lane_helpers_ = lane_section_to_lane_helpers(
                lane_section_id,
                lane_section,
                reference_line,
                lane_section_lengths[idx],
                lane_section_lengths[idx + 1],
                road.road_types,
                interpolation_step_size,
            )
            lane_helper_dict.update(lane_helpers_)

            # Collect center lane road marks (id=0)
            for center_lane in lane_section.center_lanes:
                if center_lane.id == 0 and center_lane.road_marks:
                    center_lane_marks_dict[lane_section_id] = center_lane.road_marks

    # 3. Update the connections and fill the lane helpers:
    # 3.1. From links of the roads
    _update_connection_from_links(lane_helper_dict, road_dict)
    # 3.2. From junctions
    _update_connection_from_junctions(lane_helper_dict, junction_dict, road_dict)
    # 3.3. Deduplicate connections
    _deduplicate_connections(lane_helper_dict)
    # 3.4. Remove invalid connections based on centerline distances
    _post_process_connections(lane_helper_dict, connection_distance_threshold)
    # 3.5. Propagate speed limits to junction lanes (they often lack <type> elements)
    _propagate_speed_limits_to_junction_lanes(lane_helper_dict, road_dict)
    # 3.6. Correct lanes with no connections
    _correct_lanes_with_no_connections(lane_helper_dict)

    # 4. Collect lane groups from lane helpers
    lane_group_helper_dict: Dict[str, OpenDriveLaneGroupHelper] = _collect_lane_groups(
        lane_helper_dict, junction_dict, road_dict
    )

    # 5. Collect objects, i.e. crosswalks
    crosswalk_dict = _collect_crosswalks(opendrive)

    # 6. Collect signals
    signal_dict = _collect_signals(opendrive)

    return (
        road_dict,
        junction_dict,
        lane_helper_dict,
        lane_group_helper_dict,
        crosswalk_dict,
        center_lane_marks_dict,
        signal_dict,
    )


def _update_connection_from_links(
    lane_helper_dict: Dict[str, OpenDriveLaneHelper], road_dict: Dict[int, XODRRoad]
) -> None:
    """
    Uses the links of the roads to update the connections between lane helpers.
    :param lane_helper_dict: Dictionary of lane helpers indexed by lane id.
    :param road_dict: Dictionary of roads indexed by road id.
    """

    for lane_id in lane_helper_dict.keys():  # noqa: PLC0206
        road_idx, lane_section_idx, _, lane_idx = lane_id.split("_")
        road_idx, lane_section_idx, lane_idx = int(road_idx), int(lane_section_idx), int(lane_idx)

        road = road_dict[road_idx]
        is_positive_lane = lane_idx > 0

        successor_lane_idx = lane_helper_dict[lane_id].open_drive_lane.successor
        if successor_lane_idx is not None:
            successor_lane_id: Optional[str] = None

            # Not the last lane section -> Next lane section in same road
            if lane_section_idx < road.lanes.last_lane_section_idx:
                successor_lane_id = build_lane_id(
                    road_idx,
                    lane_section_idx + 1,
                    successor_lane_idx,
                )

            # Last lane section -> Next road in first lane section
            # Try to get next road
            elif road.link.successor is not None and road.link.successor.element_type != "junction":
                successor_road = road_dict[road.link.successor.element_id]
                successor_lane_section_idx = (
                    0 if road.link.successor.contact_point == "start" else successor_road.lanes.last_lane_section_idx
                )

                successor_lane_id = build_lane_id(
                    successor_road.id,
                    successor_lane_section_idx,
                    successor_lane_idx,
                )

            # assert successor_lane_id in lane_helper_dict.keys()
            if successor_lane_id is None or successor_lane_id not in lane_helper_dict.keys():
                pass  # No valid successor, continue to predecessor check
            elif is_positive_lane:
                # Positive lanes travel opposite to s, so s-successor is traffic-predecessor
                lane_helper_dict[lane_id].predecessor_lane_ids.append(successor_lane_id)
                lane_helper_dict[successor_lane_id].successor_lane_ids.append(lane_id)
            else:
                # Negative lanes travel in s direction, so s-successor is traffic-successor
                lane_helper_dict[lane_id].successor_lane_ids.append(successor_lane_id)
                lane_helper_dict[successor_lane_id].predecessor_lane_ids.append(lane_id)

        predecessor_lane_idx = lane_helper_dict[lane_id].open_drive_lane.predecessor
        if predecessor_lane_idx is not None:
            predecessor_lane_id: Optional[str] = None

            # Not the first lane section -> Previous lane section in same road
            if lane_section_idx > 0:
                predecessor_lane_id = build_lane_id(
                    road_idx,
                    lane_section_idx - 1,
                    predecessor_lane_idx,
                )

            # First lane section -> Previous road
            # Try to get previous road
            elif road.link.predecessor is not None and road.link.predecessor.element_type != "junction":
                predecessor_road = road_dict[road.link.predecessor.element_id]
                predecessor_lane_section_idx = (
                    0
                    if road.link.predecessor.contact_point == "start"
                    else predecessor_road.lanes.last_lane_section_idx
                )

                predecessor_lane_id = build_lane_id(
                    predecessor_road.id,
                    predecessor_lane_section_idx,
                    predecessor_lane_idx,
                )

            # assert predecessor_lane_id in lane_helper_dict.keys()
            if predecessor_lane_id is None or predecessor_lane_id not in lane_helper_dict.keys():
                pass  # No valid predecessor
            elif is_positive_lane:
                # Positive lanes travel opposite to s, so s-predecessor is traffic-successor
                lane_helper_dict[lane_id].successor_lane_ids.append(predecessor_lane_id)
                lane_helper_dict[predecessor_lane_id].predecessor_lane_ids.append(lane_id)
            else:
                # Negative lanes travel in s direction, so s-predecessor is traffic-predecessor
                lane_helper_dict[lane_id].predecessor_lane_ids.append(predecessor_lane_id)
                lane_helper_dict[predecessor_lane_id].successor_lane_ids.append(lane_id)


def _update_connection_from_junctions(
    lane_helper_dict: Dict[str, OpenDriveLaneHelper],
    junction_dict: Dict[int, Junction],
    road_dict: Dict[int, XODRRoad],
) -> None:
    """
    Helper function to update the lane connections based on junctions.
    :param lane_helper_dict: Dictionary mapping lane ids to their helper objects.
    :param junction_dict: Dictionary mapping junction ids to their objects.
    :param road_dict: Dictionary mapping road ids to their objects.
    :raises ValueError: If a connection is invalid.
    """

    for junction_idx, junction in junction_dict.items():
        for connection in junction.connections:
            incoming_road = road_dict[connection.incoming_road]
            connecting_road = road_dict[connection.connecting_road]

            for lane_link in connection.lane_links:
                incoming_lane_id: Optional[str] = None
                connecting_lane_id: Optional[str] = None

                if connection.contact_point == "start":
                    incoming_lane_section_idx = incoming_road.lanes.last_lane_section_idx if lane_link.start < 0 else 0
                    incoming_lane_id = build_lane_id(incoming_road.id, incoming_lane_section_idx, lane_link.start)
                    connecting_lane_id = build_lane_id(connecting_road.id, 0, lane_link.end)
                elif connection.contact_point == "end":
                    incoming_lane_id = build_lane_id(incoming_road.id, 0, lane_link.start)
                    connecting_lane_id = build_lane_id(
                        connecting_road.id,
                        connecting_road.lanes.last_lane_section_idx,
                        lane_link.end,
                    )
                else:
                    raise ValueError(f"Unknown contact point: {connection.contact_point} in junction {junction_idx}")

                if incoming_lane_id is None or connecting_lane_id is None:
                    logger.debug(f"OpenDRIVE: Lane connection {incoming_lane_id} -> {connecting_lane_id} not valid")
                    continue
                if incoming_lane_id not in lane_helper_dict.keys() or connecting_lane_id not in lane_helper_dict.keys():
                    logger.debug(
                        f"OpenDRIVE: Lane connection {incoming_lane_id} -> {connecting_lane_id} not found in lane_helper_dict"
                    )
                    continue
                lane_helper_dict[incoming_lane_id].successor_lane_ids.append(connecting_lane_id)
                lane_helper_dict[connecting_lane_id].predecessor_lane_ids.append(incoming_lane_id)


def _deduplicate_connections(lane_helper_dict: Dict[str, OpenDriveLaneHelper]) -> None:
    """
    Helper function to deduplicate connections.
    :param lane_helper_dict: Dictionary mapping lane ids to their helper objects.
    """

    for lane_id in lane_helper_dict.keys():  # noqa: PLC0206
        lane_helper_dict[lane_id].successor_lane_ids = list(set(lane_helper_dict[lane_id].successor_lane_ids))
        lane_helper_dict[lane_id].predecessor_lane_ids = list(set(lane_helper_dict[lane_id].predecessor_lane_ids))


def _post_process_connections(
    lane_helper_dict: Dict[str, OpenDriveLaneHelper],
    connection_distance_threshold: float,
) -> None:
    """
    Helper function to post-process the connections of the lane helpers, removing invalid connections based on centerline distances.

    Connections between adjacent lane sections within the same road are always kept,
    since they come directly from XODR lane links and may have intentional lateral
    offsets (e.g. when a parking lane appears/disappears between sections).

    :param lane_helper_dict: Dictionary mapping lane ids to their helper objects.
    :param connection_distance_threshold: Threshold distance for valid connections.
    """

    for lane_id in lane_helper_dict.keys():  # noqa: PLC0206
        centerline = lane_helper_dict[lane_id].center_polyline_se2
        road_id = lane_id.split("_")[0]

        valid_successor_lane_ids: List[str] = []
        for successor_lane_id in lane_helper_dict[lane_id].successor_lane_ids:
            # Skip distance check for intra-road connections (adjacent lane sections)
            if successor_lane_id.split("_")[0] == road_id:
                valid_successor_lane_ids.append(successor_lane_id)
                continue
            successor_centerline = lane_helper_dict[successor_lane_id].center_polyline_se2
            distance = np.linalg.norm(centerline[-1, :2] - successor_centerline[0, :2])
            if distance > connection_distance_threshold:
                logger.debug(
                    f"OpenDRIVE: Removing connection {lane_id} -> {successor_lane_id} with distance {distance}"
                )
            else:
                valid_successor_lane_ids.append(successor_lane_id)
        lane_helper_dict[lane_id].successor_lane_ids = valid_successor_lane_ids

        valid_predecessor_lane_ids: List[str] = []
        for predecessor_lane_id in lane_helper_dict[lane_id].predecessor_lane_ids:
            # Skip distance check for intra-road connections (adjacent lane sections)
            if predecessor_lane_id.split("_")[0] == road_id:
                valid_predecessor_lane_ids.append(predecessor_lane_id)
                continue
            predecessor_centerline = lane_helper_dict[predecessor_lane_id].center_polyline_se2
            distance = np.linalg.norm(centerline[0, :2] - predecessor_centerline[-1, :2])
            if distance > connection_distance_threshold:
                logger.debug(
                    f"OpenDRIVE: Removing connection {predecessor_lane_id} -> {lane_id} with distance {distance}"
                )
            else:
                valid_predecessor_lane_ids.append(predecessor_lane_id)
        lane_helper_dict[lane_id].predecessor_lane_ids = valid_predecessor_lane_ids


def _propagate_speed_limits_to_junction_lanes(
    lane_helper_dict: Dict[str, OpenDriveLaneHelper],
    road_dict: Dict[int, XODRRoad],
) -> None:
    """
    Propagate speed limits from predecessor/successor lanes to junction road lanes.
    Junction roads in XODR often lack <type> elements with speed info.
    """
    for lane_id, lane_helper in lane_helper_dict.items():
        if lane_helper.speed_limit_mps is not None:
            continue

        road_id = int(lane_id.split("_")[0])
        road = road_dict.get(road_id)
        if road is None or road.junction is None:
            continue

        # Try predecessor first
        for pred_id in lane_helper.predecessor_lane_ids:
            pred_helper = lane_helper_dict.get(pred_id)
            if pred_helper and pred_helper.speed_limit_mps is not None:
                lane_helper.speed_limit_mps = pred_helper.speed_limit_mps
                break

        # Fallback to successor
        if lane_helper.speed_limit_mps is None:
            for succ_id in lane_helper.successor_lane_ids:
                succ_helper = lane_helper_dict.get(succ_id)
                if succ_helper and succ_helper.speed_limit_mps is not None:
                    lane_helper.speed_limit_mps = succ_helper.speed_limit_mps
                    break


STABLE_REGION_RATIO = 0.3


def _extend_lane_with_shoulder(
    lane_helper: OpenDriveLaneHelper,
    shoulder_helper: OpenDriveLaneHelper,
    is_predecessor: bool,
) -> OpenDriveLaneHelper:
    """
    Extend a merge/exit lane's polylines to smoothly connect with an adjacent driving lane.

    :param lane_helper: The lane to extend (must be a driving lane with missing connection)
    :param shoulder_helper: Adjacent shoulder lane whose curve guides the extension
    :param is_predecessor: True = extend start of lane (no predecessor),
                           False = extend end of lane (no successor)
    :return: New OpenDriveLaneHelper with modified polylines (deep copy, original unchanged)
    """
    lane_center = lane_helper.center_polyline_se2.array
    if lane_center.shape[0] < 2:
        return lane_helper

    def _sample_polyline_se2(polyline: PolylineSE2, count: int) -> np.ndarray:
        """Resample a polyline to have exactly `count` evenly-spaced points."""
        if count <= 1:
            return polyline.array[:count].copy()
        distances = np.linspace(0.0, polyline.length, num=count, dtype=np.float64)
        return np.array(polyline.interpolate(distances), dtype=np.float64)

    def _signed_offsets(base_xy: np.ndarray, target_xy: np.ndarray, base_yaws: np.ndarray) -> np.ndarray:
        """
        Compute signed perpendicular distance from base points to target points.
        Positive = target is to the left of base direction, Negative = to the right.
        """
        normals = np.stack(
            [np.cos(base_yaws + np.pi / 2.0), np.sin(base_yaws + np.pi / 2.0)],
            axis=-1,
        )
        return np.einsum("ij,ij->i", target_xy - base_xy, normals)

    # --- Step 1: Define stable region ---
    # The "stable region" is the part of the lane where geometry is reliable (not merging).
    # For missing predecessor: stable region is at the END of the lane.
    # For missing successor: stable region is at the START of the lane.
    count = lane_center.shape[0]
    stable_count = max(int(round(count * STABLE_REGION_RATIO)), 2)
    stable_count = min(stable_count, count)
    stable_slice = slice(count - stable_count, count) if is_predecessor else slice(0, stable_count)

    # --- Step 2: Find which shoulder boundary is closer to lane center ---
    # Shoulder has inner and outer boundaries; pick the one nearest to our lane.
    shoulder_inner = _sample_polyline_se2(shoulder_helper.inner_polyline_se2, count)
    shoulder_outer = _sample_polyline_se2(shoulder_helper.outer_polyline_se2, count)
    lane_center_xy = lane_center[:, :2]

    inner_dist = np.mean(np.linalg.norm(shoulder_inner[:, :2] - lane_center_xy, axis=1))
    outer_dist = np.mean(np.linalg.norm(shoulder_outer[:, :2] - lane_center_xy, axis=1))
    shoulder_sample = shoulder_inner if inner_dist <= outer_dist else shoulder_outer

    # --- Step 3: Compute lateral offset from shoulder to lane center ---
    # This offset tells us how far (perpendicular) the lane center is from shoulder.
    # We use the stable region to get a reliable offset value.
    shoulder_yaws = shoulder_sample[:, 2]
    shoulder_xy = shoulder_sample[:, :2]
    offsets = _signed_offsets(shoulder_xy, lane_center_xy, shoulder_yaws)
    offset_mean = float(np.mean(offsets[stable_slice]))
    if np.isclose(offset_mean, 0.0):
        offset_mean = float(np.mean(offsets))

    # Create a "target curve" by offsetting shoulder perpendicular by computed offset.
    # This curve represents where the lane SHOULD go if following the shoulder shape.
    shoulder_offset_xy = offset_points_perpendicular(shoulder_sample, offset=offset_mean)

    # --- Step 4: Blend between shoulder-based curve and original lane ---
    # Use Hermite interpolation (smooth step): 3t² - 2t³
    # This gives smooth acceleration/deceleration at blend boundaries.
    t = np.linspace(0.0, 1.0, count, dtype=np.float64)
    smooth = 3.0 * t**2 - 2.0 * t**3
    # For missing predecessor: blend from shoulder (start) to original (end)
    # For missing successor: blend from original (start) to shoulder (end)
    weight = 1.0 - smooth if is_predecessor else smooth
    new_center_xy = (weight[:, None] * shoulder_offset_xy) + ((1.0 - weight)[:, None] * lane_center_xy)
    new_center_xy = new_center_xy.astype(np.float64, copy=False)

    # --- Step 5: Smooth the blended curve with B-spline ---
    # The blending can create slight kinks; spline fitting smooths them out.
    if count >= 4 and np.sum(np.linalg.norm(np.diff(new_center_xy, axis=0), axis=1)) > 1e-6:
        tck, _ = splprep(new_center_xy.T, s=0.0, k=min(3, count - 1))
        u_new = np.linspace(0.0, 1.0, count, dtype=np.float64)
        new_center_xy = np.array(splev(u_new, tck), dtype=np.float64).T

    # Recompute yaw angles from the new XY positions
    new_center_yaws = get_points_2d_yaws(new_center_xy)
    new_center_se2 = np.column_stack([new_center_xy, new_center_yaws]).astype(np.float64, copy=False)

    # --- Step 6: Reconstruct inner/outer boundaries from new centerline ---
    # Compute average lane width from stable region
    inner_xy = lane_helper.inner_polyline_se2.array[:, :2]
    outer_xy = lane_helper.outer_polyline_se2.array[:, :2]
    widths = np.linalg.norm(inner_xy - outer_xy, axis=1)
    width_mean = float(np.mean(widths[stable_slice]))
    if width_mean <= 1e-6:
        width_mean = float(np.mean(widths))

    # Determine which side (left/right) the inner boundary is on
    center_yaws = lane_center[:, 2]
    inner_offsets = _signed_offsets(lane_center_xy, inner_xy, center_yaws)
    inner_offset_mean = float(np.mean(inner_offsets[stable_slice]))
    if np.isclose(inner_offset_mean, 0.0):
        inner_offset_mean = float(np.mean(inner_offsets))
    inner_sign = 1.0 if np.isclose(inner_offset_mean, 0.0) else float(np.sign(inner_offset_mean))

    # Offset new centerline to create inner/outer boundaries
    inner_offset = inner_sign * width_mean / 2.0
    outer_offset = -inner_offset

    new_inner_xy = offset_points_perpendicular(new_center_se2, offset=inner_offset)
    new_outer_xy = offset_points_perpendicular(new_center_se2, offset=outer_offset)

    # Compute yaws for boundary polylines
    inner_yaws = get_points_2d_yaws(new_inner_xy)
    outer_yaws = get_points_2d_yaws(new_outer_xy)
    new_inner_se2 = np.column_stack([new_inner_xy, inner_yaws])
    new_outer_se2 = np.column_stack([new_outer_xy, outer_yaws])

    # Preserve original Z coordinates (elevation)
    inner_z = lane_helper.inner_polyline_3d.array[:, 2]
    outer_z = lane_helper.outer_polyline_3d.array[:, 2]
    new_inner_3d = np.column_stack([new_inner_xy, inner_z])
    new_outer_3d = np.column_stack([new_outer_xy, outer_z])

    # --- Step 7: Create new helper with updated polylines ---
    # Use shallow copy - only the cached polyline properties need replacement,
    # the underlying boundary/reference objects are shared (read-only).
    new_helper = copy(lane_helper)
    new_helper.predecessor_lane_ids = list(lane_helper.predecessor_lane_ids)
    new_helper.successor_lane_ids = list(lane_helper.successor_lane_ids)
    new_helper.__dict__["center_polyline_se2"] = PolylineSE2.from_array(new_center_se2)
    new_helper.__dict__["inner_polyline_se2"] = PolylineSE2.from_array(new_inner_se2)
    new_helper.__dict__["outer_polyline_se2"] = PolylineSE2.from_array(new_outer_se2)
    new_helper.__dict__["inner_polyline_3d"] = Polyline3D.from_array(new_inner_3d)
    new_helper.__dict__["outer_polyline_3d"] = Polyline3D.from_array(new_outer_3d)
    return new_helper


def _correct_lanes_with_no_connections(lane_helper_dict: Dict[str, OpenDriveLaneHelper]) -> None:
    """
    Correct merge/exit lanes that have no predecessor or successor connections.

    Problem: Some OpenDRIVE maps have auxiliary lanes (merge lanes, exit ramps) that lack
    proper connectivity in the lane graph. These lanes appear as "dead ends" - they have
    no predecessor (lane starts from nowhere) or no successor (lane ends abruptly).

    Solution strategy:
    1. Find driving lanes with missing predecessor or successor connections
    2. Look for adjacent shoulder lane (provides curve shape) and driving lane (provides connections)
    3. If both exist: extend the lane geometry using shoulder curve, inherit connections from driving lane
    4. If no shoulder: remove the lane entirely (can't fix without geometric reference)

    :param lane_helper_dict: Dictionary mapping lane ids to their helper objects.
                             Modified in-place: some lanes updated, some deleted.
    """
    lanes_to_update: Dict[str, OpenDriveLaneHelper] = {}
    lanes_to_delete: List[str] = []

    for lane_id, lane_helper in lane_helper_dict.items():
        if lane_helper.type != "driving":
            continue

        road_idx, lane_section_idx, _, lane_idx = lane_id.split("_")
        road_idx, lane_section_idx, lane_idx = int(road_idx), int(lane_section_idx), int(lane_idx)

        right_lane_id = build_lane_id(road_idx, lane_section_idx, lane_idx + 1)
        left_lane_id = build_lane_id(road_idx, lane_section_idx, lane_idx - 1)

        right_lane = lane_helper_dict.get(right_lane_id)
        left_lane = lane_helper_dict.get(left_lane_id)

        # Find adjacent shoulder (for geometry) and driving lane (for connections).
        shoulder, driving = None, None
        if left_lane and left_lane.type == "shoulder":
            shoulder = left_lane
        if right_lane and right_lane.type == "shoulder":
            shoulder = right_lane
        if left_lane and left_lane.type == "driving":
            driving = left_lane
        if right_lane and right_lane.type == "driving":
            driving = right_lane

        no_predecessor = len(lane_helper.predecessor_lane_ids) == 0
        no_successor = len(lane_helper.successor_lane_ids) == 0

        # --- Handle missing predecessor (lane starts abruptly) ---
        if no_predecessor and driving:
            if shoulder:
                # Extend lane start using shoulder curve, inherit predecessor from driving lane
                new_helper = _extend_lane_with_shoulder(lane_helper, shoulder, is_predecessor=True)
                new_helper.predecessor_lane_ids = driving.predecessor_lane_ids
                # Update reverse connections: add this lane as successor to predecessors
                for pred_id in new_helper.predecessor_lane_ids:
                    pred_helper = lane_helper_dict.get(pred_id)
                    if pred_helper:
                        pred_helper.successor_lane_ids.append(lane_id)
                lanes_to_update[lane_id] = new_helper
            else:
                # No shoulder to guide extension -> remove lane entirely
                lanes_to_delete.append(lane_id)
                logger.warning(
                    f"Removing lane {lane_id} no predecessor: added {driving.lane_id}, no shoulder to extend"
                )
                continue

        # --- Handle missing successor (lane ends abruptly) ---
        if no_successor and driving:
            if shoulder:
                # Extend lane end using shoulder curve, inherit successor from driving lane
                new_helper = _extend_lane_with_shoulder(lane_helper, shoulder, is_predecessor=False)
                new_helper.successor_lane_ids = driving.successor_lane_ids
                # Update reverse connections: add this lane as predecessor to successors
                for succ_id in new_helper.successor_lane_ids:
                    succ_helper = lane_helper_dict.get(succ_id)
                    succ_helper.predecessor_lane_ids.append(lane_id)
                lanes_to_update[lane_id] = new_helper
            else:
                # No shoulder to guide extension -> remove lane entirely
                lanes_to_delete.append(lane_id)
                logger.warning(f"Removing lane {lane_id} no successor: added {driving.lane_id}, no shoulder to extend")
                continue

    # Apply all updates and deletions after iteration completes
    lane_helper_dict.update(lanes_to_update)
    for lane_id in lanes_to_delete:
        del lane_helper_dict[lane_id]


def _collect_lane_groups(
    lane_helper_dict: Dict[str, OpenDriveLaneHelper],
    junction_dict: Dict[int, Junction],
    road_dict: Dict[int, XODRRoad],
) -> Dict[str, OpenDriveLaneGroupHelper]:
    lane_group_helper_dict: Dict[str, OpenDriveLaneGroupHelper] = {}

    # Pre-build mapping: lane_group_id -> list of driving lane helpers (O(n) instead of O(n*m))
    driving_lanes_by_group: Dict[str, List[OpenDriveLaneHelper]] = {}
    road_to_group_ids: Dict[int, List[str]] = {}
    for lane_id, lane_helper in lane_helper_dict.items():
        group_id = lane_group_id_from_lane_id(lane_id)
        if lane_helper.type == "driving":
            driving_lanes_by_group.setdefault(group_id, []).append(lane_helper)

    for lane_group_id, lane_helpers in driving_lanes_by_group.items():
        if len(lane_helpers) >= 1:
            lane_group_helper_dict[lane_group_id] = OpenDriveLaneGroupHelper(lane_group_id, lane_helpers)
            road_id = int(road_id_from_lane_group_id(lane_group_id))
            road_to_group_ids.setdefault(road_id, []).append(lane_group_id)

    for junction in junction_dict.values():
        for connection in junction.connections:
            connecting_road_id = connection.connecting_road
            for connecting_lane_group_id in road_to_group_ids.get(connecting_road_id, []):
                if connecting_lane_group_id in lane_group_helper_dict:
                    lane_group_helper_dict[connecting_lane_group_id].junction_id = junction.id

    return lane_group_helper_dict


def _collect_crosswalks(opendrive: XODR) -> Dict[int, OpenDriveObjectHelper]:
    object_helper_dict: Dict[int, OpenDriveObjectHelper] = {}
    for road in opendrive.roads:
        if len(road.objects) == 0:
            continue
        reference_line = XODRReferenceLine.from_plan_view(
            road.plan_view,
            road.lanes.lane_offsets,
            road.elevation_profile.elevations,
        )
        for object in road.objects:
            if object.type in ["crosswalk"]:
                object_helper = get_object_helper(object, reference_line)
                object_helper_dict[object_helper.object_id] = object_helper

    return object_helper_dict


def _collect_signals(opendrive: XODR) -> Dict[int, OpenDriveSignalHelper]:
    """Collect signal references with lane validity info.

    Returns dict keyed by signal_id, merging lane_ids across roads/turn_relations.
    """
    signal_dict: Dict[int, OpenDriveSignalHelper] = {}

    # First pass: collect signal definitions for type lookup
    signal_lookup = {}
    for road in opendrive.roads:
        for signal in road.signals:
            signal_lookup[signal.id] = signal

    # Second pass: process signal references (have actual lane validity)
    for road in opendrive.roads:
        if not road.signal_references:
            continue

        for signal_ref in road.signal_references:
            helper = get_signal_reference_helper(signal_ref, signal_lookup, road)
            if helper and helper.lane_ids:
                key = helper.signal_id
                if key not in signal_dict:
                    signal_dict[key] = helper
                    continue

                existing = signal_dict[key]
                merged_lane_ids = sorted(set(existing.lane_ids + helper.lane_ids))
                existing.lane_ids = merged_lane_ids

    return signal_dict
