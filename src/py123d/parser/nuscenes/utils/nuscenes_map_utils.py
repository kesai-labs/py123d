from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from py123d.common.utils.dependencies import check_dependencies
from py123d.geometry.polyline import Polyline2D

check_dependencies(["nuscenes"], optional_name="nuscenes")
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.map_api import NuScenesMap


def extract_lane_and_boundaries(
    nuscenes_map: NuScenesMap, lane_record: Dict
) -> Tuple[Polyline2D, Optional[Polyline2D], Optional[Polyline2D]]:
    """Extracts the centerline and left / right boundary from a nusScenes lane.

    :param nuscenes_map: nuScenes map API instance
    :param lane_record: lane record dictionary
    :return: centerline, left boundary (optional), right boundary (optional)
    """

    # NOTE @DanielDauner: Code adapted from trajdata, Apache 2.0 License. Thank you!
    # https://github.com/NVlabs/trajdata/blob/main/src/trajdata/dataset_specific/nusc/nusc_utils.py#L281

    # Getting the bounding polygon vertices.
    lane_polygon_obj = nuscenes_map.get("polygon", lane_record["polygon_token"])
    polygon_nodes = [nuscenes_map.get("node", node_token) for node_token in lane_polygon_obj["exterior_node_tokens"]]
    polygon_outline: np.ndarray = np.array([(node["x"], node["y"]) for node in polygon_nodes])

    # Getting the lane center's points.
    centerline = extract_nuscenes_centerline(nuscenes_map, lane_record)
    centerline_array = centerline.array

    # Computing the closest lane center point to each bounding polygon vertex.
    closest_midlane_pt: np.ndarray = np.argmin(cdist(polygon_outline, centerline_array), axis=1)
    # Computing the local direction of the lane at each lane center point.
    direction_vectors: np.ndarray = np.diff(
        centerline_array,
        axis=0,
        prepend=centerline_array[[0]] - (centerline_array[[1]] - centerline_array[[0]]),
    )

    # Selecting the direction vectors at the closest lane center point per polygon vertex.
    local_dir_vecs: np.ndarray = direction_vectors[closest_midlane_pt]
    # Calculating the vectors from the the closest lane center point per polygon vertex to the polygon vertex.
    origin_to_polygon_vecs: np.ndarray = polygon_outline - centerline_array[closest_midlane_pt]

    # Computing the perpendicular dot product.
    # See https://www.xarg.org/book/linear-algebra/2d-perp-product/
    # If perp_dot_product < 0, then the associated polygon vertex is
    # on the right edge of the lane.
    perp_dot_product: np.ndarray = (
        local_dir_vecs[:, 0] * origin_to_polygon_vecs[:, 1] - local_dir_vecs[:, 1] * origin_to_polygon_vecs[:, 0]
    )

    # Determining which indices are on the right of the lane center.
    on_right: np.ndarray = perp_dot_product < 0
    # Determining the boundary between the left/right polygon vertices
    # (they will be together in blocks due to the ordering of the polygon vertices).
    idx_changes: int = np.where(np.roll(on_right, 1) < on_right)[0].item()

    if idx_changes > 0:
        # If the block of left/right points spreads across the bounds of the array,
        # roll it until the boundary between left/right points is at index 0.
        # This is important so that the following index selection orders points
        # without jumps.
        polygon_outline = np.roll(polygon_outline, shift=-idx_changes, axis=0)
        on_right = np.roll(on_right, shift=-idx_changes)

    left_polyline_array: np.ndarray = polygon_outline[~on_right]
    right_polyline_array: np.ndarray = polygon_outline[on_right]

    # Final ordering check, ensuring that left_pts and right_pts can be combined
    # into a polygon without the endpoints intersecting.
    # Reversing the one lane edge that does not match the ordering of the midline.
    if endpoints_intersect(left_polyline_array, right_polyline_array):
        if not order_matches(left_polyline_array, centerline_array):
            left_polyline_array = left_polyline_array[::-1]
        else:
            right_polyline_array = right_polyline_array[::-1]

    left_boundary = Polyline2D.from_array(left_polyline_array) if len(left_polyline_array) > 1 else None
    right_boundary = Polyline2D.from_array(right_polyline_array) if len(right_polyline_array) > 1 else None
    return centerline, left_boundary, right_boundary


def extract_nuscenes_centerline(nuscenes_map: NuScenesMap, lane_record: Dict) -> Polyline2D:
    """Extract the centerline of a nuScenes lane.

    :param nuscenes_map: nuScenes map API instance
    :param lane_record: lane record dictionary
    :return: centerline 2D polyline
    """

    # NOTE @DanielDauner: Code adapted from trajdata, Apache 2.0 License. Thank you!
    # https://github.com/NVlabs/trajdata/blob/main/src/trajdata/dataset_specific/nusc/nusc_utils.py#L262

    # Getting the lane center's points.
    curr_lane = nuscenes_map.arcline_path_3.get(lane_record["token"], [])
    centerline_array: np.ndarray = np.array(arcline_path_utils.discretize_lane(curr_lane, resolution_meters=0.25))[
        :, :2
    ]

    # For some reason, nuScenes duplicates a few entries
    # (likely how they're building their arcline representation).
    # We delete those duplicate entries here.
    duplicate_check: np.ndarray = np.where(
        np.linalg.norm(np.diff(centerline_array, axis=0, prepend=0), axis=1) < 1e-10
    )[0]
    if duplicate_check.size > 0:
        centerline_array = np.delete(centerline_array, duplicate_check, axis=0)

    return Polyline2D.from_array(centerline_array)


def endpoints_intersect(left_edge: np.ndarray, right_edge: np.ndarray) -> bool:
    """Check if the line segment connecting the endpoints of left_edge intersects
    with the line segment connecting the endpoints of right_edge.

    Forms two segments: (left_edge[-1], left_edge[0]) and (right_edge[-1], right_edge[0]),
    then tests if they intersect using the counter-clockwise (CCW) orientation test.
    """

    # NOTE @DanielDauner: Code adapted from trajdata, Apache 2.0 License. Thank you!
    # https://github.com/NVlabs/trajdata/blob/main/src/trajdata/utils/map_utils.py#L177
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    A, B = left_edge[-1], right_edge[-1]
    C, D = right_edge[0], left_edge[0]
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def order_matches(pts: np.ndarray, ref: np.ndarray) -> bool:
    """Check if two polylines have the same ordering direction, by comparing
    the distance of their start and end points to the start point of the reference polyline.
    """
    # NOTE @DanielDauner: Code adapted from trajdata, Apache 2.0 License. Thank you!
    # https://github.com/NVlabs/trajdata/blob/main/src/trajdata/utils/map_utils.py#L162
    return bool(np.linalg.norm(pts[0] - ref[0]) <= np.linalg.norm(pts[-1] - ref[0]))


def order_lanes_left_to_right(polylines: List[Polyline2D]) -> List[int]:
    """
    Order lanes from left to right based on their position.

    :param polylines: List of polylines representing lanes
    :return: List of indices representing the order of lanes from left to right
    """
    if len(polylines) == 0:
        return []

    # Step 1: Compute the average direction vector across all lanes
    all_directions = []
    for polyline in polylines:
        polyline_array = polyline.array
        if len(polyline_array) < 2:
            continue
        start = np.array(polyline_array[0])
        end = np.array(polyline_array[-1])
        direction = end - start
        all_directions.append(direction)

    avg_direction = np.mean(all_directions, axis=0)
    avg_direction /= np.linalg.norm(avg_direction)

    # Step 2: Compute perpendicular vector (left direction)
    # Rotate 90 degrees counter-clockwise: (x, y) -> (-y, x)
    left_vector = np.array([-avg_direction[1], avg_direction[0]])

    # Step 3: For each lane, use midpoint of start and end, project onto left vector
    lane_positions = []
    for i, polyline in enumerate(polylines):
        if len(polyline) == 0:
            lane_positions.append((i, 0))
            continue

        start = np.array(polyline[0])
        end = np.array(polyline[-1])
        # Use midpoint of start and end
        midpoint = (start + end) / 2

        # Project midpoint onto the left vector
        position = np.dot(midpoint, left_vector)
        lane_positions.append((i, position))

    # Step 4: Sort by position (higher values are more to the left)
    lane_positions.sort(key=lambda x: x[1], reverse=True)

    # Return ordered indices
    return [idx for idx, _ in lane_positions]
