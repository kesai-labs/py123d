import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import numpy.typing as npt
import shapely
import shapely.geometry as geom
from scipy.spatial import cKDTree

from py123d.datatypes.map_objects.map_objects import (
    BaseMapSurfaceObject,
    Carpark,
    GenericDrivable,
    Lane,
    LaneGroup,
    MapObjectIDType,
)
from py123d.geometry import Point3DIndex
from py123d.geometry.occupancy_map import OccupancyMap2D
from py123d.geometry.polyline import Polyline3D
from py123d.parser.utils.map_utils.road_edge.road_edge_2d_utils import get_road_edge_linear_rings

logger = logging.getLogger(__name__)

FLANK_MATCH_MAX_DISTANCE = 0.5  # [m] a flanking surface must share an edge with the lane group
FLANK_MATCH_MIN_POINTS = 5  # a shared edge, not just a corner touch at a section boundary
FLANK_MATCH_MAX_Z_DIFF = 2.5  # [m] and lie on the same vertical layer (not a road passing above/below)
FLANK_CAP_MIN_LENGTH = 1.5  # [m] outline segments longer than this are transverse end caps
FLANK_CAP_MEDIAN_FACTOR = 1.5  # relative to the outline's median segment length
FLANK_CAP_MAX_TURN = np.pi / 4.0  # [rad] outline corners turning sharper than this start/end a cap
FLANK_TURN_MIN_SEGMENT = 0.05  # [m] ignore turns between near-duplicate points
FLANK_RUN_MIN_FRACTION = 0.3  # pieces shorter than this fraction of the longest piece are caps
FLANK_GUIDE_MIN_OUTLINE_LENGTH = 10.0  # [m] smaller surfaces are seam slivers: merged in 2D, never lift guides
LIFT_PARALLEL_MIN_COS = np.cos(np.pi / 4.0)  # ring direction vs guide direction, modulo half-turn
DUPLICATE_EDGE_MAX_DISTANCE = 0.5  # [m]
DUPLICATE_EDGE_MAX_Z_DIFF = 1.5  # [m]
MIN_RESOLVED_EDGE_LENGTH = 2.0  # [m] drops degenerate fragments from union corner densification


def get_road_edges_3d_from_drivable_surfaces(
    lanes: List[Lane],
    lane_groups: List[LaneGroup],
    car_parks: List[Carpark],
    generic_drivables: List[GenericDrivable],
    min_interior_width: float = 0.0,
    fill_hole_points: Optional[List[Tuple[float, float]]] = None,
    non_drivable_polygons: Optional[List[shapely.Polygon]] = None,
) -> List[Polyline3D]:
    """Generates 3D road edges from drivable surfaces, i.e., lane groups, car parks, and generic drivables.
    This method merges polygons in 2D and lifts them to 3D using the boundaries/outlines of elements.
    Conflicting lane groups (e.g., bridges) are merged/lifted separately to ensure correct Z-values.

    :param lanes: A list of lanes in the map.
    :param lane_groups: A list of lane groups in the map.
    :param car_parks: A list of car parks in the map.
    :param generic_drivables: A list of generic drivable areas in the map.
    :param min_interior_width: Interior rings (holes) with a smaller mean width are dropped, defaults to 0.0.
    :param fill_hole_points: Interior rings containing one of these points are dropped, defaults to None.
    :param non_drivable_polygons: Areas subtracted from the drivable union, defaults to None.
    :return: A list of 3D interpolatable polylines representing the road edges.
    """

    # 1. Find conflicting lane groups, e.g. groups of lanes that overlap in 2D but have different Z-values (bridges)
    conflicting_lane_groups = _get_conflicting_lane_groups(lane_groups, lanes)

    # 2. Extract road edges in 2D (including conflicting lane groups)
    drivable_polygons: List[shapely.Polygon] = []
    for map_surface in lane_groups + generic_drivables:
        map_surface: BaseMapSurfaceObject
        drivable_polygons.append(map_surface.shapely_polygon)
    road_edges_2d = get_road_edge_linear_rings(
        drivable_polygons,
        min_interior_width=min_interior_width,
        fill_hole_points=fill_hole_points,
        non_drivable_polygons=non_drivable_polygons,
    )

    # 3. Collect 3D boundaries of non-conflicting lane groups and other drivable areas
    non_conflicting_boundaries: List[Polyline3D] = []
    for lane_group in lane_groups:
        lane_group_id = lane_group.object_id
        if lane_group_id not in conflicting_lane_groups.keys():
            non_conflicting_boundaries.append(lane_group.left_boundary_3d)
            non_conflicting_boundaries.append(lane_group.right_boundary_3d)
    for drivable_surface in generic_drivables:
        non_conflicting_boundaries.append(drivable_surface.outline)

    # 4. Lift road edges to 3D using the boundaries of non-conflicting elements
    non_conflicting_road_edges = lift_road_edges_to_3d(road_edges_2d, non_conflicting_boundaries)

    # 5. Add road edges from conflicting lane groups, keeping only stretches the merged union missed
    resolved_road_edges = _resolve_conflicting_lane_groups(conflicting_lane_groups, lane_groups, generic_drivables)
    resolved_road_edges = _drop_duplicate_road_edges(resolved_road_edges, non_conflicting_road_edges)

    all_road_edges = non_conflicting_road_edges + resolved_road_edges

    return all_road_edges


def _get_conflicting_lane_groups(
    lane_groups: List[LaneGroup], lanes: List[Lane], z_threshold: float = 5.0
) -> Dict[int, List[int]]:
    """Identifies conflicting lane groups based on their 2D footprints and Z-values.
    The z-values are inferred from the centerlines of the lanes within each lane group.

    :param lane_groups: List of all lane groups in the map.
    :param lanes: List of all lanes in the map.
    :param z_threshold: Z-value threshold over which a 2D overlap is considered a conflict.
    :return: A dictionary mapping lane group IDs to conflicting lane IDs.
    """

    # Convert to regular dictionaries for simpler access
    lane_group_dict: Dict[MapObjectIDType, LaneGroup] = {lane_group.object_id: lane_group for lane_group in lane_groups}
    lane_centerline_dict: Dict[MapObjectIDType, Polyline3D] = {lane.object_id: lane.centerline_3d for lane in lanes}

    # Pre-compute all centerlines
    centerlines_cache: Dict[MapObjectIDType, npt.NDArray[np.float64]] = {}
    polygons: List[geom.Polygon] = []
    ids: List[MapObjectIDType] = []

    for lane_group_id, lane_group in lane_group_dict.items():
        centerlines = [lane_centerline_dict[lane_id].array for lane_id in lane_group.lane_ids]
        centerlines_3d_array = np.concatenate(centerlines, axis=0)

        centerlines_cache[lane_group_id] = centerlines_3d_array
        polygons.append(lane_group.shapely_polygon)
        ids.append(lane_group_id)

    occupancy_map = OccupancyMap2D(polygons, ids)
    conflicting_lane_groups: Dict[MapObjectIDType, List[MapObjectIDType]] = defaultdict(list)
    processed_pairs = set()

    for i, lane_group_id in enumerate(ids):
        lane_group_polygon = polygons[i]
        lane_group_centerlines = centerlines_cache[lane_group_id]

        # Get all intersecting geometries at once
        intersecting_ids = occupancy_map.intersects(lane_group_polygon)
        intersecting_ids.remove(lane_group_id)

        for intersecting_id in intersecting_ids:
            pair_key = tuple(sorted([lane_group_id, intersecting_id]))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)

            intersecting_geometry = occupancy_map[intersecting_id]
            if intersecting_geometry.geom_type != "Polygon":
                continue
            try:
                # Compute actual intersection for better centroid
                intersection = lane_group_polygon.intersection(intersecting_geometry)
            except shapely.errors.GEOSException as e:
                logger.debug(f"Error computing intersection for {pair_key}: {e}")
                continue

            if intersection.is_empty:
                continue

            # NOTE @DanielDauner: We query the centroid of the intersection polygon to get a representative point
            # We cannot calculate the Z-difference at any area, e.g. due to arcs or complex shapes of bridges.
            intersection_centroid = np.array(intersection.centroid.coords[0], dtype=np.float64)
            intersecting_centerlines = centerlines_cache[intersecting_id]

            z_at_intersecting = _get_nearest_z_from_points_3d(intersecting_centerlines, intersection_centroid)
            z_at_lane_group = _get_nearest_z_from_points_3d(lane_group_centerlines, intersection_centroid)
            if np.abs(z_at_lane_group - z_at_intersecting) >= z_threshold:
                conflicting_lane_groups[lane_group_id].append(intersecting_id)
                conflicting_lane_groups[intersecting_id].append(lane_group_id)

    return conflicting_lane_groups


def lift_road_edges_to_3d(
    road_edges_2d: List[shapely.LinearRing],
    boundaries: List[Polyline3D],
    max_distance: float = 0.5,
    require_parallel_direction: bool = False,
) -> List[Polyline3D]:
    """Lift 2D road edges to 3D by querying elevation from boundary segments.

    :param road_edges_2d: List of 2D road edge geometries.
    :param boundaries: List of 3D boundary geometries.
    :param max_distance: Maximum 2D distance for edge-boundary association.
    :param require_parallel_direction: Only lift ring points whose local direction runs parallel
        to the matched boundary, dropping transverse construction cuts, defaults to False.
    :return: List of lifted 3D road edge geometries.
    """

    road_edges_3d: List[Polyline3D] = []

    if len(road_edges_2d) >= 1 and len(boundaries) >= 1:
        # 1. Build comprehensive spatial index with all boundary segments
        # NOTE @DanielDauner: We split each boundary polyline into small segments.
        # The spatial indexing uses axis-aligned bounding boxes, where small geometries lead to better performance.
        boundary_segments = []
        for boundary in boundaries:
            coords = boundary.array.reshape(-1, 1, 3)
            segment_coords_boundary = np.concatenate([coords[:-1], coords[1:]], axis=1)
            boundary_segments.append(segment_coords_boundary)

        boundary_segments = np.concatenate(boundary_segments, axis=0)
        boundary_segment_linestrings = shapely.creation.linestrings(boundary_segments)
        occupancy_map = OccupancyMap2D(boundary_segment_linestrings)

        for linear_ring in road_edges_2d:
            ring_edges_3d: List[Polyline3D] = []
            points_2d = np.array(linear_ring.coords, dtype=np.float64)
            points_3d = np.zeros((len(points_2d), len(Point3DIndex)), dtype=np.float64)
            points_3d[..., Point3DIndex.XY] = points_2d

            # 3. Batch query for all points
            query_points = shapely.creation.points(points_2d)
            results = occupancy_map.query_nearest(query_points, max_distance=max_distance, exclusive=True)
            if require_parallel_direction:
                results = _filter_parallel_matches(points_2d, boundary_segments, results)

            for query_idx, geometry_idx in zip(*results):
                query_point = query_points[query_idx]
                segment_coords = boundary_segments[geometry_idx]
                best_z = _interpolate_z_on_segment(query_point, segment_coords)
                points_3d[query_idx, Point3DIndex.Z] = best_z

            # Deduplicate: query_nearest with all_matches=True can return multiple geometry
            # matches per query point (equidistant segments), causing duplicate query indices.
            # _split_continuous_segments expects unique, sorted indices.
            unique_query_indices = np.unique(results[0])
            continuous_segments = _split_continuous_segments(unique_query_indices)
            for segment_indices in continuous_segments:
                if len(segment_indices) >= 2:
                    segment_points = points_3d[segment_indices]
                    ring_edges_3d.append(Polyline3D.from_array(segment_points))

            road_edges_3d.extend(_fuse_short_edges(ring_edges_3d))

    return road_edges_3d


def lift_outlines_to_3d(
    outlines_2d: List[shapely.LinearRing],
    boundaries: List[Polyline3D],
    max_distance: float = 10.0,
) -> List[Polyline3D]:
    """Lift 2D outlines to 3D by querying elevation from boundary segments.

    :param outlines_2d: List of 2D outline geometries.
    :param boundaries: List of 3D boundary geometries.
    :param max_distance: Maximum 2D distance for outline-boundary association.
    :return: List of lifted 3D outline geometries.
    """

    outlines_3d: List[Polyline3D] = []
    if len(outlines_2d) >= 1 and len(boundaries) >= 1:
        boundary_segments = []
        for boundary in boundaries:
            coords = boundary.array.reshape(-1, 1, 3)
            segment_coords_boundary = np.concatenate([coords[:-1], coords[1:]], axis=1)
            boundary_segments.append(segment_coords_boundary)

        boundary_segments = np.concatenate(boundary_segments, axis=0)
        boundary_segment_linestrings = shapely.creation.linestrings(boundary_segments)
        occupancy_map = OccupancyMap2D(boundary_segment_linestrings)

        for linear_ring in outlines_2d:
            points_2d = np.array(linear_ring.coords, dtype=np.float64)
            points_3d = np.zeros((len(points_2d), len(Point3DIndex)), dtype=np.float64)
            points_3d[..., Point3DIndex.XY] = points_2d

            # 3. Batch query for all points
            query_points = shapely.creation.points(points_2d)
            results = occupancy_map.query_nearest(query_points, max_distance=max_distance, exclusive=True)

            found_nearest = np.zeros(len(points_2d), dtype=bool)
            for query_idx, geometry_idx in zip(*results):
                query_point = query_points[query_idx]
                segment_coords = boundary_segments[geometry_idx]
                best_z = _interpolate_z_on_segment(query_point, segment_coords)
                points_3d[query_idx, Point3DIndex.Z] = best_z
                found_nearest[query_idx] = True

            if not np.all(found_nearest):
                logger.warning("Some outline points could not find a nearest boundary segment for Z-lifting.")
                points_3d[~found_nearest, Point3DIndex.Z] = np.mean(points_3d[found_nearest, Point3DIndex.Z])

            outlines_3d.append(Polyline3D.from_array(points_3d))

    return outlines_3d


def _resolve_conflicting_lane_groups(
    conflicting_lane_groups: Dict[MapObjectIDType, List[MapObjectIDType]],
    lane_groups: List[LaneGroup],
    drivable_surfaces: List[BaseMapSurfaceObject],
) -> List[Polyline3D]:
    """Resolve conflicting lane groups by merging their geometries.

    :param conflicting_lane_groups: A dictionary mapping lane group IDs to their conflicting lane group IDs.
    :param lane_groups: A list of all lane groups.
    :param drivable_surfaces: All non-lane-group drivable surfaces (shoulders, none lanes, generic drivables).
    :return: A list of merged 3D road edge geometries.
    """

    # Helper dictionary for easy access to lane group data
    lane_group_dict: Dict[MapObjectIDType, LaneGroup] = {lane_group.object_id: lane_group for lane_group in lane_groups}

    # NOTE @DanielDauner: A non-conflicting set has overlapping lane groups separated into different layers (e.g., bridges).
    # For each non-conflicting set, we can repeat the process of merging polygons in 2D and lifting to 3D.
    # For edge-continuity, we include the neighboring lane groups (predecessors and successors) as well in the 2D merging
    # but only use the original lane group boundaries for lifting to 3D.
    # Flanking surfaces (shoulders/none lanes) on the same vertical layer are merged in as well, so the
    # ring follows the drivable envelope instead of the bare driving-lane block.

    # Split conflicting lane groups into non-conflicting sets for further merging
    non_conflicting_sets = _create_non_conflicting_sets(conflicting_lane_groups)

    involved_lane_group_ids: Set[MapObjectIDType] = set()
    for non_conflicting_set in non_conflicting_sets:
        for lane_group_id in non_conflicting_set:
            involved_lane_group_ids.add(lane_group_id)
            involved_lane_group_ids.update(lane_group_dict[lane_group_id].predecessor_ids)
            involved_lane_group_ids.update(lane_group_dict[lane_group_id].successor_ids)
    flank_surfaces_by_group = _match_flanking_surfaces(involved_lane_group_ids, lane_group_dict, drivable_surfaces)

    road_edges_3d: List[Polyline3D] = []
    for non_conflicting_set in non_conflicting_sets:
        # Collect 2D polygons of non-conflicting lane group set, their neighbors, and flanking surfaces
        merge_surface_data: Dict[MapObjectIDType, geom.Polygon] = {}
        for lane_group_id in non_conflicting_set:
            member_and_neighbor_ids = (
                [lane_group_id]
                + lane_group_dict[lane_group_id].predecessor_ids
                + lane_group_dict[lane_group_id].successor_ids
            )
            for merge_id in member_and_neighbor_ids:
                merge_surface_data[merge_id] = lane_group_dict[merge_id].shapely_polygon
                for flank_surface in flank_surfaces_by_group.get(merge_id, []):
                    merge_surface_data[flank_surface.object_id] = flank_surface.shapely_polygon

        # Get 2D road edge linestrings for the non-conflicting set
        set_road_edges_2d = get_road_edge_linear_rings(list(merge_surface_data.values()))

        #  Collect 3D boundaries of non-conflicting lane groups and their flanking surfaces
        set_boundaries_3d: List[Polyline3D] = []
        for lane_group_id in non_conflicting_set:
            set_boundaries_3d.append(lane_group_dict[lane_group_id].left_boundary_3d)
            set_boundaries_3d.append(lane_group_dict[lane_group_id].right_boundary_3d)
            for flank_surface in flank_surfaces_by_group.get(lane_group_id, []):
                if _get_polyline_length(flank_surface.outline.array) < FLANK_GUIDE_MIN_OUTLINE_LENGTH:
                    continue
                set_boundaries_3d.extend(_split_outline_at_caps(flank_surface.outline))

        # Lift road edges to 3D using the boundaries of non-conflicting lane groups
        lifted_road_edges_3d = lift_road_edges_to_3d(
            set_road_edges_2d, set_boundaries_3d, require_parallel_direction=True
        )
        road_edges_3d.extend(lifted_road_edges_3d)

    return [road_edge for road_edge in road_edges_3d if _get_polyline_length(road_edge.array) >= MIN_RESOLVED_EDGE_LENGTH]


def _match_flanking_surfaces(
    lane_group_ids: Set[MapObjectIDType],
    lane_group_dict: Dict[MapObjectIDType, LaneGroup],
    drivable_surfaces: List[BaseMapSurfaceObject],
) -> Dict[MapObjectIDType, List[BaseMapSurfaceObject]]:
    """Match flanking drivable surfaces (shoulders/none lanes) to the lane groups they border.

    A surface matches a lane group when it touches the group's boundary in 2D at a consistent
    height, so surfaces of a road passing above/below never merge into the wrong layer.

    :param lane_group_ids: IDs of the lane groups to match against.
    :param lane_group_dict: Mapping of all lane group IDs to lane groups.
    :param drivable_surfaces: All non-lane-group drivable surfaces.
    :return: Mapping of lane group ID to its flanking surfaces.
    """
    flank_surfaces_by_group: Dict[MapObjectIDType, List[BaseMapSurfaceObject]] = defaultdict(list)
    if len(drivable_surfaces) == 0 or len(lane_group_ids) == 0:
        return flank_surfaces_by_group

    occupancy_map = OccupancyMap2D([surface.shapely_polygon for surface in drivable_surfaces])
    for lane_group_id in sorted(lane_group_ids, key=str):
        lane_group = lane_group_dict[lane_group_id]
        boundary_points = np.concatenate(
            [lane_group.left_boundary_3d.array, lane_group.right_boundary_3d.array], axis=0
        )
        boundary_tree = cKDTree(boundary_points[:, :2])
        candidate_region = lane_group.shapely_polygon.buffer(FLANK_MATCH_MAX_DISTANCE)
        for surface_id in sorted(occupancy_map.intersects(candidate_region), key=int):
            outline_points = drivable_surfaces[int(surface_id)].outline.array
            nearest_distances, nearest_indices = boundary_tree.query(outline_points[:, :2])
            touching_mask = nearest_distances <= FLANK_MATCH_MAX_DISTANCE
            if np.count_nonzero(touching_mask) < FLANK_MATCH_MIN_POINTS:
                continue
            outline_z = outline_points[touching_mask, 2] if outline_points.shape[-1] > 2 else 0.0
            z_diffs = np.abs(outline_z - boundary_points[nearest_indices[touching_mask], 2])
            if np.median(z_diffs) <= FLANK_MATCH_MAX_Z_DIFF:
                flank_surfaces_by_group[lane_group_id].append(drivable_surfaces[int(surface_id)])
    return flank_surfaces_by_group


def _filter_parallel_matches(
    points_2d: npt.NDArray[np.float64],
    boundary_segments: npt.NDArray[np.float64],
    results: Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]],
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Keep only matches where the ring runs parallel to the matched boundary segment.

    A transverse construction cut (e.g. where a merged patch ends across a road) touches
    longitudinal boundaries only at its corners, at a right angle; those matches are dropped.

    :param points_2d: The ring vertices, shape (N, 2).
    :param boundary_segments: The boundary segments, shape (M, 2, 3).
    :param results: Query/segment index pairs from the nearest query.
    :return: The filtered query/segment index pairs.
    """
    point_directions = np.zeros_like(points_2d)
    point_directions[:-1] = np.diff(points_2d, axis=0)
    point_directions[-1] = point_directions[-2] if len(points_2d) > 1 else 0.0

    query_indices, segment_indices = results
    ring_dirs = point_directions[query_indices]
    segment_dirs = boundary_segments[segment_indices, 1, :2] - boundary_segments[segment_indices, 0, :2]
    norms = np.linalg.norm(ring_dirs, axis=1) * np.linalg.norm(segment_dirs, axis=1)
    cos_angles = np.abs(np.sum(ring_dirs * segment_dirs, axis=1)) / np.maximum(norms, 1e-12)
    keep = (cos_angles >= LIFT_PARALLEL_MIN_COS) | (norms < 1e-12)
    return query_indices[keep], segment_indices[keep]


def _split_outline_at_caps(outline: Polyline3D) -> List[Polyline3D]:
    """Split a closed surface outline at its transverse end caps so only longitudinal runs guide lifting.

    :param outline: The closed outline of a flanking surface.
    :return: The longitudinal pieces of the outline.
    """
    points = outline.array
    if len(points) < 4:
        return [outline]
    diffs = np.diff(points[:, :2], axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cap_threshold = max(FLANK_CAP_MIN_LENGTH, FLANK_CAP_MEDIAN_FACTOR * float(np.median(segment_lengths)))
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    turns = np.diff(headings)
    turns = np.abs((turns + np.pi) % (2.0 * np.pi) - np.pi)

    piece_arrays: List[npt.NDArray[np.float64]] = []
    start = 0
    for segment_idx in range(len(diffs)):
        is_sharp_corner = (
            segment_idx > start
            and turns[segment_idx - 1] > FLANK_CAP_MAX_TURN
            and min(segment_lengths[segment_idx - 1], segment_lengths[segment_idx]) > FLANK_TURN_MIN_SEGMENT
        )
        if segment_lengths[segment_idx] > cap_threshold:
            piece_arrays.append(points[start : segment_idx + 1])
            start = segment_idx + 1
        elif is_sharp_corner:
            piece_arrays.append(points[start : segment_idx + 1])
            start = segment_idx
    piece_arrays.append(points[start:])

    if len(piece_arrays) == 1:
        return [outline]
    piece_lengths = [_get_polyline_length(piece) if len(piece) >= 2 else 0.0 for piece in piece_arrays]
    min_length = FLANK_RUN_MIN_FRACTION * max(piece_lengths)
    return [
        Polyline3D.from_array(piece)
        for piece, piece_length in zip(piece_arrays, piece_lengths)
        if len(piece) >= 2 and piece_length >= min_length
    ]


def _drop_duplicate_road_edges(
    road_edges: List[Polyline3D],
    reference_road_edges: List[Polyline3D],
) -> List[Polyline3D]:
    """Drop road-edge stretches that duplicate an already-emitted reference edge at the same height.

    :param road_edges: Candidate road edges (from conflicting lane groups).
    :param reference_road_edges: Reference road edges (from the merged drivable union).
    :return: Candidate edges with duplicated stretches removed.
    """
    if len(road_edges) == 0 or len(reference_road_edges) == 0:
        return road_edges

    reference_segments = []
    for reference_edge in reference_road_edges:
        coords = reference_edge.array.reshape(-1, 1, 3)
        reference_segments.append(np.concatenate([coords[:-1], coords[1:]], axis=1))
    reference_segments = np.concatenate(reference_segments, axis=0)
    occupancy_map = OccupancyMap2D(shapely.creation.linestrings(reference_segments))

    deduplicated: List[Polyline3D] = []
    for road_edge in road_edges:
        points_3d = road_edge.array
        query_points = shapely.creation.points(points_3d[:, :2])
        query_indices, segment_indices = occupancy_map.query_nearest(
            query_points, max_distance=DUPLICATE_EDGE_MAX_DISTANCE
        )
        duplicate_mask = np.zeros(len(points_3d), dtype=bool)
        for query_idx, segment_idx in zip(query_indices, segment_indices):
            reference_z = _interpolate_z_on_segment(query_points[query_idx], reference_segments[segment_idx])
            if abs(points_3d[query_idx, 2] - reference_z) <= DUPLICATE_EDGE_MAX_Z_DIFF:
                duplicate_mask[query_idx] = True
        kept_indices = np.nonzero(~duplicate_mask)[0]
        for segment in _split_continuous_segments(kept_indices):
            if _get_polyline_length(points_3d[segment]) >= MIN_RESOLVED_EDGE_LENGTH:
                deduplicated.append(Polyline3D.from_array(points_3d[segment]))
    return deduplicated


def _get_polyline_length(points: npt.NDArray[np.float64]) -> float:
    """Helper function to compute 3D polyline length from point arrays."""
    return Polyline3D.from_array(points, copy=False).length


def _get_edge_gap(first: npt.NDArray[np.float64], second: npt.NDArray[np.float64]) -> float:
    """Helper function to compute the 2D gap between consecutive edge fragments."""
    return float(np.linalg.norm(first[-1, :2] - second[0, :2]))


def _fuse_short_edge_sequence(
    edge_arrays: List[npt.NDArray[np.float64]], min_length: float, max_gap: float
) -> List[npt.NDArray[np.float64]]:
    """Fuse short edge fragments in sequence while preserving valid leftovers."""
    if not edge_arrays:
        return edge_arrays

    fused: List[npt.NDArray[np.float64]] = []
    buf = edge_arrays[0]

    for edge_array in edge_arrays[1:]:
        if _get_edge_gap(buf, edge_array) < max_gap and _get_polyline_length(buf) < min_length:
            buf = np.concatenate([buf, edge_array], axis=0)
            continue

        fused.append(buf)
        buf = edge_array

    return fused + [buf]


def _fuse_short_edges(edges: List[Polyline3D], min_length: float = 2.0, max_gap: float = 0.5) -> List[Polyline3D]:
    """Merge adjacent short road edges, including across the LinearRing seam."""
    if not edges:
        return edges

    fused_arrays = _fuse_short_edge_sequence([edge.array for edge in edges], min_length=min_length, max_gap=max_gap)

    if len(fused_arrays) >= 2:
        first_edge = fused_arrays[0]
        last_edge = fused_arrays[-1]
        should_fuse_ring_closure = _get_edge_gap(last_edge, first_edge) < max_gap and (
            _get_polyline_length(last_edge) < min_length or _get_polyline_length(first_edge) < min_length
        )

        if should_fuse_ring_closure:
            fused_arrays = _fuse_short_edge_sequence(
                [np.concatenate([last_edge, first_edge], axis=0)] + fused_arrays[1:-1],
                min_length=min_length,
                max_gap=max_gap,
            )

    return [Polyline3D.from_array(edge_array) for edge_array in fused_arrays]


def _get_nearest_z_from_points_3d(points_3d: npt.NDArray[np.float64], query_point: npt.NDArray[np.float64]) -> float:
    """Helpers function to get the Z-value of the nearest 3D point to a query point."""
    assert points_3d.ndim == 2 and points_3d.shape[1] == len(Point3DIndex), (
        "points_3d must be a 2D array with shape (N, 3)"
    )
    distances = np.linalg.norm(points_3d[..., Point3DIndex.XY] - query_point[..., Point3DIndex.XY], axis=1)
    closest_point = points_3d[np.argmin(distances)]
    return closest_point[2]


def _interpolate_z_on_segment(point: shapely.Point, segment_coords: npt.NDArray[np.float64]) -> float:
    """Helpers function to interpolate the Z-value on a 3D segment given a 2D point."""
    p1, p2 = segment_coords[0], segment_coords[1]

    # Project point onto segment
    segment_vec = p2[:2] - p1[:2]
    point_vec = np.array([point.x, point.y]) - p1[:2]

    # Handle degenerate case
    segment_length_sq = np.dot(segment_vec, segment_vec)
    if segment_length_sq == 0:
        return p1[2]

    # Calculate projection parameter
    t = np.dot(point_vec, segment_vec) / segment_length_sq
    t = np.clip(t, 0, 1)  # Clamp to segment bounds

    # Interpolate Z
    return p1[2] + t * (p2[2] - p1[2])


def _split_continuous_segments(indices: npt.NDArray[np.int64]) -> List[npt.NDArray[np.int64]]:
    """Helper function to find continuous segments in a list of indices."""
    if len(indices) == 0:
        return []

    # Find breaks in continuity
    breaks = np.where(np.diff(indices) != 1)[0] + 1
    segments = np.split(indices, breaks)

    # Filter segments with at least 2 points
    return [seg for seg in segments if len(seg) >= 2]


def _create_non_conflicting_sets(conflicts: Dict[MapObjectIDType, List[MapObjectIDType]]) -> List[Set[MapObjectIDType]]:
    """Helper function to create non-conflicting sets from a conflict dictionary."""

    # NOTE @DanielDauner: The conflict problem is a graph coloring problem. Map objects are nodes, conflicts are edges.
    # https://en.wikipedia.org/wiki/Graph_coloring

    # Create graph from conflicts
    G = nx.Graph()
    for idx, conflict_list in conflicts.items():
        for conflict_idx in conflict_list:
            G.add_edge(idx, conflict_idx)

    result: List[Set[MapObjectIDType]] = []

    # Process each connected component
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)

        # Try bipartite coloring first
        if nx.is_bipartite(subgraph):
            sets = nx.bipartite.sets(subgraph)
            result.extend([set(s) for s in sets])
        else:
            # Fall back to greedy coloring for non-bipartite graphs
            coloring = nx.greedy_color(subgraph, strategy="largest_first")
            color_groups = {}
            for node, color in coloring.items():
                if color not in color_groups:
                    color_groups[color] = set()
                color_groups[color].add(node)
            result.extend(color_groups.values())

    return result
