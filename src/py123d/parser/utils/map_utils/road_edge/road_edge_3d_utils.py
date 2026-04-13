import logging
from collections import defaultdict
from typing import Dict, List, Set

import networkx as nx
import numpy as np
import numpy.typing as npt
import shapely
import shapely.geometry as geom

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


def get_road_edges_3d_from_drivable_surfaces(
    lanes: List[Lane],
    lane_groups: List[LaneGroup],
    car_parks: List[Carpark],
    generic_drivables: List[GenericDrivable],
) -> List[Polyline3D]:
    """Generates 3D road edges from drivable surfaces, i.e., lane groups, car parks, and generic drivables.
    This method merges polygons in 2D and lifts them to 3D using the boundaries/outlines of elements.
    Conflicting lane groups (e.g., bridges) are merged/lifted separately to ensure correct Z-values.

    :param lanes: A list of lanes in the map.
    :param lane_groups: A list of lane groups in the map.
    :param car_parks: A list of car parks in the map.
    :param generic_drivables: A list of generic drivable areas in the map.
    :return: A list of 3D interpolatable polylines representing the road edges.
    """

    # 1. Find conflicting lane groups, e.g. groups of lanes that overlap in 2D but have different Z-values (bridges)
    conflicting_lane_groups = _get_conflicting_lane_groups(lane_groups, lanes)

    # 2. Extract road edges in 2D (including conflicting lane groups)
    drivable_polygons: List[shapely.Polygon] = []
    for map_surface in lane_groups + generic_drivables:
        map_surface: BaseMapSurfaceObject
        drivable_polygons.append(map_surface.shapely_polygon)
    road_edges_2d = get_road_edge_linear_rings(drivable_polygons)

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

    # 5. Add road edges from conflicting lane groups
    resolved_road_edges = _resolve_conflicting_lane_groups(conflicting_lane_groups, lane_groups)

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
) -> List[Polyline3D]:
    """Lift 2D road edges to 3D by querying elevation from boundary segments.

    :param road_edges_2d: List of 2D road edge geometries.
    :param boundaries: List of 3D boundary geometries.
    :param max_distance: Maximum 2D distance for edge-boundary association.
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
) -> List[Polyline3D]:
    """Resolve conflicting lane groups by merging their geometries.

    :param conflicting_lane_groups: A dictionary mapping lane group IDs to their conflicting lane group IDs.
    :param lane_groups: A list of all lane groups.
    :return: A list of merged 3D road edge geometries.
    """

    # Helper dictionary for easy access to lane group data
    lane_group_dict: Dict[MapObjectIDType, LaneGroup] = {lane_group.object_id: lane_group for lane_group in lane_groups}

    # NOTE @DanielDauner: A non-conflicting set has overlapping lane groups separated into different layers (e.g., bridges).
    # For each non-conflicting set, we can repeat the process of merging polygons in 2D and lifting to 3D.
    # For edge-continuity, we include the neighboring lane groups (predecessors and successors) as well in the 2D merging
    # but only use the original lane group boundaries for lifting to 3D.

    # Split conflicting lane groups into non-conflicting sets for further merging
    non_conflicting_sets = _create_non_conflicting_sets(conflicting_lane_groups)

    road_edges_3d: List[Polyline3D] = []
    for non_conflicting_set in non_conflicting_sets:
        # Collect 2D polygons of non-conflicting lane group set and their neighbors
        merge_lane_group_data: Dict[MapObjectIDType, geom.Polygon] = {}
        for lane_group_id in non_conflicting_set:
            merge_lane_group_data[lane_group_id] = lane_group_dict[lane_group_id].shapely_polygon
            for neighbor_id in (
                lane_group_dict[lane_group_id].predecessor_ids + lane_group_dict[lane_group_id].successor_ids
            ):
                merge_lane_group_data[neighbor_id] = lane_group_dict[neighbor_id].shapely_polygon

        # Get 2D road edge linestrings for the non-conflicting set
        set_road_edges_2d = get_road_edge_linear_rings(list(merge_lane_group_data.values()))

        #  Collect 3D boundaries only of non-conflicting lane groups
        set_boundaries_3d: List[Polyline3D] = []
        for lane_group_id in non_conflicting_set:
            set_boundaries_3d.append(lane_group_dict[lane_group_id].left_boundary_3d)
            set_boundaries_3d.append(lane_group_dict[lane_group_id].right_boundary_3d)

        # Lift road edges to 3D using the boundaries of non-conflicting lane groups
        lifted_road_edges_3d = lift_road_edges_to_3d(set_road_edges_2d, set_boundaries_3d)
        road_edges_3d.extend(lifted_road_edges_3d)

    return road_edges_3d


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
