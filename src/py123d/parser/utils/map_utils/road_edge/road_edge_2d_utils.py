from typing import List, Optional, Tuple, Union

import numpy as np
import shapely
from shapely import LinearRing, LineString, Point, Polygon, box, union_all
from shapely.strtree import STRtree


def get_road_edge_linear_rings(
    drivable_polygons: List[Polygon],
    buffer_distance: float = 0.05,
    add_interiors: bool = True,
    min_interior_width: float = 0.0,
    fill_hole_points: Optional[List[Tuple[float, float]]] = None,
    non_drivable_polygons: Optional[List[Polygon]] = None,
) -> List[LinearRing]:
    """
    Helper function to extract road edges (i.e. linear rings) from drivable area polygons.
    Interior rings (holes) with a mean width (2 * area / perimeter) below min_interior_width are
    skipped; such hairline slivers between adjacent drivable surfaces are artifacts, not real holes.
    Interior rings containing one of fill_hole_points are also skipped; this patches known source-map
    bugs where an area is drivable in reality but has no surface coverage in the map data.
    non_drivable_polygons are subtracted from the drivable union (even where other surfaces overlap
    them), and the resulting holes are kept regardless of their width.
    TODO: Move and rename for general use.
    """
    fill_points = [Point(p) for p in fill_hole_points] if fill_hole_points else []
    non_drivable_union = (
        union_all([polygon.buffer(0) for polygon in non_drivable_polygons]) if non_drivable_polygons else None
    )

    def _is_deliberate_hole(hole: Polygon) -> bool:
        if non_drivable_union is None:
            return False
        return hole.intersection(non_drivable_union).area > 0.3 * hole.area

    def _polygon_to_linear_rings(polygon: Polygon) -> List[LinearRing]:
        assert polygon.geom_type == "Polygon"
        linear_ring_list = []
        linear_ring_list.append(polygon.exterior)
        if add_interiors:
            for interior in polygon.interiors:
                hole = Polygon(interior)
                if not _is_deliberate_hole(hole):
                    if min_interior_width > 0.0 and 2.0 * hole.area / hole.exterior.length < min_interior_width:
                        continue
                    if any(hole.contains(point) for point in fill_points):
                        continue
                linear_ring_list.append(interior)
        return linear_ring_list

    # Round join on the erosion: mitre projections on spiky interior rings (e.g. the ring enclosing
    # a roundabout center island) can cross the hole and fill it entirely.
    union_polygon = union_all([polygon.buffer(buffer_distance, join_style=2) for polygon in drivable_polygons]).buffer(
        -buffer_distance
    )
    if non_drivable_union is not None:
        union_polygon = union_polygon.difference(non_drivable_union)

    linear_ring_list = []
    if union_polygon.geom_type == "Polygon":
        for polyline in _polygon_to_linear_rings(union_polygon):
            linear_ring_list.append(LinearRing(polyline))
    elif union_polygon.geom_type == "MultiPolygon":
        for polygon in union_polygon.geoms:
            for polyline in _polygon_to_linear_rings(polygon):
                linear_ring_list.append(LinearRing(polyline))

    return linear_ring_list


def split_line_geometry_by_max_length(
    geometries: Union[LineString, LinearRing, List[Union[LineString, LinearRing]]],
    max_length_meters: float,
) -> List[LineString]:
    """
    Splits LineString or LinearRing geometries into smaller segments based on a maximum length.
    TODO: Move and rename for general use.
    """

    if not isinstance(geometries, list):
        geometries = [geometries]

    all_segments = []
    for geom in geometries:
        if geom.length <= max_length_meters:
            all_segments.append(LineString(geom.coords))
            continue

        num_segments = int(np.ceil(geom.length / max_length_meters))
        segment_length = geom.length / num_segments

        for i in range(num_segments):
            start_dist = i * segment_length
            end_dist = min((i + 1) * segment_length, geom.length)
            segment = shapely.ops.substring(geom, start_dist, end_dist)
            all_segments.append(segment)

    return all_segments


def split_polygon_by_grid(polygon: Polygon, cell_size: float) -> List[Polygon]:
    """
    Split a polygon by grid-like cells of given size.
    TODO: Move and rename for general use.
    """

    minx, miny, maxx, maxy = polygon.bounds

    # Generate all grid cells
    x_coords = np.arange(minx, maxx, cell_size)
    y_coords = np.arange(miny, maxy, cell_size)

    grid_cells = [box(x, y, x + cell_size, y + cell_size) for x in x_coords for y in y_coords]

    # Build spatial index for fast queries
    tree = STRtree(grid_cells)

    # Query cells that potentially intersect
    candidate_indices = tree.query(polygon, predicate="intersects")

    cells = []
    for idx in candidate_indices:
        cell = grid_cells[idx]
        intersection = polygon.intersection(cell)

        if intersection.is_empty:
            continue

        if intersection.geom_type == "Polygon":
            cells.append(intersection)
        elif intersection.geom_type == "MultiPolygon":
            cells.extend(intersection.geoms)

    return cells
