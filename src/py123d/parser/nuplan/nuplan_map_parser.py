from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Final, Iterator, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from shapely import LineString

from py123d.datatypes.map_objects.base_map_objects import BaseMapObject
from py123d.datatypes.map_objects.map_layer_types import IntersectionType, LaneType, RoadEdgeType, StopZoneType
from py123d.datatypes.map_objects.map_objects import (
    Carpark,
    Crosswalk,
    GenericDrivable,
    Intersection,
    Lane,
    LaneGroup,
    RoadEdge,
    RoadLine,
    StopZone,
    Walkway,
)
from py123d.datatypes.metadata import MapMetadata
from py123d.geometry import Polyline2D, Polyline3D
from py123d.parser.base_dataset_parser import BaseMapParser
from py123d.parser.nuplan.utils.nuplan_constants import (
    NUPLAN_INTERSECTION_TYPE_CONVERSION,
    NUPLAN_LANE_TYPE_CONVERSION,
    NUPLAN_MAP_GPKG_LAYERS,
    NUPLAN_MAP_LOCATION_FILES,
    NUPLAN_ROAD_LINE_CONVERSION,
    NUPLAN_STOP_ZONE_TYPE_CONVERSION,
)
from py123d.parser.utils.map_utils.road_edge.road_edge_2d_utils import (
    get_road_edge_linear_rings,
    split_line_geometry_by_max_length,
)

# TODO: Refactor and remove all of the "type: ignore" comment.

MAX_ROAD_EDGE_LENGTH: Final[float] = 100.0  # TODO: Add to config


class NuplanMapParser(BaseMapParser):
    def __init__(self, nuplan_maps_root: Path, location: str) -> None:
        self._nuplan_maps_root = nuplan_maps_root
        self._location = location

    def get_map_metadata(self) -> MapMetadata:
        """Inherited, see superclass."""
        return MapMetadata(
            dataset="nuplan",
            split=None,
            log_name=None,
            location=self._location,
            map_has_z=False,
            map_is_per_log=False,
        )

    def iter_map_objects(self) -> Iterator[BaseMapObject]:
        """Inherited, see superclass."""
        assert self._location in NUPLAN_MAP_LOCATION_FILES, f"Map name {self._location} is not supported."
        source_map_path = self._nuplan_maps_root / NUPLAN_MAP_LOCATION_FILES[self._location]
        assert source_map_path.exists(), f"Map file {source_map_path} does not exist."

        nuplan_gdf = _load_nuplan_gdf(source_map_path)
        yield from _iter_nuplan_lanes(nuplan_gdf)
        yield from _iter_nuplan_lane_connectors(nuplan_gdf)
        yield from _iter_nuplan_lane_groups(nuplan_gdf)
        yield from _iter_nuplan_lane_connector_groups(nuplan_gdf)
        yield from _iter_nuplan_intersections(nuplan_gdf)
        yield from _iter_nuplan_crosswalks(nuplan_gdf)
        yield from _iter_nuplan_walkways(nuplan_gdf)
        yield from _iter_nuplan_carparks(nuplan_gdf)
        yield from _iter_nuplan_generic_drivables(nuplan_gdf)
        yield from _iter_nuplan_stop_zones(nuplan_gdf)
        yield from _iter_nuplan_road_edges(nuplan_gdf)
        yield from _iter_nuplan_road_lines(nuplan_gdf)
        del nuplan_gdf


# Map object iterators
# ----------------------------------------------------------------------------------------------------------------------


def _iter_nuplan_lanes(nuplan_gdf: Dict[str, gpd.GeoDataFrame]) -> Iterator[Lane]:
    """Yield Lane objects from nuPlan lane data."""
    all_ids = nuplan_gdf["lanes_polygons"].lane_fid.to_list()
    all_lane_group_ids = nuplan_gdf["lanes_polygons"].lane_group_fid.to_list()
    all_speed_limits_mps = nuplan_gdf["lanes_polygons"].speed_limit_mps.to_list()
    all_lane_type_fids = nuplan_gdf["lanes_polygons"].lane_type_fid.to_list()
    all_geometries = nuplan_gdf["lanes_polygons"].geometry.to_list()

    for idx, lane_id in enumerate(all_ids):
        # 1. predecessor_ids, successor_ids
        predecessor_ids = get_all_rows_with_value(
            nuplan_gdf["lane_connectors"],
            "entry_lane_fid",
            lane_id,
        )["fid"].tolist()  # type: ignore
        successor_ids = get_all_rows_with_value(
            nuplan_gdf["lane_connectors"],
            "exit_lane_fid",
            lane_id,
        )["fid"].tolist()  # type: ignore

        # 2. left_boundary, right_boundary
        lane_series = get_row_with_value(nuplan_gdf["lanes_polygons"], "fid", str(lane_id))
        left_boundary_fid = lane_series["left_boundary_fid"]  # type: ignore
        left_boundary = get_row_with_value(nuplan_gdf["boundaries"], "fid", str(left_boundary_fid))["geometry"]  # type: ignore

        right_boundary_fid = lane_series["right_boundary_fid"]  # type: ignore
        right_boundary = get_row_with_value(nuplan_gdf["boundaries"], "fid", str(right_boundary_fid))["geometry"]  # type: ignore

        # 3. left_lane_id, right_lane_id
        lane_index = lane_series["lane_index"]  # type: ignore
        all_group_lanes = get_all_rows_with_value(
            nuplan_gdf["lanes_polygons"],
            "lane_group_fid",
            lane_series["lane_group_fid"],  # type: ignore
        )
        left_lane_id = all_group_lanes[all_group_lanes["lane_index"] == int(lane_index) - 1]["fid"]  # type: ignore
        right_lane_id = all_group_lanes[all_group_lanes["lane_index"] == int(lane_index) + 1]["fid"]  # type: ignore
        left_lane_id = left_lane_id.item() if not left_lane_id.empty else None
        right_lane_id = right_lane_id.item() if not right_lane_id.empty else None

        # 3. centerline (aka. baseline_path)
        centerline = get_row_with_value(nuplan_gdf["baseline_paths"], "lane_fid", float(lane_id))["geometry"]  # type: ignore

        # Ensure the left/right boundaries are aligned with the baseline path direction.
        left_boundary = align_boundary_direction(centerline, left_boundary)
        right_boundary = align_boundary_direction(centerline, right_boundary)

        lane_type = NUPLAN_LANE_TYPE_CONVERSION.get(all_lane_type_fids[idx], LaneType.UNDEFINED)

        yield Lane(
            object_id=int(lane_id),
            lane_type=lane_type,
            lane_group_id=all_lane_group_ids[idx],
            left_boundary=Polyline3D.from_linestring(left_boundary),
            right_boundary=Polyline3D.from_linestring(right_boundary),
            centerline=Polyline3D.from_linestring(centerline),
            left_lane_id=left_lane_id,
            right_lane_id=right_lane_id,
            predecessor_ids=predecessor_ids,
            successor_ids=successor_ids,
            speed_limit_mps=all_speed_limits_mps[idx],
            outline=None,
            shapely_polygon=all_geometries[idx],  # type: ignore
        )


def _iter_nuplan_lane_connectors(nuplan_gdf: Dict[str, gpd.GeoDataFrame]) -> Iterator[Lane]:
    """Yield Lane objects from nuPlan lane connectors (lanes on intersections)."""
    all_ids = nuplan_gdf["lane_connectors"].fid.to_list()
    all_lane_group_ids = nuplan_gdf["lane_connectors"].lane_group_connector_fid.to_list()
    all_speed_limits_mps = nuplan_gdf["lane_connectors"].speed_limit_mps.to_list()

    for idx, lane_id in enumerate(all_ids):
        # 1. predecessor_ids, successor_ids
        lane_connector_row = get_row_with_value(nuplan_gdf["lane_connectors"], "fid", str(lane_id))
        assert lane_connector_row is not None, f"Could not find lane connector with id {lane_id}"
        predecessor_ids = [lane_connector_row["entry_lane_fid"]]
        successor_ids = [lane_connector_row["exit_lane_fid"]]

        # 2. left_boundaries, right_boundaries
        lane_connector_polygons_row = get_row_with_value(
            nuplan_gdf["gen_lane_connectors_scaled_width_polygons"], "lane_connector_fid", str(lane_id)
        )
        assert lane_connector_polygons_row is not None, f"Could not find lane connector polygon with id {lane_id}"
        left_boundary_fid = lane_connector_polygons_row["left_boundary_fid"]
        left_boundary = get_row_with_value(nuplan_gdf["boundaries"], "fid", str(left_boundary_fid))["geometry"]  # type: ignore

        right_boundary_fid = lane_connector_polygons_row["right_boundary_fid"]
        right_boundary = get_row_with_value(nuplan_gdf["boundaries"], "fid", str(right_boundary_fid))["geometry"]  # type: ignore

        # 3. baseline_paths
        centerline = get_row_with_value(nuplan_gdf["baseline_paths"], "lane_connector_fid", float(lane_id))["geometry"]  # type: ignore

        left_boundary = align_boundary_direction(centerline, left_boundary)
        right_boundary = align_boundary_direction(centerline, right_boundary)

        yield Lane(
            object_id=int(lane_id),
            lane_type=LaneType.UNDEFINED,
            lane_group_id=all_lane_group_ids[idx],
            left_boundary=Polyline3D.from_linestring(left_boundary),
            right_boundary=Polyline3D.from_linestring(right_boundary),
            centerline=Polyline3D.from_linestring(centerline),
            left_lane_id=None,
            right_lane_id=None,
            predecessor_ids=predecessor_ids,
            successor_ids=successor_ids,
            speed_limit_mps=all_speed_limits_mps[idx],
            outline=None,
            shapely_polygon=lane_connector_polygons_row.geometry,
        )


def _iter_nuplan_lane_groups(nuplan_gdf: Dict[str, gpd.GeoDataFrame]) -> Iterator[LaneGroup]:
    """Yield LaneGroup objects from nuPlan lane group data."""
    ids = nuplan_gdf["lane_groups_polygons"].fid.to_list()

    for lane_group_id in ids:
        # 1. lane_ids
        lane_ids = get_all_rows_with_value(
            nuplan_gdf["lanes_polygons"],
            "lane_group_fid",
            lane_group_id,
        )["fid"].tolist()  # type: ignore

        # 2. predecessor_lane_group_ids, successor_lane_group_ids
        predecessor_lane_group_ids = get_all_rows_with_value(
            nuplan_gdf["lane_group_connectors"],
            "to_lane_group_fid",
            lane_group_id,
        )["fid"].tolist()  # type: ignore
        successor_lane_group_ids = get_all_rows_with_value(
            nuplan_gdf["lane_group_connectors"],
            "from_lane_group_fid",
            lane_group_id,
        )["fid"].tolist()  # type: ignore

        # 3. left_boundaries, right_boundaries
        lane_group_row = get_row_with_value(nuplan_gdf["lane_groups_polygons"], "fid", str(lane_group_id))
        left_boundary_fid = lane_group_row["left_boundary_fid"]  # type: ignore
        left_boundary = get_row_with_value(nuplan_gdf["boundaries"], "fid", str(left_boundary_fid))["geometry"]  # type: ignore

        right_boundary_fid = lane_group_row["right_boundary_fid"]  # type: ignore
        right_boundary = get_row_with_value(nuplan_gdf["boundaries"], "fid", str(right_boundary_fid))["geometry"]  # type: ignore

        # Flip the boundaries to align with the first lane's baseline path direction.
        repr_centerline = get_row_with_value(nuplan_gdf["baseline_paths"], "lane_fid", float(lane_ids[0]))["geometry"]  # type: ignore

        left_boundary = align_boundary_direction(repr_centerline, left_boundary)
        right_boundary = align_boundary_direction(repr_centerline, right_boundary)

        yield LaneGroup(
            object_id=lane_group_id,
            lane_ids=lane_ids,
            left_boundary=Polyline3D.from_linestring(left_boundary),
            right_boundary=Polyline3D.from_linestring(right_boundary),
            intersection_id=None,
            predecessor_ids=predecessor_lane_group_ids,
            successor_ids=successor_lane_group_ids,
            outline=None,
            shapely_polygon=lane_group_row.geometry,  # type: ignore
        )


def _iter_nuplan_lane_connector_groups(nuplan_gdf: Dict[str, gpd.GeoDataFrame]) -> Iterator[LaneGroup]:
    """Yield LaneGroup objects from nuPlan lane connector groups (lane groups on intersections)."""
    ids = nuplan_gdf["lane_group_connectors"].fid.to_list()
    all_intersection_ids = nuplan_gdf["lane_group_connectors"].intersection_fid.to_list()

    for idx, lane_group_connector_id in enumerate(ids):
        # 1. lane_ids
        lane_ids = get_all_rows_with_value(
            nuplan_gdf["lane_connectors"], "lane_group_connector_fid", lane_group_connector_id
        )["fid"].tolist()  # type: ignore

        # 2. predecessor_lane_group_ids, successor_lane_group_ids
        lane_group_connector_row = get_row_with_value(
            nuplan_gdf["lane_group_connectors"], "fid", lane_group_connector_id
        )
        predecessor_lane_group_ids = [str(lane_group_connector_row["from_lane_group_fid"])]  # type: ignore
        successor_lane_group_ids = [str(lane_group_connector_row["to_lane_group_fid"])]  # type: ignore

        # 3. left_boundaries, right_boundaries
        left_boundary_fid = lane_group_connector_row["left_boundary_fid"]  # type: ignore
        left_boundary = get_row_with_value(nuplan_gdf["boundaries"], "fid", str(left_boundary_fid))["geometry"]  # type: ignore
        right_boundary_fid = lane_group_connector_row["right_boundary_fid"]  # type: ignore
        right_boundary = get_row_with_value(nuplan_gdf["boundaries"], "fid", str(right_boundary_fid))["geometry"]  # type: ignore

        yield LaneGroup(
            object_id=int(lane_group_connector_id),
            lane_ids=lane_ids,
            left_boundary=Polyline3D.from_linestring(left_boundary),
            right_boundary=Polyline3D.from_linestring(right_boundary),
            intersection_id=all_intersection_ids[idx],
            predecessor_ids=predecessor_lane_group_ids,  # type: ignore
            successor_ids=successor_lane_group_ids,  # type: ignore
            outline=None,
            shapely_polygon=lane_group_connector_row.geometry,  # type: ignore
        )


def _iter_nuplan_intersections(nuplan_gdf: Dict[str, gpd.GeoDataFrame]) -> Iterator[Intersection]:
    """Yield Intersection objects from nuPlan intersection data."""
    all_ids = nuplan_gdf["intersections"].fid.to_list()
    all_intersection_type_fids = nuplan_gdf["intersections"].intersection_type_fid.to_list()
    all_geometries = nuplan_gdf["intersections"].geometry.to_list()
    for idx, intersection_id in enumerate(all_ids):
        lane_group_connector_ids = get_all_rows_with_value(
            nuplan_gdf["lane_group_connectors"], "intersection_fid", str(intersection_id)
        )["fid"].tolist()  # type: ignore

        intersection_type = NUPLAN_INTERSECTION_TYPE_CONVERSION.get(
            all_intersection_type_fids[idx], IntersectionType.DEFAULT
        )

        yield Intersection(
            object_id=int(intersection_id),
            intersection_type=intersection_type,
            lane_group_ids=lane_group_connector_ids,
            shapely_polygon=all_geometries[idx],  # type: ignore
        )


def _iter_nuplan_crosswalks(nuplan_gdf: Dict[str, gpd.GeoDataFrame]) -> Iterator[Crosswalk]:
    """Yield Crosswalk objects from nuPlan crosswalk data."""
    for id, geometry in zip(nuplan_gdf["crosswalks"].fid.to_list(), nuplan_gdf["crosswalks"].geometry.to_list()):
        yield Crosswalk(object_id=int(id), shapely_polygon=geometry)  # type: ignore


def _iter_nuplan_walkways(nuplan_gdf: Dict[str, gpd.GeoDataFrame]) -> Iterator[Walkway]:
    """Yield Walkway objects from nuPlan walkway data."""
    for id, geometry in zip(nuplan_gdf["walkways"].fid.to_list(), nuplan_gdf["walkways"].geometry.to_list()):
        yield Walkway(object_id=int(id), shapely_polygon=geometry)  # type: ignore


def _iter_nuplan_carparks(nuplan_gdf: Dict[str, gpd.GeoDataFrame]) -> Iterator[Carpark]:
    """Yield Carpark objects from nuPlan carpark data."""
    for id, geometry in zip(nuplan_gdf["carpark_areas"].fid.to_list(), nuplan_gdf["carpark_areas"].geometry.to_list()):
        yield Carpark(object_id=int(id), shapely_polygon=geometry)  # type: ignore


def _iter_nuplan_generic_drivables(nuplan_gdf: Dict[str, gpd.GeoDataFrame]) -> Iterator[GenericDrivable]:
    """Yield GenericDrivable objects from nuPlan generic drivable area data."""
    for id, geometry in zip(
        nuplan_gdf["generic_drivable_areas"].fid.to_list(), nuplan_gdf["generic_drivable_areas"].geometry.to_list()
    ):
        yield GenericDrivable(object_id=int(id), shapely_polygon=geometry)  # type: ignore


def _iter_nuplan_stop_zones(nuplan_gdf: Dict[str, gpd.GeoDataFrame]) -> Iterator[StopZone]:
    """Yield StopZone objects from nuPlan stop zone data."""
    stop_polygons = nuplan_gdf["stop_polygons"]
    all_ids = stop_polygons.fid.to_list()
    all_types = stop_polygons.stop_polygon_type_fid.to_list()
    all_lane_fids = stop_polygons.lane_fids.to_list()
    all_geometries = stop_polygons.geometry.to_list()

    for idx, stop_id in enumerate(all_ids):
        stop_zone_type = NUPLAN_STOP_ZONE_TYPE_CONVERSION.get(all_types[idx], StopZoneType.UNKNOWN)
        if pd.notna(all_lane_fids[idx]) and all_lane_fids[idx]:
            lane_ids = [int(fid) for fid in all_lane_fids[idx].split(",") if fid]
        else:
            lane_ids = None

        yield StopZone(
            object_id=int(stop_id),
            stop_zone_type=stop_zone_type,
            shapely_polygon=all_geometries[idx],  # type: ignore
            lane_ids=lane_ids,
        )


def _iter_nuplan_road_edges(nuplan_gdf: Dict[str, gpd.GeoDataFrame]) -> Iterator[RoadEdge]:
    """Yield RoadEdge objects derived from nuPlan drivable area boundaries."""
    drivable_polygons = (
        nuplan_gdf["intersections"].geometry.to_list()
        + nuplan_gdf["lane_groups_polygons"].geometry.to_list()
        + nuplan_gdf["carpark_areas"].geometry.to_list()
        + nuplan_gdf["generic_drivable_areas"].geometry.to_list()
    )
    road_edge_linear_rings = get_road_edge_linear_rings(drivable_polygons)  # type: ignore
    road_edges = split_line_geometry_by_max_length(road_edge_linear_rings, MAX_ROAD_EDGE_LENGTH)  # type: ignore

    for idx in range(len(road_edges)):
        yield RoadEdge(
            object_id=int(idx),
            road_edge_type=RoadEdgeType.ROAD_EDGE_BOUNDARY,
            polyline=Polyline2D.from_linestring(road_edges[idx]),
        )


def _iter_nuplan_road_lines(nuplan_gdf: Dict[str, gpd.GeoDataFrame]) -> Iterator[RoadLine]:
    """Yield RoadLine objects from nuPlan boundary data."""
    boundaries = nuplan_gdf["boundaries"].geometry.to_list()
    fids = nuplan_gdf["boundaries"].fid.to_list()
    boundary_types = nuplan_gdf["boundaries"].boundary_type_fid.to_list()

    for idx in range(len(boundary_types)):
        yield RoadLine(
            object_id=int(fids[idx]),
            road_line_type=NUPLAN_ROAD_LINE_CONVERSION[boundary_types[idx]],
            polyline=Polyline2D.from_linestring(boundaries[idx]),  # type: ignore
        )


# GeoDataFrame loading and geometry helpers
# ----------------------------------------------------------------------------------------------------------------------


def _load_nuplan_gdf(map_file_path: Path) -> Dict[str, gpd.GeoDataFrame]:
    """Load nuPlan map data from a GPKG file into a dictionary of GeoDataFrames."""

    # The projected coordinate system depends on which UTM zone the mapped location is in.
    map_meta = gpd.read_file(map_file_path, layer="meta", engine="pyogrio")
    projection_system = map_meta[map_meta["key"] == "projectedCoordSystem"]["value"].iloc[0]

    nuplan_gdf: Dict[str, gpd.GeoDataFrame] = {}
    for layer_name in NUPLAN_MAP_GPKG_LAYERS:
        with warnings.catch_warnings():
            # Suppress the warnings from the GPKG operations below so that they don't spam the training logs.
            warnings.filterwarnings("ignore")

            gdf_in_pixel_coords = pyogrio.read_dataframe(map_file_path, layer=layer_name, fid_as_index=True)
            gdf_in_utm_coords = gdf_in_pixel_coords.to_crs(projection_system)  # type: ignore

            # For backwards compatibility, cast the index to string datatype.
            #   and mirror it to the "fid" column.
            gdf_in_utm_coords.index = gdf_in_utm_coords.index.map(str)
            gdf_in_utm_coords["fid"] = gdf_in_utm_coords.index

        nuplan_gdf[layer_name] = gdf_in_utm_coords

    return nuplan_gdf


def _flip_linestring(linestring: LineString) -> LineString:
    """Flips the direction of a shapely LineString."""
    return LineString(linestring.coords[::-1])


def lines_same_direction(centerline: LineString, boundary: LineString) -> bool:
    """Check if the boundary LineString is in the same direction as the centerline LineString."""
    center_start = np.array(centerline.coords[0])
    center_end = np.array(centerline.coords[-1])
    boundary_start = np.array(boundary.coords[0])
    boundary_end = np.array(boundary.coords[-1])

    # Distance from centerline start to boundary start + centerline end to boundary end
    same_dir_dist = np.linalg.norm(center_start - boundary_start) + np.linalg.norm(center_end - boundary_end)
    opposite_dir_dist = np.linalg.norm(center_start - boundary_end) + np.linalg.norm(center_end - boundary_start)

    return bool(same_dir_dist <= opposite_dir_dist)


def align_boundary_direction(centerline: LineString, boundary: LineString) -> LineString:
    """Aligns the boundary LineString direction to be the same as the centerline LineString direction."""
    if not lines_same_direction(centerline, boundary):
        return _flip_linestring(boundary)
    return boundary


def get_all_rows_with_value(
    elements: gpd.geodataframe.GeoDataFrame, column_label: str, desired_value: str
) -> Optional[gpd.geodataframe.GeoDataFrame]:
    """Extract all matching elements.

    :param elements: data frame from MapsDb.
    :param column_label: key to extract from a column.
    :param desired_value: key which is compared with the values of column_label entry.
    :return: a subset of the original GeoDataFrame containing the matching key.
    """
    if desired_value is None or pd.isna(desired_value):
        return None

    mask = elements[column_label].notna()
    valid_elements = elements[mask]

    return valid_elements.iloc[np.where(valid_elements[column_label].to_numpy().astype(int) == int(desired_value))]  # type: ignore


def get_row_with_value(
    elements: gpd.geodataframe.GeoDataFrame, column_label: str, desired_value: str
) -> Optional[pd.Series]:
    """Extract a matching element.

    :param elements: data frame from MapsDb.
    :param column_label: key to extract from a column.
    :param desired_value: key which is compared with the values of column_label entry.
    :return row from GeoDataFrame.
    """
    if column_label == "fid":
        return elements.loc[desired_value]  # pyright: ignore[reportReturnType]

    geo_series: Optional[pd.Series] = None
    matching_rows = get_all_rows_with_value(elements, column_label, desired_value)
    if matching_rows is not None:
        assert len(matching_rows) > 0, f"Could not find the desired key = {desired_value}"
        assert len(matching_rows) == 1, (
            f"{len(matching_rows)} matching keys found. Expected to only find one. Try using get_all_rows_with_value"
        )
        geo_series = matching_rows.iloc[0]
    return geo_series
