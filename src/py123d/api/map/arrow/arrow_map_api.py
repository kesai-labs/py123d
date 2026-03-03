from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Final, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
import shapely
import shapely.geometry as geom

from py123d.api.map.map_api import MapAPI
from py123d.api.scene.arrow.utils.arrow_metadata_utils import get_map_metadata_from_arrow_table
from py123d.api.utils.arrow_helper import get_lru_cached_arrow_table
from py123d.common.dataset_paths import get_dataset_paths
from py123d.common.utils.msgpack_utils import msgpack_decode_with_numpy
from py123d.datatypes.map_objects.base_map_objects import BaseMapObject, MapObjectIDType
from py123d.datatypes.map_objects.map_layer_types import MapLayer, RoadEdgeType, RoadLineType, StopZoneType
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
from py123d.datatypes.metadata.map_metadata import MapMetadata
from py123d.geometry import OccupancyMap2D, Point2D, Point3D, Polyline3D

# TODO: add to some configs
MAX_LRU_CACHED_TABLES: Final[int] = 128
MAP_OBJECT_CACHE_SIZE: Final[int] = 10_000


def _load_map_layers_from_arrow_table(
    arrow_table: pa.Table,
) -> Tuple[Dict[MapLayer, OccupancyMap2D], Dict[MapLayer, Dict[MapObjectIDType, int]]]:
    """Helper function to load map layers from an Arrow table into occupancy maps and object ID mappings."""

    all_object_ids = arrow_table.column("object_id").to_numpy()
    all_map_layers = arrow_table.column("map_layer").to_numpy()
    all_wkbs = arrow_table.column("wkb").to_numpy()
    geometries = shapely.from_wkb(all_wkbs)

    occupancy_map_dict: Dict[MapLayer, OccupancyMap2D] = {}
    object_ids_to_row_idx: Dict[MapLayer, Dict[MapObjectIDType, int]] = {}

    layer_indices = {}
    for layer in MapLayer:
        layer_mask = all_map_layers == int(layer)
        layer_idx = np.where(layer_mask)[0]
        if len(layer_idx) > 0:
            layer_indices[layer] = layer_idx

    for layer, indices in layer_indices.items():
        layer_object_ids = all_object_ids[indices].tolist()
        layer_geometries = geometries[indices].tolist()
        object_ids_to_row_idx[layer] = {obj_id: idx for obj_id, idx in zip(layer_object_ids, indices)}
        occupancy_map = OccupancyMap2D(geometries=layer_geometries, ids=layer_object_ids)
        occupancy_map_dict[layer] = occupancy_map

    return occupancy_map_dict, object_ids_to_row_idx


class ArrowMapAPI(MapAPI):
    def __init__(self, file_path: Union[Path, str]) -> None:
        """Initialize a ArrowMapAPI instance.

        :param file_path: The file path to the Arrow file.
        """

        self._file_path = Path(file_path)
        self._map_object_getter: Dict[MapLayer, Callable[[MapObjectIDType], Optional[BaseMapObject]]] = {
            MapLayer.LANE: self._get_lane,
            MapLayer.LANE_GROUP: self._get_lane_group,
            MapLayer.INTERSECTION: self._get_intersection,
            MapLayer.CROSSWALK: self._get_crosswalk,
            MapLayer.CARPARK: self._get_carpark,
            MapLayer.WALKWAY: self._get_walkway,
            MapLayer.GENERIC_DRIVABLE: self._get_generic_drivable,
            MapLayer.STOP_ZONE: self._get_stop_zone,
            MapLayer.ROAD_EDGE: self._get_road_edge,
            MapLayer.ROAD_LINE: self._get_road_line,
        }

        _map_table = get_lru_cached_arrow_table(str(self._file_path))
        self._map_metadata: MapMetadata = get_map_metadata_from_arrow_table(_map_table)
        _occupancy_maps, _object_ids_to_row_idx = _load_map_layers_from_arrow_table(_map_table)
        self._occupancy_maps: Dict[MapLayer, OccupancyMap2D] = _occupancy_maps
        self._object_ids_to_row_idx: Dict[MapLayer, Dict[MapObjectIDType, int]] = _object_ids_to_row_idx
        self._features = _map_table["features"].to_numpy()

    def get_map_metadata(self):
        """Inherited, see superclass."""
        return self._map_metadata

    def get_available_map_layers(self) -> List[MapLayer]:
        """Inherited, see superclass."""
        return list(self._occupancy_maps.keys())

    def get_map_object(self, object_id: MapObjectIDType, layer: MapLayer) -> Optional[BaseMapObject]:
        """Inherited, see superclass."""
        return self._map_object_getter[layer](object_id)

    def get_map_objects_in_radius(
        self,
        point: Union[Point2D, Point3D],
        radius: float,
        layers: List[MapLayer],
    ) -> Dict[MapLayer, List[BaseMapObject]]:
        """Inherited, see superclass."""
        x_min, x_max = point.x - radius, point.x + radius
        y_min, y_max = point.y - radius, point.y + radius
        patch = geom.box(x_min, y_min, x_max, y_max)
        return self.query(geometry=patch, layers=layers, predicate="intersects")  # type: ignore

    def query(
        self,
        geometry: Union[geom.base.BaseGeometry, Iterable[geom.base.BaseGeometry]],
        layers: List[MapLayer],
        predicate: Optional[
            Literal[
                "contains",
                "contains_properly",
                "covered_by",
                "covers",
                "crosses",
                "intersects",
                "overlaps",
                "touches",
                "within",
                "dwithin",
            ]
        ] = None,
        distance: Optional[float] = None,
    ) -> Dict[MapLayer, Union[List[BaseMapObject], Dict[int, List[BaseMapObject]]]]:
        """Inherited, see superclass."""
        object_map: Dict[MapLayer, Union[List[BaseMapObject], Dict[int, List[BaseMapObject]]]] = defaultdict(list)
        for layer in layers:
            object_map[layer] = self._query_layer(geometry, layer, predicate, distance)
        return object_map

    def query_object_ids(
        self,
        geometry: Union[geom.base.BaseGeometry, Iterable[geom.base.BaseGeometry]],
        layers: List[MapLayer],
        predicate: Optional[
            Literal[
                "contains",
                "contains_properly",
                "covered_by",
                "covers",
                "crosses",
                "intersects",
                "overlaps",
                "touches",
                "within",
                "dwithin",
            ]
        ] = None,
        distance: Optional[float] = None,
    ) -> Dict[MapLayer, Union[List[MapObjectIDType], Dict[int, List[MapObjectIDType]]]]:
        """Inherited, see superclass."""
        object_map: Dict[MapLayer, Union[List[MapObjectIDType], Dict[int, List[MapObjectIDType]]]] = defaultdict(list)
        for layer in layers:
            object_map[layer] = self._query_layer_objects_ids(geometry, layer, predicate, distance)
        return object_map

    def _get_map_table(self) -> pa.Table:
        """Helper method to get the Arrow map table."""
        return get_lru_cached_arrow_table(str(self._file_path))

    def _query_layer(
        self,
        geometry: Union[shapely.Geometry, Iterable[shapely.Geometry]],
        layer: MapLayer,
        predicate: Optional[str] = None,
        distance: Optional[float] = None,
    ) -> Union[List[BaseMapObject], Dict[int, List[BaseMapObject]]]:
        """Helper method to query a single layer."""
        if layer not in self._occupancy_maps.keys():
            return {} if not isinstance(geometry, Iterable) else []
        occupancy_map = self._occupancy_maps[layer]
        query_result = occupancy_map.query(geometry, predicate=predicate, distance=distance)  # type: ignore
        if query_result.ndim == 2:
            query_dict: Dict[int, List[BaseMapObject]] = defaultdict(list)
            for geometry_idx, occ_idx in zip(query_result[0], query_result[1]):
                map_object_id = occupancy_map.ids[occ_idx]
                map_object = self.get_map_object(map_object_id, layer)
                assert map_object is not None, (
                    f"Queried map object should exist. Cannot find object {map_object_id} in layer {layer.name}"
                )
                query_dict[int(geometry_idx)].append(map_object)
            return query_dict
        else:
            query_list: List[BaseMapObject] = []
            for idx in query_result:
                map_object = self._map_object_getter[layer](occupancy_map.ids[idx])
                assert map_object is not None, (
                    f"Queried map object should exist. Cannot find object {occupancy_map.ids[idx]} in layer {layer.name}"
                )
                query_list.append(map_object)
            return query_list

    def _query_layer_objects_ids(
        self,
        geometry: Union[geom.base.BaseGeometry, Iterable[geom.base.BaseGeometry]],
        layer: MapLayer,
        predicate: Optional[str] = None,
        distance: Optional[float] = None,
    ) -> Union[List[MapObjectIDType], Dict[int, List[MapObjectIDType]]]:
        """Helper method to query a single layer, while only returning object IDs."""
        if layer not in self._occupancy_maps.keys():
            return {} if not isinstance(geometry, Iterable) else []

        occupancy_map = self._occupancy_maps[layer]
        query_result = occupancy_map.query(geometry, predicate=predicate, distance=distance)  # type: ignore

        if query_result.ndim == 2:
            query_dict: Dict[int, List[MapObjectIDType]] = defaultdict(list)
            for geometry_idx, occ_idx in zip(query_result[0], query_result[1]):
                map_object_id = occupancy_map.ids[occ_idx]
                query_dict[int(geometry_idx)].append(map_object_id)
            return query_dict
        else:
            query_list = [occupancy_map.ids[idx] for idx in query_result]
            return query_list

    @lru_cache(maxsize=MAP_OBJECT_CACHE_SIZE)
    def _get_lane(self, object_id: MapObjectIDType) -> Optional[Lane]:
        """Helper method for getting a lane by its ID."""
        lane: Optional[Lane] = None
        table_row_idx = self._object_ids_to_row_idx[MapLayer.LANE].get(object_id, None)
        if table_row_idx is not None and object_id in self._occupancy_maps[MapLayer.LANE].ids:
            lane_features_binary = self._features[table_row_idx]
            lane_features = msgpack_decode_with_numpy(lane_features_binary)
            lane_polygon = self._occupancy_maps[MapLayer.LANE][object_id]
            assert isinstance(lane_polygon, geom.Polygon)
            lane = Lane(
                object_id=object_id,
                lane_group_id=lane_features["lane_group_id"],
                left_boundary=Polyline3D.from_array(lane_features["left_boundary"], copy=False),
                right_boundary=Polyline3D.from_array(lane_features["right_boundary"], copy=False),
                centerline=Polyline3D.from_array(lane_features["centerline"], copy=False),
                left_lane_id=lane_features["left_lane_id"],
                right_lane_id=lane_features["right_lane_id"],
                predecessor_ids=lane_features["predecessor_ids"],
                successor_ids=lane_features["successor_ids"],
                speed_limit_mps=lane_features["speed_limit_mps"],
                outline=Polyline3D.from_array(lane_features["outline"], copy=False),
                shapely_polygon=lane_polygon,
                map_api=self,
            )
        return lane

    @lru_cache(maxsize=MAP_OBJECT_CACHE_SIZE)
    def _get_lane_group(self, object_id: MapObjectIDType) -> Optional[LaneGroup]:
        """Helper method for getting a lane group by its ID."""
        lane_group: Optional[LaneGroup] = None
        table_row_idx = self._object_ids_to_row_idx[MapLayer.LANE_GROUP].get(object_id, None)
        if table_row_idx is not None and object_id in self._occupancy_maps[MapLayer.LANE_GROUP].ids:
            lane_group_features_binary = self._features[table_row_idx]
            lane_group_features = msgpack_decode_with_numpy(lane_group_features_binary)
            lane_group_polygon = self._occupancy_maps[MapLayer.LANE_GROUP][object_id]
            assert isinstance(lane_group_polygon, geom.Polygon)
            lane_group = LaneGroup(
                object_id=object_id,
                lane_ids=lane_group_features["lane_ids"],
                left_boundary=Polyline3D.from_array(lane_group_features["left_boundary"], copy=False),
                right_boundary=Polyline3D.from_array(lane_group_features["right_boundary"], copy=False),
                intersection_id=lane_group_features["intersection_id"],
                predecessor_ids=lane_group_features["predecessor_ids"],
                successor_ids=lane_group_features["successor_ids"],
                outline=Polyline3D.from_array(lane_group_features["outline"], copy=False),
                shapely_polygon=lane_group_polygon,
                map_api=self,
            )
        return lane_group

    @lru_cache(maxsize=MAP_OBJECT_CACHE_SIZE)
    def _get_intersection(self, object_id: MapObjectIDType) -> Optional[Intersection]:
        """Helper method for getting an intersection by its ID."""
        intersection: Optional[Intersection] = None
        table_row_idx = self._object_ids_to_row_idx[MapLayer.INTERSECTION].get(object_id, None)
        if table_row_idx is not None and object_id in self._occupancy_maps[MapLayer.INTERSECTION].ids:
            intersection_features_binary = self._features[table_row_idx]
            intersection_features = msgpack_decode_with_numpy(intersection_features_binary)
            intersection_polygon = self._occupancy_maps[MapLayer.INTERSECTION][object_id]
            assert isinstance(intersection_polygon, geom.Polygon)
            intersection = Intersection(
                object_id=object_id,
                lane_group_ids=intersection_features["lane_group_ids"],
                outline=Polyline3D.from_array(intersection_features["outline"], copy=False),
                shapely_polygon=intersection_polygon,
                map_api=self,
            )
        return intersection

    @lru_cache(maxsize=MAP_OBJECT_CACHE_SIZE)
    def _get_crosswalk(self, object_id: MapObjectIDType) -> Optional[Crosswalk]:
        """Helper method for getting a crosswalk by its ID."""
        crosswalk: Optional[Crosswalk] = None
        table_row_idx = self._object_ids_to_row_idx[MapLayer.CROSSWALK].get(object_id, None)
        if table_row_idx is not None and object_id in self._occupancy_maps[MapLayer.CROSSWALK].ids:
            crosswalk_features_binary = self._features[table_row_idx]
            crosswalk_features = msgpack_decode_with_numpy(crosswalk_features_binary)
            crosswalk_polygon = self._occupancy_maps[MapLayer.CROSSWALK][object_id]
            assert isinstance(crosswalk_polygon, geom.Polygon)
            crosswalk = Crosswalk(
                object_id=object_id,
                outline=Polyline3D.from_array(crosswalk_features["outline"], copy=False),
                shapely_polygon=crosswalk_polygon,
            )
        return crosswalk

    @lru_cache(maxsize=MAP_OBJECT_CACHE_SIZE)
    def _get_carpark(self, object_id: MapObjectIDType) -> Optional[Carpark]:
        """Helper method for getting a carpark by its ID."""
        carpark: Optional[Carpark] = None
        table_row_idx = self._object_ids_to_row_idx[MapLayer.CARPARK].get(object_id, None)
        if table_row_idx is not None and object_id in self._occupancy_maps[MapLayer.CARPARK].ids:
            carpark_features_binary = self._features[table_row_idx]
            carpark_features = msgpack_decode_with_numpy(carpark_features_binary)
            carpark_polygon = self._occupancy_maps[MapLayer.CARPARK][object_id]
            assert isinstance(carpark_polygon, geom.Polygon)
            carpark = Carpark(
                object_id=object_id,
                outline=Polyline3D.from_array(carpark_features["outline"], copy=False),
                shapely_polygon=carpark_polygon,
            )
        return carpark

    @lru_cache(maxsize=MAP_OBJECT_CACHE_SIZE)
    def _get_walkway(self, object_id: MapObjectIDType) -> Optional[Walkway]:
        """Helper method for getting a walkway by its ID."""
        walkway: Optional[Walkway] = None
        table_row_idx = self._object_ids_to_row_idx[MapLayer.WALKWAY].get(object_id, None)
        if table_row_idx is not None and object_id in self._occupancy_maps[MapLayer.WALKWAY].ids:
            walkway_features_binary = self._features[table_row_idx]
            walkway_features = msgpack_decode_with_numpy(walkway_features_binary)
            walkway_polygon = self._occupancy_maps[MapLayer.WALKWAY][object_id]
            assert isinstance(walkway_polygon, geom.Polygon)
            walkway = Walkway(
                object_id=object_id,
                outline=Polyline3D.from_array(walkway_features["outline"], copy=False),
                shapely_polygon=walkway_polygon,
            )
        return walkway

    @lru_cache(maxsize=MAP_OBJECT_CACHE_SIZE)
    def _get_generic_drivable(self, object_id: MapObjectIDType) -> Optional[GenericDrivable]:
        """Helper method for getting a generic drivable area by its ID."""
        generic_drivable: Optional[GenericDrivable] = None
        table_row_idx = self._object_ids_to_row_idx[MapLayer.GENERIC_DRIVABLE].get(object_id, None)
        if table_row_idx is not None and object_id in self._occupancy_maps[MapLayer.GENERIC_DRIVABLE].ids:
            generic_drivable_features_binary = self._features[table_row_idx]
            generic_drivable_features = msgpack_decode_with_numpy(generic_drivable_features_binary)
            generic_drivable_polygon = self._occupancy_maps[MapLayer.GENERIC_DRIVABLE][object_id]
            assert isinstance(generic_drivable_polygon, geom.Polygon)
            generic_drivable = GenericDrivable(
                object_id=object_id,
                outline=Polyline3D.from_array(generic_drivable_features["outline"], copy=False),
                shapely_polygon=generic_drivable_polygon,
            )
        return generic_drivable

    @lru_cache(maxsize=MAP_OBJECT_CACHE_SIZE)
    def _get_stop_zone(self, object_id: MapObjectIDType) -> Optional[StopZone]:
        """Helper method for getting a stop zone area by its ID."""
        stop_zone: Optional[StopZone] = None
        table_row_idx = self._object_ids_to_row_idx[MapLayer.STOP_ZONE].get(object_id, None)
        if table_row_idx is not None and object_id in self._occupancy_maps[MapLayer.STOP_ZONE].ids:
            stop_zone_features_binary = self._features[table_row_idx]
            stop_zone_features = msgpack_decode_with_numpy(stop_zone_features_binary)
            stop_zone_polygon = self._occupancy_maps[MapLayer.STOP_ZONE][object_id]
            assert isinstance(stop_zone_polygon, geom.Polygon)
            stop_zone = StopZone(
                object_id=object_id,
                stop_zone_type=StopZoneType(stop_zone_features["stop_zone_type"]),
                outline=Polyline3D.from_array(stop_zone_features["outline"], copy=False),
                shapely_polygon=stop_zone_polygon,
                lane_ids=stop_zone_features.get("lane_ids", []),
            )
        return stop_zone

    @lru_cache(maxsize=MAP_OBJECT_CACHE_SIZE)
    def _get_road_edge(self, object_id: MapObjectIDType) -> Optional[RoadEdge]:
        """Helper method for getting a road edge by its ID."""
        road_edge: Optional[RoadEdge] = None
        table_row_idx = self._object_ids_to_row_idx[MapLayer.ROAD_EDGE].get(object_id, None)
        if table_row_idx is not None and object_id in self._occupancy_maps[MapLayer.ROAD_EDGE].ids:
            road_edge_features_binary = self._features[table_row_idx]
            road_edge_features = msgpack_decode_with_numpy(road_edge_features_binary)
            road_edge_linestring = self._occupancy_maps[MapLayer.ROAD_EDGE][object_id]
            assert isinstance(road_edge_linestring, geom.LineString)
            road_edge = RoadEdge(
                object_id=object_id,
                road_edge_type=RoadEdgeType(road_edge_features["road_edge_type"]),
                polyline=Polyline3D.from_linestring(road_edge_linestring),
            )
        return road_edge

    @lru_cache(maxsize=MAP_OBJECT_CACHE_SIZE)
    def _get_road_line(self, object_id: MapObjectIDType) -> Optional[RoadLine]:
        """Helper method for getting a road line by its ID."""
        road_line: Optional[RoadLine] = None
        table_row_idx = self._object_ids_to_row_idx[MapLayer.ROAD_LINE].get(object_id, None)
        if table_row_idx is not None and object_id in self._occupancy_maps[MapLayer.ROAD_LINE].ids:
            road_line_features_binary = self._features[table_row_idx]
            road_line_features = msgpack_decode_with_numpy(road_line_features_binary)
            road_line_linestring = self._occupancy_maps[MapLayer.ROAD_LINE][object_id]
            assert isinstance(road_line_linestring, geom.LineString)
            road_line = RoadLine(
                object_id=object_id,
                road_line_type=RoadLineType(road_line_features["road_line_type"]),
                polyline=Polyline3D.from_linestring(road_line_linestring),
            )
        return road_line


@lru_cache(maxsize=MAX_LRU_CACHED_TABLES)
def get_global_map_api(dataset: str, location: str) -> ArrowMapAPI:
    """Get the global map API for a given dataset and location."""
    PY123D_MAPS_ROOT: Optional[Path] = get_dataset_paths().py123d_maps_root
    assert PY123D_MAPS_ROOT is not None, (
        "PY123D_MAPS_ROOT is not configured. Please set the environment variable or configure dataset paths accordingly."
    )
    arrow_path = PY123D_MAPS_ROOT / dataset / f"{dataset}_{location}.arrow"
    assert arrow_path.is_file(), f"{dataset}_{location}.arrow not found in {str(PY123D_MAPS_ROOT)}."
    map_api = ArrowMapAPI(arrow_path)
    return map_api


def get_local_map_api(split_name: str, log_name: str) -> ArrowMapAPI:
    """Get the local map API for a given split name and log name."""
    PY123D_MAPS_ROOT: Optional[Path] = get_dataset_paths().py123d_maps_root
    assert PY123D_MAPS_ROOT is not None, (
        "PY123D_MAPS_ROOT is not configured. Please set the environment variable or configure dataset paths accordingly."
    )
    arrow_path = PY123D_MAPS_ROOT / split_name / f"{log_name}.arrow"
    assert arrow_path.is_file(), f"{log_name}.arrow not found in {str(PY123D_MAPS_ROOT)}."
    map_api = ArrowMapAPI(arrow_path)
    return map_api
