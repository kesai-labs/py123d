from typing import Dict, Iterable, List, Optional, Union

import shapely.geometry as geom
from typing_extensions import Literal

from py123d.api import MapAPI
from py123d.datatypes.map_objects import BaseMapObject, MapLayer
from py123d.datatypes.map_objects.base_map_objects import MapObjectIDType
from py123d.datatypes.map_objects.map_objects import (
    Carpark,
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
from py123d.geometry import Point2D, Point3D


class MockMapAPI(MapAPI):
    def __init__(
        self,
        lanes: List[Lane] = [],
        lane_groups: List[LaneGroup] = [],
        intersections: List[Intersection] = [],
        crosswalks: List[Intersection] = [],
        carparks: List[Carpark] = [],
        walkways: List[Walkway] = [],
        generic_drivables: List[GenericDrivable] = [],
        stop_zones: List[StopZone] = [],
        road_edges: List[RoadEdge] = [],
        road_lines: List[RoadLine] = [],
        add_map_api_links: bool = False,
    ):
        self._layers: Dict[MapLayer, List[BaseMapObject]] = {
            MapLayer.LANE: lanes,
            MapLayer.LANE_GROUP: lane_groups,
            MapLayer.INTERSECTION: intersections,
            MapLayer.CROSSWALK: crosswalks,
            MapLayer.WALKWAY: walkways,
            MapLayer.CARPARK: carparks,
            MapLayer.GENERIC_DRIVABLE: generic_drivables,
            MapLayer.STOP_ZONE: stop_zones,
            MapLayer.ROAD_EDGE: road_edges,
            MapLayer.ROAD_LINE: road_lines,
        }

        for layer, layer_objects in self._layers.items():
            if layer in [
                MapLayer.LANE,
                MapLayer.LANE_GROUP,
                MapLayer.INTERSECTION,
            ]:
                for obj in layer_objects:
                    if add_map_api_links:
                        obj._map_api = self  # type: ignore
                    else:
                        obj._map_api = None  # type: ignore

    def get_map_metadata(self) -> MapMetadata:
        return MapMetadata(
            dataset="test",
            split="test_split",
            log_name="test_log_name",
            location="test_location",
            map_has_z=True,
            map_is_local=True,
        )

    def get_available_map_layers(self) -> List[MapLayer]:
        return list(self._layers.keys())

    def get_map_object(self, object_id: MapObjectIDType, layer: MapLayer) -> Optional[BaseMapObject]:
        target_layer = self._layers.get(layer, [])
        map_object: Optional[BaseMapObject] = None
        for obj in target_layer:
            if obj.object_id == object_id:
                map_object = obj
                break
        return map_object

    def get_map_objects_in_radius(
        self,
        point: Union[Point2D, Point3D],
        radius: float,
        layers: List[MapLayer],
    ) -> Dict[MapLayer, List[BaseMapObject]]:
        return {}

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
        return {}

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
        return {}
