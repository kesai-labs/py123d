from __future__ import annotations

import abc
from typing import Dict, FrozenSet, Iterable, Iterator, List, Literal, Optional, Sequence, Union, overload

import networkx as nx
import shapely.geometry as geom

from py123d.datatypes import MapMetadata
from py123d.datatypes.map_objects import (
    BaseMapObject,
    Carpark,
    Crosswalk,
    GenericDrivable,
    Intersection,
    Lane,
    LaneGroup,
    MapLayer,
    RoadEdge,
    RoadLine,
    SpeedBump,
    StopZone,
    TrafficLight,
    Walkway,
)
from py123d.datatypes.map_objects.base_map_objects import MapObjectIDType
from py123d.geometry import Point2D, Point3D

# Layers whose objects expose predecessor_ids/successor_ids — those that form a directed graph.
GRAPH_LAYERS: FrozenSet[MapLayer] = frozenset({MapLayer.LANE, MapLayer.LANE_GROUP})


class MapAPI(abc.ABC):
    """The base class for all map APIs in 123D."""

    # Abstract Methods, to be implemented by subclasses
    # ------------------------------------------------------------------------------------------------------------------

    @abc.abstractmethod
    def get_map_metadata(self) -> MapMetadata:
        """Returns the :class:`~py123d.datatypes.MapMetadata` of the map api.

        :return: The map metadata, e.g. location, dataset, etc.
        """

    @abc.abstractmethod
    def get_available_map_layers(self) -> List[MapLayer]:
        """Returns the available :class:`~py123d.datatypes.MapLayer`,
            e.g. LANE, LANE_GROUP, etc.

        :return: A list of available map layers.
        """

    @abc.abstractmethod
    def get_map_object_in_layer(
        self, object_id: MapObjectIDType, layer: Union[str, MapLayer]
    ) -> Optional[BaseMapObject]:
        """Returns a :class:`~py123d.datatypes.BaseMapObject` by its ID
            and :class:`~py123d.datatypes.MapLayer`.

        :param object_id: The ID of the map object.
        :param layer: The layer the map object belongs to. Accepts a :class:`MapLayer` or its lowercase name
            (e.g. ``"lane"``).
        :return: The map object if found, None otherwise.
        """

    @abc.abstractmethod
    def get_all_map_object_ids_in_layer(self, layer: Union[str, MapLayer]) -> List[MapObjectIDType]:
        """Returns a list of all map object IDs in a given layer.

        :param layer: The layer to retrieve object IDs from. Accepts a :class:`MapLayer` or its lowercase name
            (e.g. ``"lane"``).
        :return: A list of map object IDs in the specified layer.
        """

    @abc.abstractmethod
    def get_all_map_objects_in_layer(self, layer: Union[str, MapLayer]) -> Iterator[BaseMapObject]:
        """Returns an iterator of all :class:`~py123d.datatypes.BaseMapObject` in a given
            :class:`~py123d.datatypes.MapLayer`.

        :param layer: The map layer to retrieve objects from. Accepts a :class:`MapLayer` or its lowercase name
            (e.g. ``"lane"``).
        :return: An iterator of all map objects in the specified layer.
        """

    @abc.abstractmethod
    def get_all_map_objects_in_layers(self, layers: List[Union[str, MapLayer]]) -> Iterator[BaseMapObject]:
        """Returns an iterator of all :class:`~py123d.datatypes.BaseMapObject` in
            the specified layers.

        :param layers: A list of map layers. Each entry may be a :class:`MapLayer` or its lowercase name
            (e.g. ``"lane"``).
        :return: An iterator of all map objects in the specified layers
        """

    @abc.abstractmethod
    def get_map_objects_in_radius(
        self,
        point: Union[Point2D, Point3D],
        radius: float,
        layers: List[Union[str, MapLayer]],
    ) -> Dict[MapLayer, List[BaseMapObject]]:
        """Returns a dictionary of :class:`~py123d.datatypes.MapLayer` to a list of
            :class:`~py123d.datatypes.BaseMapObject` within a given radius
            around a center point.

        :param point: The center point to search around.
        :param radius: The radius to search within.
        :param layers: The map layers to search in. Each entry may be a :class:`MapLayer` or its lowercase name
            (e.g. ``"lane"``).
        :return: A dictionary mapping each layer to a list of map objects within the radius.
        """

    @abc.abstractmethod
    def query(
        self,
        geometry: Union[geom.base.BaseGeometry, Iterable[geom.base.BaseGeometry]],
        layers: List[Union[str, MapLayer]],
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
        """Queries geometries against the map objects in the specified layers using an optional spatial predicate.

        Notes
        -----
        The syntax is aligned with STRtree implementation of shapely and the corresponding ``query`` function [2]_.

        References
        ----------
        .. [2] https://shapely.readthedocs.io/en/latest/strtree.html#shapely.STRtree.query

        :param geometry: A shapely geometry or an iterable of shapely geometries to query against.
        :param layers: The map layers to query against. Each entry may be a :class:`MapLayer` or its lowercase name
            (e.g. ``"lane"``).
        :param predicate: An optional spatial predicate to filter the results.
        :param distance: An optional maximum distance to filter the results, defaults to None.
        :return:
            If geometry is a single geometry, a dictionary mapping each layer to a list of map objects.

            If geometry is an iterable of geometries, a dictionary mapping each layer to a dictionary of indices
            (of the input geometries) to lists of map objects (found in map).
        """

    @abc.abstractmethod
    def query_object_ids(
        self,
        geometry: Union[geom.base.BaseGeometry, Iterable[geom.base.BaseGeometry]],
        layers: List[Union[str, MapLayer]],
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
        """Queries geometries against the map objects in the specified layers using an optional spatial predicate.
        Only output the IDs of the matching map objects.

        Notes
        -----
        The syntax is aligned with STRtree implementation of shapely and the corresponding ``query`` function [3]_.

        References
        ----------
        .. [3] https://shapely.readthedocs.io/en/latest/strtree.html#shapely.STRtree.query


        :param geometry: A shapely geometry or an iterable of shapely geometries to query against.
        :param layers: The map layers to query against. Each entry may be a :class:`MapLayer` or its lowercase name
            (e.g. ``"lane"``).
        :param predicate: An optional spatial predicate to filter the results.
        :param distance: An optional maximum distance to filter the results, defaults to None.
        :return:
            If geometry is a single geometry, a dictionary mapping each layer to a list of map object ids.

            If geometry is an iterable of geometries, a dictionary mapping each layer to a dictionary of indices
            (of the input geometries) to lists of map object ids (found in map).
        """

    @abc.abstractmethod
    def get_layer_graph(self, layer: Union[str, MapLayer]) -> nx.DiGraph:
        """Returns a :class:`networkx.DiGraph` of predecessor/successor relations for a layer.

        Nodes are the object IDs of the layer; edges go from each predecessor to its successor.
        Only layers with predecessor/successor topology (see :data:`GRAPH_LAYERS`) are supported.

        :param layer: The map layer to build the graph for. Accepts a :class:`MapLayer` or its
            lowercase name (e.g. ``"lane"``).
        :raises ValueError: If the layer does not expose predecessor/successor topology.
        :return: A directed graph whose node set equals the IDs in the given layer.
        """

    # Syntactic Sugar / Properties, for easier access to common attributes
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def map_metadata(self) -> MapMetadata:
        """The :class:`~py123d.datatypes.MapMetadata` of the map api."""
        return self.get_map_metadata()

    @property
    def dataset(self) -> str:
        """The dataset name from the map metadata."""
        return self.map_metadata.dataset

    @property
    def location(self) -> Optional[str]:
        """The location from the map metadata."""
        return self.map_metadata.location

    @property
    def map_is_per_log(self) -> bool:
        """Indicates if the map is per-log (map for each log) or global (map for multiple logs in dataset)."""
        return self.map_metadata.map_is_per_log

    @property
    def map_has_z(self) -> bool:
        """Indicates if the map includes Z (elevation) data."""
        return self.map_metadata.map_has_z

    @property
    def version(self) -> str:
        """The version of the py123d library used to create this map metadata."""
        return self.map_metadata.version

    @property
    def available_map_layers(self) -> List[MapLayer]:
        """The available :class:`~py123d.datatypes.MapLayer` in the map."""
        return self.get_available_map_layers()

    @overload
    def get_layer_objects_in_radius(
        self, point: Union[Point2D, Point3D], radius: float, layer: Literal[MapLayer.LANE]
    ) -> Sequence[Lane]: ...

    @overload
    def get_layer_objects_in_radius(
        self, point: Union[Point2D, Point3D], radius: float, layer: Literal[MapLayer.LANE_GROUP]
    ) -> Sequence[LaneGroup]: ...

    @overload
    def get_layer_objects_in_radius(
        self, point: Union[Point2D, Point3D], radius: float, layer: Literal[MapLayer.INTERSECTION]
    ) -> Sequence[Intersection]: ...

    @overload
    def get_layer_objects_in_radius(
        self, point: Union[Point2D, Point3D], radius: float, layer: Literal[MapLayer.CROSSWALK]
    ) -> Sequence[Crosswalk]: ...

    @overload
    def get_layer_objects_in_radius(
        self, point: Union[Point2D, Point3D], radius: float, layer: Literal[MapLayer.WALKWAY]
    ) -> Sequence[Walkway]: ...

    @overload
    def get_layer_objects_in_radius(
        self, point: Union[Point2D, Point3D], radius: float, layer: Literal[MapLayer.CARPARK]
    ) -> Sequence[Carpark]: ...

    @overload
    def get_layer_objects_in_radius(
        self, point: Union[Point2D, Point3D], radius: float, layer: Literal[MapLayer.GENERIC_DRIVABLE]
    ) -> Sequence[GenericDrivable]: ...

    @overload
    def get_layer_objects_in_radius(
        self, point: Union[Point2D, Point3D], radius: float, layer: Literal[MapLayer.STOP_ZONE]
    ) -> Sequence[StopZone]: ...

    @overload
    def get_layer_objects_in_radius(
        self, point: Union[Point2D, Point3D], radius: float, layer: Literal[MapLayer.ROAD_EDGE]
    ) -> Sequence[RoadEdge]: ...

    @overload
    def get_layer_objects_in_radius(
        self, point: Union[Point2D, Point3D], radius: float, layer: Literal[MapLayer.ROAD_LINE]
    ) -> Sequence[RoadLine]: ...

    @overload
    def get_layer_objects_in_radius(
        self, point: Union[Point2D, Point3D], radius: float, layer: Literal[MapLayer.SPEED_BUMP]
    ) -> Sequence[SpeedBump]: ...

    @overload
    def get_layer_objects_in_radius(
        self, point: Union[Point2D, Point3D], radius: float, layer: Literal[MapLayer.TRAFFIC_LIGHT]
    ) -> Sequence[TrafficLight]: ...

    def get_layer_objects_in_radius(
        self, point: Union[Point2D, Point3D], radius: float, layer: MapLayer
    ) -> Sequence[BaseMapObject]:
        """Returns the map objects of one :class:`~py123d.datatypes.MapLayer` within a given radius
            around a center point, typed as the layer's concrete object class.

        :param point: The center point to search around.
        :param radius: The radius to search within.
        :param layer: The map layer to search in.
        :return: The map objects of the layer within the radius.
        """
        layer = MapLayer.from_arbitrary(layer)
        return self.get_map_objects_in_radius(point, radius, [layer]).get(layer, [])
