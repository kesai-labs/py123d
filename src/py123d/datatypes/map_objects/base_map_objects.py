from __future__ import annotations

import abc
from typing import Optional, Union

import numpy as np
import shapely.geometry as geom
import trimesh
from typing_extensions import TypeAlias

from py123d.datatypes.map_objects.map_layer_types import MapLayer
from py123d.geometry import Point2D, Point3D, Point3DIndex, Polyline2D, Polyline3D

MapObjectIDType: TypeAlias = Union[str, int]


class BaseMapObject(abc.ABC):
    """Base interface representation of all map objects."""

    __slots__ = ("_object_id",)

    def __init__(self, object_id: MapObjectIDType):
        """Constructor of the base map object type.

        :param object_id: unique identifier of the map object.
        """
        self._object_id: MapObjectIDType = object_id

    @property
    def object_id(self) -> MapObjectIDType:
        """The unique identifier of the map object (unique within a map layer)."""
        return self._object_id

    @property
    @abc.abstractmethod
    def layer(self) -> MapLayer:
        """The :class:`~py123d.datatypes.MapLayer` of the map object."""


class BaseMapSurfaceObject(BaseMapObject):
    """Base interface representation of all map objects that represent surfaces."""

    __slots__ = ("_outline", "_shapely_polygon")

    def __init__(
        self,
        object_id: MapObjectIDType,
        outline: Optional[Union[Polyline2D, Polyline3D]] = None,
        shapely_polygon: Optional[geom.Polygon] = None,
    ) -> None:
        """Initialize a BaseMapSurfaceObject instance. Either outline or shapely_polygon must be provided.

        :param object_id: Unique identifier for the map object.
        :param outline: Outline of the surface, either 2D or 3D, defaults to None.
        :param shapely_polygon: A shapely Polygon representing the surface geometry, defaults to None.
        :raises ValueError: If both outline and shapely_polygon are not provided.
        """
        super().__init__(object_id)

        if outline is None and shapely_polygon is None:
            raise ValueError("Either outline or shapely_polygon must be provided.")
        if outline is None:
            outline = Polyline3D.from_linestring(shapely_polygon.exterior)  # type: ignore
        if shapely_polygon is None:
            shapely_polygon = geom.Polygon(outline.array[:, :2])

        self._object_id = object_id
        self._outline = outline
        self._shapely_polygon = shapely_polygon

    @property
    def outline(self) -> Union[Polyline2D, Polyline3D]:
        """The outline of the surface as either :class:`~py123d.geometry.Polyline2D`
        or :class:`~py123d.geometry.Polyline3D`."""
        return self._outline

    @property
    def outline_2d(self) -> Polyline2D:
        """The outline of the surface as :class:`~py123d.geometry.Polyline2D`."""
        if isinstance(self._outline, Polyline2D):
            return self._outline
        # Converts 3D polyline to 2D by dropping the z-coordinate
        return Polyline2D.from_linestring(self._outline.linestring)

    @property
    def outline_3d(self) -> Polyline3D:
        """The outline of the surface as :class:`~py123d.geometry.Polyline3D` (zero-padded to 3D if necessary)."""
        if isinstance(self._outline, Polyline3D):
            return self._outline
        # Converts 2D polyline to 3D by adding a default (zero) z-coordinate
        return Polyline3D.from_linestring(self._outline.linestring)

    @property
    def shapely_polygon(self) -> geom.Polygon:
        """The shapely polygon of the surface."""
        return self._shapely_polygon

    @property
    def trimesh_mesh(self) -> trimesh.Trimesh:
        """The trimesh mesh representation of the surface."""
        # Fallback to geometry if no boundaries are available
        outline_3d_array = self.outline_3d.array
        vertices_2d, faces = trimesh.creation.triangulate_polygon(
            geom.Polygon(outline_3d_array[:, Point3DIndex.XY]), engine="earcut"
        )
        if len(vertices_2d) == len(outline_3d_array):
            # Regular case, where vertices match outline_3d_array
            vertices_3d = outline_3d_array
        elif len(vertices_2d) == len(outline_3d_array) + 1:
            # outline array was not closed, so we need to add the first vertex again
            vertices_3d = np.vstack((outline_3d_array, outline_3d_array[0]))
        else:
            raise ValueError("No vertices found for triangulation.")
        trimesh_mesh = trimesh.Trimesh(vertices=vertices_3d, faces=faces)
        return trimesh_mesh


class BaseMapLineObject(BaseMapObject):
    """Base interface representation of all line map objects."""

    __slots__ = ("_polyline",)

    def __init__(self, object_id: MapObjectIDType, polyline: Union[Polyline2D, Polyline3D]) -> None:
        """Initialize a BaseMapLineObject instance.

        :param object_id: Unique identifier for the map object.
        :param polyline: The polyline representation of the line object.
        """
        super().__init__(object_id)
        self._polyline = polyline

    @property
    def polyline(self) -> Union[Polyline2D, Polyline3D]:
        """The polyline representation, either :class:`~py123d.geometry.Polyline2D` or
        :class:`~py123d.geometry.Polyline3D`."""
        return self._polyline

    @property
    def polyline_2d(self) -> Polyline2D:
        """The polyline representation as :class:`~py123d.geometry.Polyline2D`."""
        if isinstance(self._polyline, Polyline2D):
            return self._polyline
        # Converts 3D polyline to 2D by dropping the z-coordinate
        return Polyline2D.from_linestring(self._polyline.linestring)

    @property
    def polyline_3d(self) -> Polyline3D:
        """The polyline representation as :class:`~py123d.geometry.Polyline3D` (zero-padded to 3D if necessary)."""
        if isinstance(self._polyline, Polyline3D):
            return self._polyline
        # Converts 2D polyline to 3D by adding a default (zero) z-coordinate
        return Polyline3D.from_linestring(self._polyline.linestring)

    @property
    def shapely_linestring(self) -> geom.LineString:
        """The shapely LineString representation of the polyline."""
        return self._polyline.linestring


class BaseMapPointObject(BaseMapObject):
    """Base interface representation of all map objects that are located by a single point."""

    __slots__ = ("_point",)

    def __init__(self, object_id: MapObjectIDType, point: Union[Point2D, Point3D]) -> None:
        """Initialize a BaseMapPointObject instance.

        :param object_id: Unique identifier for the map object.
        :param point: The location of the point object, either 2D or 3D.
        """
        super().__init__(object_id)
        self._point = point

    @property
    def point(self) -> Union[Point2D, Point3D]:
        """The location, either :class:`~py123d.geometry.Point2D` or :class:`~py123d.geometry.Point3D`."""
        return self._point

    @property
    def point_2d(self) -> Point2D:
        """The location as :class:`~py123d.geometry.Point2D`."""
        if isinstance(self._point, Point2D):
            return self._point
        # Converts 3D point to 2D by dropping the z-coordinate
        return self._point.point_2d

    @property
    def point_3d(self) -> Point3D:
        """The location as :class:`~py123d.geometry.Point3D` (zero-padded to 3D if necessary)."""
        if isinstance(self._point, Point3D):
            return self._point
        # Converts 2D point to 3D by adding a default (zero) z-coordinate
        return Point3D(self._point.x, self._point.y, 0.0)

    @property
    def shapely_point(self) -> geom.Point:
        """The shapely Point representation of the location."""
        return self._point.shapely_point
