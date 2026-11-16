import numpy as np
import pytest
import shapely.geometry as geom

from py123d.datatypes.map_objects.base_map_objects import (
    BaseMapLineObject,
    BaseMapObject,
    BaseMapPointObject,
    BaseMapSurfaceObject,
)
from py123d.datatypes.map_objects.map_layer_types import MapLayer
from py123d.geometry import Point2D, Point3D, Polyline2D, Polyline3D


class ConcreteMapObject(BaseMapObject):
    """Concrete implementation for testing BaseMapObject."""

    def __init__(self, object_id, layer_type=MapLayer.GENERIC_DRIVABLE):
        super().__init__(object_id)
        self._layer = layer_type

    @property
    def layer(self) -> MapLayer:
        return self._layer


class ConcreteMapSurfaceObject(BaseMapSurfaceObject):
    """Concrete implementation for testing BaseMapSurfaceObject."""

    def __init__(self, object_id, outline=None, shapely_polygon=None, layer_type=MapLayer.GENERIC_DRIVABLE):
        super().__init__(object_id, outline, shapely_polygon)
        self._layer = layer_type

    @property
    def layer(self) -> MapLayer:
        return self._layer


class ConcreteMapLineObject(BaseMapLineObject):
    """Concrete implementation for testing BaseMapLineObject."""

    def __init__(self, object_id, polyline, layer_type=MapLayer.GENERIC_DRIVABLE):
        super().__init__(object_id, polyline)
        self._layer = layer_type

    @property
    def layer(self) -> MapLayer:
        return self._layer


class ConcreteMapPointObject(BaseMapPointObject):
    """Concrete implementation for testing BaseMapPointObject."""

    def __init__(self, object_id, point, layer_type=MapLayer.TRAFFIC_LIGHT):
        super().__init__(object_id, point)
        self._layer = layer_type

    @property
    def layer(self) -> MapLayer:
        return self._layer


class TestBaseMapObject:
    """Test cases for BaseMapObject class."""

    def test_init_with_string_id(self):
        """Test initialization with string object ID."""
        obj = ConcreteMapObject("test_id_123")
        assert obj.object_id == "test_id_123"

    def test_init_with_int_id(self):
        """Test initialization with integer object ID."""
        obj = ConcreteMapObject(42)
        assert obj.object_id == 42

    def test_object_id_property(self):
        """Test object_id property."""
        obj = ConcreteMapObject("unique_id")
        assert obj.object_id == "unique_id"

    def test_layer_property(self):
        """Test layer property."""
        obj = ConcreteMapObject("id1", MapLayer.GENERIC_DRIVABLE)
        assert obj.layer == MapLayer.GENERIC_DRIVABLE

    def test_abstract_instantiation_fails(self):
        """Test that instantiating BaseMapObject directly raises TypeError."""
        with pytest.raises(TypeError):
            BaseMapObject("test_id")


class TestBaseMapSurfaceObject:
    """Test cases for BaseMapSurfaceObject class."""

    def test_init_with_polyline2d(self):
        coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        polyline = Polyline2D.from_array(coords)
        obj = ConcreteMapSurfaceObject("surf_1", outline=polyline)
        assert obj.object_id == "surf_1"
        assert isinstance(obj.outline, Polyline2D)

    def test_init_with_polyline3d(self):
        coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]])
        polyline = Polyline3D.from_array(coords)
        obj = ConcreteMapSurfaceObject("surf_2", outline=polyline)
        assert isinstance(obj.outline, Polyline3D)

    def test_init_with_shapely_polygon(self):
        polygon = geom.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        obj = ConcreteMapSurfaceObject("surf_3", shapely_polygon=polygon)
        assert obj.shapely_polygon.equals(polygon)

    def test_init_without_outline_or_polygon_raises_error(self):
        with pytest.raises(ValueError):
            ConcreteMapSurfaceObject("surf_4")

    def test_outline_property(self):
        coords = np.array([[0, 0], [1, 0], [1, 1], [0, 0]])
        polyline = Polyline2D.from_array(coords)
        obj = ConcreteMapSurfaceObject("surf_5", outline=polyline)
        assert obj.outline is polyline

    def test_outline_2d_from_2d_polyline(self):
        coords = np.array([[0, 0], [1, 0], [1, 1], [0, 0]])
        polyline = Polyline2D.from_array(coords)
        obj = ConcreteMapSurfaceObject("surf_6", outline=polyline)
        assert isinstance(obj.outline_2d, Polyline2D)

    def test_outline_2d_from_3d_polyline(self):
        coords = np.array([[0, 0, 5], [1, 0, 5], [1, 1, 5], [0, 0, 5]])
        polyline = Polyline3D.from_array(coords)
        obj = ConcreteMapSurfaceObject("surf_7", outline=polyline)
        outline_2d = obj.outline_2d
        assert isinstance(outline_2d, Polyline2D)

    def test_outline_3d_from_3d_polyline(self):
        coords = np.array([[0, 0, 2], [1, 0, 2], [1, 1, 2], [0, 0, 2]])
        polyline = Polyline3D.from_array(coords)
        obj = ConcreteMapSurfaceObject("surf_8", outline=polyline)
        assert isinstance(obj.outline_3d, Polyline3D)

    def test_outline_3d_from_2d_polyline(self):
        coords = np.array([[0, 0], [1, 0], [1, 1], [0, 0]])
        polyline = Polyline2D.from_array(coords)
        obj = ConcreteMapSurfaceObject("surf_9", outline=polyline)
        outline_3d = obj.outline_3d
        assert isinstance(outline_3d, Polyline3D)

    def test_shapely_polygon_property(self):
        polygon = geom.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        obj = ConcreteMapSurfaceObject("surf_10", shapely_polygon=polygon)
        assert obj.shapely_polygon.equals(polygon)

    def test_trimesh_mesh_property(self):
        coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]])
        polyline = Polyline3D.from_array(coords)
        obj = ConcreteMapSurfaceObject("surf_11", outline=polyline)
        mesh = obj.trimesh_mesh
        assert mesh is not None
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0


class TestBaseMapLineObject:
    """Test cases for BaseMapLineObject class."""

    def test_init_with_polyline2d(self):
        coords = np.array([[0, 0], [1, 1], [2, 2]])
        polyline = Polyline2D.from_array(coords)
        obj = ConcreteMapLineObject("line_1", polyline)
        assert obj.object_id == "line_1"
        assert isinstance(obj.polyline, Polyline2D)

    def test_init_with_polyline3d(self):
        coords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        polyline = Polyline3D(coords)
        obj = ConcreteMapLineObject("line_2", polyline)
        assert isinstance(obj.polyline, Polyline3D)

    def test_polyline_property(self):
        coords = np.array([[0, 0], [1, 1], [2, 2]])
        polyline = Polyline2D.from_array(coords)
        obj = ConcreteMapLineObject("line_3", polyline)
        assert obj.polyline == polyline

    def test_polyline_2d_from_2d_polyline(self):
        coords = np.array([[0, 0], [1, 1], [2, 2]])
        polyline = Polyline2D.from_array(coords)
        obj = ConcreteMapLineObject("line_4", polyline)
        assert isinstance(obj.polyline_2d, Polyline2D)
        assert obj.polyline_2d is polyline

    def test_polyline_2d_from_3d_polyline(self):
        coords = np.array([[0, 0, 5], [1, 1, 5], [2, 2, 5]])
        polyline = Polyline3D.from_array(coords)
        obj = ConcreteMapLineObject("line_5", polyline)
        polyline_2d = obj.polyline_2d
        assert isinstance(polyline_2d, Polyline2D)

    def test_polyline_3d_from_3d_polyline(self):
        coords = np.array([[0, 0, 3], [1, 1, 3], [2, 2, 3]])
        polyline = Polyline3D.from_array(coords)
        obj = ConcreteMapLineObject("line_6", polyline)
        assert isinstance(obj.polyline_3d, Polyline3D)
        assert obj.polyline_3d is polyline

    def test_polyline_3d_from_2d_polyline(self):
        coords = np.array([[0, 0], [1, 1], [2, 2]])
        polyline = Polyline2D.from_array(coords)
        obj = ConcreteMapLineObject("line_7", polyline)
        polyline_3d = obj.polyline_3d
        assert isinstance(polyline_3d, Polyline3D)

    def test_shapely_linestring_property(self):
        coords = np.array([[0, 0], [1, 1], [2, 2]])
        polyline = Polyline2D.from_array(coords)
        obj = ConcreteMapLineObject("line_8", polyline)
        linestring = obj.shapely_linestring
        assert isinstance(linestring, geom.LineString)

    def test_object_id_with_integer(self):
        coords = np.array([[0, 0], [1, 1]])
        polyline = Polyline2D.from_array(coords)
        obj = ConcreteMapLineObject(999, polyline)
        assert obj.object_id == 999


class TestBaseMapPointObject:
    """Tests for BaseMapPointObject."""

    def test_init_with_point2d(self):
        obj = ConcreteMapPointObject("point_1", Point2D(1.0, 2.0))
        assert obj.object_id == "point_1"
        assert obj.layer == MapLayer.TRAFFIC_LIGHT
        assert isinstance(obj.point, Point2D)

    def test_init_with_point3d(self):
        obj = ConcreteMapPointObject("point_2", Point3D(1.0, 2.0, 3.0))
        assert isinstance(obj.point, Point3D)
        assert obj.point.z == 3.0

    def test_point_2d_from_3d_point(self):
        obj = ConcreteMapPointObject("point_3", Point3D(1.0, 2.0, 3.0))
        point_2d = obj.point_2d
        assert isinstance(point_2d, Point2D)
        np.testing.assert_array_equal(point_2d.array, [1.0, 2.0])

    def test_point_3d_from_2d_point(self):
        obj = ConcreteMapPointObject("point_4", Point2D(1.0, 2.0))
        point_3d = obj.point_3d
        assert isinstance(point_3d, Point3D)
        np.testing.assert_array_equal(point_3d.array, [1.0, 2.0, 0.0])

    def test_point_2d_and_3d_identity(self):
        point_2d, point_3d = Point2D(1.0, 2.0), Point3D(1.0, 2.0, 3.0)
        assert ConcreteMapPointObject("point_5", point_2d).point_2d is point_2d
        assert ConcreteMapPointObject("point_6", point_3d).point_3d is point_3d

    def test_shapely_point_property(self):
        obj = ConcreteMapPointObject("point_7", Point3D(1.0, 2.0, 3.0))
        shapely_point = obj.shapely_point
        assert isinstance(shapely_point, geom.Point)
        assert shapely_point.has_z
        assert (shapely_point.x, shapely_point.y, shapely_point.z) == (1.0, 2.0, 3.0)
        assert not ConcreteMapPointObject("point_8", Point2D(1.0, 2.0)).shapely_point.has_z
