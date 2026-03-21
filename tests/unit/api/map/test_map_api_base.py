"""Tests for the abstract base classes MapAPI and BaseMapWriter."""

from __future__ import annotations

from typing import List

import pytest

from py123d.api.map.base_map_writer import BaseMapWriter
from py123d.api.map.map_api import MapAPI
from py123d.datatypes.map_objects import BaseMapObject, MapLayer
from py123d.datatypes.metadata import MapMetadata

from ...datatypes.map_objects.mock_map_api import MockMapAPI


class TestMapAPIAbstract:
    """Test that MapAPI cannot be instantiated directly and partial subclasses raise TypeError."""

    def test_cannot_instantiate_directly(self) -> None:
        """MapAPI is abstract and must raise TypeError when instantiated directly."""
        with pytest.raises(TypeError):
            MapAPI()  # type: ignore[abstract]

    def test_partial_subclass_missing_all_methods_raises_type_error(self) -> None:
        """A subclass that implements none of the abstract methods cannot be instantiated."""

        class EmptyMapAPI(MapAPI):
            pass

        with pytest.raises(TypeError):
            EmptyMapAPI()  # type: ignore[abstract]

    def test_partial_subclass_missing_some_methods_raises_type_error(self) -> None:
        """A subclass that implements only some abstract methods cannot be instantiated."""

        class PartialMapAPI(MapAPI):
            def get_map_metadata(self) -> MapMetadata:
                return MapMetadata(
                    dataset="partial",
                    location=None,
                    map_has_z=False,
                    map_is_per_log=False,
                )

            def get_available_map_layers(self) -> List[MapLayer]:
                return []

        with pytest.raises(TypeError):
            PartialMapAPI()  # type: ignore[abstract]


class TestMapAPIProperties:
    """Test that MapAPI convenience properties delegate correctly to the underlying methods."""

    @pytest.fixture()
    def mock_api(self) -> MockMapAPI:
        """Create a default MockMapAPI instance."""
        return MockMapAPI()

    def test_map_metadata_returns_map_metadata(self, mock_api: MockMapAPI) -> None:
        """The map_metadata property should return the MapMetadata from get_map_metadata."""
        metadata = mock_api.map_metadata
        assert isinstance(metadata, MapMetadata)

    def test_dataset(self, mock_api: MockMapAPI) -> None:
        """The dataset property should return 'test'."""
        result = mock_api.dataset
        assert result == "test"

    def test_location(self, mock_api: MockMapAPI) -> None:
        """The location property should return 'test_location'."""
        result = mock_api.location
        assert result == "test_location"

    def test_map_is_per_log(self, mock_api: MockMapAPI) -> None:
        """The map_is_per_log property should return True."""
        result = mock_api.map_is_per_log
        assert result is True

    def test_map_has_z(self, mock_api: MockMapAPI) -> None:
        """The map_has_z property should return True."""
        result = mock_api.map_has_z
        assert result is True

    def test_version_is_string(self, mock_api: MockMapAPI) -> None:
        """The version property should return a string."""
        result = mock_api.version
        assert isinstance(result, str)

    def test_available_map_layers_returns_all_10_layers(self, mock_api: MockMapAPI) -> None:
        """The available_map_layers property should return all 10 MapLayer values."""
        layers = mock_api.available_map_layers
        assert len(layers) == 10
        expected_layers = {
            MapLayer.LANE,
            MapLayer.LANE_GROUP,
            MapLayer.INTERSECTION,
            MapLayer.CROSSWALK,
            MapLayer.WALKWAY,
            MapLayer.CARPARK,
            MapLayer.GENERIC_DRIVABLE,
            MapLayer.STOP_ZONE,
            MapLayer.ROAD_EDGE,
            MapLayer.ROAD_LINE,
        }
        assert set(layers) == expected_layers

    def test_available_map_layers_delegates_to_get_available_map_layers(self, mock_api: MockMapAPI) -> None:
        """The available_map_layers property should return the same result as get_available_map_layers."""
        result = mock_api.available_map_layers
        assert result == mock_api.get_available_map_layers()


class TestBaseMapWriter:
    """Test that BaseMapWriter cannot be instantiated directly and partial subclasses raise TypeError."""

    def test_cannot_instantiate_directly(self) -> None:
        """BaseMapWriter is abstract and must raise TypeError when instantiated directly."""
        with pytest.raises(TypeError):
            BaseMapWriter()  # type: ignore[abstract]

    def test_partial_subclass_missing_all_methods_raises_type_error(self) -> None:
        """A subclass that implements none of the abstract methods cannot be instantiated."""

        class EmptyWriter(BaseMapWriter):
            pass

        with pytest.raises(TypeError):
            EmptyWriter()  # type: ignore[abstract]

    def test_partial_subclass_missing_some_methods_raises_type_error(self) -> None:
        """A subclass that implements only reset and write_map_object but not close cannot be instantiated."""

        class PartialWriter(BaseMapWriter):
            def reset(self, map_metadata: MapMetadata) -> bool:
                return True

            def write_map_object(self, map_object: BaseMapObject) -> None:
                pass

        with pytest.raises(TypeError):
            PartialWriter()  # type: ignore[abstract]

    def test_complete_subclass_can_be_instantiated(self) -> None:
        """A subclass implementing all abstract methods can be instantiated."""

        class CompleteWriter(BaseMapWriter):
            def reset(self, map_metadata: MapMetadata) -> bool:
                return True

            def write_map_object(self, map_object: BaseMapObject) -> None:
                pass

            def close(self) -> None:
                pass

        writer = CompleteWriter()
        assert isinstance(writer, BaseMapWriter)
