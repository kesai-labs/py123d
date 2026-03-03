import abc
from abc import abstractmethod

from py123d.conversion.dataset_converter_config import DatasetConverterConfig
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


class AbstractMapWriter(abc.ABC):
    """Abstract base class for map writers."""

    @abstractmethod
    def reset(self, dataset_converter_config: DatasetConverterConfig, map_metadata: MapMetadata) -> bool:
        """Reset the writer to its initial state."""

    @abstractmethod
    def write_lane(self, lane: Lane) -> None:
        """Write a lane to the map."""

    @abstractmethod
    def write_lane_group(self, lane_group: LaneGroup) -> None:
        """Write a group of lanes to the map."""

    @abstractmethod
    def write_intersection(self, intersection: Intersection) -> None:
        """Write an intersection to the map."""

    @abstractmethod
    def write_crosswalk(self, crosswalk: Crosswalk) -> None:
        """Write a crosswalk to the map."""

    @abstractmethod
    def write_carpark(self, carpark: Carpark) -> None:
        """Write a car park to the map."""

    @abstractmethod
    def write_walkway(self, walkway: Walkway) -> None:
        """Write a walkway to the map."""

    @abstractmethod
    def write_generic_drivable(self, obj: GenericDrivable) -> None:
        """Write a generic drivable area to the map."""

    @abstractmethod
    def write_stop_zone(self, stop_zone: StopZone) -> None:
        """Write a stop zone to the map."""

    @abstractmethod
    def write_road_edge(self, road_edge: RoadEdge) -> None:
        """Write a road edge to the map."""

    @abstractmethod
    def write_road_line(self, road_line: RoadLine) -> None:
        """Write a road line to the map."""

    @abstractmethod
    def close(self) -> None:
        """Close the writer and finalize any resources."""
