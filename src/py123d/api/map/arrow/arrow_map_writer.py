from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import msgpack
import numpy as np
import pyarrow as pa

from py123d.api.map.arrow.arrow_id_utils import ToIntMapping
from py123d.api.map.base_map_writer import BaseMapWriter
from py123d.api.utils.arrow_helper import write_arrow_table
from py123d.common.utils.msgpack_utils import msgpack_encode_with_numpy
from py123d.datatypes import (
    BaseMapLineObject,
    BaseMapSurfaceObject,
    Carpark,
    Crosswalk,
    GenericDrivable,
    Intersection,
    Lane,
    LaneGroup,
    MapLayer,
    MapMetadata,
    RoadEdge,
    RoadLine,
    StopZone,
    Walkway,
)
from py123d.datatypes.map_objects.base_map_objects import BaseMapObject
from py123d.geometry import Polyline2D, Polyline3D


class ArrowMapWriter(BaseMapWriter):
    """Arrow-based map writer."""

    def __init__(
        self,
        force_map_conversion: bool,
        maps_root: Union[str, Path],
        logs_root: Union[str, Path],
    ) -> None:
        self._force_map_conversion = force_map_conversion
        self._maps_root = Path(maps_root)
        self._logs_root = Path(logs_root)

        # Data to be written to the map for each object type
        self._map_data: Dict[MapLayer, Dict[str, Any]] = {}
        self._map_file: Optional[Path] = None
        self._map_metadata: Optional[MapMetadata] = None

    def reset(self, map_metadata: MapMetadata) -> bool:
        """Inherited, see superclass."""

        map_needs_writing: bool = False

        if map_metadata.map_is_per_log:
            split, log_name = map_metadata.split, map_metadata.log_name
            assert split is not None, "For per-log maps, split must be provided in map metadata."
            assert log_name is not None, "For per-log maps, log_name must be provided in map metadata."
            map_file = self._logs_root / split / log_name / "map.arrow"
        else:
            dataset, location = map_metadata.dataset, map_metadata.location
            assert location is not None, "For global maps, location must be provided in map metadata."
            map_file = self._maps_root / dataset / f"{dataset}_{location}.arrow"

        map_needs_writing = self._force_map_conversion or not map_file.exists()
        if map_needs_writing:
            # Reset all map layers and update map file / metadata
            self._map_data = {map_layer: defaultdict(list) for map_layer in MapLayer}
            self._map_file = Path(map_file)
            self._map_metadata = map_metadata
        else:
            self._map_file = None
            self._map_data = {}
            self._map_metadata = None

        return map_needs_writing

    def write_map_object(self, map_object: BaseMapObject) -> None:
        """Inherited, see superclass."""

        if isinstance(map_object, Lane):
            self._write_lane(map_object)
        elif isinstance(map_object, LaneGroup):
            self._write_lane_group(map_object)
        elif isinstance(map_object, Intersection):
            self._write_intersection(map_object)
        elif isinstance(map_object, Crosswalk):
            self._write_crosswalk(map_object)
        elif isinstance(map_object, Carpark):
            self._write_carpark(map_object)
        elif isinstance(map_object, Walkway):
            self._write_walkway(map_object)
        elif isinstance(map_object, GenericDrivable):
            self._write_generic_drivable(map_object)
        elif isinstance(map_object, StopZone):
            self._write_stop_zone(map_object)
        elif isinstance(map_object, RoadEdge):
            self._write_road_edge(map_object)
        elif isinstance(map_object, RoadLine):
            self._write_road_line(map_object)
        else:
            raise ValueError(f"Unsupported map object type: {type(map_object)}")

    def _write_lane(self, lane: Lane) -> None:
        self._write_surface_layer(MapLayer.LANE, lane)
        self._map_data[MapLayer.LANE]["lane_type"].append(int(lane.lane_type))
        self._map_data[MapLayer.LANE]["lane_group_id"].append(lane.lane_group_id)
        self._map_data[MapLayer.LANE]["left_boundary"].append(lane.left_boundary.array)
        self._map_data[MapLayer.LANE]["right_boundary"].append(lane.right_boundary.array)
        self._map_data[MapLayer.LANE]["centerline"].append(lane.centerline.array)
        self._map_data[MapLayer.LANE]["left_lane_id"].append(lane.left_lane_id)
        self._map_data[MapLayer.LANE]["right_lane_id"].append(lane.right_lane_id)
        self._map_data[MapLayer.LANE]["predecessor_ids"].append(lane.predecessor_ids)
        self._map_data[MapLayer.LANE]["successor_ids"].append(lane.successor_ids)
        self._map_data[MapLayer.LANE]["speed_limit_mps"].append(lane.speed_limit_mps)

    def _write_lane_group(self, lane_group: LaneGroup) -> None:
        self._write_surface_layer(MapLayer.LANE_GROUP, lane_group)
        self._map_data[MapLayer.LANE_GROUP]["lane_ids"].append(lane_group.lane_ids)
        self._map_data[MapLayer.LANE_GROUP]["intersection_id"].append(lane_group.intersection_id)
        self._map_data[MapLayer.LANE_GROUP]["predecessor_ids"].append(lane_group.predecessor_ids)
        self._map_data[MapLayer.LANE_GROUP]["successor_ids"].append(lane_group.successor_ids)
        self._map_data[MapLayer.LANE_GROUP]["left_boundary"].append(lane_group.left_boundary.array)
        self._map_data[MapLayer.LANE_GROUP]["right_boundary"].append(lane_group.right_boundary.array)

    def _write_intersection(self, intersection: Intersection) -> None:
        self._write_surface_layer(MapLayer.INTERSECTION, intersection)
        self._map_data[MapLayer.INTERSECTION]["intersection_type"].append(int(intersection.intersection_type))
        self._map_data[MapLayer.INTERSECTION]["lane_group_ids"].append(intersection.lane_group_ids)

    def _write_crosswalk(self, crosswalk: Crosswalk) -> None:
        self._write_surface_layer(MapLayer.CROSSWALK, crosswalk)

    def _write_carpark(self, carpark: Carpark) -> None:
        """Inherited, see superclass."""
        self._write_surface_layer(MapLayer.CARPARK, carpark)

    def _write_walkway(self, walkway: Walkway) -> None:
        self._write_surface_layer(MapLayer.WALKWAY, walkway)

    def _write_generic_drivable(self, obj: GenericDrivable) -> None:
        """Inherited, see superclass."""
        self._write_surface_layer(MapLayer.GENERIC_DRIVABLE, obj)

    def _write_stop_zone(self, stop_zone: StopZone) -> None:
        self._write_surface_layer(MapLayer.STOP_ZONE, stop_zone)
        self._map_data[MapLayer.STOP_ZONE]["stop_zone_type"].append(int(stop_zone.stop_zone_type))
        self._map_data[MapLayer.STOP_ZONE]["lane_ids"].append(stop_zone.lane_ids)

    def _write_road_edge(self, road_edge: RoadEdge) -> None:
        """Inherited, see superclass."""
        self._write_line_layer(MapLayer.ROAD_EDGE, road_edge)
        self._map_data[MapLayer.ROAD_EDGE]["road_edge_type"].append(int(road_edge.road_edge_type))

    def _write_road_line(self, road_line: RoadLine) -> None:
        self._write_line_layer(MapLayer.ROAD_LINE, road_line)
        self._map_data[MapLayer.ROAD_LINE]["road_line_type"].append(int(road_line.road_line_type))

    def close(self) -> None:
        """Inherited, see superclass."""

        if self._map_file is not None and self._map_data is not None:
            assert isinstance(self._map_file, Path), "Map file path is not set."
            assert isinstance(self._map_metadata, MapMetadata), "Map metadata is not set."

            if not self._map_file.parent.exists():
                self._map_file.parent.mkdir(parents=True, exist_ok=True)

            # NOTE @DanielDauner: Currently, we enforce remapping of map IDs to integers for Arrow maps.
            # In the future, string IDs could be supported as well, but this complicates the implementation and
            # the benefits are unclear.
            _map_ids_to_integer(self._map_data)
            object_id_type = pa.int64()

            all_object_ids = []
            all_wkbs: List[bytes] = []
            all_map_layers: List[int] = []
            all_features: List[bytes] = []

            # 1. Lanes
            for idx in range(len(self._map_data[MapLayer.LANE]["id"])):
                all_object_ids.append(self._map_data[MapLayer.LANE]["id"][idx])
                all_wkbs.append(self._map_data[MapLayer.LANE]["wkb"][idx])
                all_map_layers.append(int(MapLayer.LANE))

                lane_dict = {
                    "lane_type": self._map_data[MapLayer.LANE]["lane_type"][idx],
                    "lane_group_id": self._map_data[MapLayer.LANE]["lane_group_id"][idx],
                    "left_boundary": self._map_data[MapLayer.LANE]["left_boundary"][idx],
                    "right_boundary": self._map_data[MapLayer.LANE]["right_boundary"][idx],
                    "centerline": self._map_data[MapLayer.LANE]["centerline"][idx],
                    "left_lane_id": self._map_data[MapLayer.LANE]["left_lane_id"][idx],
                    "right_lane_id": self._map_data[MapLayer.LANE]["right_lane_id"][idx],
                    "predecessor_ids": self._map_data[MapLayer.LANE]["predecessor_ids"][idx],
                    "successor_ids": self._map_data[MapLayer.LANE]["successor_ids"][idx],
                    "speed_limit_mps": self._map_data[MapLayer.LANE]["speed_limit_mps"][idx],
                    "outline": self._map_data[MapLayer.LANE]["outline"][idx],
                }
                all_features.append(msgpack_encode_with_numpy(lane_dict))  # type: ignore

            # 2. Lane groups
            for idx in range(len(self._map_data[MapLayer.LANE_GROUP]["id"])):
                all_object_ids.append(self._map_data[MapLayer.LANE_GROUP]["id"][idx])
                all_wkbs.append(self._map_data[MapLayer.LANE_GROUP]["wkb"][idx])
                all_map_layers.append(int(MapLayer.LANE_GROUP))

                lane_group_dict = {
                    "lane_ids": self._map_data[MapLayer.LANE_GROUP]["lane_ids"][idx],
                    "left_boundary": self._map_data[MapLayer.LANE_GROUP]["left_boundary"][idx],
                    "right_boundary": self._map_data[MapLayer.LANE_GROUP]["right_boundary"][idx],
                    "intersection_id": self._map_data[MapLayer.LANE_GROUP]["intersection_id"][idx],
                    "predecessor_ids": self._map_data[MapLayer.LANE_GROUP]["predecessor_ids"][idx],
                    "successor_ids": self._map_data[MapLayer.LANE_GROUP]["successor_ids"][idx],
                    "outline": self._map_data[MapLayer.LANE_GROUP]["outline"][idx],
                }
                all_features.append(msgpack_encode_with_numpy(lane_group_dict))  # type: ignore

            # 3. Intersections
            for idx in range(len(self._map_data[MapLayer.INTERSECTION]["id"])):
                all_object_ids.append(self._map_data[MapLayer.INTERSECTION]["id"][idx])
                all_wkbs.append(self._map_data[MapLayer.INTERSECTION]["wkb"][idx])
                all_map_layers.append(int(MapLayer.INTERSECTION))

                intersection_dict = {
                    "intersection_type": self._map_data[MapLayer.INTERSECTION]["intersection_type"][idx],
                    "lane_group_ids": self._map_data[MapLayer.INTERSECTION]["lane_group_ids"][idx],
                    "outline": self._map_data[MapLayer.INTERSECTION]["outline"][idx],
                }
                all_features.append(msgpack_encode_with_numpy(intersection_dict))  # type: ignore

            # 4. Crosswalks
            for idx in range(len(self._map_data[MapLayer.CROSSWALK]["id"])):
                all_object_ids.append(self._map_data[MapLayer.CROSSWALK]["id"][idx])
                all_wkbs.append(self._map_data[MapLayer.CROSSWALK]["wkb"][idx])
                all_map_layers.append(int(MapLayer.CROSSWALK))

                crosswalk_dict = {
                    "outline": self._map_data[MapLayer.CROSSWALK]["outline"][idx],
                }
                all_features.append(msgpack_encode_with_numpy(crosswalk_dict))  # type: ignore

            # 5. Carparks
            for idx in range(len(self._map_data[MapLayer.CARPARK]["id"])):
                all_object_ids.append(self._map_data[MapLayer.CARPARK]["id"][idx])
                all_wkbs.append(self._map_data[MapLayer.CARPARK]["wkb"][idx])
                all_map_layers.append(int(MapLayer.CARPARK))

                carpark_dict = {
                    "outline": self._map_data[MapLayer.CARPARK]["outline"][idx],
                }
                all_features.append(msgpack_encode_with_numpy(carpark_dict))  # type: ignore

            # 6. Walkways
            for idx in range(len(self._map_data[MapLayer.WALKWAY]["id"])):
                all_object_ids.append(self._map_data[MapLayer.WALKWAY]["id"][idx])
                all_wkbs.append(self._map_data[MapLayer.WALKWAY]["wkb"][idx])
                all_map_layers.append(int(MapLayer.WALKWAY))

                walkway_dict = {
                    "outline": self._map_data[MapLayer.WALKWAY]["outline"][idx],
                }
                all_features.append(msgpack_encode_with_numpy(walkway_dict))  # type: ignore

            # 7. Generic Drivables
            for idx in range(len(self._map_data[MapLayer.GENERIC_DRIVABLE]["id"])):
                all_object_ids.append(self._map_data[MapLayer.GENERIC_DRIVABLE]["id"][idx])
                all_wkbs.append(self._map_data[MapLayer.GENERIC_DRIVABLE]["wkb"][idx])
                all_map_layers.append(int(MapLayer.GENERIC_DRIVABLE))

                generic_drivable_dict = {
                    "outline": self._map_data[MapLayer.GENERIC_DRIVABLE]["outline"][idx],
                }
                all_features.append(msgpack_encode_with_numpy(generic_drivable_dict))  # type: ignore

            # 8. Stop zones
            for idx in range(len(self._map_data[MapLayer.STOP_ZONE]["id"])):
                all_object_ids.append(self._map_data[MapLayer.STOP_ZONE]["id"][idx])
                all_wkbs.append(self._map_data[MapLayer.STOP_ZONE]["wkb"][idx])
                all_map_layers.append(int(MapLayer.STOP_ZONE))

                stop_zone_dict = {
                    "outline": self._map_data[MapLayer.STOP_ZONE]["outline"][idx],
                    "stop_zone_type": self._map_data[MapLayer.STOP_ZONE]["stop_zone_type"][idx],
                    "lane_ids": self._map_data[MapLayer.STOP_ZONE]["lane_ids"][idx],
                }
                all_features.append(msgpack_encode_with_numpy(stop_zone_dict))  # type: ignore

            # 8. Road edges
            for idx in range(len(self._map_data[MapLayer.ROAD_EDGE]["id"])):
                all_object_ids.append(self._map_data[MapLayer.ROAD_EDGE]["id"][idx])
                all_wkbs.append(self._map_data[MapLayer.ROAD_EDGE]["wkb"][idx])
                all_map_layers.append(int(MapLayer.ROAD_EDGE))

                road_edge_dict = {
                    "road_edge_type": self._map_data[MapLayer.ROAD_EDGE]["road_edge_type"][idx],
                }
                all_features.append(msgpack_encode_with_numpy(road_edge_dict))  # type: ignore

            # 9. Road lines
            for idx in range(len(self._map_data[MapLayer.ROAD_LINE]["id"])):
                all_object_ids.append(self._map_data[MapLayer.ROAD_LINE]["id"][idx])
                all_wkbs.append(self._map_data[MapLayer.ROAD_LINE]["wkb"][idx])
                all_map_layers.append(int(MapLayer.ROAD_LINE))

                road_line_dict = {
                    "road_line_type": self._map_data[MapLayer.ROAD_LINE]["road_line_type"][idx],
                }
                all_features.append(msgpack_encode_with_numpy(road_line_dict))  # type: ignore

            # Create final table and write to file
            object_ids_ = pa.array(all_object_ids, type=object_id_type)
            map_layers_ = pa.array(all_map_layers, type=pa.int8())
            features_ = pa.array(all_features, type=pa.binary())
            wkbs_ = pa.array(all_wkbs, type=pa.binary())

            table = pa.Table.from_arrays(
                [object_ids_, map_layers_, features_, wkbs_],
                names=["object_id", "map_layer", "features", "wkb"],
            )
            # Add metadata to the table and write to file
            table = table.replace_schema_metadata(
                {b"metadata": msgpack.packb(self._map_metadata.to_dict(), use_bin_type=True)}
            )
            write_arrow_table(table, self._map_file)

        del self._map_file, self._map_data, self._map_metadata
        self._map_file = None
        self._map_data = {}
        self._map_metadata = None

    def _assert_initialized(self) -> None:
        assert len(self._map_data) > 0, "Call reset() before writing data."
        assert self._map_file is not None, "Call reset() before writing data."
        assert self._map_metadata is not None, "Call reset() before writing data."

    def _write_surface_layer(self, layer: MapLayer, surface_object: BaseMapSurfaceObject) -> None:
        """Helper to write surface map objects.

        :param layer: map layer of surface object
        :param surface_object: surface map object to write
        """
        self._assert_initialized()
        if surface_object.object_id in self._map_data[layer]["id"]:
            raise ValueError(
                f"Duplicate object ID {surface_object.object_id!r} in layer {layer.name}. "
                f"Object IDs must be unique within each map layer."
            )
        self._map_data[layer]["id"].append(surface_object.object_id)
        if isinstance(surface_object.outline, Polyline3D):
            self._map_data[layer]["outline"].append(surface_object.outline.array)
        elif isinstance(surface_object.outline, Polyline2D):
            array_2d = surface_object.outline.array
            self._map_data[layer]["outline"].append(np.column_stack([array_2d, np.zeros(len(array_2d))]))
        else:
            # Infer 3D outline from polygon exterior coordinates
            coords = np.array(surface_object.shapely_polygon.exterior.coords)
            if coords.shape[1] == 2:
                coords = np.column_stack([coords, np.zeros(len(coords))])
            self._map_data[layer]["outline"].append(coords)
        self._map_data[layer]["wkb"].append(surface_object.shapely_polygon.wkb)

    def _write_line_layer(self, layer: MapLayer, line_object: BaseMapLineObject) -> None:
        """Helper to write line map objects.

        :param layer: map layer of line object
        :param line_object: line map object to write
        """
        self._assert_initialized()
        if line_object.object_id in self._map_data[layer]["id"]:
            raise ValueError(
                f"Duplicate object ID {line_object.object_id!r} in layer {layer.name}. "
                f"Object IDs must be unique within each map layer."
            )
        self._map_data[layer]["id"].append(line_object.object_id)
        self._map_data[layer]["wkb"].append(line_object.shapely_linestring.wkb)


def _map_ids_to_integer(map_data: Dict[MapLayer, Dict[str, Any]]) -> None:
    # initialize id mappings
    lane_id_mapping = ToIntMapping.from_list(map_data[MapLayer.LANE]["id"])
    lane_group_id_mapping = ToIntMapping.from_list(map_data[MapLayer.LANE_GROUP]["id"])
    intersection_id_mapping = ToIntMapping.from_list(map_data[MapLayer.INTERSECTION]["id"])

    crosswalk_id_mapping = ToIntMapping.from_list(map_data[MapLayer.CROSSWALK]["id"])
    walkway_id_mapping = ToIntMapping.from_list(map_data[MapLayer.WALKWAY]["id"])
    carpark_id_mapping = ToIntMapping.from_list(map_data[MapLayer.CARPARK]["id"])
    generic_drivable_id_mapping = ToIntMapping.from_list(map_data[MapLayer.GENERIC_DRIVABLE]["id"])
    stop_zone_id_mapping = ToIntMapping.from_list(map_data[MapLayer.STOP_ZONE]["id"])
    road_line_id_mapping = ToIntMapping.from_list(map_data[MapLayer.ROAD_LINE]["id"])
    road_edge_id_mapping = ToIntMapping.from_list(map_data[MapLayer.ROAD_EDGE]["id"])

    # 1. Remap lane ids in LANE layer
    if len(map_data[MapLayer.LANE]["id"]) > 0:
        for lane_idx in range(len(map_data[MapLayer.LANE]["id"])):
            map_data[MapLayer.LANE]["id"][lane_idx] = lane_id_mapping.map(map_data[MapLayer.LANE]["id"][lane_idx])
            map_data[MapLayer.LANE]["lane_group_id"][lane_idx] = lane_group_id_mapping.map(
                map_data[MapLayer.LANE]["lane_group_id"][lane_idx]
            )
            for column in ["predecessor_ids", "successor_ids"]:
                map_data[MapLayer.LANE][column][lane_idx] = lane_id_mapping.map_list(
                    map_data[MapLayer.LANE][column][lane_idx]
                )
            for column in ["left_lane_id", "right_lane_id"]:
                map_data[MapLayer.LANE][column][lane_idx] = lane_id_mapping.map(
                    map_data[MapLayer.LANE][column][lane_idx]
                )

    # 2. Remap lane group ids in LANE_GROUP
    if len(map_data[MapLayer.LANE_GROUP]["id"]) > 0:
        for lg_idx in range(len(map_data[MapLayer.LANE_GROUP]["id"])):
            map_data[MapLayer.LANE_GROUP]["id"][lg_idx] = lane_group_id_mapping.map(
                map_data[MapLayer.LANE_GROUP]["id"][lg_idx]
            )
            map_data[MapLayer.LANE_GROUP]["lane_ids"][lg_idx] = lane_id_mapping.map_list(
                map_data[MapLayer.LANE_GROUP]["lane_ids"][lg_idx]
            )
            map_data[MapLayer.LANE_GROUP]["intersection_id"][lg_idx] = intersection_id_mapping.map(
                map_data[MapLayer.LANE_GROUP]["intersection_id"][lg_idx]
            )
            for column in ["predecessor_ids", "successor_ids"]:
                map_data[MapLayer.LANE_GROUP][column][lg_idx] = lane_group_id_mapping.map_list(
                    map_data[MapLayer.LANE_GROUP][column][lg_idx]
                )

    # 3. Remap lane group ids in INTERSECTION
    if len(map_data[MapLayer.INTERSECTION]["id"]) > 0:
        for inter_idx in range(len(map_data[MapLayer.INTERSECTION]["id"])):
            map_data[MapLayer.INTERSECTION]["id"][inter_idx] = intersection_id_mapping.map(
                map_data[MapLayer.INTERSECTION]["id"][inter_idx]
            )
            map_data[MapLayer.INTERSECTION]["lane_group_ids"][inter_idx] = lane_group_id_mapping.map_list(
                map_data[MapLayer.INTERSECTION]["lane_group_ids"][inter_idx]
            )

    # 4. Remap ids in other layers
    if len(map_data[MapLayer.CROSSWALK]["id"]) > 0:
        for idx in range(len(map_data[MapLayer.CROSSWALK]["id"])):
            map_data[MapLayer.CROSSWALK]["id"][idx] = crosswalk_id_mapping.map(map_data[MapLayer.CROSSWALK]["id"][idx])
    if len(map_data[MapLayer.CARPARK]["id"]) > 0:
        for idx in range(len(map_data[MapLayer.CARPARK]["id"])):
            map_data[MapLayer.CARPARK]["id"][idx] = carpark_id_mapping.map(map_data[MapLayer.CARPARK]["id"][idx])
    if len(map_data[MapLayer.WALKWAY]["id"]) > 0:
        for idx in range(len(map_data[MapLayer.WALKWAY]["id"])):
            map_data[MapLayer.WALKWAY]["id"][idx] = walkway_id_mapping.map(map_data[MapLayer.WALKWAY]["id"][idx])
    if len(map_data[MapLayer.GENERIC_DRIVABLE]["id"]) > 0:
        for idx in range(len(map_data[MapLayer.GENERIC_DRIVABLE]["id"])):
            map_data[MapLayer.GENERIC_DRIVABLE]["id"][idx] = generic_drivable_id_mapping.map(
                map_data[MapLayer.GENERIC_DRIVABLE]["id"][idx]
            )
    if len(map_data[MapLayer.STOP_ZONE]["id"]) > 0:
        for idx in range(len(map_data[MapLayer.STOP_ZONE]["id"])):
            map_data[MapLayer.STOP_ZONE]["id"][idx] = stop_zone_id_mapping.map(map_data[MapLayer.STOP_ZONE]["id"][idx])
            map_data[MapLayer.STOP_ZONE]["lane_ids"][idx] = lane_id_mapping.map_list(
                map_data[MapLayer.STOP_ZONE]["lane_ids"][idx]
            )
    if len(map_data[MapLayer.ROAD_LINE]["id"]) > 0:
        for idx in range(len(map_data[MapLayer.ROAD_LINE]["id"])):
            map_data[MapLayer.ROAD_LINE]["id"][idx] = road_line_id_mapping.map(map_data[MapLayer.ROAD_LINE]["id"][idx])
    if len(map_data[MapLayer.ROAD_EDGE]["id"]) > 0:
        for idx in range(len(map_data[MapLayer.ROAD_EDGE]["id"])):
            map_data[MapLayer.ROAD_EDGE]["id"][idx] = road_edge_id_mapping.map(map_data[MapLayer.ROAD_EDGE]["id"][idx])
