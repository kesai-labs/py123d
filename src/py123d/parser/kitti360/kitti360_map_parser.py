import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator, List

import numpy as np
import shapely.geometry as geom

from py123d.datatypes import BaseMapObject, Carpark, GenericDrivable, MapMetadata, RoadEdge, RoadEdgeType, Walkway
from py123d.geometry import Polyline3D
from py123d.parser.base_dataset_parser import BaseMapParser
from py123d.parser.kitti360.utils.kitti360_helper import KITTI360MapBbox3D
from py123d.parser.utils.map_utils.road_edge.road_edge_2d_utils import (
    get_road_edge_linear_rings,
    split_line_geometry_by_max_length,
)
from py123d.parser.utils.map_utils.road_edge.road_edge_3d_utils import lift_road_edges_to_3d

MAX_ROAD_EDGE_LENGTH = 100.0  # meters, used to filter out very long road edges
KITTI360_MAP_BBOX = [
    "road",
    "sidewalk",
    # "railtrack",
    # "ground",
    "driveway",
]


class Kitti360MapParser(BaseMapParser):
    """Lightweight, picklable handle to one KITTI-360 map."""

    def __init__(self, log_name: str, split: str, bbox_root: Path) -> None:
        self._log_name = log_name
        self._split = split
        self._bbox_root = bbox_root

    def get_map_metadata(self) -> MapMetadata:
        """Returns metadata for this KITTI-360 map."""
        return MapMetadata(
            dataset="kitti360",
            split=self._split,
            log_name=self._log_name,
            location=self._log_name,
            map_has_z=True,
            map_is_per_log=True,
        )

    def iter_map_objects(self) -> Iterator[BaseMapObject]:
        """Yields map objects lazily from the KITTI-360 3D bounding box XML files."""
        yield from iter_kitti360_map_objects(self._log_name, self._bbox_root)


def iter_kitti360_map_objects(log_name: str, bbox_root: Path) -> Iterator[BaseMapObject]:
    """Yield KITTI-360 map objects from the 3D bounding box XML files.

    :param log_name: The name of the log/sequence
    :param bbox_root: Path to the data_3d_bboxes directory
    :yields: BaseMapObject instances (GenericDrivable, Walkway, Carpark, RoadEdge)
    """
    xml_path = bbox_root / "train_full" / f"{log_name}.xml"
    if not xml_path.exists():
        xml_path = bbox_root / "train" / f"{log_name}.xml"

    if not xml_path.exists():
        raise FileNotFoundError(f"BBox 3D file not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs: List[KITTI360MapBbox3D] = []

    for child in root:
        label = child.find("label").text
        if child.find("transform") is None or label not in KITTI360_MAP_BBOX:
            continue
        obj = KITTI360MapBbox3D()
        obj.parse_bbox(child)
        objs.append(obj)

    # 1. Yield roads, sidewalks, driveways, and collect road geometries
    road_outlines_3d: List[Polyline3D] = []
    for obj in objs:
        if obj.label == "road":
            yield GenericDrivable(
                object_id=obj.id,
                outline=obj.vertices,
                shapely_polygon=geom.Polygon(obj.vertices.array[:, :3]),
            )
            road_outline_array = np.concatenate([obj.vertices.array[:, :3], obj.vertices.array[0:, :3]])
            road_outlines_3d.append(Polyline3D.from_array(road_outline_array))
        elif obj.label == "sidewalk":
            yield Walkway(
                object_id=obj.id,
                outline=obj.vertices,
                shapely_polygon=geom.Polygon(obj.vertices.array[:, :3]),
            )
        elif obj.label == "driveway":
            yield Carpark(
                object_id=obj.id,
                outline=obj.vertices,
                shapely_polygon=geom.Polygon(obj.vertices.array[:, :3]),
            )

    # 2. Use road geometries to create road edges
    # NOTE @DanielDauner: We merge all drivable areas in 2D and lift the outlines to 3D.
    # Currently the method assumes that the drivable areas do not overlap and all road surfaces are included.
    road_polygons_2d = [geom.Polygon(road_outline.array[:, :2]) for road_outline in road_outlines_3d]
    road_edges_2d = get_road_edge_linear_rings(road_polygons_2d)
    road_edges_3d = lift_road_edges_to_3d(road_edges_2d, road_outlines_3d)
    road_edges_linestrings_3d = [polyline.linestring for polyline in road_edges_3d]
    road_edges_linestrings_3d = split_line_geometry_by_max_length(road_edges_linestrings_3d, MAX_ROAD_EDGE_LENGTH)

    for idx in range(len(road_edges_linestrings_3d)):
        yield RoadEdge(
            object_id=idx,
            road_edge_type=RoadEdgeType.ROAD_EDGE_BOUNDARY,
            polyline=Polyline3D.from_linestring(road_edges_linestrings_3d[idx]),
        )
