"""Smoke tests that verify all parser modules can be imported.

Modules with optional dataset-specific dependencies (nuplan, nuscenes, waymo/tensorflow)
are expected to raise ImportError when those packages are not installed, so we skip them.
"""

import importlib

import pytest

# Modules that require no optional dataset-specific dependencies.
CORE_MODULES = [
    "py123d.parser",
    "py123d.parser.abstract_dataset_parser",
    "py123d.parser.dataset_converter_config",
    "py123d.parser.registry",
    # av2
    "py123d.parser.av2",
    "py123d.parser.av2.av2_sensor_parser",
    "py123d.parser.av2.av2_sensor_io",
    "py123d.parser.av2.av2_map_parser",
    "py123d.parser.av2.utils.av2_helper",
    "py123d.parser.av2.utils.av2_constants",
    # opendrive
    "py123d.parser.opendrive",
    "py123d.parser.opendrive.opendrive_parser",
    "py123d.parser.opendrive.opendrive_map_parser",
    "py123d.parser.opendrive.xodr_parser.opendrive",
    "py123d.parser.opendrive.xodr_parser.geometry",
    "py123d.parser.opendrive.xodr_parser.reference",
    "py123d.parser.opendrive.xodr_parser.road",
    "py123d.parser.opendrive.xodr_parser.lane",
    "py123d.parser.opendrive.xodr_parser.elevation",
    "py123d.parser.opendrive.xodr_parser.polynomial",
    "py123d.parser.opendrive.xodr_parser.objects",
    "py123d.parser.opendrive.xodr_parser.signals",
    "py123d.parser.opendrive.utils.lane_helper",
    "py123d.parser.opendrive.utils.stop_zone_helper",
    "py123d.parser.opendrive.utils.collection",
    "py123d.parser.opendrive.utils.signal_helper",
    "py123d.parser.opendrive.utils.id_system",
    "py123d.parser.opendrive.utils.objects_helper",
    # kitti360
    "py123d.parser.kitti360",
    "py123d.parser.kitti360.kitti360_parser",
    "py123d.parser.kitti360.kitti360_map_parser",
    "py123d.parser.kitti360.kitti360_sensor_io",
    "py123d.parser.kitti360.utils.kitti360_helper",
    "py123d.parser.kitti360.utils.kitti360_labels",
    "py123d.parser.kitti360.utils.preprocess_detection",
    # pandaset
    "py123d.parser.pandaset",
    "py123d.parser.pandaset.pandaset_parser",
    "py123d.parser.pandaset.pandaset_sensor_io",
    "py123d.parser.pandaset.utils.pandaset_utlis",
    "py123d.parser.pandaset.utils.pandaset_constants",
    # utils
    "py123d.parser.utils.sensor_utils.camera_conventions",
    "py123d.parser.utils.map_utils.road_edge.road_edge_2d_utils",
    "py123d.parser.utils.map_utils.road_edge.road_edge_3d_utils",
]

# Modules that call check_dependencies at module level for optional packages.
OPTIONAL_MODULES = [
    # nuscenes
    "py123d.parser.nuscenes.nuscenes_parser",
    "py123d.parser.nuscenes.nuscenes_interpolated_parser",
    "py123d.parser.nuscenes.nuscenes_map_parser",
    "py123d.parser.nuscenes.nuscenes_sensor_io",
    "py123d.parser.nuscenes.utils.nuscenes_map_utils",
    "py123d.parser.nuscenes.utils.nuscenes_constants",
    # nuplan
    "py123d.parser.nuplan.nuplan_parser",
    "py123d.parser.nuplan.nuplan_sensor_io",
    "py123d.parser.nuplan.nuplan_map_parser",
    "py123d.parser.nuplan.utils.nuplan_sql_helper",
    "py123d.parser.nuplan.utils.nuplan_constants",
    # waymo (requires tensorflow / protobuf)
    "py123d.parser.wod.wod_motion_parser",
    "py123d.parser.wod.wod_perception_parser",
    "py123d.parser.wod.wod_perception_sensor_io",
    "py123d.parser.wod.wod_map_parser",
    "py123d.parser.wod.utils.wod_constants",
    "py123d.parser.wod.utils.wod_boundary_utils",
]


@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_import_core_parser_module(module_name: str) -> None:
    """Core parser modules must import successfully without optional dependencies."""
    importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", OPTIONAL_MODULES)
def test_import_optional_parser_module(module_name: str) -> None:
    """Parser modules with optional deps should either import or raise ImportError."""
    try:
        importlib.import_module(module_name)
    except (ImportError, TypeError):
        pytest.skip(f"Optional dependency not available for {module_name}")
