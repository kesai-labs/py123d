import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import trimesh
import viser

from py123d.api import SceneAPI
from py123d.common.utils.enums import resolve_enum_arguments
from py123d.datatypes.map_objects.base_map_objects import BaseMapLineObject, BaseMapSurfaceObject
from py123d.datatypes.map_objects.map_layer_types import MapLayer, StopZoneType
from py123d.datatypes.map_objects.map_objects import Lane, StopZone
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.geometry import Point3D, Point3DIndex, Polyline3D
from py123d.visualization.color.default import MAP_SURFACE_CONFIG
from py123d.visualization.viser.elements.base_element import ElementContext, ViewerElement

logger = logging.getLogger(__name__)


@dataclass
class MapConfig:
    visible: bool = True
    radius: float = 200.0
    non_road_z_offset: float = 0.1
    opacity: float = 1.0
    requery: bool = True
    centerline_dash_length: float = 1 / 3
    show_road_edges: bool = True
    show_centerlines: bool = True
    visible_layers: List[MapLayer] = field(default_factory=lambda: [])

    def __post_init__(self):
        self.visible_layers = resolve_enum_arguments(MapLayer, self.visible_layers)


_MAP_DISPLAY_LAYERS: List[MapLayer] = [
    MapLayer.LANE,
    MapLayer.LANE_GROUP,
    MapLayer.INTERSECTION,
    MapLayer.WALKWAY,
    MapLayer.CROSSWALK,
    MapLayer.CARPARK,
    MapLayer.GENERIC_DRIVABLE,
    MapLayer.STOP_ZONE,
]

_ROAD_EDGE_COLOR_DARK: Tuple[int, int, int] = (255, 255, 255)
_ROAD_EDGE_COLOR_LIGHT: Tuple[int, int, int] = (0, 0, 0)
_CENTERLINE_COLOR_DARK: Tuple[int, int, int] = (255, 255, 255)
_CENTERLINE_COLOR_LIGHT: Tuple[int, int, int] = (0, 0, 0)
_LINE_Z_OFFSET: float = 0.15


class MapElement(ViewerElement):
    """Visualizes map layers (lanes, crosswalks, etc.) in the 3D scene."""

    def __init__(self, context: ElementContext, config: MapConfig) -> None:
        self._context = context
        self._config = config
        self._server: Optional[viser.ViserServer] = None
        self._handles: Dict[str, Optional[Union[viser.MeshHandle, viser.LineSegmentsHandle]]] = {}
        self._last_query_position: Optional[Point3D] = None
        self._force_update: bool = False
        self._current_iteration: int = 0
        self._gui_visible: Optional[viser.GuiCheckboxHandle] = None
        self._gui_radius: Optional[viser.GuiSliderHandle] = None
        self._gui_opacity: Optional[viser.GuiSliderHandle] = None
        self._gui_layer_checkboxes: Dict[MapLayer, viser.GuiCheckboxHandle] = {}
        self._gui_road_edges: Optional[viser.GuiCheckboxHandle] = None
        self._gui_centerlines: Optional[viser.GuiCheckboxHandle] = None
        self._dark_mode: bool = context.dark_mode

    @property
    def name(self) -> str:
        return "Map"

    def create_gui(self, server: viser.ViserServer) -> None:
        self._server = server
        self._gui_visible = server.gui.add_checkbox("Visible", self._config.visible)
        self._gui_radius = server.gui.add_slider(
            "Radius", min=10.0, max=1000.0, step=1.0, initial_value=self._config.radius
        )
        gui_radius_options = server.gui.add_button_group("Radius Options.", ("25", "50", "100", "500"))

        self._gui_opacity = server.gui.add_slider(
            "Opacity",
            min=0.0,
            max=1.0,
            step=0.05,
            initial_value=self._config.opacity,
        )

        self._gui_visible.on_update(self._on_visibility_changed)
        self._gui_radius.on_update(self._on_radius_changed)
        self._gui_opacity.on_update(self._on_opacity_changed)
        gui_radius_options.on_click(self._on_radius_preset_clicked)

        server.gui.add_markdown("**Layers**")
        for layer in _MAP_DISPLAY_LAYERS:
            label = layer.serialize(lower=False).replace("_", " ").title()
            checked = layer in self._config.visible_layers
            cb = server.gui.add_checkbox(label, checked)
            cb.on_update(self._on_layer_visibility_changed)
            self._gui_layer_checkboxes[layer] = cb

        server.gui.add_markdown("**Lines**")
        self._gui_road_edges = server.gui.add_checkbox("Road Edges", self._config.show_road_edges)
        self._gui_centerlines = server.gui.add_checkbox("Centerlines", self._config.show_centerlines)
        self._gui_road_edges.on_update(self._on_line_visibility_changed)
        self._gui_centerlines.on_update(self._on_line_visibility_changed)

    def update(self, iteration: int) -> None:
        self._current_iteration = iteration
        if not self._gui_visible.value:
            return

        needs_update = len(self._handles) == 0 or self._force_update
        current_ego_state = self._context.initial_ego_state

        if not needs_update and self._config.requery:
            current_ego_state = self._context.scene.get_ego_state_se3_at_iteration(iteration)
            current_position = current_ego_state.center_se3.point_3d
            if np.linalg.norm(current_position.array - self._last_query_position.array) > self._config.radius / 2:
                needs_update = True

        if not needs_update:
            return

        map_data = _get_map_data(
            self._context.scene,
            self._context.initial_ego_state,
            current_ego_state,
            self._config.radius,
            self._config.non_road_z_offset,
            self._config.centerline_dash_length,
        )
        self._last_query_position = current_ego_state.center_se3.point_3d
        self._force_update = False

        opacity = self._gui_opacity.value

        # Surface layers
        for map_layer, mesh in map_data["surfaces"].items():
            layer_cb = self._gui_layer_checkboxes.get(map_layer)
            is_visible = self._gui_visible.value and (layer_cb is None or layer_cb.value)
            color = MAP_SURFACE_CONFIG[map_layer].fill_color.rgb
            self._handles[f"surface/{map_layer.serialize()}"] = self._server.scene.add_mesh_simple(
                f"/map/{map_layer.serialize()}",
                vertices=mesh.vertices.astype(np.float32),
                faces=mesh.faces.astype(np.uint32),
                color=color,
                opacity=opacity,
                flat_shading=False,
                side="front",
                cast_shadow=False,
                receive_shadow=False,
                visible=is_visible,
            )

        # Road edge lines
        road_edge_color = _ROAD_EDGE_COLOR_DARK if self._dark_mode else _ROAD_EDGE_COLOR_LIGHT
        road_edge_segments = map_data["road_edges"]
        if road_edge_segments is not None and len(road_edge_segments) > 0:
            colors = np.full(road_edge_segments.shape, np.array(road_edge_color) / 255.0, dtype=np.float32)
            self._handles["lines/road_edges"] = self._server.scene.add_line_segments(
                "/map/road_edges",
                points=road_edge_segments,
                colors=colors,
                line_width=2.0,
                visible=self._gui_road_edges.value,
            )

        # Lane centerlines
        centerline_color = _CENTERLINE_COLOR_DARK if self._dark_mode else _CENTERLINE_COLOR_LIGHT
        centerline_segments = map_data["centerlines"]
        if centerline_segments is not None and len(centerline_segments) > 0:
            colors = np.full(centerline_segments.shape, np.array(centerline_color) / 255.0, dtype=np.float32)
            self._handles["lines/centerlines"] = self._server.scene.add_line_segments(
                "/map/centerlines",
                points=centerline_segments,
                colors=colors,
                line_width=1.5,
                visible=self._gui_centerlines.value,
            )

    def remove(self) -> None:
        for handle in self._handles.values():
            if handle is not None:
                handle.remove()
        self._handles.clear()
        self._last_query_position = None

    def _sync_visibility(self) -> None:
        master = self._gui_visible.value
        for layer in _MAP_DISPLAY_LAYERS:
            key = f"surface/{layer.serialize()}"
            handle = self._handles.get(key)
            if handle is not None:
                layer_cb = self._gui_layer_checkboxes.get(layer)
                handle.visible = master and (layer_cb is None or layer_cb.value)

        road_edge_handle = self._handles.get("lines/road_edges")
        if road_edge_handle is not None:
            road_edge_handle.visible = master and self._gui_road_edges.value

        centerline_handle = self._handles.get("lines/centerlines")
        if centerline_handle is not None:
            centerline_handle.visible = master and self._gui_centerlines.value

    def _sync_config_visible_layers(self) -> None:
        """Update config.visible_layers to match current GUI checkbox state."""
        self._config.visible_layers = [layer for layer, cb in self._gui_layer_checkboxes.items() if cb.value]

    def _on_visibility_changed(self, _) -> None:
        self._config.visible = self._gui_visible.value
        self._sync_visibility()

    def _on_layer_visibility_changed(self, _) -> None:
        self._sync_config_visible_layers()
        self._sync_visibility()

    def _on_line_visibility_changed(self, _) -> None:
        self._config.show_road_edges = self._gui_road_edges.value
        self._config.show_centerlines = self._gui_centerlines.value
        self._sync_visibility()

    def _on_radius_changed(self, _) -> None:
        self._config.radius = self._gui_radius.value
        self._force_update = True
        self.update(self._current_iteration)

    def _on_opacity_changed(self, _) -> None:
        self._config.opacity = self._gui_opacity.value
        self._force_update = True
        self.update(self._current_iteration)

    def _on_radius_preset_clicked(self, event) -> None:
        self._gui_radius.value = float(event.target.value)
        self._force_update = True
        self.update(self._current_iteration)

    def on_dark_mode_changed(self, dark_mode: bool) -> None:
        self._dark_mode = dark_mode
        self._force_update = True
        self.update(self._current_iteration)


def _polyline_to_segments(points: np.ndarray) -> np.ndarray:
    """Convert a polyline (N, 3) to line segments (N-1, 2, 3) for viser."""
    return np.stack([points[:-1], points[1:]], axis=1)


def _polyline_to_dashed_segments(polyline: Polyline3D, dash_length: float) -> np.ndarray:
    """Resample a polyline at fixed intervals and return every other segment to produce dashes.

    :param polyline: The 3D polyline to dash.
    :param dash_length: Length of each dash (and gap) in world units (meters).
    :return: (M, 2, 3) dashed line segments.
    """
    total_length = polyline.length
    if total_length < 1e-6 or dash_length <= 0:
        return _polyline_to_segments(polyline.array)

    sample_distances = np.arange(0, total_length, dash_length, dtype=np.float64)
    resampled = np.asarray(polyline.interpolate(sample_distances), dtype=np.float64)

    segments = _polyline_to_segments(resampled)
    return segments[::2]


def _get_map_data(
    scene: SceneAPI,
    initial_ego_state: EgoStateSE3,
    current_ego_state: EgoStateSE3,
    radius: float,
    non_road_z_offset: float,
    centerline_dash_length: float = 1.0,
) -> dict:
    output: dict = {"surfaces": {}, "road_edges": None, "centerlines": None}

    scene_center: Point3D = initial_ego_state.center_se3.point_3d
    scene_center_array = scene_center.array
    scene_query_position = current_ego_state.center_se3.point_3d

    surface_layers = [
        MapLayer.LANE,
        MapLayer.LANE_GROUP,
        MapLayer.INTERSECTION,
        MapLayer.WALKWAY,
        MapLayer.CROSSWALK,
        MapLayer.CARPARK,
        MapLayer.GENERIC_DRIVABLE,
        MapLayer.STOP_ZONE,
    ]
    line_layers = [MapLayer.ROAD_EDGE]

    map_api = scene.get_map_api()
    if map_api is None:
        return output

    all_layers = surface_layers + line_layers
    map_objects_dict = map_api.get_map_objects_in_radius(
        scene_query_position.point_2d,
        radius=radius,
        layers=all_layers,
    )

    # Save lane objects for centerline extraction before the lane/lane_group pop
    lane_objects = map_objects_dict.get(MapLayer.LANE, [])

    # Handle lane vs lane_group preference for surface rendering
    if len(map_objects_dict.get(MapLayer.LANE_GROUP, [])) == 0:
        map_objects_dict.pop(MapLayer.LANE_GROUP, None)
    else:
        map_objects_dict.pop(MapLayer.LANE, None)

    z_offset_no_z = 0.0
    if not map_api.map_metadata.map_has_z:
        z_offset_no_z = scene_query_position.z - initial_ego_state.metadata.height / 2

    # Surface meshes
    for map_layer in surface_layers:
        if map_layer not in map_objects_dict:
            continue
        surface_meshes = []
        for map_surface in map_objects_dict[map_layer]:
            map_surface: BaseMapSurfaceObject

            if isinstance(map_surface, StopZone) and map_surface.stop_zone_type == StopZoneType.TURN_STOP:
                continue

            trimesh_mesh = map_surface.trimesh_mesh
            trimesh_mesh.vertices -= scene_center_array

            if map_layer in {MapLayer.WALKWAY, MapLayer.CROSSWALK, MapLayer.CARPARK, MapLayer.STOP_ZONE}:
                trimesh_mesh.vertices[..., Point3DIndex.Z] += non_road_z_offset

            if z_offset_no_z != 0.0:
                trimesh_mesh.vertices[..., Point3DIndex.Z] += z_offset_no_z

            surface_meshes.append(trimesh_mesh)

        if len(surface_meshes) > 0:
            output["surfaces"][map_layer] = trimesh.util.concatenate(surface_meshes)

    # Road edge line segments
    road_edge_objects = map_objects_dict.get(MapLayer.ROAD_EDGE, [])
    if len(road_edge_objects) > 0:
        all_segments = []
        for road_edge in road_edge_objects:
            road_edge: BaseMapLineObject
            pts = road_edge.polyline_3d.array.copy()
            pts -= scene_center_array
            pts[..., Point3DIndex.Z] += _LINE_Z_OFFSET + z_offset_no_z
            if len(pts) >= 2:
                all_segments.append(_polyline_to_segments(pts))
        if len(all_segments) > 0:
            output["road_edges"] = np.concatenate(all_segments, axis=0).astype(np.float64)

    # Lane centerlines (extracted from Lane objects saved before lane/lane_group pop)
    if len(lane_objects) > 0:
        all_segments = []
        for lane_obj in lane_objects:
            if not isinstance(lane_obj, Lane):
                continue
            centerline = lane_obj.centerline_3d
            if len(centerline.array) < 2:
                continue
            dashed = _polyline_to_dashed_segments(centerline, dash_length=centerline_dash_length)
            dashed[..., :3] -= scene_center_array
            dashed[..., Point3DIndex.Z] += _LINE_Z_OFFSET + z_offset_no_z
            all_segments.append(dashed)
        if len(all_segments) > 0:
            output["centerlines"] = np.concatenate(all_segments, axis=0).astype(np.float64)

    return output
