import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np
import viser

from py123d.common.utils.enums import resolve_enum_arguments
from py123d.datatypes.sensors.radar import RadarID
from py123d.geometry import PoseSE3Index
from py123d.geometry.transform.transform_se3 import (
    abs_to_rel_se3_array,
    rel_to_abs_points_3d_array,
    rel_to_abs_se3_array,
)
from py123d.visualization.matplotlib.radar import get_radar_pc_color
from py123d.visualization.viser.elements.base_element import ElementContext, ViewerElement
from py123d.visualization.viser.utils.parallel_fetch import fetch_parallel
from py123d.visualization.viser.utils.view_utils import get_scene_center_pose

logger = logging.getLogger(__name__)

# Signal-quality features (rcs, snr, confidence) directly after "none": they are the most
# relevant radar statistics and stay grouped in the dropdown.
_RADAR_COLOR_OPTIONS = (
    "none",
    "rcs",
    "snr",
    "confidence",
    "height",
    "distance",
    "ids",
    "cluster_id",
    "velocity",
    "velocity_comp",
    "timestamps",
)


@dataclass
class RadarConfig:
    visible: bool = False
    # Sensors that start checked in the per-sensor checklist; every id available in the
    # scene gets its own checkbox and its own point-cloud node.
    ids: List[RadarID] = field(default_factory=lambda: [RadarID.RADAR_MERGED])
    point_size: float = 0.1
    point_shape: Literal["square", "diamond", "circle", "rounded", "sparkle"] = "circle"
    point_color: Literal[
        "none",
        "rcs",
        "snr",
        "confidence",
        "height",
        "distance",
        "ids",
        "cluster_id",
        "velocity",
        "velocity_comp",
        "timestamps",
    ] = "none"
    stride_step: int = 1
    show_sensor_frames: bool = False

    def __post_init__(self):
        self.ids = resolve_enum_arguments(RadarID, self.ids)  # type: ignore


class RadarElement(ViewerElement):
    """Visualizes radar point clouds in the 3D scene. Kept fully separate from the lidar element."""

    def __init__(self, context: ElementContext, config: RadarConfig) -> None:
        self._context = context
        self._config = config
        self._server: Optional[viser.ViserServer] = None
        self._handles: Dict[RadarID, Optional[viser.PointCloudHandle]] = {}
        self._frame_handles: List[viser.FrameHandle] = []
        self._gui_visible: Optional[viser.GuiCheckboxHandle] = None
        self._gui_coloring: Optional[viser.GuiDropdownHandle] = None
        self._gui_sensor_checkboxes: Dict[RadarID, viser.GuiCheckboxHandle] = {}
        self._gui_point_size: Optional[viser.GuiInputHandle] = None
        self._gui_stride_step: Optional[viser.GuiInputHandle] = None
        self._gui_show_sensor_frames: Optional[viser.GuiCheckboxHandle] = None
        self._dark_mode: bool = context.dark_mode
        self._current_iteration: int = 0

    @property
    def name(self) -> str:
        return "Radar"

    def create_gui(self, server: viser.ViserServer) -> None:
        self._server = server
        radar_id_list = self._context.scene.available_radar_ids

        self._gui_visible = server.gui.add_checkbox("Visible", self._config.visible)
        self._gui_coloring = server.gui.add_dropdown(
            "Coloring",
            _RADAR_COLOR_OPTIONS,
            initial_value=self._config.point_color,
        )
        self._gui_point_size = server.gui.add_slider(
            "Point Size",
            min=0.001,
            max=1.0,
            step=0.001,
            initial_value=self._config.point_size,
        )

        self._gui_stride_step = server.gui.add_slider(
            "Stride Step",
            min=1,
            max=10,
            step=1,
            initial_value=self._config.stride_step,
        )

        self._gui_show_sensor_frames = server.gui.add_checkbox("Show Sensor Frames", self._config.show_sensor_frames)

        # One checkbox per available sensor, same style as the camera frustum checklist.
        server.gui.add_markdown("**Radars**")
        # A scene without any of the configured sensors (e.g. no merged radar) starts with all of its sensors checked.
        checked_ids = [radar_id for radar_id in radar_id_list if radar_id in self._config.ids] or radar_id_list
        for radar_id in radar_id_list:
            checkbox = server.gui.add_checkbox(radar_id.serialize(lower=False), radar_id in checked_ids)
            checkbox.on_update(self._on_sensor_selection_changed)
            self._gui_sensor_checkboxes[radar_id] = checkbox

        self._gui_visible.on_update(self._on_visibility_changed)
        self._gui_coloring.on_update(self._on_coloring_changed)
        self._gui_point_size.on_update(self._on_point_size_changed)
        self._gui_stride_step.on_update(self._on_stride_step_changed)
        self._gui_show_sensor_frames.on_update(self._on_show_sensor_frames_changed)

    def _selected_ids(self) -> List[RadarID]:
        """Sensors whose checklist entry is currently checked."""
        return [radar_id for radar_id, checkbox in self._gui_sensor_checkboxes.items() if checkbox.value]

    def update(self, iteration: int) -> None:
        assert self._server is not None
        assert self._gui_visible is not None
        assert self._gui_coloring is not None
        self._current_iteration = iteration

        selected = set(self._selected_ids()) if self._gui_visible.value else set()

        # Hide every sensor that is deselected (or the whole element invisible) first, so a
        # selection change never leaves a stale cloud behind.
        for radar_id, handle in self._handles.items():
            if handle is not None and radar_id not in selected:
                handle.visible = False

        if not selected:
            return

        ego_state_se3 = self._context.scene.get_ego_state_se3_at_iteration(iteration)
        assert ego_state_se3 is not None, f"Ego state SE3 should be available at iteration {iteration}."
        ego_pose = ego_state_se3.imu_se3.array
        ego_pose = ego_pose.astype(np.float64)
        ego_pose[PoseSE3Index.XYZ] -= self._context.scene_center_array.astype(np.float64)

        # Same float16-transport mitigation as the lidar element: ego-centered points,
        # node anchored at the ego position (full-precision float32 node transform).
        ego_position = ego_pose[PoseSE3Index.XYZ].copy()
        rotation_only_pose = ego_pose.copy()
        rotation_only_pose[PoseSE3Index.XYZ] = 0.0

        def _fetch_cloud(radar_id: RadarID):
            """Heavy part (arrow read, transform, coloring); runs on the worker pool."""
            radar = self._context.scene.get_radar_at_iteration(iteration, radar_id=radar_id)
            if radar is not None:
                xyz = np.array(radar.xyz, dtype=np.float64)
                points = rel_to_abs_points_3d_array(rotation_only_pose, xyz)
                colors = get_radar_pc_color(
                    radar,
                    color_feature=self._config.point_color,
                    dark_mode=self._dark_mode,
                    range_smoothing_key=radar_id.serialize(),
                )
            else:
                points = np.zeros((0, 3), dtype=np.float32)
                colors = np.zeros((0, 3), dtype=np.uint8)
            return self._downsample(points, colors)

        selected_list = list(selected)
        results = fetch_parallel(_fetch_cloud, selected_list)

        for radar_id, (points, colors) in zip(selected_list, results):
            handle = self._handles.get(radar_id)
            if handle is not None:
                handle.points = points  # type: ignore
                handle.colors = colors  # type: ignore
                handle.position = ego_position
                handle.visible = True
            else:
                # One uniquely named node per sensor; a shared name would make every added
                # cloud replace the previous sensor's node server-side while its stale
                # handle lives on, which is what made selection changes apply erratically.
                self._handles[radar_id] = self._server.scene.add_point_cloud(
                    f"radar_points/{radar_id.serialize()}",
                    points=points,
                    colors=colors,
                    point_size=self._config.point_size,
                    point_shape=self._config.point_shape,
                    position=ego_position,
                )

        self._update_sensor_frames(iteration)

    def remove(self) -> None:
        for handle in self._handles.values():
            if handle is not None:
                handle.remove()
        self._handles.clear()
        self._remove_sensor_frames()

    def _on_visibility_changed(self, _) -> None:
        assert self._gui_visible is not None
        self._config.visible = self._gui_visible.value
        if self._gui_visible.value:
            # The element starts hidden, so no cloud may exist yet (update() skips hidden
            # elements); rebuilding for the current iteration creates and shows them.
            # Toggling handle.visible alone would show nothing until the next frame change.
            self.update(self._current_iteration)
        else:
            for handle in self._handles.values():
                if handle is not None:
                    handle.visible = False
            self._remove_sensor_frames()

    def _on_coloring_changed(self, _) -> None:
        assert self._gui_coloring is not None
        self._config.point_color = self._gui_coloring.value
        self.update(self._current_iteration)

    def _on_sensor_selection_changed(self, _) -> None:
        self._config.ids = self._selected_ids()
        self.update(self._current_iteration)

    def _on_point_size_changed(self, _) -> None:
        assert self._gui_point_size is not None
        self._config.point_size = self._gui_point_size.value
        for handle in self._handles.values():
            if handle is not None:
                handle.point_size = self._gui_point_size.value

    def _on_stride_step_changed(self, _) -> None:
        assert self._gui_stride_step is not None
        self._config.stride_step = self._gui_stride_step.value
        self.update(self._current_iteration)

    def on_dark_mode_changed(self, dark_mode: bool) -> None:
        self._dark_mode = dark_mode
        self.update(self._current_iteration)

    def _on_show_sensor_frames_changed(self, _) -> None:
        assert self._gui_show_sensor_frames is not None
        self._config.show_sensor_frames = self._gui_show_sensor_frames.value
        self.update(self._current_iteration)

    def _remove_sensor_frames(self) -> None:
        assert self._server is not None
        for handle in self._frame_handles:
            handle.remove()
        self._frame_handles.clear()

    def _update_sensor_frames(self, iteration: int) -> None:
        assert self._server is not None
        assert self._gui_show_sensor_frames is not None
        assert self._gui_visible is not None

        self._remove_sensor_frames()

        if not self._gui_visible.value or not self._gui_show_sensor_frames.value:
            return

        radar_metadatas = {}
        for selected_id in self._selected_ids():
            radar = self._context.scene.get_radar_at_iteration(iteration, radar_id=selected_id)
            if radar is not None:
                radar_metadatas.update(radar.radar_metadatas)
        if not radar_metadatas:
            return

        ego_pose = self._context.scene.get_ego_state_se3_at_iteration(iteration).imu_se3  # type: ignore
        scene_center_pose = get_scene_center_pose(self._context.scene_center_array)

        for radar_id, radar_meta in radar_metadatas.items():
            radar_world_pose = rel_to_abs_se3_array(ego_pose, radar_meta.radar_to_imu_se3.array)
            radar_scene_pose = abs_to_rel_se3_array(origin=scene_center_pose, pose_se3_array=radar_world_pose)
            position = radar_scene_pose[PoseSE3Index.XYZ]
            wxyz = radar_scene_pose[PoseSE3Index.QUATERNION]

            frame_handle = self._server.scene.add_frame(
                f"radar_sensor_frames/{radar_id.name}",
                axes_length=0.5,
                axes_radius=0.01,
                position=position,
                wxyz=wxyz,
            )
            self._frame_handles.append(frame_handle)

    def _downsample(self, points: np.ndarray, colors: np.ndarray) -> tuple:
        if len(points) == 0 or self._config.stride_step <= 1:
            return points, colors
        step = self._config.stride_step
        return points[::step], colors[::step]
