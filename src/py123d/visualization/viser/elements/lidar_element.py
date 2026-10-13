import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np
import viser

from py123d.common.utils.enums import resolve_enum_arguments
from py123d.datatypes.sensors.lidar import LidarID
from py123d.geometry import PoseSE3Index
from py123d.geometry.transform.transform_se3 import (
    abs_to_rel_se3_array,
    rel_to_abs_points_3d_array,
    rel_to_abs_se3_array,
)
from py123d.visualization.matplotlib.lidar import get_lidar_pc_color
from py123d.visualization.viser.elements.base_element import ElementContext, ViewerElement
from py123d.visualization.viser.utils.parallel_fetch import fetch_parallel
from py123d.visualization.viser.utils.view_utils import get_scene_center_pose

logger = logging.getLogger(__name__)


@dataclass
class LidarConfig:
    visible: bool = True
    # Sensors that start checked in the per-sensor checklist; every id available in the
    # scene gets its own checkbox and its own point-cloud node.
    ids: List[LidarID] = field(default_factory=lambda: [LidarID.LIDAR_MERGED])
    point_size: float = 0.02
    point_shape: Literal["square", "diamond", "circle", "rounded", "sparkle"] = "circle"
    point_color: Literal[
        "none",
        "height",
        "distance",
        "ids",
        "intensity",
        "channel",
        "timestamps",
        "range",
        "elongation",
        "semantic",
        "instance",
    ] = "height"
    stride_step: int = 1
    # Per-sensor display cap. viser transmits every point on every timestep, so
    # uncapped high-resolution sensors (800k+ points) saturate the connection and
    # starve playback. Applied after stride_step; <=0 disables the cap.
    max_points: int = 100_000
    show_sensor_frames: bool = False

    def __post_init__(self):
        self.ids = resolve_enum_arguments(LidarID, self.ids)  # type: ignore


class LidarElement(ViewerElement):
    """Visualizes lidar point clouds in the 3D scene."""

    def __init__(self, context: ElementContext, config: LidarConfig) -> None:
        self._context = context
        self._config = config
        self._server: Optional[viser.ViserServer] = None
        self._handles: Dict[LidarID, Optional[viser.PointCloudHandle]] = {}
        self._frame_handles: List[viser.FrameHandle] = []
        self._gui_visible: Optional[viser.GuiCheckboxHandle] = None
        self._gui_coloring: Optional[viser.GuiDropdownHandle] = None
        self._gui_sensor_checkboxes: Dict[LidarID, viser.GuiCheckboxHandle] = {}
        self._gui_point_size: Optional[viser.GuiInputHandle] = None
        self._gui_stride_step: Optional[viser.GuiInputHandle] = None
        self._gui_max_points: Optional[viser.GuiSliderHandle] = None
        self._gui_show_sensor_frames: Optional[viser.GuiCheckboxHandle] = None
        self._dark_mode: bool = context.dark_mode
        self._current_iteration: int = 0

    @property
    def name(self) -> str:
        return "Lidar"

    def create_gui(self, server: viser.ViserServer) -> None:
        self._server = server
        lidar_id_list = self._context.scene.available_lidar_ids

        self._gui_visible = server.gui.add_checkbox("Visible", self._config.visible)
        self._gui_coloring = server.gui.add_dropdown(
            "Coloring",
            (
                "none",
                "height",
                "distance",
                "ids",
                "intensity",
                "channel",
                "timestamps",
                "range",
                "elongation",
                "semantic",
                "instance",
            ),
            initial_value=self._config.point_color,
        )
        self._gui_point_size = server.gui.add_slider(
            "Point Size",
            min=0.001,
            max=0.2,
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

        self._gui_max_points = server.gui.add_slider(
            "Max Points",
            min=0,
            max=1_000_000,
            step=10_000,
            initial_value=max(0, self._config.max_points),
            hint="Per-sensor display cap; 0 disables the cap.",
        )

        self._gui_show_sensor_frames = server.gui.add_checkbox("Show Sensor Frames", self._config.show_sensor_frames)

        # One checkbox per available sensor, same style as the camera frustum checklist.
        server.gui.add_markdown("**Lidars**")
        # A scene without any of the configured sensors (e.g. no merged lidar) starts with all of its sensors checked.
        checked_ids = [lidar_id for lidar_id in lidar_id_list if lidar_id in self._config.ids] or lidar_id_list
        for lidar_id in lidar_id_list:
            checkbox = server.gui.add_checkbox(lidar_id.serialize(lower=False), lidar_id in checked_ids)
            checkbox.on_update(self._on_sensor_selection_changed)
            self._gui_sensor_checkboxes[lidar_id] = checkbox

        self._gui_visible.on_update(self._on_visibility_changed)
        self._gui_coloring.on_update(self._on_coloring_changed)
        self._gui_point_size.on_update(self._on_point_size_changed)
        self._gui_stride_step.on_update(self._on_stride_step_changed)
        self._gui_max_points.on_update(self._on_max_points_changed)
        self._gui_show_sensor_frames.on_update(self._on_show_sensor_frames_changed)

    def _selected_ids(self) -> List[LidarID]:
        """Sensors whose checklist entry is currently checked."""
        return [lidar_id for lidar_id, checkbox in self._gui_sensor_checkboxes.items() if checkbox.value]

    def update(self, iteration: int) -> None:
        assert self._server is not None
        assert self._gui_visible is not None
        assert self._gui_coloring is not None
        self._current_iteration = iteration

        selected = set(self._selected_ids()) if self._gui_visible.value else set()

        # Hide every sensor that is deselected (or the whole element invisible) first, so a
        # selection change never leaves a stale cloud behind.
        for lidar_id, handle in self._handles.items():
            if handle is not None and lidar_id not in selected:
                handle.visible = False

        if not selected:
            return

        ego_state_se3 = self._context.scene.get_ego_state_se3_at_iteration(iteration)
        assert ego_state_se3 is not None, f"Ego state SE3 should be available at iteration {iteration}."
        ego_pose = ego_state_se3.imu_se3.array
        ego_pose = ego_pose.astype(np.float64)
        ego_pose[PoseSE3Index.XYZ] -= self._context.scene_center_array.astype(np.float64)

        # viser transmits point positions as float16: coordinates far from the origin
        # quantize visibly (~1 m at 1.5 km). Points are therefore kept ego-centered
        # (world-oriented, small magnitudes) and the node itself is anchored at the ego
        # position, which travels as a full-precision float32 node transform.
        ego_position = ego_pose[PoseSE3Index.XYZ].copy()
        rotation_only_pose = ego_pose.copy()
        rotation_only_pose[PoseSE3Index.XYZ] = 0.0

        def _fetch_cloud(lidar_id: LidarID):
            """Heavy part (arrow read, transform, coloring); runs on the worker pool."""
            lidar = self._context.scene.get_lidar_at_iteration(iteration, lidar_id=lidar_id)
            if lidar is None:
                return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
            # Subsample BEFORE transforming and coloring: with the display cap active,
            # only the kept points pay for the heavy per-point work.
            num_points = len(lidar.xyz)
            step = max(1, self._config.stride_step)
            max_points = self._config.max_points
            if max_points > 0:
                remaining = int(np.ceil(num_points / step))
                if remaining > max_points:
                    step *= int(np.ceil(remaining / max_points))
            xyz = np.array(lidar.xyz[::step], dtype=np.float64)
            points = rel_to_abs_points_3d_array(rotation_only_pose, xyz)
            colors = get_lidar_pc_color(
                lidar,
                color_feature=self._config.point_color,
                dark_mode=self._dark_mode,
                stride=step,
                range_smoothing_key=lidar_id.serialize(),
            )
            return points, colors

        selected_list = list(selected)
        results = fetch_parallel(_fetch_cloud, selected_list)

        for lidar_id, (points, colors) in zip(selected_list, results):
            handle = self._handles.get(lidar_id)
            if handle is not None:
                handle.points = points  # type: ignore
                handle.colors = colors  # type: ignore
                handle.position = ego_position
                handle.visible = True
            else:
                # One uniquely named node per sensor; a shared name would make every added
                # cloud replace the previous sensor's node server-side while its stale
                # handle lives on, which is what made selection changes apply erratically.
                self._handles[lidar_id] = self._server.scene.add_point_cloud(
                    f"lidar_points/{lidar_id.serialize()}",
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
            # A cloud may not exist yet when the element was launched hidden (update()
            # skips hidden elements); rebuild for the current iteration instead of only
            # toggling handle visibility.
            self.update(self._current_iteration)
        else:
            for handle in self._handles.values():
                if handle is not None:
                    handle.visible = False
        if not self._gui_visible.value:
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

    def _on_max_points_changed(self, _) -> None:
        assert self._gui_max_points is not None
        self._config.max_points = self._gui_max_points.value
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

        lidar_metadatas = {}
        for selected_id in self._selected_ids():
            lidar = self._context.scene.get_lidar_at_iteration(iteration, lidar_id=selected_id)
            if lidar is not None:
                lidar_metadatas.update(lidar.lidar_metadatas)
        if not lidar_metadatas:
            return

        ego_pose = self._context.scene.get_ego_state_se3_at_iteration(iteration).imu_se3  # type: ignore
        scene_center_pose = get_scene_center_pose(self._context.scene_center_array)

        for lidar_id, lidar_meta in lidar_metadatas.items():
            lidar_world_pose = rel_to_abs_se3_array(ego_pose, lidar_meta.lidar_to_imu_se3.array)
            lidar_scene_pose = abs_to_rel_se3_array(origin=scene_center_pose, pose_se3_array=lidar_world_pose)
            position = lidar_scene_pose[PoseSE3Index.XYZ]
            wxyz = lidar_scene_pose[PoseSE3Index.QUATERNION]

            frame_handle = self._server.scene.add_frame(
                f"lidar_sensor_frames/{lidar_id.name}",
                axes_length=0.5,
                axes_radius=0.01,
                position=position,
                wxyz=wxyz,
            )
            self._frame_handles.append(frame_handle)
