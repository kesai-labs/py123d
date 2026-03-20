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
from py123d.visualization.viser.utils.view_utils import get_scene_center_pose

logger = logging.getLogger(__name__)


@dataclass
class LidarConfig:
    visible: bool = True
    ids: List[LidarID] = field(default_factory=lambda: [LidarID.LIDAR_MERGED])
    point_size: float = 0.02
    point_shape: Literal["square", "diamond", "circle", "rounded", "sparkle"] = "circle"
    point_color: Literal[
        "none", "height", "distance", "ids", "intensity", "channel", "timestamps", "range", "elongation"
    ] = "none"
    stride_step: int = 1
    show_sensor_frames: bool = False

    def __post_init__(self):
        self.ids = resolve_enum_arguments(LidarID, self.ids)  # type: ignore


class LidarElement(ViewerElement):
    """Visualizes lidar point clouds in the 3D scene."""

    def __init__(self, context: ElementContext, config: LidarConfig) -> None:
        self._context = context
        self._config = config
        self._server: Optional[viser.ViserServer] = None
        self._handles: Dict[LidarID, Optional[viser.PointCloudHandle]] = {config.ids[0]: None}
        self._frame_handles: List[viser.FrameHandle] = []
        self._gui_visible: Optional[viser.GuiCheckboxHandle] = None
        self._gui_coloring: Optional[viser.GuiDropdownHandle] = None
        self._gui_lidar_id: Optional[viser.GuiDropdownHandle] = None
        self._gui_point_size: Optional[viser.GuiInputHandle] = None
        self._gui_stride_step: Optional[viser.GuiInputHandle] = None
        self._gui_show_sensor_frames: Optional[viser.GuiCheckboxHandle] = None
        self._dark_mode: bool = context.dark_mode
        self._current_iteration: int = 0

    @property
    def name(self) -> str:
        return "Lidar"

    def create_gui(self, server: viser.ViserServer) -> None:
        self._server = server
        lidar_id_list = self._context.scene.available_lidar_ids
        lidar_id_names = tuple(lid.name for lid in lidar_id_list)

        self._gui_visible = server.gui.add_checkbox("Visible", self._config.visible)
        self._gui_coloring = server.gui.add_dropdown(
            "Coloring",
            ("none", "height", "distance", "ids", "intensity", "channel", "timestamps", "range", "elongation"),
            initial_value=self._config.point_color,
        )
        self._gui_lidar_id = server.gui.add_dropdown(
            "Lidar ID",
            lidar_id_names,
            initial_value=self._config.ids[0].name,
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

        self._gui_show_sensor_frames = server.gui.add_checkbox("Show Sensor Frames", self._config.show_sensor_frames)

        self._gui_visible.on_update(self._on_visibility_changed)
        self._gui_coloring.on_update(self._on_coloring_changed)
        self._gui_lidar_id.on_update(self._on_lidar_id_changed)
        self._gui_point_size.on_update(self._on_point_size_changed)
        self._gui_stride_step.on_update(self._on_stride_step_changed)
        self._gui_show_sensor_frames.on_update(self._on_show_sensor_frames_changed)

    def update(self, iteration: int) -> None:
        assert self._server is not None
        assert self._gui_visible is not None
        assert self._gui_coloring is not None
        self._current_iteration = iteration
        active_id = self._config.ids[0]

        if active_id not in self._handles:
            self._handles[active_id] = None

        if not self._gui_visible.value:
            if self._handles[active_id] is not None:
                self._handles[active_id].visible = False  # type: ignore
            return

        ego_state_se3 = self._context.scene.get_ego_state_se3_at_iteration(iteration)
        assert ego_state_se3 is not None, f"Ego state SE3 should be available at iteration {iteration}."
        ego_pose = ego_state_se3.imu_se3.array
        ego_pose[PoseSE3Index.XYZ] -= self._context.scene_center_array

        lidar = self._context.scene.get_lidar_at_iteration(iteration, lidar_id=active_id)
        if lidar is not None:
            points = rel_to_abs_points_3d_array(ego_pose, lidar.xyz.astype(np.float64))
            colors = get_lidar_pc_color(lidar, feature=self._config.point_color, dark_mode=self._dark_mode)
        else:
            points = np.zeros((0, 3), dtype=np.float32)
            colors = np.zeros((0, 3), dtype=np.uint8)

        points, colors = self._downsample(points, colors)

        if self._handles[active_id] is not None:
            self._handles[active_id].points = points  # type: ignore
            self._handles[active_id].colors = colors  # type: ignore
            self._handles[active_id].visible = True  # type: ignore
        else:
            self._handles[active_id] = self._server.scene.add_point_cloud(  # type: ignore
                "lidar_points",
                points=points,
                colors=colors,
                point_size=self._config.point_size,
                point_shape=self._config.point_shape,
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
        for handle in self._handles.values():
            if handle is not None:
                handle.visible = self._gui_visible.value
        if not self._gui_visible.value:
            self._remove_sensor_frames()

    def _on_coloring_changed(self, _) -> None:
        assert self._gui_coloring is not None
        self._config.point_color = self._gui_coloring.value
        self.update(self._current_iteration)

    def _on_lidar_id_changed(self, _) -> None:
        assert self._gui_lidar_id is not None
        self._config.ids = [LidarID[self._gui_lidar_id.value]]
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

        active_id = self._config.ids[0]
        lidar = self._context.scene.get_lidar_at_iteration(iteration, lidar_id=active_id)
        if lidar is None:
            return

        ego_pose = self._context.scene.get_ego_state_se3_at_iteration(iteration).imu_se3  # type: ignore
        scene_center_pose = get_scene_center_pose(self._context.scene_center_array)

        for lidar_id, lidar_meta in lidar.lidar_metadatas.items():
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

    def _downsample(self, points: np.ndarray, colors: np.ndarray) -> tuple:
        if len(points) == 0 or self._config.stride_step <= 1:
            return points, colors
        step = self._config.stride_step
        return points[::step], colors[::step]
