import logging
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Union

import numpy as np
import trimesh
import trimesh.visual.material
import viser

from py123d.datatypes.detections.box_detection_label import DefaultBoxDetectionLabel
from py123d.geometry.geometry_index import BoundingBoxSE3Index, Corners3DIndex, PoseSE3Index
from py123d.geometry.utils.bounding_box_utils import (
    bbse3_array_to_corners_array,
    corners_array_to_3d_mesh,
    corners_array_to_edge_lines,
)
from py123d.visualization.color.default import BOX_DETECTION_CONFIG
from py123d.visualization.viser.elements.base_element import ElementContext, ViewerElement

logger = logging.getLogger(__name__)


@dataclass
class EgoConfig:
    visible: bool = True
    type: Literal["mesh", "lines", "mesh+lines"] = "mesh+lines"
    line_width: float = 2.0
    opacity: float = 0.5
    show_imu_frame: bool = False


class EgoElement(ViewerElement):
    """Visualizes the ego vehicle bounding box in the scene."""

    def __init__(self, context: ElementContext, config: EgoConfig) -> None:
        self._context = context
        self._config = config
        self._server: Optional[viser.ViserServer] = None
        self._handles: Dict[str, Optional[Union[viser.GlbHandle, viser.LineSegmentsHandle]]] = {
            "mesh": None,
            "lines": None,
        }
        self._imu_frame_handle: Optional[viser.FrameHandle] = None
        self._gui_visible: Optional[viser.GuiCheckboxHandle] = None
        self._gui_type: Optional[viser.GuiDropdownHandle] = None
        self._gui_opacity: Optional[viser.GuiSliderHandle] = None
        self._gui_show_imu_frame: Optional[viser.GuiCheckboxHandle] = None
        self._current_iteration: int = 0

    @property
    def name(self) -> str:
        return "Ego State (SE3)"

    def create_gui(self, server: viser.ViserServer) -> None:
        self._server = server
        self._gui_visible = server.gui.add_checkbox("Visible", self._config.visible)
        self._gui_type = server.gui.add_dropdown(
            "Type", ("mesh", "lines", "mesh+lines"), initial_value=self._config.type
        )
        self._gui_opacity = server.gui.add_slider(
            "Opacity", min=0.0, max=1.0, step=0.05, initial_value=self._config.opacity
        )
        self._gui_show_imu_frame = server.gui.add_checkbox("Show IMU Frame", self._config.show_imu_frame)
        self._gui_visible.on_update(self._on_visibility_changed)
        self._gui_type.on_update(self._on_type_changed)
        self._gui_opacity.on_update(self._on_opacity_changed)
        self._gui_show_imu_frame.on_update(self._on_show_imu_frame_changed)

    def update(self, iteration: int) -> None:
        assert self._server is not None, "Server must be set before updating element."
        assert self._gui_visible is not None, "GUI must be created before updating element."
        assert self._gui_type is not None, "GUI must be created before updating element."
        assert self._gui_opacity is not None, "GUI must be created before updating element."
        self._current_iteration = iteration
        visible_handle_keys = []
        display_type = self._gui_type.value

        if self._gui_visible.value:
            ego_vehicle_state = self._context.scene.get_ego_state_se3_at_iteration(iteration)
            assert ego_vehicle_state is not None, "Ego vehicle state must be available at the specified iteration."
            box_se3_array = np.array([ego_vehicle_state.bounding_box_se3.array])
            box_se3_array[..., BoundingBoxSE3Index.XYZ] -= self._context.initial_ego_state.center_se3.array[
                PoseSE3Index.XYZ
            ]
            box_corners_array = bbse3_array_to_corners_array(box_se3_array)

            if display_type in {"mesh", "mesh+lines"}:
                opacity = self._gui_opacity.value
                alpha = int(np.clip(opacity * 255, 0, 255))
                box_vertices, box_faces = corners_array_to_3d_mesh(box_corners_array)
                r, g, b, _ = BOX_DETECTION_CONFIG[DefaultBoxDetectionLabel.EGO].fill_color.rgba
                vertex_colors = np.tile(np.array([r, g, b, alpha]), (len(Corners3DIndex), 1))
                mesh = trimesh.Trimesh(vertices=box_vertices, faces=box_faces)
                mesh.visual.vertex_colors = vertex_colors  # type: ignore
                mesh.visual.material = trimesh.visual.material.PBRMaterial(alphaMode="BLEND")  # type: ignore
                self._handles["mesh"] = self._server.scene.add_mesh_trimesh(
                    "ego_mesh",
                    mesh=mesh,
                    visible=True,
                    cast_shadow=False,
                )
                visible_handle_keys.append("mesh")

            if display_type in {"lines", "mesh+lines"}:
                box_outlines = corners_array_to_edge_lines(box_corners_array).reshape(-1, 2, 3)
                colors = np.broadcast_to(
                    np.array(BOX_DETECTION_CONFIG[DefaultBoxDetectionLabel.EGO].fill_color.rgb),
                    (len(box_outlines), 2, 3),
                )
                self._handles["lines"] = self._server.scene.add_line_segments(
                    "ego_lines",
                    points=box_outlines,
                    colors=colors,
                    line_width=self._config.line_width,
                    visible=True,
                )
                visible_handle_keys.append("lines")

        self._update_imu_frame(iteration)

        for key in self._handles:
            if key not in visible_handle_keys and self._handles[key] is not None:
                self._handles[key].visible = False  # type: ignore

    def remove(self) -> None:
        for handle in self._handles.values():
            if handle is not None:
                handle.remove()
        self._handles = {"mesh": None, "lines": None}
        self._remove_imu_frame()

    def _on_visibility_changed(self, _) -> None:
        assert self._gui_visible is not None, "GUI must be created before handling visibility change."
        self._config.visible = self._gui_visible.value
        if self._gui_visible.value:
            self.update(self._current_iteration)
        else:
            for handle in self._handles.values():
                if handle is not None:
                    handle.visible = False
            self._remove_imu_frame()

    def _on_type_changed(self, _) -> None:
        assert self._gui_type is not None, "GUI must be created before handling type change."
        self._config.type = self._gui_type.value
        self.update(self._current_iteration)

    def _on_opacity_changed(self, _) -> None:
        assert self._gui_opacity is not None, "GUI must be created before handling opacity change."
        self._config.opacity = self._gui_opacity.value
        self.update(self._current_iteration)

    def _on_show_imu_frame_changed(self, _) -> None:
        assert self._gui_show_imu_frame is not None, "GUI must be created before handling IMU frame change."
        self._config.show_imu_frame = self._gui_show_imu_frame.value
        self.update(self._current_iteration)

    def _remove_imu_frame(self) -> None:
        if self._imu_frame_handle is not None:
            self._imu_frame_handle.remove()
            self._imu_frame_handle = None

    def _update_imu_frame(self, iteration: int) -> None:
        assert self._server is not None
        assert self._gui_visible is not None
        assert self._gui_show_imu_frame is not None

        self._remove_imu_frame()

        if not self._gui_visible.value or not self._gui_show_imu_frame.value:
            return

        ego_vehicle_state = self._context.scene.get_ego_state_se3_at_iteration(iteration)
        assert ego_vehicle_state is not None, "Ego vehicle state must be available at the specified iteration."
        imu_pose = ego_vehicle_state.imu_se3.array.copy()
        imu_pose[PoseSE3Index.XYZ] -= self._context.initial_ego_state.center_se3.array[PoseSE3Index.XYZ]
        position = imu_pose[PoseSE3Index.XYZ]
        wxyz = imu_pose[PoseSE3Index.QUATERNION]

        self._imu_frame_handle = self._server.scene.add_frame(
            "ego_imu_frame",
            axes_length=0.5,
            axes_radius=0.01,
            position=position,
            wxyz=wxyz,
        )
