import logging
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import trimesh
import trimesh.visual.material
import viser

from py123d.api.scene.scene_api import SceneAPI
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
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
class DetectionConfig:
    visible: bool = True
    type: Literal["mesh", "lines", "mesh+lines"] = "mesh+lines"
    line_width: float = 2.0
    opacity: float = 0.5
    show_center_frames: bool = False


class BoxDetectionsSE3Element(ViewerElement):
    """Visualizes 3D bounding box detections in the scene."""

    def __init__(self, context: ElementContext, config: DetectionConfig) -> None:
        self._context = context
        self._config = config
        self._server: Optional[viser.ViserServer] = None
        self._handles: Dict[str, Optional[Union[viser.GlbHandle, viser.LineSegmentsHandle]]] = {
            "mesh": None,
            "lines": None,
        }
        self._frame_handles: List[viser.FrameHandle] = []
        self._gui_visible: Optional[viser.GuiCheckboxHandle] = None
        self._gui_type: Optional[viser.GuiDropdownHandle] = None
        self._gui_opacity: Optional[viser.GuiSliderHandle] = None
        self._gui_show_center_frames: Optional[viser.GuiCheckboxHandle] = None
        self._current_iteration: int = 0

    @property
    def name(self) -> str:
        return "Box Detections (SE3)"

    def create_gui(self, server: viser.ViserServer) -> None:
        self._server = server
        self._gui_visible = server.gui.add_checkbox("Visible", self._config.visible)
        self._gui_type = server.gui.add_dropdown(
            "Type", ("mesh", "lines", "mesh+lines"), initial_value=self._config.type
        )
        self._gui_opacity = server.gui.add_slider(
            "Opacity", min=0.0, max=1.0, step=0.05, initial_value=self._config.opacity
        )
        self._gui_show_center_frames = server.gui.add_checkbox("Show Center Frames", self._config.show_center_frames)
        self._gui_visible.on_update(self._on_visibility_changed)
        self._gui_type.on_update(self._on_type_changed)
        self._gui_opacity.on_update(self._on_opacity_changed)
        self._gui_show_center_frames.on_update(self._on_show_center_frames_changed)

    def update(self, iteration: int) -> None:
        assert self._server is not None, "Server must be set before updating element."
        assert self._gui_visible is not None, "GUI must be created before updating element."
        assert self._gui_type is not None, "GUI must be created before updating element."
        assert self._gui_opacity is not None, "GUI must be created before updating element"
        # assert self._context.initial_ego_state is not None, "Initial ego state must be set in context."

        self._current_iteration = iteration
        visible_handle_keys: List[str] = []
        display_type = self._gui_type.value

        if self._gui_visible.value:
            if display_type in {"mesh", "mesh+lines"}:
                opacity = self._gui_opacity.value
                mesh = _get_bounding_box_meshes(
                    self._context.scene, iteration, self._context.initial_ego_state, opacity=opacity
                )
                self._handles["mesh"] = self._server.scene.add_mesh_trimesh(
                    "box_detections_mesh",
                    mesh=mesh,
                    visible=True,
                    cast_shadow=False,
                )
                visible_handle_keys.append("mesh")
            if display_type in {"lines", "mesh+lines"}:
                lines, colors, _ = _get_bounding_box_outlines(
                    self._context.scene, iteration, self._context.initial_ego_state
                )
                self._handles["lines"] = self._server.scene.add_line_segments(
                    "box_detections_lines",
                    points=lines,
                    colors=colors,
                    line_width=self._config.line_width,
                    visible=True,
                )
                visible_handle_keys.append("lines")

        self._update_center_frames(iteration)

        for key in self._handles:
            if key not in visible_handle_keys and self._handles[key] is not None:
                self._handles[key].visible = False  # type: ignore

    def remove(self) -> None:
        for handle in self._handles.values():
            if handle is not None:
                handle.remove()
        self._handles = {"mesh": None, "lines": None}
        self._remove_center_frames()

    def _on_visibility_changed(self, _) -> None:
        assert self._gui_visible is not None, "GUI must be created before handling visibility change."
        self._config.visible = self._gui_visible.value
        if self._gui_visible.value:
            self.update(self._current_iteration)
        else:
            for handle in self._handles.values():
                if handle is not None:
                    handle.visible = False
            self._remove_center_frames()

    def _on_type_changed(self, _) -> None:
        assert self._gui_type is not None, "GUI must be created before handling type change."
        self._config.type = self._gui_type.value
        self.update(self._current_iteration)

    def _on_opacity_changed(self, _) -> None:
        assert self._gui_opacity is not None, "GUI must be created before handling opacity change."
        self._config.opacity = self._gui_opacity.value
        self.update(self._current_iteration)

    def _on_show_center_frames_changed(self, _) -> None:
        assert self._gui_show_center_frames is not None, "GUI must be created before handling center frames change."
        self._config.show_center_frames = self._gui_show_center_frames.value
        self.update(self._current_iteration)

    def _remove_center_frames(self) -> None:
        for handle in self._frame_handles:
            handle.remove()
        self._frame_handles.clear()

    def _update_center_frames(self, iteration: int) -> None:
        assert self._server is not None
        assert self._gui_visible is not None
        assert self._gui_show_center_frames is not None

        self._remove_center_frames()

        if not self._gui_visible.value or not self._gui_show_center_frames.value:
            return

        box_detections = self._context.scene.get_box_detections_se3_at_iteration(iteration)
        box_detections_list = box_detections.box_detections if box_detections is not None else []

        for i, bd in enumerate(box_detections_list):
            box_array = bd.bounding_box_se3.array
            position = box_array[BoundingBoxSE3Index.X : BoundingBoxSE3Index.Z + 1].copy()
            position -= self._context.initial_ego_state.center_se3.array[PoseSE3Index.XYZ]
            wxyz = box_array[BoundingBoxSE3Index.QW : BoundingBoxSE3Index.QZ + 1]

            frame_handle = self._server.scene.add_frame(
                f"box_detection_frames/{i}",
                axes_length=0.5,
                axes_radius=0.01,
                position=position,
                wxyz=wxyz,
            )
            self._frame_handles.append(frame_handle)


def _get_bounding_box_meshes(
    scene: SceneAPI, iteration: int, initial_ego_state: EgoStateSE3, opacity: float = 1.0
) -> trimesh.Trimesh:
    box_detections = scene.get_box_detections_se3_at_iteration(iteration)

    if box_detections is None:
        box_detections_list = []
    else:
        box_detections_list = box_detections.box_detections

    boxes = [bd.bounding_box_se3 for bd in box_detections_list]
    boxes_labels = [bd.attributes.default_label for bd in box_detections_list]

    box_se3_array = np.array([box.array for box in boxes])
    box_se3_array[..., BoundingBoxSE3Index.XYZ] -= initial_ego_state.center_se3.array[PoseSE3Index.XYZ]
    box_corners_array = bbse3_array_to_corners_array(box_se3_array)
    box_vertices, box_faces = corners_array_to_3d_mesh(box_corners_array)

    alpha = int(np.clip(opacity * 255, 0, 255))
    box_colors = []
    for box_label in boxes_labels:
        r, g, b, _ = BOX_DETECTION_CONFIG[box_label].fill_color.rgba
        box_colors.append((r, g, b, alpha))

    box_colors = np.array(box_colors)
    vertex_colors = np.repeat(box_colors, len(Corners3DIndex), axis=0)

    mesh = trimesh.Trimesh(vertices=box_vertices, faces=box_faces)
    mesh.visual.vertex_colors = vertex_colors  # type: ignore
    if opacity < 1.0:
        mesh.visual.material = trimesh.visual.material.PBRMaterial(alphaMode="BLEND")  # type: ignore

    return mesh


def _get_bounding_box_outlines(scene: SceneAPI, iteration: int, initial_ego_state: EgoStateSE3) -> tuple:
    box_detections = scene.get_box_detections_se3_at_iteration(iteration)

    box_detections_list = box_detections.box_detections if box_detections is not None else []
    boxes = [bd.bounding_box_se3 for bd in box_detections_list]
    boxes_labels = [bd.attributes.default_label for bd in box_detections_list]

    box_se3_array = np.array([box.array for box in boxes])
    box_se3_array[..., BoundingBoxSE3Index.XYZ] -= initial_ego_state.center_se3.array[PoseSE3Index.XYZ]
    box_corners_array = bbse3_array_to_corners_array(box_se3_array)
    box_outlines = corners_array_to_edge_lines(box_corners_array)

    box_colors = np.zeros(box_outlines.shape, dtype=np.float32)
    for i, box_label in enumerate(boxes_labels):
        box_colors[i, ...] = BOX_DETECTION_CONFIG[box_label].fill_color.rgb_norm

    box_outlines = box_outlines.reshape(-1, *box_outlines.shape[2:])
    box_colors = box_colors.reshape(-1, *box_colors.shape[2:])

    return box_outlines, box_colors, box_se3_array
