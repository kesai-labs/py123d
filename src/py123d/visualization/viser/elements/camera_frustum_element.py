import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import viser

from py123d.datatypes import CameraID, FisheyeMEICameraMetadata, PinholeCameraMetadata
from py123d.datatypes.sensors.base_camera import ALL_FISHEYE_MEI_CAMERA_IDS, ALL_PINHOLE_CAMERA_IDS
from py123d.visualization.viser.elements.base_element import ElementContext, ViewerElement
from py123d.visualization.viser.utils.view_utils import decompose_camera_pose, get_scene_center_pose

logger = logging.getLogger(__name__)


@dataclass
class CameraFrustumConfig:
    visible: bool = True
    frustum_scale: float = 1.0
    image_scale: int = 4
    fisheye_fov: float = 185.0
    show_frames: bool = False
    visible_camera_ids: List[CameraID] = field(
        default_factory=lambda: ALL_PINHOLE_CAMERA_IDS + ALL_FISHEYE_MEI_CAMERA_IDS
    )


class CameraFrustumElement(ViewerElement):
    """Visualizes camera frustums (pinhole and fisheye) in the 3D scene."""

    def __init__(self, context: ElementContext, config: CameraFrustumConfig) -> None:
        self._context = context
        self._config = config
        self._dark_mode: bool = context.dark_mode
        self._server: Optional[viser.ViserServer] = None
        self._frustum_handles: Dict[CameraID, viser.CameraFrustumHandle] = {}
        self._frame_handles: Dict[CameraID, viser.FrameHandle] = {}
        self._gui_checkbox_handle: Optional[viser.GuiCheckboxHandle] = None
        self._gui_frustum_scale_handle: Optional[viser.GuiInputHandle] = None
        self._gui_image_scale_handle: Optional[viser.GuiDropdownHandle] = None
        self._gui_show_frames_handle: Optional[viser.GuiCheckboxHandle] = None
        self._gui_camera_checkboxes: Dict[CameraID, viser.GuiCheckboxHandle] = {}
        self._current_iteration: int = 0

    @property
    def name(self) -> str:
        return "Cameras"

    def create_gui(self, server: viser.ViserServer) -> None:
        self._server = server
        self._gui_checkbox_handle = server.gui.add_checkbox("Visible", self._config.visible)
        self._gui_frustum_scale_handle = server.gui.add_slider(
            "Frustum Scale",
            min=0.1,
            max=5.0,
            step=0.1,
            initial_value=self._config.frustum_scale,
        )
        self._gui_image_scale_handle = server.gui.add_dropdown(
            "Image Scale",
            (
                "1",
                "2",
                "4",
                "8",
            ),
            initial_value=str(self._config.image_scale),
        )
        self._gui_show_frames_handle = server.gui.add_checkbox("Show Frames", self._config.show_frames)
        self._gui_checkbox_handle.on_update(self._on_visibility_changed)
        self._gui_frustum_scale_handle.on_update(self._on_frustum_scale_changed)
        self._gui_image_scale_handle.on_update(self._on_image_scale_changed)
        self._gui_show_frames_handle.on_update(self._on_show_frames_changed)

        available_ids = set(self._context.scene.available_camera_ids)
        server.gui.add_markdown("**Cameras**")
        for camera_id in available_ids:
            label = camera_id.serialize(lower=False)
            initially_visible = camera_id in self._config.visible_camera_ids
            cb = server.gui.add_checkbox(label, initially_visible)
            cb.on_update(self._on_camera_visibility_changed)
            self._gui_camera_checkboxes[camera_id] = cb

    def update(self, iteration: int) -> None:
        assert self._server is not None
        assert self._gui_checkbox_handle is not None

        self._current_iteration = iteration
        if not self._gui_checkbox_handle.value:
            return

        scene_center_pose = get_scene_center_pose(self._context.scene_center_array)

        def _update_frustum(camera_type: CameraID) -> None:
            assert self._server is not None
            assert self._server.scene is not None

            camera_cb = self._gui_camera_checkboxes.get(camera_type)
            if camera_cb is not None and not camera_cb.value:
                return

            camera = self._context.scene.get_camera_at_iteration(iteration, camera_type, scale=self._config.image_scale)
            if camera is None:
                return

            camera_position, camera_quaternion = decompose_camera_pose(camera, scene_center_pose)

            # Determine FOV based on camera model
            if isinstance(camera.metadata, PinholeCameraMetadata):
                fov = camera.metadata.fov_y
            elif isinstance(camera.metadata, FisheyeMEICameraMetadata):
                fov = self._config.fisheye_fov
            else:
                raise ValueError(f"Unsupported camera metadata type: {type(camera.metadata)}")

            if camera_type in self._frustum_handles:
                self._frustum_handles[camera_type].position = camera_position
                self._frustum_handles[camera_type].wxyz = camera_quaternion
                self._frustum_handles[camera_type].image = camera.image
            else:
                self._frustum_handles[camera_type] = self._server.scene.add_camera_frustum(
                    f"camera_frustums/{camera_type.serialize()}",
                    fov=fov,  # type: ignore
                    aspect=camera.metadata.aspect_ratio,
                    scale=self._config.frustum_scale,
                    image=camera.image,
                    position=camera_position,
                    cast_shadow=False,
                    receive_shadow=False,
                    wxyz=camera_quaternion,
                )

            show_frames = self._gui_show_frames_handle is not None and self._gui_show_frames_handle.value
            if show_frames:
                if camera_type in self._frame_handles:
                    self._frame_handles[camera_type].position = camera_position
                    self._frame_handles[camera_type].wxyz = camera_quaternion
                else:
                    self._frame_handles[camera_type] = self._server.scene.add_frame(
                        f"camera_frames/{camera_type.serialize()}",
                        axes_length=0.5,
                        axes_radius=0.01,
                        position=camera_position,
                        wxyz=camera_quaternion,
                    )

        for camera_id, cb in self._gui_camera_checkboxes.items():
            if cb.value:
                _update_frustum(camera_id)

    def remove(self) -> None:
        for handle in self._frustum_handles.values():
            handle.remove()
        self._frustum_handles.clear()
        self._remove_frames()

    def _remove_frames(self) -> None:
        for handle in self._frame_handles.values():
            handle.remove()
        self._frame_handles.clear()

    def _sync_config_visible_camera_ids(self) -> None:
        """Update config.visible_camera_ids to match current GUI checkbox state."""
        self._config.visible_camera_ids = [
            camera_id for camera_id, cb in self._gui_camera_checkboxes.items() if cb.value
        ]

    def _sync_visibility(self) -> None:
        assert self._gui_checkbox_handle is not None
        master = self._gui_checkbox_handle.value
        for camera_id, handle in self._frustum_handles.items():
            camera_cb = self._gui_camera_checkboxes.get(camera_id)
            handle.visible = master and (camera_cb is None or camera_cb.value)

    def _on_visibility_changed(self, _) -> None:
        assert self._gui_checkbox_handle is not None
        self._config.visible = self._gui_checkbox_handle.value
        self._sync_visibility()

    def _on_camera_visibility_changed(self, _) -> None:
        self._sync_config_visible_camera_ids()
        self._sync_visibility()

    def _on_frustum_scale_changed(self, _) -> None:
        assert self._gui_frustum_scale_handle is not None
        self._config.frustum_scale = self._gui_frustum_scale_handle.value
        self.remove()
        self.update(self._current_iteration)

    def _on_show_frames_changed(self, _) -> None:
        assert self._gui_show_frames_handle is not None
        self._config.show_frames = self._gui_show_frames_handle.value
        if self._config.show_frames:
            self.update(self._current_iteration)
        else:
            self._remove_frames()

    def _on_image_scale_changed(self, _) -> None:
        assert self._gui_image_scale_handle is not None
        self._config.image_scale = int(self._gui_image_scale_handle.value)
        self.remove()
        self.update(self._current_iteration)
