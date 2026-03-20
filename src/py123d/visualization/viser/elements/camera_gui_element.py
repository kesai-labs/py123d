import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import viser

from py123d.common.utils.enums import resolve_enum_arguments
from py123d.datatypes.sensors.base_camera import CameraID
from py123d.visualization.viser.elements.base_element import ElementContext, ViewerElement

logger = logging.getLogger(__name__)


@dataclass
class CameraGuiConfig:
    visible: bool = True
    types: List[CameraID] = field(default_factory=lambda: [CameraID.PCAM_F0])
    image_scale: int = 4

    def __post_init__(self):
        self.types = resolve_enum_arguments(CameraID, self.types)  # type: ignore
        self.image_scale = int(self.image_scale)


_IMAGE_SCALE_OPTIONS = ("1", "2", "4", "8")


class CameraGuiElement(ViewerElement):
    """Displays camera images as embedded GUI panels."""

    def __init__(self, context: ElementContext, config: CameraGuiConfig) -> None:
        self._context = context
        self._config = config
        self._server: Optional[viser.ViserServer] = None
        self._handles: Dict[CameraID, viser.GuiImageHandle] = {}
        self._gui_visible: Optional[viser.GuiCheckboxHandle] = None
        self._gui_image_scale: Optional[viser.GuiDropdownHandle] = None
        self._gui_camera_checkboxes: Dict[CameraID, viser.GuiCheckboxHandle] = {}
        self._current_iteration: int = 0

    @property
    def name(self) -> str:
        return "Camera Images"

    def create_gui(self, server: viser.ViserServer) -> None:
        self._server = server
        self._gui_visible = server.gui.add_checkbox("Visible", self._config.visible)
        self._gui_image_scale = server.gui.add_dropdown(
            "Image Scale",
            _IMAGE_SCALE_OPTIONS,
            initial_value=str(self._config.image_scale),
        )

        self._gui_visible.on_update(self._on_visibility_changed)
        self._gui_image_scale.on_update(self._on_image_scale_changed)

        available_ids = set(self._context.scene.available_camera_ids)
        server.gui.add_markdown("**Cameras**")
        for camera_id in self._config.types:
            if camera_id not in available_ids:
                continue
            label = camera_id.serialize(lower=False)
            cb = server.gui.add_checkbox(label, True)
            cb.on_update(self._on_camera_visibility_changed)
            self._gui_camera_checkboxes[camera_id] = cb

    def update(self, iteration: int) -> None:
        assert self._server is not None, "Server must be set before updating element."
        assert self._gui_visible is not None, "GUI must be created before updating element."
        self._current_iteration = iteration
        if not self._gui_visible.value:
            return

        for camera_type in self._config.types:
            camera_cb = self._gui_camera_checkboxes.get(camera_type)
            if camera_cb is not None and not camera_cb.value:
                continue

            camera = self._context.scene.get_camera_at_iteration(iteration, camera_type, scale=self._config.image_scale)
            if camera is None:
                continue

            if camera_type in self._handles:
                self._handles[camera_type].image = camera.image
            else:
                with self._server.gui.add_folder(f"Camera {camera_type.serialize()}"):
                    self._handles[camera_type] = self._server.gui.add_image(
                        image=camera.image, label=camera_type.serialize()
                    )

    def remove(self) -> None:
        self._handles.clear()

    def _sync_visibility(self) -> None:
        assert self._gui_visible is not None, "GUI must be created before syncing visibility."
        master = self._gui_visible.value
        for camera_id, handle in self._handles.items():
            camera_cb = self._gui_camera_checkboxes.get(camera_id)
            handle.visible = master and (camera_cb is None or camera_cb.value)

    def _on_visibility_changed(self, _) -> None:
        self._sync_visibility()

    def _on_camera_visibility_changed(self, _) -> None:
        self._sync_visibility()

    def _on_image_scale_changed(self, _) -> None:
        assert self._gui_image_scale is not None, "GUI must be created before handling image scale change."
        self._config.image_scale = int(self._gui_image_scale.value)
        # Clear existing handles so they get recreated at new scale
        self._handles.clear()
        self.update(self._current_iteration)
