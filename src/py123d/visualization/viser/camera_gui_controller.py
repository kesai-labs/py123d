import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import viser

from py123d.datatypes.sensors.base_camera import CameraID
from py123d.visualization.viser.elements.base_element import ElementContext

logger = logging.getLogger(__name__)

_IMAGE_SCALE_OPTIONS = ("1", "2", "4", "8")


@dataclass
class CameraGuiConfig:
    visible: bool = False
    image_scale: int = 2
    selected_camera: Optional[CameraID] = None


class CameraGuiController:
    """Displays a single selected camera image in its own GUI folder with a dropdown for camera selection."""

    def __init__(self, server: viser.ViserServer, config: CameraGuiConfig, context: ElementContext) -> None:
        self._server = server
        self._config = config
        self._context = context
        self._image_handle: Optional[viser.GuiImageHandle] = None
        self._folder: Optional[viser.GuiFolderHandle] = None
        self._gui_camera_dropdown: Optional[viser.GuiDropdownHandle] = None
        self._gui_image_scale: Optional[viser.GuiDropdownHandle] = None
        self._gui_visible: Optional[viser.GuiCheckboxHandle] = None
        self._current_iteration: int = 0

        # Build camera ID lookup from available cameras
        metadatas = context.scene.get_camera_metadatas()
        self._camera_ids: Dict[str, CameraID] = {cam_id.serialize(lower=False): cam_id for cam_id in metadatas}
        self._camera_names: List[str] = list(self._camera_ids.keys())

    def create_gui(self) -> None:
        """Create the Camera Image folder with dropdown and image display."""
        if len(self._camera_names) == 0:
            return

        self._folder = self._server.gui.add_folder("Camera Image", expand_by_default=False)
        with self._folder:
            self._gui_visible = self._server.gui.add_checkbox("Visible", self._config.visible)
            self._gui_camera_dropdown = self._server.gui.add_dropdown(
                "Camera",
                self._camera_names,
                initial_value=self._camera_names[0]
                if self._config.selected_camera is None
                else self._config.selected_camera.serialize(lower=False),
            )
            self._gui_image_scale = self._server.gui.add_dropdown(
                "Image Scale",
                _IMAGE_SCALE_OPTIONS,
                initial_value=str(self._config.image_scale),
            )

            @self._gui_visible.on_update
            def _on_visible_changed(_) -> None:
                assert self._gui_visible is not None, "GUI must be created before handling visibility change."
                self._config.visible = self._gui_visible.value
                if self._image_handle is not None:
                    self._image_handle.visible = self._gui_visible.value
                else:
                    self._refresh_image()

            @self._gui_camera_dropdown.on_update
            def _on_camera_changed(_) -> None:
                assert self._gui_camera_dropdown is not None, "GUI must be created before handling camera change."
                # self._image_handle = None
                self._config.selected_camera = self._camera_ids[self._gui_camera_dropdown.value]
                self._refresh_image()

            @self._gui_image_scale.on_update
            def _on_scale_changed(_) -> None:
                assert self._gui_image_scale is not None, "GUI must be created before handling scale change."
                self._config.image_scale = int(self._gui_image_scale.value)
                # self._image_handle = None
                self._refresh_image()

    def update(self, iteration: int) -> None:
        """Update the displayed image for the current iteration."""
        self._current_iteration = iteration
        if self._gui_visible is None or not self._gui_visible.value:
            return
        self._refresh_image()

    def remove(self) -> None:
        """Clean up handles."""
        self._image_handle = None

    def _refresh_image(self) -> None:
        """Fetch and display the image for the currently selected camera."""
        if self._gui_camera_dropdown is None or self._gui_visible is None or self._folder is None:
            return
        if not self._gui_visible.value:
            return

        camera_name = self._gui_camera_dropdown.value
        camera_id = self._camera_ids.get(camera_name)
        if camera_id is None:
            return

        camera = self._context.scene.get_camera_at_iteration(
            self._current_iteration, camera_id, scale=self._config.image_scale
        )
        if camera is None:
            return

        if self._image_handle is not None:
            self._image_handle.image = camera.image
        else:
            with self._folder:
                self._image_handle = self._server.gui.add_image(
                    image=camera.image,
                    label=camera_name,
                )
