import logging
from typing import List, Literal, Tuple

import viser
from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage

from py123d.api.scene.scene_api import SceneAPI
from py123d.visualization.viser.camera_gui_controller import CameraGuiController
from py123d.visualization.viser.element_manager import ElementManager
from py123d.visualization.viser.elements.base_element import ElementContext
from py123d.visualization.viser.elements.box_detections_se3_element import BoxDetectionsSE3Element
from py123d.visualization.viser.elements.camera_frustum_element import CameraFrustumElement
from py123d.visualization.viser.elements.ego_state_se3_element import EgoElement
from py123d.visualization.viser.elements.lidar_element import LidarElement
from py123d.visualization.viser.elements.map_element import MapElement
from py123d.visualization.viser.playback_controller import PlaybackController
from py123d.visualization.viser.render_controller import RenderController
from py123d.visualization.viser.viser_config import ViserConfig

logger = logging.getLogger(__name__)

HDRI: Literal[
    "apartment",
    "city",
    "dawn",
    "forest",
    "lobby",
    "night",
    "park",
    "studio",
    "sunset",
    "warehouse",
] = "warehouse"


def _build_titlebar() -> TitlebarConfig:
    buttons = (
        TitlebarButton(
            text="Getting Started",
            icon=None,
            href="https://autonomousvision.github.io/py123d",
        ),
        TitlebarButton(
            text="Github",
            icon="GitHub",
            href="https://github.com/autonomousvision/py123d",
        ),
        TitlebarButton(
            text="Documentation",
            icon="Description",
            href="https://autonomousvision.github.io/py123d",
        ),
    )
    image = TitlebarImage(
        image_url_light="https://autonomousvision.github.io/py123d/_static/123D_logo_transparent_black.svg",
        image_url_dark="https://autonomousvision.github.io/py123d/_static/123D_logo_transparent_white.svg",
        image_alt="123D",
        href="https://autonomousvision.github.io/py123d/",
    )
    return TitlebarConfig(buttons=buttons, image=image)


def _build_viser_server(config: ViserConfig) -> Tuple[viser.ViserServer, TitlebarConfig]:
    server = viser.ViserServer(
        host=config.server.host,
        port=config.server.port,
        label=config.server.label,
        verbose=config.server.verbose,
    )

    titlebar_theme = _build_titlebar()

    server.gui.configure_theme(
        titlebar_content=titlebar_theme,
        control_layout=config.theme.control_layout,
        control_width=config.theme.control_width,
        dark_mode=config.theme.dark_mode,
        show_logo=config.theme.show_logo,
        show_share_button=config.theme.show_share_button,
        brand_color=config.theme.brand_color,
    )

    server.scene.configure_environment_map(
        hdri=HDRI,
        environment_intensity=0.75,  # down from default 1.0
    )
    return server, titlebar_theme


class ViserViewer:
    """Orchestrates the viser 3D viewer: wires elements, playback, and rendering together."""

    def __init__(
        self,
        scenes: List[SceneAPI],
        viser_config: ViserConfig = ViserConfig(),
        scene_index: int = 0,
    ) -> None:
        if len(scenes) == 0:
            raise ValueError("At least one scene must be provided.")

        self._scenes = scenes
        self._config = viser_config
        self._scene_index = scene_index
        self._server, self._titlebar = _build_viser_server(self._config)
        self._dark_mode = self._config.theme.dark_mode
        self._environment_intensity = 0.25
        self._run_scene(self._scenes[self._scene_index % len(self._scenes)])

    def _run_scene(self, scene: SceneAPI) -> None:
        """Set up and run the viewer for a single scene. Blocks until scene switch."""
        context = ElementContext.from_scene(scene, dark_mode=self._dark_mode)

        # Build elements based on available data
        self._element_manager = self._build_elements(context)

        # Build controllers
        playback = PlaybackController(
            self._server,
            self._config.playback,
            context,
            on_dark_mode_changed=self._on_dark_mode_changed,
        )
        render = RenderController(self._server, self._config.render, context, playback)

        # Build camera GUI controller
        self._camera_gui = CameraGuiController(self._server, self._config.camera_gui, context)

        # Create GUI in order: Playback -> Modality Tabs -> Camera Image -> Render
        playback.create_gui(scene)
        self._element_manager.create_all_gui(self._server)
        self._camera_gui.create_gui()
        render.create_gui()

        # Re-apply persisted environment intensity (scene.reset() clears it)
        self._server.scene.configure_environment_map(
            hdri=HDRI,
            environment_intensity=self._environment_intensity,
        )

        # Wire iteration callback
        def _on_iteration_changed(iteration: int) -> None:
            self._element_manager.update_all(iteration)
            self._camera_gui.update(iteration)

        playback.set_on_iteration_changed(_on_iteration_changed)

        # Initial render at frame 0
        _on_iteration_changed(0)

        # Blocking playback loop -- returns on Next Scene
        playback.run_loop()

        # Cleanup and advance to next scene
        self._camera_gui.remove()
        self._element_manager.remove_all()
        self._server.flush()
        self._server.gui.reset()
        self._server.scene.reset()
        self._scene_index = (self._scene_index + 1) % len(self._scenes)
        self._run_scene(self._scenes[self._scene_index])

    def _on_dark_mode_changed(self, dark_mode: bool) -> None:
        """Handle dark mode toggle from playback controller."""
        theme = self._config.theme
        self._dark_mode = dark_mode
        self._server.gui.configure_theme(
            titlebar_content=self._titlebar,
            control_layout=theme.control_layout,
            control_width=theme.control_width,
            dark_mode=dark_mode,
            show_logo=theme.show_logo,
            show_share_button=theme.show_share_button,
            brand_color=theme.brand_color,
        )
        self._element_manager.notify_dark_mode_changed(dark_mode)

    def _build_elements(self, context: ElementContext) -> ElementManager:
        """Conditionally register elements based on what the scene supports."""
        manager = ElementManager()
        scene = context.scene

        if len(scene.get_lidar_metadatas()) > 0:
            manager.register(LidarElement(context, self._config.lidar))

        if scene.get_box_detections_se3_metadata() is not None:
            manager.register(BoxDetectionsSE3Element(context, self._config.detection))

        if len(scene.get_camera_metadatas()) > 0:
            manager.register(CameraFrustumElement(context, self._config.camera_frustum))

        if scene.get_ego_state_se3_metadata() is not None:
            manager.register(EgoElement(context, self._config.ego))

        if scene.get_map_api() is not None:
            manager.register(MapElement(context, self._config.map))

        return manager
