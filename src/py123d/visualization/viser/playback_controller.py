import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

import viser

from py123d.api.scene.scene_api import SceneAPI
from py123d.visualization.viser.elements.base_element import ElementContext

logger = logging.getLogger(__name__)


@dataclass
class PlaybackConfig:
    is_playing: bool = False
    speed: float = 1.0
    atomic: bool = False
    dark_mode: bool = False


class PlaybackController:
    """Manages playback state, timestep navigation, and the blocking playback loop."""

    def __init__(
        self,
        server: viser.ViserServer,
        config: PlaybackConfig,
        context: ElementContext,
        on_dark_mode_changed: Optional[Callable[[bool], None]] = None,
    ) -> None:
        self._server = server
        self._config = config
        self._context = context
        self._should_stop: bool = False
        self._rendering: bool = False
        self._on_iteration_changed: Optional[Callable[[int], None]] = None
        self._on_dark_mode_changed = on_dark_mode_changed

        # GUI handles (created in create_gui)
        self._gui_timestep: Optional[viser.GuiSliderHandle] = None
        self._gui_playing: Optional[viser.GuiCheckboxHandle] = None
        self._gui_speed: Optional[viser.GuiSliderHandle] = None
        self._gui_atomic: Optional[viser.GuiCheckboxHandle] = None

    @property
    def current_iteration(self) -> int:
        """Current timestep value."""
        return self._gui_timestep.value if self._gui_timestep is not None else 0

    @property
    def is_rendering(self) -> bool:
        return self._rendering

    @is_rendering.setter
    def is_rendering(self, value: bool) -> None:
        self._rendering = value

    def set_on_iteration_changed(self, callback: Callable[[int], None]) -> None:
        """Set the callback invoked when the timestep changes."""
        self._on_iteration_changed = callback

    def set_timestep(self, value: int) -> None:
        """Programmatically set the timestep (used by render controller)."""
        self._gui_timestep.value = value

    def create_gui(self, scene: SceneAPI) -> None:
        """Create the Playback folder with all controls."""
        num_frames = self._context.num_frames

        with self._server.gui.add_folder("Playback"):
            self._gui_timestep = self._server.gui.add_slider(
                "Timestep", min=0, max=num_frames - 1, step=1, initial_value=0, disabled=True
            )
            gui_next_frame = self._server.gui.add_button("Next Frame", disabled=True)
            gui_prev_frame = self._server.gui.add_button("Prev Frame", disabled=True)
            gui_next_scene = self._server.gui.add_button("Next Scene", disabled=False)
            self._gui_playing = self._server.gui.add_checkbox("Playing", self._config.is_playing)
            self._gui_speed = self._server.gui.add_slider(
                "Playback speed", min=0.1, max=10.0, step=0.1, initial_value=self._config.speed
            )
            gui_speed_options = self._server.gui.add_button_group("Options.", ("0.5", "1.0", "2.0", "5.0", "10.0"))
            self._gui_atomic = self._server.gui.add_checkbox("Atomic Updates", self._config.atomic)
            gui_dark_mode = self._server.gui.add_checkbox("Dark Mode", initial_value=self._config.dark_mode)

            @self._gui_atomic.on_update
            def _on_atomic_changed(_) -> None:
                self._config.atomic = self._gui_atomic.value

            @self._gui_speed.on_update
            def _on_speed_changed(_) -> None:
                self._config.speed = self._gui_speed.value

            @gui_dark_mode.on_update
            def _on_dark_mode_changed(_) -> None:
                self._config.dark_mode = gui_dark_mode.value
                if self._on_dark_mode_changed is not None:
                    self._on_dark_mode_changed(gui_dark_mode.value)

            # Timestep change -> update all elements
            @self._gui_timestep.on_update
            def _on_timestep_changed(_) -> None:
                if self._on_iteration_changed is not None:
                    start = time.perf_counter()
                    if self._gui_atomic.value:
                        with self._server.atomic():
                            self._on_iteration_changed(self._gui_timestep.value)
                    else:
                        self._on_iteration_changed(self._gui_timestep.value)
                    rendering_time = time.perf_counter() - start

                    base_frame_time = scene.scene_metadata.iteration_duration_s
                    target_frame_time = base_frame_time / self._gui_speed.value
                    sleep_time = target_frame_time - rendering_time

                    if sleep_time > 0 and not self._rendering:
                        time.sleep(max(sleep_time, 0.0))

            @gui_next_frame.on_click
            def _on_next_frame(_) -> None:
                self._gui_timestep.value = (self._gui_timestep.value + 1) % num_frames

            @gui_prev_frame.on_click
            def _on_prev_frame(_) -> None:
                self._gui_timestep.value = (self._gui_timestep.value - 1) % num_frames

            @gui_next_scene.on_click
            def _on_next_scene(_) -> None:
                self._should_stop = True

            @self._gui_playing.on_update
            def _on_playing_changed(_) -> None:
                self._gui_timestep.disabled = self._gui_playing.value
                gui_next_frame.disabled = self._gui_playing.value
                gui_prev_frame.disabled = self._gui_playing.value
                self._config.is_playing = self._gui_playing.value

            @gui_speed_options.on_click
            def _on_speed_preset(_) -> None:
                self._gui_speed.value = float(gui_speed_options.value)
                self._config.speed = self._gui_speed.value

    def run_loop(self) -> None:
        """Blocking playback loop. Returns when the user clicks Next Scene."""
        num_frames = self._context.num_frames
        while not self._should_stop:
            if self._gui_playing.value and not self._rendering:
                self._gui_timestep.value = (self._gui_timestep.value + 1) % num_frames
            else:
                time.sleep(0.1)

    def stop(self) -> None:
        """Signal the playback loop to exit."""
        self._should_stop = True


# def _get_scene_info_markdown(scene: SceneAPI) -> str:
#     markdown = f"""
#     - Dataset: {scene.log_metadata.split}
#     - Location: {scene.log_metadata.location if scene.log_metadata.location else "N/A"}
#     - Log: {scene.log_metadata.log_name}
#     - UUID: {scene.scene_uuid}
#     """
#     return markdown
