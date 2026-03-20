import abc
import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import viser

from py123d.api.scene.scene_api import SceneAPI
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ElementContext:
    """Immutable per-scene context shared with all viewer elements."""

    scene: SceneAPI
    initial_ego_state: EgoStateSE3
    num_frames: int
    scene_center_array: npt.NDArray[np.float64]
    dark_mode: bool = False

    @staticmethod
    def from_scene(scene: SceneAPI, dark_mode: bool = False) -> "ElementContext":
        """Create an ElementContext from a SceneAPI instance."""
        initial_ego_state = scene.get_ego_state_se3_at_iteration(0)
        if initial_ego_state is None:
            raise ValueError("Scene must have an ego state at iteration 0.")
        scene_center_array = initial_ego_state.center_se3.point_3d.array
        return ElementContext(
            scene=scene,
            initial_ego_state=initial_ego_state,
            num_frames=scene.number_of_iterations,
            scene_center_array=scene_center_array,
            dark_mode=dark_mode,
        )


class ViewerElement(abc.ABC):
    """Self-contained visualization element that owns its handles, GUI controls, and data fetching."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable name used for the GUI folder."""

    @abc.abstractmethod
    def create_gui(self, server: viser.ViserServer) -> None:
        """Create GUI controls inside the element's own folder. Called once per scene load."""

    @abc.abstractmethod
    def update(self, iteration: int) -> None:
        """Fetch data from SceneAPI and update viser scene handles. Called on every timestep change."""

    @abc.abstractmethod
    def remove(self) -> None:
        """Remove all scene handles and clean up. Called before scene switch."""

    def on_dark_mode_changed(self, dark_mode: bool) -> None:
        """Called when the viewer's dark mode setting changes. Override to adapt colors."""
