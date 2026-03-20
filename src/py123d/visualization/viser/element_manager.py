import logging
from typing import List

import viser

from py123d.visualization.viser.elements.base_element import ViewerElement

logger = logging.getLogger(__name__)


class ElementManager:
    """Ordered registry of ViewerElement instances. Creates a tab group where each element gets its own tab."""

    def __init__(self) -> None:
        self._elements: List[ViewerElement] = []

    def register(self, element: ViewerElement) -> None:
        """Register a viewer element."""
        self._elements.append(element)

    def create_all_gui(self, server: viser.ViserServer) -> None:
        """Create a tab group with one tab per registered element."""
        folder = server.gui.add_folder("Modalities", expand_by_default=True)
        with folder:
            tab_group = server.gui.add_tab_group()
            for element in self._elements:
                tab = tab_group.add_tab(element.name)
                with tab:
                    element.create_gui(server)

    def update_all(self, iteration: int) -> None:
        """Update all registered elements for the given iteration."""
        for element in self._elements:
            try:
                element.update(iteration)
            except Exception:
                logger.warning("Failed to update element '%s' at iteration %d", element.name, iteration, exc_info=True)

    def notify_dark_mode_changed(self, dark_mode: bool) -> None:
        """Notify all registered elements that dark mode has changed."""
        for element in self._elements:
            element.on_dark_mode_changed(dark_mode)

    def remove_all(self) -> None:
        """Remove all scene handles from all registered elements."""
        for element in self._elements:
            try:
                element.remove()
            except Exception:
                logger.warning("Failed to remove element '%s'", element.name, exc_info=True)
        self._elements.clear()
