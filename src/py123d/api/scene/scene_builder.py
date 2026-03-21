import abc
from typing import List

from py123d.api.scene.scene_api import SceneAPI
from py123d.api.scene.scene_filter import SceneFilter
from py123d.common.execution import Executor


class SceneBuilder(abc.ABC):
    """Base class for all scene builders. The scene builder is responsible for building scene given a \
        :class:`~py123d.api.scene.scene_filter.SceneFilter`.
    """

    @abc.abstractmethod
    def get_scenes(self, filter: SceneFilter, executor: Executor) -> List[SceneAPI]:
        """Returns a list of scenes that match the given filter.

        :param filter: SceneFilter object to filter the scenes.
        :param executor: Executor to parallelize the scene extraction.
        :return: List of SceneAPI objects.
        """
