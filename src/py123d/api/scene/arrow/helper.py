from pathlib import Path
from typing import List, Optional, Union

from py123d.api.scene.arrow.arrow_scene_builder import ArrowSceneBuilder
from py123d.api.scene.scene_api import SceneAPI
from py123d.api.scene.scene_filter import SceneFilter
from py123d.common.execution import Executor, ThreadPoolExecutor


def get_filtered_scenes(
    scene_filter: SceneFilter,
    data_root: Optional[Union[str, Path]] = None,
    executor: Executor = ThreadPoolExecutor(),
) -> List[SceneAPI]:
    """Retrieve a list of scenes that match the given filter criteria.

    :param scene_filter: Filter class describing criteria for scene selection.
    :param data_root: Root directory for py123d data, defaults to None
    :param executor: Executor for parallel execution, defaults to ThreadPoolExecutor()
    :return: List of scenes matching the filter criteria
    """

    if data_root is not None:
        data_root = Path(data_root)

    scenes = ArrowSceneBuilder(
        logs_root=data_root / "logs" if data_root is not None else None,
        maps_root=data_root / "maps" if data_root is not None else None,
    ).get_scenes(filter=scene_filter, executor=executor)

    return scenes
