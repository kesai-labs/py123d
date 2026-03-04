import logging
import random
import traceback
from functools import partial
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

from py123d.api.scene.arrow.arrow_scene_api import ArrowSceneAPI
from py123d.api.scene.arrow.utils.arrow_metadata_utils import get_metadata_from_arrow_schema
from py123d.api.scene.scene_api import SceneAPI
from py123d.api.scene.scene_builder import SceneBuilder
from py123d.api.scene.scene_filter import SceneFilter
from py123d.api.scene.scene_metadata import SceneMetadata
from py123d.api.utils.arrow_helper import open_arrow_table
from py123d.api.utils.arrow_schema import FISHEYE_MEI, LIDAR, PINHOLE_CAMERA, SYNC
from py123d.common.dataset_paths import get_dataset_paths
from py123d.common.execution import Executor, executor_map_chunked_list
from py123d.common.utils.uuid_utils import convert_to_str_uuid
from py123d.datatypes.metadata.log_metadata import LogMetadata


class ArrowSceneBuilder(SceneBuilder):
    """Class for building scenes from Arrow log directories."""

    def __init__(
        self,
        logs_root: Optional[Union[str, Path]] = None,
        maps_root: Optional[Union[str, Path]] = None,
    ):
        """Initializes the ArrowSceneBuilder.

        :param logs_root: The root directory fo log files, defaults to None
        :param maps_root: The root directory for map files, defaults to None
        """
        if logs_root is None:
            logs_root = get_dataset_paths().py123d_logs_root
        if maps_root is None:
            maps_root = get_dataset_paths().py123d_maps_root

        assert logs_root is not None, "logs_root must be provided or PY123D_DATA_ROOT must be set."
        assert maps_root is not None, "maps_root must be provided or PY123D_DATA_ROOT must be set."
        self._logs_root = Path(logs_root)
        self._maps_root = Path(maps_root)

    def get_scenes(self, filter: SceneFilter, executor: Executor) -> List[SceneAPI]:
        """Inherited, see superclass."""

        split_names = set(filter.split_names) if filter.split_names else _discover_split_names(self._logs_root, filter)
        filter_log_names = set(filter.log_names) if filter.log_names else None
        log_paths = _discover_log_paths(self._logs_root, split_names, filter_log_names)

        if len(log_paths) == 0:
            return []

        scenes = executor_map_chunked_list(executor, partial(_extract_scenes_from_logs, filter=filter), log_paths)
        if filter.shuffle:
            random.shuffle(scenes)

        if filter.max_num_scenes is not None:
            scenes = scenes[: filter.max_num_scenes]
        return scenes


def _discover_split_names(logs_root: Path, filter: SceneFilter) -> List[str]:
    split_types = set(filter.split_types) if filter.split_types else {"train", "val", "test"}
    assert set(split_types).issubset({"train", "val", "test"}), (
        f"Invalid split types: {split_types}. Valid split types are 'train', 'val', 'test'."
    )
    split_names: List[str] = []
    for split in logs_root.iterdir():
        split_name = split.name
        dataset_name = split_name.split("_")[0]

        if filter.datasets is not None and dataset_name not in filter.datasets:
            continue

        if split.is_dir():
            if any(split_type in split_name for split_type in split_types):
                split_names.append(split_name)

    return split_names


def _discover_log_paths(logs_root: Path, split_names: List[str], log_names: Optional[List[str]]) -> List[Path]:
    """Discovers log directory paths in the logs root directory based on the specified split names and log names."""
    log_paths: List[Path] = []
    for split_name in split_names:
        split_dir = logs_root / split_name
        if not split_dir.exists():
            continue
        for log_path in split_dir.iterdir():
            if log_path.is_dir() and (log_path / f"{SYNC.prefix()}.arrow").exists():
                if log_names is None or log_path.name in log_names:
                    log_paths.append(log_path)
    return log_paths


def _extract_scenes_from_logs(log_paths: List[Path], filter: SceneFilter) -> List[SceneAPI]:
    """Extracts scenes from log directories based on the given filter."""
    scenes: List[SceneAPI] = []
    for log_dir in log_paths:
        try:
            scene_extraction_metadatas = _get_scene_extraction_metadatas(log_dir, filter)
        except Exception:
            # logger.warning("Error extracting scenes from %s: %s", log_dir, e)
            # logger.debug("Full traceback for %s:", log_dir, exc_info=True)
            traceback.print_exc()  # noqa: F821
            continue
        for scene_extraction_metadata in scene_extraction_metadatas:
            scenes.append(
                ArrowSceneAPI(
                    log_dir=log_dir,
                    scene_metadata=scene_extraction_metadata,
                )
            )
    return scenes


def _get_scene_extraction_metadatas(log_dir: Union[str, Path], filter: SceneFilter) -> List[SceneMetadata]:
    """Gets the scene metadatas from a log directory based on the given filter.

    TODO: This needs refactoring, clean-up, and tests. It's a mess.
    """

    log_dir = Path(log_dir)
    sync_path = log_dir / f"{SYNC.prefix()}.arrow"

    scene_metadatas: List[SceneMetadata] = []
    sync_table = open_arrow_table(str(sync_path))
    log_metadata = get_metadata_from_arrow_schema(sync_table.schema, LogMetadata)
    num_log_iterations = len(sync_table)

    start_idx = int(filter.history_s / log_metadata.timestep_seconds) if filter.history_s is not None else 0
    end_idx = (
        num_log_iterations - int(filter.duration_s / log_metadata.timestep_seconds)
        if filter.duration_s is not None
        else num_log_iterations
    )

    # 1. Filter location & whether map API is required
    if filter.map_api_required and log_metadata.location is None:
        pass
    elif (
        filter.locations is not None
        and log_metadata.location is not None
        and log_metadata.location not in filter.locations
    ):
        pass

    elif filter.duration_s is None:
        scene_metadatas.append(
            SceneMetadata(
                initial_uuid=convert_to_str_uuid(sync_table[SYNC.col("uuid")][start_idx].as_py()),
                initial_idx=start_idx,
                duration_s=(end_idx - start_idx) * log_metadata.timestep_seconds,
                history_s=filter.history_s if filter.history_s is not None else 0.0,
                iteration_duration_s=log_metadata.timestep_seconds,
            )
        )
    else:
        scene_uuid_set = set(filter.scene_uuids) if filter.scene_uuids is not None else None
        step_idx = int(filter.duration_s / log_metadata.timestep_seconds)
        all_row_uuids = sync_table[SYNC.col("uuid")].to_pylist()
        history_s = filter.history_s if filter.history_s is not None else 0.0

        for idx in range(start_idx, end_idx, step_idx):
            scene_extraction_metadata: Optional[SceneMetadata] = None
            current_uuid = convert_to_str_uuid(all_row_uuids[idx])

            if scene_uuid_set is None:
                scene_extraction_metadata = SceneMetadata(
                    initial_uuid=current_uuid,
                    initial_idx=idx,
                    duration_s=filter.duration_s,
                    history_s=history_s,
                    iteration_duration_s=log_metadata.timestep_seconds,
                )
            elif current_uuid in scene_uuid_set:
                scene_extraction_metadata = SceneMetadata(
                    initial_uuid=current_uuid,
                    initial_idx=idx,
                    duration_s=filter.duration_s,
                    history_s=history_s,
                    iteration_duration_s=log_metadata.timestep_seconds,
                )

            if scene_extraction_metadata is not None:
                # Check of timestamp threshold exceeded between previous scene, if specified in filter
                if filter.timestamp_threshold_s is not None and len(scene_metadatas) > 0:
                    iteration_delta = idx - scene_metadatas[-1].initial_idx
                    if (iteration_delta * log_metadata.timestep_seconds) < filter.timestamp_threshold_s:
                        continue

                scene_metadatas.append(scene_extraction_metadata)

    scene_extraction_metadatas_ = []
    for scene_extraction_metadata in scene_metadatas:
        add_scene = True
        start_idx = scene_extraction_metadata.initial_idx
        if filter.pinhole_camera_ids is not None:
            cam_file = log_dir / f"{PINHOLE_CAMERA.prefix()}.arrow"
            if not cam_file.exists():
                add_scene = False

        if filter.fisheye_mei_camera_ids is not None:
            cam_file = log_dir / f"{FISHEYE_MEI.prefix()}.arrow"
            if not cam_file.exists():
                add_scene = False

        if filter.lidar_ids is not None:
            lidar_file = log_dir / f"{LIDAR.prefix()}.arrow"
            if not lidar_file.exists():
                add_scene = False
        if add_scene:
            scene_extraction_metadatas_.append(scene_extraction_metadata)

    del sync_table
    return scene_extraction_metadatas_
