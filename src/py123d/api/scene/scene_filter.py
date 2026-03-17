import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

from py123d.api.scene.scene_api import SceneAPI
from py123d.common.utils.uuid_utils import convert_to_str_uuid

logger = logging.getLogger(__name__)


@dataclass
class SceneFilter:
    """Class to filter scenes when building scenes from logs."""

    # 1. Category: Filter options to resolve log directions (does not require reading).
    # Answers: What logs to read from?

    datasets: Optional[List[str]] = None
    """List of dataset names to filter scenes by."""

    split_types: Optional[List[str]] = None
    """List of split types to filter scenes by (e.g. `train`, `val`, `test`)."""

    split_names: Optional[List[str]] = None
    """List of split names to filter scenes by (in the form `{dataset-name}_{split_type}`)."""

    log_names: Optional[List[str]] = None
    """Name of logs to include scenes from."""

    # 2. Category: Filter options that can be resolved parsing the log / map (without temporal data).
    # Answers: What is available in the log and map?

    # 2.1 Map-related
    has_map: Optional[bool] = None
    """Filter scenes by map availability. True: only with map, False: only without map, None: no filter."""

    map_has_z: Optional[bool] = None
    """Filter scenes with available map elevation. True: only with 3D maps. False: only 2D maps. None (default): no filter."""

    map_locations: Optional[List[str]] = None
    """List of locations as stored in the map metadata to filter scenes by."""

    map_version: Optional[str] = None
    """Version of the map metadata to filter by. If None (default), does not filter by version."""

    # 2.1 Log-related
    log_locations: Optional[List[str]] = None
    """List of locations as stored in the log metadata to filter scenes by."""

    log_version: Optional[str] = None
    """Version of the log metadata to filter by. If None (default), does not filter by version."""

    required_log_modalities: Optional[List[str]] = None
    """List of modality ids or keys that must be present in the log for a scene to be included."""

    # 3. Category: Filter options that require reading scene data.
    # Answers: How to sample scenes from the log?

    scene_uuids: Optional[List[str]] = None
    """List of scene UUIDs to include."""

    future_duration_s: Optional[float] = None
    """Duration of each scene in seconds."""

    history_duration_s: Optional[float] = None
    """History duration of each scene in seconds."""

    timestamp_threshold_s: Optional[float] = None
    """Minimum time between the start timestamps of two consecutive scenes in seconds."""

    # TODO: Implement in the future.
    # iteration_stride: int = 1
    # """Redefines the unit of one iteration by skipping every N raw log frames.
    # A stride of 1 (default) uses the native log frequency.
    # A stride of 5 on a 10 Hz log yields an effective 2 Hz iteration rate.
    # Only affects fields expressed in iterations (future_num_iterations,
    # history_num_iterations, iteration_threshold); duration-based fields
    # (future_duration_s etc.) are unaffected."""

    future_num_iterations: Optional[int] = None
    """Number of iterations in the future for each scene, ignored if future_duration_s is provided."""

    history_num_iterations: Optional[int] = None
    """Number of iterations in the history for each scene, ignored if history_duration_s is provided."""

    iteration_threshold: Optional[float] = None
    """Minimum number of iterations between two consecutive scenes, ignored if timestamp_threshold_s is provided."""

    required_scene_modalities: Optional[List[str]] = None
    """List of modality ids or keys that must be present in a scene for it to be included."""

    # 4. Category: Post-filtering options (applied after scenes are filtered by the above criteria, e.g. for sampling or shuffling).
    # Answers: What to do with the scenes after filtering?

    # 4.1 Custom filter functions
    # NOTE @DanielDauner: Not compatible with Hydra override.
    custom_filter_fns: Optional[List[Callable[[SceneAPI], bool]]] = None
    """List of custom filter functions that take a SceneAPI object and return True if the scene should be included."""

    # 4.2. Chunking
    num_chunks: Optional[int] = None
    """Number of chunks to split the returned scenes into. If None (default), does not split into chunks."""

    chunk_idx: Optional[int] = None
    """Index of the chunk to return (0-indexed). Only used if num_chunks is not None."""

    # 4.3. Other post-filtering options
    max_num_scenes: Optional[int] = None
    """Maximum number of scenes to return."""

    shuffle: bool = False
    """Whether to shuffle the returned scenes."""

    def __post_init__(self):
        """Post-initialization to validate filter options and resolve any conflicts."""

        def _deduplicate_optional(values: Optional[List]) -> Optional[List]:
            _deduplicate_list: Optional[List] = None
            if values is not None:
                _deduplicate_list = list(dict.fromkeys(values))  # preserves order, removes duplicates
            return _deduplicate_list

        # 1. Category
        self.datasets = _deduplicate_optional(self.datasets)
        self.split_types = _deduplicate_optional(self.split_types)
        self.split_names = _deduplicate_optional(self.split_names)
        self.log_names = _deduplicate_optional(self.log_names)

        # 2. Category
        if self.map_has_z is not None and self.has_map is False:
            raise ValueError(
                "Cannot filter by map elevation (map_has_z) if filtering for scenes without maps (has_map=False)."
            )
        self.map_locations = _deduplicate_optional(self.map_locations)
        self.log_locations = _deduplicate_optional(self.log_locations)

        # 3. Category
        if self.scene_uuids is not None:
            self.scene_uuids = [convert_to_str_uuid(s) for s in self.scene_uuids]
        self.scene_uuids = _deduplicate_optional(self.scene_uuids)
        self.required_scene_modalities = _deduplicate_optional(self.required_scene_modalities)

        if self.future_duration_s is not None and self.future_num_iterations is not None:
            logger.warning("Both future_duration_s and future_num_iterations set; future_duration_s takes priority.")
        if self.history_duration_s is not None and self.history_num_iterations is not None:
            logger.warning("Both history_duration_s and history_num_iterations set; history_duration_s takes priority.")
        if self.timestamp_threshold_s is not None and self.iteration_threshold is not None:
            logger.warning(
                "Both timestamp_threshold_s and iteration_threshold set; timestamp_threshold_s takes priority."
            )

        # Passing any required log modalities upward to category 2, for earlier log-level filtering.
        _required_log_modalities = self.required_log_modalities or []
        if self.required_scene_modalities is not None:
            _required_log_modalities += self.required_scene_modalities
        self.required_log_modalities = _deduplicate_optional(_required_log_modalities)

        # 4. Category
        if (
            self.num_chunks is not None
            and self.chunk_idx is None
            or self.num_chunks is None
            and self.chunk_idx is not None
        ):
            raise ValueError("Both num_chunks and chunk_idx must be set together to enable chunking.")

        if self.num_chunks is not None and self.chunk_idx is not None:
            if self.chunk_idx >= self.num_chunks:
                raise ValueError(f"chunk_idx ({self.chunk_idx}) must be < num_chunks ({self.num_chunks}).")
