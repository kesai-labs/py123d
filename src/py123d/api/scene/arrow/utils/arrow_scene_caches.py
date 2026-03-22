from pathlib import Path
from typing import Final, Union

from py123d.api.scene.arrow.utils.scene_builder_utils import infer_iteration_duration_s
from py123d.api.utils.arrow_helper import get_lru_cached_arrow_table
from py123d.common.utils.uuid_utils import convert_to_str_uuid
from py123d.datatypes import LogMetadata
from py123d.datatypes.metadata import SceneMetadata

MAX_LRU_CACHED_LOG_METADATA: Final[int] = 1_000


def _get_complete_log_scene_metadata(log_dir: Union[Path, str], log_metadata: LogMetadata) -> SceneMetadata:
    """Helper function to get the scene metadata for a complete log from a log directory."""
    sync_path = Path(log_dir) / "sync.arrow"
    table = get_lru_cached_arrow_table(sync_path)
    initial_uuid = convert_to_str_uuid(table["sync.uuid"][0].as_py())
    num_rows = table.num_rows

    # When < 2 rows, num_future_iterations == 0, so duration is unused; default to 0.0
    iteration_duration_s = infer_iteration_duration_s(table) if num_rows >= 2 else 0.0
    num_future_iterations = max(num_rows - 1, 0)

    return SceneMetadata(
        dataset=log_metadata.dataset,
        split=log_metadata.split,
        initial_uuid=initial_uuid,
        initial_idx=0,
        num_future_iterations=num_future_iterations,
        num_history_iterations=0,
        future_duration_s=num_future_iterations * iteration_duration_s,
        history_duration_s=0.0,
        iteration_duration_s=iteration_duration_s,
        target_iteration_stride=1,
    )
