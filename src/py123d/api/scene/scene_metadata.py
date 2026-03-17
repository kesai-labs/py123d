import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SceneMetadata:
    """Metadata for a scene extracted from a log."""

    dataset: str
    """Name of the dataset the scene belongs to."""

    split: str
    """Name of the split the scene belongs to."""

    initial_uuid: str
    """UUID of the scene, i.e., the UUID of the starting frame of the scene."""

    initial_idx: int
    """Index of the starting frame of the scene in the log."""

    num_future_iterations: int
    """Number of future iterations in the scene (excludes the current frame)."""

    num_history_iterations: int
    """Number of history iterations in the scene (excludes the current frame)."""

    future_duration_s: float
    """Forward reach of the scene from the current frame in seconds."""

    history_duration_s: float
    """Backward reach of history before the current frame in seconds."""

    iteration_duration_s: float
    """Approximate duration of each iteration in seconds (inferred from sync table timestamps).
    When target_iteration_stride > 1, this reflects the effective strided duration (raw_duration * stride)."""

    target_iteration_stride: int = 1
    """Number of raw sync-table frames per logical iteration. 1 (default) uses the native log frequency."""

    @property
    def end_idx(self) -> int:
        """Index of the end frame of the scene (exclusive), accounting for stride."""
        return self.initial_idx + self.num_future_iterations * self.target_iteration_stride + 1

    @property
    def total_iterations(self) -> int:
        """Total number of iterations including history, current frame, and future."""
        return self.num_history_iterations + 1 + self.num_future_iterations

    def __post_init__(self):
        """Validate consistency between iteration counts and durations."""
        if self.iteration_duration_s > 0:
            expected_future = round(self.future_duration_s / self.iteration_duration_s)
            expected_history = round(self.history_duration_s / self.iteration_duration_s)
            if expected_future != self.num_future_iterations:
                logger.debug(
                    "SceneMetadata: num_future_iterations=%d != round(future_duration_s/iteration_duration_s)=%d",
                    self.num_future_iterations,
                    expected_future,
                )
            if expected_history != self.num_history_iterations:
                logger.debug(
                    "SceneMetadata: num_history_iterations=%d != round(history_duration_s/iteration_duration_s)=%d",
                    self.num_history_iterations,
                    expected_history,
                )

    def __repr__(self) -> str:
        stride_str = (
            f", target_iteration_stride={self.target_iteration_stride}" if self.target_iteration_stride != 1 else ""
        )
        return (
            f"SceneMetadata(dataset={self.dataset}, split={self.split}, "
            f"initial_uuid={self.initial_uuid}, initial_idx={self.initial_idx}, "
            f"num_future_iterations={self.num_future_iterations}, num_history_iterations={self.num_history_iterations}, "
            f"future_duration_s={self.future_duration_s}, history_duration_s={self.history_duration_s}, "
            f"iteration_duration_s={self.iteration_duration_s}{stride_str})"
        )
