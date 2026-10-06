"""Scene enumeration that stays columnar until a scene is actually asked for.

The eager builder materializes one :class:`SceneMetadata` and one
:class:`ArrowSceneAPI` per candidate frame before any of them is used, which
costs a Python object per tick of the dataset. This module keeps the same
information as a few numpy arrays per log and builds a scene on ``__getitem__``.
"""

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet, List, Optional, Sequence, Set, Tuple, Union, overload

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from py123d.api.scene.arrow.arrow_scene_api import ArrowSceneAPI
from py123d.api.scene.arrow.utils.scene_builder_utils import (
    _resolve_requirement,
    check_log_passes_metadata_filters,
    infer_iteration_duration_s,
    keep_anchors_with_min_remaining_route,
    resolve_iteration_counts,
    resolve_iteration_stride,
    resolve_scene_step_size,
    resolve_scene_uuid_indices,
)
from py123d.api.scene.scene_api import SceneAPI
from py123d.api.scene.scene_filter import AnchorFilterContext, SceneFilter
from py123d.api.utils.arrow_helper import get_lru_cached_arrow_table
from py123d.api.utils.arrow_metadata_utils import get_metadata_from_arrow_schema
from py123d.api.utils.cache_source_utils import check_cache_source_modalities
from py123d.common.utils.uuid_utils import convert_to_str_uuid
from py123d.datatypes.metadata import SceneMetadata
from py123d.datatypes.metadata.log_metadata import LogMetadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LogSceneIndex:
    """The scenes of one log, as columns instead of objects.

    Every scene of a log shares its stride and iteration counts, so only the
    anchor frames differ between them.
    """

    log_dir: Path
    """Directory the scenes are read from."""

    maps_root: Optional[Path]
    """Maps directory the scenes resolve their map under before the global dataset paths."""

    dataset: str
    """Dataset the log belongs to."""

    split: str
    """Split the log belongs to."""

    anchor_indices: np.ndarray
    """Sync-table row of each scene's initial frame, ascending, int64."""

    anchor_timestamps_us: np.ndarray
    """Simulation timestamp of each scene's initial frame in microseconds, int64."""

    num_rows: int
    """Rows in the log's sync table."""

    stride: int
    """Raw sync frames per logical iteration."""

    history_iterations: int
    """History iterations every scene of this log carries."""

    future_iterations: Optional[int]
    """Future iterations every scene carries, or None when scenes run to the log's end."""

    effective_duration_s: float
    """Duration of one logical iteration in seconds, stride included."""

    def num_future_at(self, anchor_index: int) -> int:
        """Future iterations of the scene anchored at a frame.

        Args:
            anchor_index: Sync-table row of the scene's initial frame.

        Returns:
            The scene's future iteration count.
        """
        if self.future_iterations is not None:
            return self.future_iterations
        return max((self.num_rows - anchor_index - 1) // self.stride, 0)

    def scene_metadata_at(self, anchor_index: int) -> SceneMetadata:
        """Build the metadata of the scene anchored at a frame.

        Args:
            anchor_index: Sync-table row of the scene's initial frame.

        Returns:
            The scene metadata, matching what the eager builder produces.
        """
        sync_table = get_lru_cached_arrow_table(str(self.log_dir / "sync.arrow"))
        num_future = self.num_future_at(anchor_index)
        return SceneMetadata(
            dataset=self.dataset,
            split=self.split,
            initial_uuid=convert_to_str_uuid(sync_table["sync.uuid"][anchor_index].as_py()),
            initial_idx=anchor_index,
            num_future_iterations=num_future,
            num_history_iterations=self.history_iterations,
            future_duration_s=num_future * self.effective_duration_s,
            history_duration_s=self.history_iterations * self.effective_duration_s,
            iteration_duration_s=self.effective_duration_s,
            target_iteration_stride=self.stride,
        )


class LazySceneSequence(Sequence[SceneAPI]):
    """A scene list that builds each :class:`SceneAPI` when it is indexed.

    Indexing is the only operation that touches a log, so holding the sequence
    costs a few arrays per log rather than an object per scene.
    """

    def __init__(self, log_indices: List[LogSceneIndex]) -> None:
        """Initialize the sequence over per-log indices.

        :param log_indices: The indexed logs, in the order their scenes appear.
        """
        self._log_indices = [index for index in log_indices if len(index.anchor_indices) > 0]
        counts = [len(index.anchor_indices) for index in self._log_indices]
        # One extra leading zero, so a scene's log is a searchsorted away.
        self._log_starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self._order: Optional[np.ndarray] = None

    def __len__(self) -> int:
        """Inherited, see superclass."""
        if self._order is not None:
            return len(self._order)
        return int(self._log_starts[-1])

    @overload
    def __getitem__(self, index: int) -> SceneAPI: ...

    @overload
    def __getitem__(self, index: slice) -> List[SceneAPI]: ...

    def __getitem__(self, index: Union[int, slice]) -> Union[SceneAPI, List[SceneAPI]]:
        """Inherited, see superclass."""
        if isinstance(index, slice):
            return [self._build(position) for position in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(f"scene index {index} out of range for {len(self)} scenes")
        return self._build(index)

    def _build(self, position: int) -> SceneAPI:
        """Materialize the scene at a position in the sequence.

        :param position: Position in this sequence.
        :return: The scene.
        """
        flat_index = int(self._order[position]) if self._order is not None else position
        log_position = int(np.searchsorted(self._log_starts, flat_index, side="right")) - 1
        log_index = self._log_indices[log_position]
        anchor_index = int(log_index.anchor_indices[flat_index - self._log_starts[log_position]])
        return ArrowSceneAPI(
            log_dir=log_index.log_dir,
            scene_metadata=log_index.scene_metadata_at(anchor_index),
            maps_root=log_index.maps_root,
        )

    def anchor_columns(self) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Identify every scene by its log and initial timestamp, building none of them.

        Pairing two sensor views of the same drive needs exactly this, and the
        index already holds it, so no scene has to be read back. Columns rather
        than pairs, so a whole split can be matched without Python-level work.

        :return: Tuple of (log names, index into those names per scene, initial timestamp
            in microseconds per scene), the last two in sequence order.
        """
        log_names = [log_index.log_dir.name for log_index in self._log_indices]
        if not self._log_indices:
            return log_names, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

        counts = [len(log_index.anchor_indices) for log_index in self._log_indices]
        name_indices = np.repeat(np.arange(len(log_names), dtype=np.int64), counts)
        timestamps = np.concatenate([log_index.anchor_timestamps_us for log_index in self._log_indices])
        if self._order is not None:
            name_indices, timestamps = name_indices[self._order], timestamps[self._order]
        return log_names, name_indices, timestamps

    def anchor_keys(self) -> List[Tuple[str, int]]:
        """Identify every scene by its log and initial timestamp, as pairs.

        :return: One (log name, initial timestamp in microseconds) pair per scene, in sequence order.
        """
        log_names, name_indices, timestamps = self.anchor_columns()
        return [(log_names[int(name_index)], int(timestamp)) for name_index, timestamp in zip(name_indices, timestamps)]

    def reorder(self, order: np.ndarray) -> None:
        """Restrict and reorder the sequence without materializing anything.

        :param order: Positions into the current flat order, in their new order.
        """
        self._order = order if self._order is None else self._order[order]


def _scope_offsets(scope: FrozenSet[str], history_iterations: int, future_iterations: int, stride: int) -> np.ndarray:
    """Frame offsets from a scene's anchor that a modality scope covers.

    :param scope: Temporal segments from :data:`~py123d.api.scene.scene_filter.VALID_MODALITY_SCOPES`.
    :param history_iterations: History iterations of the scene.
    :param future_iterations: Future iterations of the scene.
    :param stride: Raw sync frames per logical iteration.
    :return: Sorted offsets, int64.
    """
    offsets: Set[int] = set()
    if "history" in scope:
        offsets.update(range(-history_iterations * stride, 0, stride))
    if "initial" in scope:
        offsets.add(0)
    if "future" in scope:
        offsets.update(range(stride, future_iterations * stride + 1, stride))
    return np.array(sorted(offsets), dtype=np.int64)


def _null_mask(sync_table: pa.Table, column_name: str, cache: dict) -> np.ndarray:
    """Per-row null mask of a sync column, computed once per log.

    :param sync_table: The sync Arrow table.
    :param column_name: Column to mask.
    :param cache: Per-log store of already-computed masks.
    :return: Boolean array, True where the column is null.
    """
    if column_name not in cache:
        # to_numpy rather than asarray: the column is chunked, and this is flat by contract.
        cache[column_name] = pc.is_null(sync_table.column(column_name)).to_numpy(zero_copy_only=False)
    return cache[column_name]


def _keep_anchors(
    sync_table: pa.Table,
    anchors: np.ndarray,
    filter: SceneFilter,
    history_iterations: int,
    future_iterations: int,
    stride: int,
    log_dir: Path,
) -> np.ndarray:
    """Select the anchors whose scenes satisfy every modality, route, and custom requirement.

    Checks all anchors of a log at once: the scoped frames of a scene are its
    anchor plus a fixed set of offsets, so completeness is one lookup into the
    column's null mask.

    :param sync_table: The sync Arrow table.
    :param anchors: Candidate anchor rows, int64.
    :param filter: The scene filter.
    :param history_iterations: History iterations every candidate carries.
    :param future_iterations: Future iterations every candidate carries.
    :param stride: Raw sync frames per logical iteration.
    :param log_dir: The log directory, handed to the custom anchor filter functions.
    :return: Boolean array over ``anchors``, True for the scenes to keep.
    """
    keep = np.ones(len(anchors), dtype=bool)
    if len(anchors) == 0:
        return keep

    if filter.min_remaining_route_m is not None:
        keep &= keep_anchors_with_min_remaining_route(sync_table, anchors, filter.min_remaining_route_m, log_dir)

    if filter.required_scene_modalities is not None and keep.any():
        sync_column_set = set(sync_table.column_names)
        mask_cache: dict = {}
        for requirement in filter.required_scene_modalities:
            columns, quantifier, scope = _resolve_requirement(requirement, sync_column_set)
            if len(columns) == 0:
                # An "all" requirement over no columns is vacuously satisfied; an
                # "any" requirement over no columns keeps nothing.
                if quantifier != "all":
                    keep[:] = False
                    break
                continue

            offsets = _scope_offsets(scope, history_iterations, future_iterations, stride)
            frames = anchors[:, None] + offsets[None, :]
            complete: Optional[np.ndarray] = None
            for column_name in columns:
                column_complete = ~_null_mask(sync_table, column_name, mask_cache)[frames].any(axis=1)
                if complete is None:
                    complete = column_complete
                elif quantifier == "all":
                    complete &= column_complete
                else:
                    complete |= column_complete
            assert complete is not None
            keep &= complete
            if not keep.any():
                break

    if filter.custom_anchor_filter_fns is not None:
        for filter_fn in filter.custom_anchor_filter_fns:
            if not keep.any():
                break
            context = AnchorFilterContext(
                log_dir=log_dir,
                sync_table=sync_table,
                anchors=anchors[keep],
                history_iterations=history_iterations,
                future_iterations=future_iterations,
                stride=stride,
            )
            result = np.asarray(filter_fn(context), dtype=bool)
            if result.shape != context.anchors.shape:
                raise ValueError(
                    f"custom anchor filter returned shape {result.shape} for {len(context.anchors)} anchors"
                )
            keep[keep] = result
    return keep


def build_log_scene_index(
    log_dir: Path,
    filter: SceneFilter,
    target_uuids_binary: Optional[pa.Array] = None,
    maps_root: Optional[Path] = None,
) -> Optional[LogSceneIndex]:
    """Index one log's scenes without building a scene object.

    :param log_dir: The log directory.
    :param filter: The scene filter.
    :param target_uuids_binary: Pre-converted binary(16) Arrow array of target UUIDs, or None.
    :return: The log's index, or None when the log contributes no scene.
    :raises StaleModalityError: If a source modality changed after a derived modality was written.
    """
    check_cache_source_modalities(log_dir)

    try:
        sync_table = get_lru_cached_arrow_table(str(log_dir / "sync.arrow"))
        log_metadata = get_metadata_from_arrow_schema(sync_table.schema, LogMetadata)
        if not check_log_passes_metadata_filters(log_metadata, sync_table.column_names, filter):
            return None

        raw_duration_s = infer_iteration_duration_s(sync_table)
        stride = resolve_iteration_stride(filter, raw_duration_s)
        if stride is None:
            return None
        future_iterations, history_iterations = resolve_iteration_counts(filter, raw_duration_s, stride)
        step_idx = max(1, resolve_scene_step_size(filter, raw_duration_s, stride))

        uuid_indices: Optional[Set[int]] = None
        if target_uuids_binary is not None:
            uuid_indices = resolve_scene_uuid_indices(sync_table, target_uuids_binary)
            if uuid_indices is None:
                return None

        num_rows = sync_table.num_rows
        initial_idx = history_iterations * stride
        if future_iterations is None:
            # Scenes reach to the end of the log, so their future length differs
            # per anchor and the scoped frames cannot share one offset array.
            candidates = (
                np.array(sorted(index for index in uuid_indices if index >= initial_idx), dtype=np.int64)
                if uuid_indices is not None
                else np.array([initial_idx], dtype=np.int64)
            )
            keep = np.array(
                [
                    _keep_anchors(
                        sync_table,
                        np.array([anchor], dtype=np.int64),
                        filter,
                        history_iterations,
                        max((num_rows - int(anchor) - 1) // stride, 0),
                        stride,
                        log_dir,
                    )[0]
                    for anchor in candidates
                ],
                dtype=bool,
            )
        else:
            end_idx = num_rows - future_iterations * stride
            candidates = (
                np.array(
                    sorted(index for index in uuid_indices if initial_idx <= index < end_idx),
                    dtype=np.int64,
                )
                if uuid_indices is not None
                else np.arange(initial_idx, max(end_idx, initial_idx), step_idx, dtype=np.int64)
            )
            keep = _keep_anchors(sync_table, candidates, filter, history_iterations, future_iterations, stride, log_dir)

        anchors = candidates[keep] if len(candidates) else candidates
        if len(anchors) == 0:
            return None
        timestamps_us = np.asarray(sync_table["sync.timestamp_us"].to_numpy(), dtype=np.int64)
        return LogSceneIndex(
            log_dir=log_dir,
            maps_root=maps_root,
            dataset=log_metadata.dataset,
            split=log_metadata.split,
            anchor_indices=anchors,
            anchor_timestamps_us=timestamps_us[anchors],
            num_rows=num_rows,
            stride=stride,
            history_iterations=history_iterations,
            future_iterations=future_iterations,
            effective_duration_s=raw_duration_s * stride,
        )
    except Exception as error:
        logger.warning("Error indexing scenes from %s: %s", log_dir, error)
        logger.debug("Full traceback for %s:", log_dir, exc_info=True)
        return None


def apply_lazy_post_filters(scenes: LazySceneSequence, filter: SceneFilter) -> Sequence[SceneAPI]:
    """Apply chunking, shuffling and the scene cap without materializing scenes.

    Custom filter functions take a scene, so they force every scene to be built
    and turn the result back into a plain list.

    :param scenes: The lazy sequence to filter.
    :param filter: The scene filter.
    :return: The filtered scenes, lazy unless custom filter functions are set.
    """
    if filter.custom_filter_fns is not None:
        materialized: List[SceneAPI] = list(scenes)
        for filter_fn in filter.custom_filter_fns:
            materialized = [scene for scene in materialized if filter_fn(scene)]
        return _apply_order_filters(materialized, filter)

    order = _apply_order_filters(np.arange(len(scenes), dtype=np.int64), filter)
    scenes.reorder(np.asarray(order, dtype=np.int64))
    return scenes


def _apply_order_filters(items: Sequence, filter: SceneFilter) -> Sequence:
    """Apply chunking, shuffling and the scene cap, in the eager builder's order.

    :param items: Scenes, or positions standing in for them.
    :param filter: The scene filter.
    :return: The kept items.
    """
    if filter.num_chunks is not None and filter.chunk_idx is not None:
        chunk_size = max(1, len(items) // filter.num_chunks)
        start = filter.chunk_idx * chunk_size
        end = start + chunk_size if filter.chunk_idx < filter.num_chunks - 1 else len(items)
        items = items[start:end]

    if filter.shuffle:
        # Shuffling positions rather than scenes draws the permutation the eager
        # builder draws, because the swaps do not depend on the contents.
        positions = list(range(len(items)))
        random.shuffle(positions)
        items = [items[position] for position in positions] if isinstance(items, list) else items[positions]

    if filter.max_num_scenes is not None:
        items = items[: filter.max_num_scenes]
    return items
