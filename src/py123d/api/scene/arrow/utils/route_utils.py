"""Route computation, ``route_position.arrow`` I/O, and polyline interpolation.

A log's route is one polyline resampled at a fixed arc-length resolution — the driven
path derived from ego odometry, or an explicitly provided (planned) route. It is stored
entirely in the regular ``route_position`` modality: the static polyline in the modality
metadata, the dynamic per-frame arc positions in the rows.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from py123d.api.utils.arrow_metadata_utils import add_metadata_to_arrow_schema, get_metadata_from_arrow_schema
from py123d.datatypes.metadata.cache_source import CacheSourceInfo
from py123d.datatypes.metadata.route_metadata import RouteMetadata
from py123d.datatypes.modalities.base_modality import ModalityType

logger = logging.getLogger(__name__)

ROUTE_POSITION_KEY = ModalityType.ROUTE_POSITION.serialize()
"""Modality key of the per-frame route positions (``route_position.arrow``); also the
sync-table column holding each frame's row index into that file."""

ROUTE_PRODUCER = "py123d:route_from_ego_state_se3@1"
"""``computed_by`` recorded in the route's ``CacheSourceInfo``. Bump the version suffix
whenever the route computation changes in a way that makes existing routes wrong."""

_MIN_SEGMENT_M = 0.05
"""Positions closer than this to the previously kept one are pruned, so standstill
odometry jitter accumulates no arc-length."""

_PROJECTION_WINDOW_M = 50.0
"""Forward search window when projecting ego positions onto a provided route; keeps the
projection monotone even where the route crosses itself."""

_MAX_PLAUSIBLE_SPEED_MPS = 100.0
"""Implied speeds above this (360 km/h) are treated as localization jumps."""


@dataclass(frozen=True)
class RouteData:
    """A log's route polyline and the per-frame positions on it.

    ``polyline_arc_m`` (K,) holds the arc-length of each vertex in ``polyline_xyz`` (K, 3);
    ``progress_m`` (N,) is each ego frame's monotone arc-length position, or None when the
    log has no ego frames to attach progress to.
    """

    polyline_arc_m: npt.NDArray[np.float64]
    polyline_xyz: npt.NDArray[np.float64]
    progress_m: Optional[npt.NDArray[np.float64]]
    total_arc_m: float

    def to_route_metadata(self, resolution_m: float, source: str, cache_source_info: CacheSourceInfo) -> RouteMetadata:
        """Pack the polyline into the modality metadata (static over the log)."""
        return RouteMetadata(
            resolution_m=resolution_m,
            total_arc_m=self.total_arc_m,
            polyline_x=self.polyline_xyz[:, 0].tolist(),
            polyline_y=self.polyline_xyz[:, 1].tolist(),
            polyline_z=self.polyline_xyz[:, 2].tolist(),
            cache_source_info=cache_source_info,
            source=source,
        )


def _resampled_polyline(
    positions_xyz: npt.NDArray[np.float64], resolution_m: float, min_segment_m: float
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Prune positions within ``min_segment_m`` of the previously kept one (strictly
    increasing arc parameterization), then resample at ``resolution_m`` steps with the
    final vertex landing exactly on the total arc.

    :return: (kept indices, arc per kept position, resampled arc, resampled vertices).
    """
    kept = [0]
    for index in range(1, len(positions_xyz)):
        if float(np.linalg.norm(positions_xyz[index] - positions_xyz[kept[-1]])) >= min_segment_m:
            kept.append(index)
    kept_indices = np.asarray(kept, dtype=np.int64)
    kept_xyz = positions_xyz[kept_indices]
    kept_arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(kept_xyz, axis=0), axis=1), dtype=np.float64)))

    total_arc_m = float(kept_arc[-1])
    sample_arc = np.arange(0.0, total_arc_m, resolution_m, dtype=np.float64)
    if len(sample_arc) == 0 or total_arc_m - sample_arc[-1] > 1e-9:
        sample_arc = np.concatenate((sample_arc, [total_arc_m]))
    return kept_indices, kept_arc, sample_arc, interpolate_route_at_arc(kept_arc, kept_xyz, sample_arc)


def _as_positions(positions_xyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    positions_xyz = np.asarray(positions_xyz, dtype=np.float64)
    assert positions_xyz.ndim == 2 and positions_xyz.shape[1] == 3, (
        f"Expected positions of shape (N, 3), got {positions_xyz.shape}."
    )
    return positions_xyz


def compute_route_data(
    positions_xyz: npt.NDArray[np.float64],
    resolution_m: float,
    min_segment_m: float = _MIN_SEGMENT_M,
) -> Optional[RouteData]:
    """Compute the driven route and per-frame progress from ego positions (N, 3) in log order.

    A frame's progress is the arc-length of the last kept (pruned) position at or before
    it: piecewise-constant while standing, exact while driving. Returns None when no
    position is available.
    """
    assert resolution_m > 0.0, f"resolution_m must be > 0, got {resolution_m}."
    positions_xyz = _as_positions(positions_xyz)
    if len(positions_xyz) == 0:
        return None

    kept_indices, kept_arc, sample_arc, sample_xyz = _resampled_polyline(positions_xyz, resolution_m, min_segment_m)
    frame_positions = np.searchsorted(kept_indices, np.arange(len(positions_xyz), dtype=np.int64), side="right") - 1
    return RouteData(sample_arc, sample_xyz, kept_arc[frame_positions], float(kept_arc[-1]))


def compute_route_data_from_waypoints(
    route_xyz: npt.NDArray[np.float64],
    ego_positions_xyz: Optional[npt.NDArray[np.float64]],
    resolution_m: float,
    min_segment_m: float = _MIN_SEGMENT_M,
) -> Optional[RouteData]:
    """Compute route data from explicitly provided waypoints (M, 3) in driving order.

    Per-frame progress is the ego's monotone projection onto the route, well-defined even
    where the ego deviates from the route or the route crosses itself; pass
    ``ego_positions_xyz=None`` to skip it. Returns None when no waypoint is available.
    """
    assert resolution_m > 0.0, f"resolution_m must be > 0, got {resolution_m}."
    route_xyz = _as_positions(route_xyz)
    if len(route_xyz) == 0:
        return None

    _, kept_arc, sample_arc, sample_xyz = _resampled_polyline(route_xyz, resolution_m, min_segment_m)
    progress_m: Optional[npt.NDArray[np.float64]] = None
    if ego_positions_xyz is not None and len(ego_positions_xyz) > 0:
        progress_m = project_onto_polyline(sample_arc, sample_xyz, _as_positions(ego_positions_xyz))

    return RouteData(sample_arc, sample_xyz, progress_m, float(kept_arc[-1]))


def project_onto_polyline(
    polyline_arc_m: npt.NDArray[np.float64],
    polyline_xyz: npt.NDArray[np.float64],
    positions_xyz: npt.NDArray[np.float64],
    window_m: float = _PROJECTION_WINDOW_M,
) -> npt.NDArray[np.float64]:
    """Project positions (N, 3, temporal order) onto a polyline as monotone arc-length values.

    The first position searches the whole polyline; each subsequent one only from the
    previous match forward by ``window_m``, which keeps progress monotone on self-crossing
    routes.
    """
    progress = np.empty(len(positions_xyz), dtype=np.float64)
    start = 0
    for index, position in enumerate(positions_xyz):
        end = int(np.searchsorted(polyline_arc_m, polyline_arc_m[start] + window_m, side="right"))
        distances = np.linalg.norm(polyline_xyz[start : end + 1] - position, axis=1)
        nearest = start + int(np.argmin(distances))
        best_arc, best_dist = float(polyline_arc_m[nearest]), float(distances[nearest - start])
        # Exact projection onto the two segments adjacent to the nearest vertex.
        for a in (nearest - 1, nearest):
            if a < 0 or a + 1 >= len(polyline_xyz):
                continue
            segment = polyline_xyz[a + 1] - polyline_xyz[a]
            length_sq = float(segment @ segment)
            if length_sq == 0.0:  # out-and-back turnaround can coincide spatially
                continue
            t = float(np.clip((position - polyline_xyz[a]) @ segment / length_sq, 0.0, 1.0))
            dist = float(np.linalg.norm(polyline_xyz[a] + t * segment - position))
            if dist < best_dist:
                best_arc = float(polyline_arc_m[a] + t * (polyline_arc_m[a + 1] - polyline_arc_m[a]))
                best_dist = dist
        progress[index] = max(best_arc, progress[index - 1]) if index > 0 else best_arc
        start = max(int(np.searchsorted(polyline_arc_m, progress[index], side="right")) - 1, 0)
    return progress


def interpolate_route_at_arc(
    polyline_arc_m: npt.NDArray[np.float64],
    polyline_xyz: npt.NDArray[np.float64],
    query_arc_m: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Interpolate route positions (M, 3) at the given arc-length values (M,).

    Queries outside ``[0, total_arc_m]`` clamp to the endpoints; callers that must not
    read past the route's end check the query against the total arc-length first.
    """
    query_arc_m = np.asarray(query_arc_m, dtype=np.float64)
    return np.stack([np.interp(query_arc_m, polyline_arc_m, polyline_xyz[:, axis]) for axis in range(3)], axis=1)


def write_route_position_arrow(
    log_dir: Path,
    timestamps_us: npt.NDArray[np.int64],
    progress_m: npt.NDArray[np.float64],
    route_metadata: RouteMetadata,
    ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
    ipc_compression_level: Optional[int] = None,
) -> None:
    """Write the ``route_position`` modality file: per-frame progress and remaining route,
    with the polyline in the schema metadata."""
    progress_m = np.asarray(progress_m, dtype=np.float64)
    schema = pa.schema(
        [
            pa.field(f"{ROUTE_POSITION_KEY}.timestamp_us", pa.int64()),
            pa.field(f"{ROUTE_POSITION_KEY}.progress_m", pa.float64()),
            pa.field(f"{ROUTE_POSITION_KEY}.remaining_m", pa.float64()),
        ]
    )
    schema = add_metadata_to_arrow_schema(schema, route_metadata)
    table = pa.table(
        {
            f"{ROUTE_POSITION_KEY}.timestamp_us": np.asarray(timestamps_us, dtype=np.int64),
            f"{ROUTE_POSITION_KEY}.progress_m": progress_m,
            f"{ROUTE_POSITION_KEY}.remaining_m": route_metadata.total_arc_m - progress_m,
        },
        schema=schema,
    )

    options = None
    if ipc_compression is not None:
        options = pa.ipc.IpcWriteOptions(compression=pa.Codec(ipc_compression, compression_level=ipc_compression_level))
    with pa.OSFile(str(log_dir / f"{ROUTE_POSITION_KEY}.arrow"), "wb") as source:
        with pa.ipc.new_file(source, schema=schema, options=options) as writer:
            writer.write_table(table)


def read_route_position(
    log_dir: Path,
) -> Optional[Tuple[RouteMetadata, npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """Read (route metadata, progress per row (N,), remaining per row (N,)) from
    ``route_position.arrow``, or None when the log has no route positions."""
    from py123d.api.utils.arrow_helper import get_lru_cached_arrow_table

    path = Path(log_dir) / f"{ROUTE_POSITION_KEY}.arrow"
    if not path.exists():
        return None

    table = get_lru_cached_arrow_table(str(path))
    route_metadata = get_metadata_from_arrow_schema(table.schema, RouteMetadata)
    return (
        route_metadata,
        table[f"{ROUTE_POSITION_KEY}.progress_m"].to_numpy(),
        table[f"{ROUTE_POSITION_KEY}.remaining_m"].to_numpy(),
    )


def read_remaining_route_m(log_dir: Path) -> Optional[npt.NDArray[np.float64]]:
    """Read only the per-row remaining route (N,) from ``route_position.arrow``, or None.

    Filtering uses this to stay off the metadata: the polyline blob is never decoded
    during scene enumeration.
    """
    from py123d.api.utils.arrow_helper import get_lru_cached_arrow_table

    path = Path(log_dir) / f"{ROUTE_POSITION_KEY}.arrow"
    if not path.exists():
        return None
    table = get_lru_cached_arrow_table(str(path))
    return table[f"{ROUTE_POSITION_KEY}.remaining_m"].to_numpy()


def warn_on_position_jumps(
    positions_xyz: npt.NDArray[np.float64],
    timestamps_us: npt.NDArray[np.int64],
    context: str,
    max_speed_mps: float = _MAX_PLAUSIBLE_SPEED_MPS,
) -> None:
    """Warn once when consecutive positions imply physically implausible speed.

    Localization teleports silently inflate the driven route's arc-length; this flags
    them without correcting anything, since the right fix is upstream in the data.
    """
    dt_s = np.diff(np.asarray(timestamps_us, dtype=np.float64)) / 1e6
    distances = np.linalg.norm(np.diff(positions_xyz, axis=0), axis=1)
    valid = dt_s > 0.0
    jumps = valid & (distances > max_speed_mps * dt_s)
    if jumps.any():
        worst = int(np.argmax(np.where(jumps, distances / np.where(valid, dt_s, np.inf), 0.0)))
        logger.warning(
            "%s: %d ego position jump(s) above %.0f m/s (worst: %.1f m in %.3f s at row %d); "
            "the route arc-length is inflated accordingly.",
            context,
            int(jumps.sum()),
            max_speed_mps,
            float(distances[worst]),
            float(dt_s[worst]),
            worst,
        )
