"""Grid coding of point positions, the kedec mode ``"grid"`` (for scans that store a beam and a column index).

The positions are given in the sensor frame. The points are sorted by beam, then by column. Per scan, three
float32 tables are stored: one elevation per beam (the median elevation of its points), one azimuth per column (the
circular median of its points after the beam offsets) and one azimuth offset per beam (the median deviation of its
points from the column table). Per point, the range is rounded to ``range_step`` meters, and the deviation of each
angle from the tables is stored as a sideways distance in steps of ``side_step`` meters: the elevation deviation in
radians times the decoded range, and the azimuth deviation times the decoded range and the cosine of the decoded
elevation. Each part of the error is at most half a step, so the largest position error is about
``sqrt(3) / 2`` steps (1.73 cm for steps of 2 cm), for points more than a few degrees away from the poles.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt

# Lower bound of the cosine that scales azimuth deviations, so that points at the poles do not divide by zero.
_COS_MIN = 1e-6


def _wrap(angle: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Angles in radians wrapped to [-pi, pi)."""
    return np.remainder(angle + np.pi, 2.0 * np.pi) - np.pi


def _circular_median(angle: npt.NDArray[np.float64]) -> float:
    """The median of angles (radians) around their circular mean."""
    center = float(np.arctan2(np.sin(angle).mean(), np.cos(angle).mean()))
    return center + float(np.median(_wrap(angle - center)))


def _groups(sorted_index: npt.NDArray[np.int64]) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """(group values, start positions) of a sorted index array; the groups end at the next start."""
    starts = np.flatnonzero(np.diff(sorted_index)) + 1
    starts = np.concatenate(([0], starts)).astype(np.int64)
    return sorted_index[starts], starts


def build_tables(
    azimuth: npt.NDArray[np.float64],
    elevation: npt.NDArray[np.float64],
    beam: npt.NDArray[np.int64],
    column: npt.NDArray[np.int64],
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """The float32 tables of one scan (radians): elevation per beam, azimuth per column, azimuth offset per beam.

    The points must be sorted by beam, then by column.
    """
    n_beams = int(beam.max()) + 1
    n_columns = int(column.max()) + 1
    elevation_table = np.zeros(n_beams)
    offset_table = np.zeros(n_beams)
    azimuth_table = np.zeros(n_columns)

    beam_values, beam_starts = _groups(beam)
    beam_ends = np.append(beam_starts[1:], len(beam))
    column_order = np.argsort(column, kind="stable")
    column_values, column_starts = _groups(column[column_order])
    column_ends = np.append(column_starts[1:], len(column))

    for value, start, end in zip(beam_values, beam_starts, beam_ends):
        elevation_table[value] = np.median(elevation[start:end])
    for value, start, end in zip(column_values, column_starts, column_ends):  # first pass, without beam offsets
        azimuth_table[value] = _circular_median(azimuth[column_order[start:end]])
    residual = _wrap(azimuth - azimuth_table[column])
    for value, start, end in zip(beam_values, beam_starts, beam_ends):
        offset_table[value] = np.median(residual[start:end])
    corrected = azimuth - offset_table[beam]
    for value, start, end in zip(column_values, column_starts, column_ends):  # second pass, after the beam offsets
        azimuth_table[value] = _circular_median(corrected[column_order[start:end]])
    return elevation_table.astype(np.float32), azimuth_table.astype(np.float32), offset_table.astype(np.float32)


def encode_positions(
    points: npt.NDArray[np.float64],
    beam: npt.NDArray,
    column: npt.NDArray,
    range_step: float,
    side_step: float,
) -> Tuple[npt.NDArray[np.int64], bytes, Dict[str, npt.NDArray[np.int64]], Tuple[int, int]]:
    """Build the tables, quantize the positions and choose the stored order.

    :param points: (n, 3) positions in the sensor frame, n > 0.
    :param beam: Beam index per point (non-negative integers).
    :param column: Column index per point (non-negative integers).
    :param range_step: Range step in meters.
    :param side_step: Step of the sideways deviations in meters.
    :return: The order (indices into the input), the table bytes (float32 elevations, azimuths, offsets), the
        integer channels ``r``, ``s_a``, ``s_e`` in that order, and (number of beams, number of columns).
    """
    beam = np.asarray(beam).astype(np.int64)
    column = np.asarray(column).astype(np.int64)
    order = np.lexsort((column, beam))
    p = points[order]
    beam, column = beam[order], column[order]
    r = np.sqrt((p * p).sum(axis=1))
    azimuth = np.arctan2(p[:, 1], p[:, 0])
    elevation = np.arcsin(np.clip(p[:, 2] / np.maximum(r, 1e-12), -1.0, 1.0))
    elevation_table, azimuth_table, offset_table = build_tables(azimuth, elevation, beam, column)
    e64 = elevation_table.astype(np.float64)
    a64 = azimuth_table.astype(np.float64)
    o64 = offset_table.astype(np.float64)

    rq = np.round(r / range_step).astype(np.int64)
    decoded_range = np.maximum(rq * range_step, range_step)
    se = np.round((elevation - e64[beam]) * decoded_range / side_step).astype(np.int64)
    decoded_elevation = e64[beam] + se * side_step / decoded_range
    scale = np.maximum(np.cos(decoded_elevation), _COS_MIN)
    sa = np.round(_wrap(azimuth - a64[column] - o64[beam]) * decoded_range * scale / side_step).astype(np.int64)
    tables = elevation_table.tobytes() + azimuth_table.tobytes() + offset_table.tobytes()
    return order, tables, {"r": rq, "s_a": sa, "s_e": se}, (len(elevation_table), len(azimuth_table))


def decode_positions(
    tables: bytes,
    channels: Dict[str, npt.NDArray[np.int64]],
    beam: npt.NDArray,
    column: npt.NDArray,
    n_beams: int,
    n_columns: int,
    range_step: float,
    side_step: float,
) -> npt.NDArray[np.float64]:
    """Positions in the sensor frame from the tables, the integer channels and the beam and column indices.

    :raises ValueError: If the tables have the wrong size or an index lies outside them (a corrupt blob).
    """
    values = np.frombuffer(tables, dtype=np.float32).astype(np.float64)
    if len(values) != 2 * n_beams + n_columns:
        raise ValueError("kedec: the grid tables have the wrong size.")
    elevation_table = values[:n_beams]
    azimuth_table = values[n_beams : n_beams + n_columns]
    offset_table = values[n_beams + n_columns :]
    beam = np.asarray(beam).astype(np.int64)
    column = np.asarray(column).astype(np.int64)
    if len(beam) and (
        int(beam.min()) < 0 or int(beam.max()) >= n_beams or int(column.min()) < 0 or int(column.max()) >= n_columns
    ):
        raise ValueError("kedec: a beam or column index lies outside the grid tables.")

    r = channels["r"] * range_step
    decoded_range = np.maximum(r, range_step)
    elevation = elevation_table[beam] + channels["s_e"] * side_step / decoded_range
    scale = np.maximum(np.cos(elevation), _COS_MIN)
    azimuth = azimuth_table[column] + offset_table[beam] + channels["s_a"] * side_step / (decoded_range * scale)
    horizontal = r * np.cos(elevation)
    out = np.empty((len(r), 3), dtype=np.float64)
    out[:, 0] = horizontal * np.cos(azimuth)
    out[:, 1] = horizontal * np.sin(azimuth)
    out[:, 2] = r * np.sin(elevation)
    return out
