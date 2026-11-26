"""Spherical coding of point positions, the kedec mode ``"spherical"`` (for any lidar).

The positions are given in the coding frame of the sensor (centered at the sensor position). Per point, the range
is rounded to ``range_step`` meters and the azimuth and elevation to ``angle_step`` degrees. The points are sorted
by a stored ring index (when the scan has one), else by rows of ``row_step`` degrees in elevation, and then by
azimuth (beam order), so that neighboring points have similar values. The three integer channels are stored by the
integer coding of :mod:`~py123d.common.io.lidar.kedec.int_coding`.

The largest position error of a point at range r is about
``sqrt((range_step / 2)**2 + 2 * (r * radians(angle_step / 2))**2)``: 2 cm up to 175 m and 5.04 cm at 500 m for the
default steps (2 cm, 0.008 degrees).
"""

from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

_TABLES: Dict[float, Tuple[int, npt.NDArray[np.float64], npt.NDArray[np.float64]]] = {}
_TABLES_LOCK = threading.Lock()


def _angle_tables(angle_step: float) -> Tuple[int, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """(offset, cosine, sine) of every angle step k * angle_step degrees for k from -offset to +offset (360 degrees
    on each side)."""
    tables = _TABLES.get(angle_step)
    if tables is None:
        with _TABLES_LOCK:
            tables = _TABLES.get(angle_step)
            if tables is None:
                offset = int(np.ceil(360.0 / angle_step)) + 1
                grid = np.radians(np.arange(-offset, offset + 1, dtype=np.float64) * angle_step)
                tables = (offset, np.cos(grid), np.sin(grid))
                _TABLES[angle_step] = tables
    return tables


def spherical_angles(points: npt.NDArray[np.float64]) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Range (meters), azimuth and elevation (degrees) of points in the coding frame."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    azimuth = np.degrees(np.arctan2(y, x))
    elevation = np.degrees(np.arcsin(np.clip(z / np.maximum(r, 1e-12), -1.0, 1.0)))
    return r, azimuth, elevation


def encode_positions(
    points: npt.NDArray[np.float64],
    ring: Optional[npt.NDArray],
    range_step: float,
    angle_step: float,
    row_step: float,
) -> Tuple[npt.NDArray[np.int64], Dict[str, npt.NDArray[np.int64]]]:
    """Quantize positions and choose the stored order.

    :param points: (n, 3) positions in the coding frame.
    :param ring: Optional ring (beam) index per point; it orders the points instead of the elevation rows.
    :param range_step: Range step in meters.
    :param angle_step: Azimuth and elevation step in degrees.
    :param row_step: Height of the elevation rows in degrees, used for the order when no ring is given.
    :return: The order (indices into the input) and the integer channels ``r``, ``a``, ``e`` in that order.
    """
    r, azimuth, elevation = spherical_angles(points)
    if ring is not None:
        order = np.lexsort((azimuth, np.asarray(ring)))
    else:
        order = np.lexsort((azimuth, np.round(elevation / row_step)))
    channels = {
        "r": np.round(r[order] / range_step).astype(np.int64),
        "a": np.round(azimuth[order] / angle_step).astype(np.int64),
        "e": np.round(elevation[order] / angle_step).astype(np.int64),
    }
    return order, channels


def decode_positions(
    channels: Dict[str, npt.NDArray[np.int64]], range_step: float, angle_step: float
) -> npt.NDArray[np.float64]:
    """Positions in the coding frame from the integer channels ``r``, ``a``, ``e``.

    :raises ValueError: If an angle lies outside -360 to 360 degrees (a corrupt blob).
    """
    offset, cos, sin = _angle_tables(angle_step)
    ia = channels["a"] + offset
    ie = channels["e"] + offset
    if len(ia) and (min(int(ia.min()), int(ie.min())) < 0 or max(int(ia.max()), int(ie.max())) >= len(cos)):
        raise ValueError("kedec: a spherical angle lies outside -360 to 360 degrees.")
    r = channels["r"] * range_step
    horizontal = r * cos[ie]
    out = np.empty((len(r), 3), dtype=np.float64)
    out[:, 0] = horizontal * cos[ia]
    out[:, 1] = horizontal * sin[ia]
    out[:, 2] = r * sin[ie]
    return out
