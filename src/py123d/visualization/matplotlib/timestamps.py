from collections import OrderedDict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from py123d.api.scene.scene_api import SceneAPI
from py123d.datatypes import ModalityType

# Display order for modality types (top to bottom).
_MODALITY_TYPE_ORDER: List[ModalityType] = [
    ModalityType.EGO_STATE_SE3,
    ModalityType.BOX_DETECTIONS_SE3,
    ModalityType.TRAFFIC_LIGHT_DETECTIONS,
    ModalityType.LIDAR,
    ModalityType.CAMERA,
    ModalityType.CUSTOM,
]

# Color palette per modality type.
_MODALITY_TYPE_COLORS: Dict[ModalityType, str] = {
    ModalityType.EGO_STATE_SE3: "#4C78A8",
    ModalityType.BOX_DETECTIONS_SE3: "#F58518",
    ModalityType.TRAFFIC_LIGHT_DETECTIONS: "#E45756",
    ModalityType.LIDAR: "#72B7B2",
    ModalityType.CAMERA: "#54A24B",
    ModalityType.CUSTOM: "#B279A2",
}


# Human-readable names for known modality types (without the technical suffix).
_MODALITY_TYPE_LABELS: Dict[ModalityType, str] = {
    ModalityType.EGO_STATE_SE3: "Ego State",
    ModalityType.BOX_DETECTIONS_SE3: "Box Detections",
    ModalityType.TRAFFIC_LIGHT_DETECTIONS: "Traffic Lights",
    ModalityType.LIDAR: "Lidar",
    ModalityType.CAMERA: "Camera",
    ModalityType.CUSTOM: "Custom",
}


def plot_scene_timestamps(
    scene: SceneAPI,
    include_history: bool = False,
    figsize_width: float = 10.0,
    row_height: float = 0.55,
    time_unit: str = "ms",
) -> Tuple[plt.Figure, plt.Axes]:  # type: ignore
    """Plot the timestamps of all available modalities in a scene.

    Creates a horizontal timeline chart where each row represents one sensor or modality
    and vertical tick marks show when data is available. Sync (iteration) timestamps are
    shown as subtle reference lines in the background.

    :param scene: The scene API to read timestamps from.
    :param include_history: Whether to include history timestamps.
    :param figsize_width: Width of the figure in inches.
    :param row_height: Height per modality row in inches.
    :param time_unit: Time unit for the x-axis. One of ``"ms"``, ``"s"``, or ``"us"``.
    :return: Tuple of (figure, axes).
    """
    time_divisors = {"us": 1.0, "ms": 1e3, "s": 1e6}
    assert time_unit in time_divisors, f"time_unit must be one of {list(time_divisors.keys())}"
    divisor = time_divisors[time_unit]

    # -- Collect sync timestamps as the reference baseline -----------------------------------------------
    sync_timestamps = scene.get_all_iteration_timestamps(include_history)
    initial_time_us = scene.get_timestamp_at_iteration(0).time_us
    sync_times = np.array([t.time_us for t in sync_timestamps], dtype=np.float64)
    sync_times = (sync_times - initial_time_us) / divisor

    # -- Collect per-modality timestamps ----------------------------------------------------------------
    all_metadatas = scene.get_all_modality_metadatas()

    # Group modality keys by type, preserving type order.
    grouped: Dict[ModalityType, List[str]] = OrderedDict()
    for mt in _MODALITY_TYPE_ORDER:
        grouped[mt] = []
    for key, meta in all_metadatas.items():
        mt = meta.modality_type
        if mt not in grouped:
            grouped[mt] = []
        grouped[mt].append(key)
    # Collect timestamps for every modality key up front.
    key_timestamps: Dict[str, np.ndarray] = {}
    for key, meta in all_metadatas.items():
        timestamps = scene.get_all_modality_timestamps(
            meta.modality_type, meta.modality_id, include_history=include_history
        )
        if len(timestamps) > 0:
            times = np.array([t.time_us for t in timestamps], dtype=np.float64)
            key_timestamps[key] = (times - initial_time_us) / divisor

    # Sort instance keys within each group by their earliest timestamp (closest to iteration 0).
    for mt in grouped:
        grouped[mt] = [k for k in grouped[mt] if k in key_timestamps]
        grouped[mt].sort(
            key=lambda k: key_timestamps[k][key_timestamps[k] >= 0][0]
            if np.any(key_timestamps[k] >= 0)
            else float("inf")
        )

    # (label, modality_type_name, times, color)
    rows: List[Tuple[str, str, np.ndarray, str]] = []
    # Track row indices where a new modality type group starts (for visual separators).
    group_boundaries: List[int] = []

    for modality_type, keys in grouped.items():
        if len(keys) == 0:
            continue

        group_start = len(rows)
        base_color = _MODALITY_TYPE_COLORS.get(modality_type, "#888888")

        for key in keys:
            times = key_timestamps[key]
            rows.append((key, modality_type.serialize(), times, base_color))

        # Record boundary if this group added any rows.
        if len(rows) > group_start > 0:
            group_boundaries.append(group_start)

    n_rows = len(rows)
    if n_rows == 0:
        fig, ax = plt.subplots(figsize=(figsize_width, 2))
        ax.text(0.5, 0.5, "No modality timestamps found", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    # -- Figure setup -----------------------------------------------------------------------------------
    top_margin = 0.8
    bottom_margin = 0.6
    fig_height = n_rows * row_height + top_margin + bottom_margin
    fig, ax = plt.subplots(figsize=(figsize_width, fig_height))

    half_height = 0.35
    linewidth = 2.0

    # -- Draw sync reference lines (behind everything) --------------------------------------------------
    ax.vlines(
        sync_times,
        ymin=-0.5,
        ymax=n_rows - 0.5,
        color="#737373",
        alpha=0.7,
        linewidth=0.5,
        linestyle="--",
        zorder=0,
        label="Iteration",
    )

    # -- Draw each modality row -------------------------------------------------------------------------
    y_labels: List[str] = []
    for y_idx, (label, _type_name, times, color) in enumerate(rows):
        ax.vlines(
            times,
            ymin=y_idx - half_height,
            ymax=y_idx + half_height,
            color=color,
            alpha=0.9,
            linewidth=linewidth,
            zorder=2,
        )
        y_labels.append(label)

    # -- Styling ----------------------------------------------------------------------------------------
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_ylim(-0.7, n_rows - 0.3)
    ax.invert_yaxis()

    # -- Group separators (thin horizontal lines between modality types) ----------------------------------
    for boundary_idx in group_boundaries:
        ax.axhline(y=boundary_idx - 0.5, color="#E0E0E0", linewidth=0.8, linestyle="-", zorder=1)

    ax.set_xlabel(f"Time ({time_unit})", fontsize=10)
    ax.set_title(f"Timestamps — {scene.split} / {scene.log_name}", fontsize=12, pad=10)

    # Subtle grid on x-axis only.
    ax.xaxis.grid(True, linestyle=":", alpha=0.3, zorder=0)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)

    # Thin spines.
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#CCCCCC")
    ax.tick_params(axis="both", which="both", length=3, width=0.5, color="#CCCCCC")

    # -- Build legend by modality type ------------------------------------------------------------------
    legend_handles = _build_type_legend(rows)
    if len(legend_handles) > 0:
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            fontsize=8,
            framealpha=0.9,
            edgecolor="#CCCCCC",
            fancybox=True,
        )

    fig.tight_layout()
    return fig, ax


def _build_type_legend(
    rows: List[Tuple[str, str, np.ndarray, str]],
) -> list:
    """Build one legend entry per modality type that appears in the rows."""
    from matplotlib.lines import Line2D

    seen_types: Dict[str, str] = OrderedDict()
    for _label, type_name, _times, color in rows:
        if type_name not in seen_types:
            seen_types[type_name] = color

    # Add sync timestamps entry.
    handles = [
        Line2D([0], [0], color="#D0D0D0", linewidth=1, linestyle="--", label="Iteration"),
    ]
    for type_name, color in seen_types.items():
        modality_type = ModalityType.deserialize(type_name)
        pretty = _MODALITY_TYPE_LABELS.get(modality_type, type_name.replace("_", " ").title())
        handles.append(Line2D([0], [0], color=color, linewidth=3, label=pretty))
    return handles
