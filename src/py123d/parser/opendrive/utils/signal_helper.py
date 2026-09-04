import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from py123d.parser.opendrive.utils.id_system import build_lane_id
from py123d.parser.opendrive.xodr_parser.road import XODRRoad
from py123d.parser.opendrive.xodr_parser.signals import XODRSignal, XODRSignalReference

logger = logging.getLogger(__name__)


@dataclass
class OpenDriveSignalHelper:
    signal_id: int
    signal_type: str
    lane_ids: List[str]  # Lane IDs controlled by this signal
    turn_relation: Optional[str]
    xodr_signal: XODRSignal
    junction_id: Optional[int] = None  # junction whose controller cycles this signal
    phase_idx: Optional[int] = None  # controller sequence within that junction's cycle


def _lane_section_idx_from_s(road: XODRRoad, s: float) -> int:
    lane_section_idx = 0
    for idx, lane_section in enumerate(road.lanes.lane_sections):
        if s < lane_section.s:
            break
        lane_section_idx = idx
    return lane_section_idx


def _lane_ids_from_signal_ref_validity(
    signal_ref: XODRSignalReference,
    road: XODRRoad,
) -> List[str]:
    """Extract lane IDs from signal reference validity elements."""
    lane_section_idx = _lane_section_idx_from_s(road, signal_ref.s)

    if not signal_ref.validity:
        return []

    lane_indices = set()
    for validity in signal_ref.validity:
        if validity.from_lane == validity.to_lane:
            lane_indices.add(validity.from_lane)
            continue
        step = 1 if validity.to_lane > validity.from_lane else -1
        lane_indices.update(range(validity.from_lane, validity.to_lane + step, step))

    # Remove center lane (not drivable)
    lane_indices.discard(0)

    if not lane_indices:
        return []

    return [build_lane_id(road.id, lane_section_idx, lane_idx) for lane_idx in sorted(lane_indices)]


def get_signal_reference_helper(
    signal_ref: XODRSignalReference,
    signal_lookup: Dict[int, XODRSignal],
    road: XODRRoad,
) -> Optional[OpenDriveSignalHelper]:
    """Create helper from signal reference (has lane validity) and signal definition (has type)."""
    signal = signal_lookup.get(signal_ref.id)
    if signal is None:
        logger.debug(f"Signal definition not found for signal_ref.id={signal_ref.id} on road {road.id}")
        return None

    lane_ids = _lane_ids_from_signal_ref_validity(signal_ref, road)

    return OpenDriveSignalHelper(
        signal_id=signal_ref.id,
        signal_type=signal.type,
        lane_ids=lane_ids,
        turn_relation=signal_ref.turn_relation,
        xodr_signal=signal,
    )
