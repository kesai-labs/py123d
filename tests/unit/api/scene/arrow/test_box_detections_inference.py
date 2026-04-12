"""Tests for finite-difference box-velocity inference in global frame."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyarrow as pa

from py123d.api.scene.arrow.modalities.arrow_box_detections_se3 import (
    ArrowBoxDetectionsSE3Reader,
    ArrowBoxDetectionsSE3Writer,
)
from py123d.datatypes import Timestamp
from py123d.datatypes.detections.box_detection_label import DefaultBoxDetectionLabel
from py123d.datatypes.detections.box_detections import (
    BoxDetectionAttributes,
    BoxDetectionSE3,
    BoxDetectionsSE3,
)
from py123d.geometry.bounding_box import BoundingBoxSE3
from py123d.geometry.pose import PoseSE3
from py123d.geometry.vector import Vector3D

from ..conftest import make_box_detections_metadata


def _make_detection(track_token: str, x: float, y: float, z: float) -> BoxDetectionSE3:
    pose = PoseSE3(x=x, y=y, z=z, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
    bbox = BoundingBoxSE3(center_se3=pose, length=4.5, width=2.0, height=1.6)
    return BoxDetectionSE3(
        attributes=BoxDetectionAttributes(
            label=DefaultBoxDetectionLabel.VEHICLE,
            track_token=track_token,
            num_lidar_points=100,
        ),
        bounding_box_se3=bbox,
        velocity_3d=Vector3D(999.0, 999.0, 999.0),
    )


def _make_frame(ts_us: int, entries: List[Tuple[str, float, float, float]]) -> BoxDetectionsSE3:
    return BoxDetectionsSE3(
        box_detections=[_make_detection(*e) for e in entries],
        timestamp=Timestamp.from_us(ts_us),
        metadata=make_box_detections_metadata(),
    )


def _write_and_read(log_dir: Path, frames: List[BoxDetectionsSE3]) -> List[BoxDetectionsSE3]:
    metadata = make_box_detections_metadata()
    writer = ArrowBoxDetectionsSE3Writer(log_dir=log_dir, metadata=metadata, infer_box_dynamics=True)
    for frame in frames:
        writer.write_modality(frame)
    writer.close()

    table: pa.Table = pa.ipc.open_file(str(log_dir / f"{metadata.modality_key}.arrow")).read_all()
    assert table.num_rows == len(frames)
    results = []
    for i in range(table.num_rows):
        result = ArrowBoxDetectionsSE3Reader.read_at_index(i, table, metadata, "test-dataset")
        assert result is not None
        results.append(result)
    return results


def _velocity_by_token(frame: BoxDetectionsSE3, track_token: str) -> np.ndarray:
    det = frame.get_detection_by_track_token(track_token)
    assert det is not None and det.velocity_3d is not None
    return det.velocity_3d.array


class TestArrowBoxDetectionsSE3Inference:
    def test_constant_velocity_global_frame(self, tmp_path: Path):
        """Single track moving at 10 m/s along global X."""
        dt = 0.1
        frames = [_make_frame(int(i * dt * 1e6), [("A", float(i), 0.0, 0.0)]) for i in range(5)]
        results = _write_and_read(tmp_path, frames)
        for result in results:
            np.testing.assert_allclose(_velocity_by_token(result, "A"), [10.0, 0.0, 0.0], atol=1e-9)

    def test_track_appearing_gets_one_sided(self, tmp_path: Path):
        """A track that appears mid-log should still get a non-zero velocity from the neighbor it has."""
        dt = 0.1
        frames = [
            _make_frame(int(0 * dt * 1e6), [("A", 0.0, 0.0, 0.0)]),
            _make_frame(int(1 * dt * 1e6), [("A", 1.0, 0.0, 0.0), ("B", 0.0, 0.0, 0.0)]),
            _make_frame(int(2 * dt * 1e6), [("A", 2.0, 0.0, 0.0), ("B", 0.5, 0.0, 0.0)]),
        ]
        results = _write_and_read(tmp_path, frames)
        # Track A present in all three frames -> centered/forward diffs around 10.0.
        np.testing.assert_allclose(_velocity_by_token(results[1], "A"), [10.0, 0.0, 0.0], atol=1e-9)
        # Track B appears at frame 1 (no prev neighbor) -> forward diff against frame 2 = 5 m/s.
        np.testing.assert_allclose(_velocity_by_token(results[1], "B"), [5.0, 0.0, 0.0], atol=1e-9)

    def test_track_disappearing_gets_one_sided(self, tmp_path: Path):
        """A track only present in the first two frames gets a forward-diff velocity on frame 0 and
        a backward-diff velocity on frame 1; on frame 2 it is absent entirely."""
        dt = 0.1
        frames = [
            _make_frame(int(0 * dt * 1e6), [("A", 0.0, 0.0, 0.0), ("B", 0.0, 0.0, 0.0)]),
            _make_frame(int(1 * dt * 1e6), [("A", 1.0, 0.0, 0.0), ("B", 2.0, 0.0, 0.0)]),
            _make_frame(int(2 * dt * 1e6), [("A", 2.0, 0.0, 0.0)]),
        ]
        results = _write_and_read(tmp_path, frames)
        # Frame 0: B forward diff -> 20.0 m/s.
        np.testing.assert_allclose(_velocity_by_token(results[0], "B"), [20.0, 0.0, 0.0], atol=1e-9)
        # Frame 1: B only has a prev neighbor (A is still in frame 2 but B is not)
        # -> backward diff against frame 0 = 20.0 m/s.
        np.testing.assert_allclose(_velocity_by_token(results[1], "B"), [20.0, 0.0, 0.0], atol=1e-9)

    def test_single_frame_zero_velocity(self, tmp_path: Path):
        frames = [_make_frame(1000, [("A", 1.0, 2.0, 3.0)])]
        results = _write_and_read(tmp_path, frames)
        np.testing.assert_allclose(_velocity_by_token(results[0], "A"), [0.0, 0.0, 0.0])

    def test_inference_disabled_passthrough(self, tmp_path: Path):
        """With the flag off, the velocity field on the input is preserved verbatim."""
        frame = _make_frame(1000, [("A", 0.0, 0.0, 0.0)])  # velocity_3d = (999, 999, 999)
        metadata = make_box_detections_metadata()
        writer = ArrowBoxDetectionsSE3Writer(log_dir=tmp_path, metadata=metadata, infer_box_dynamics=False)
        writer.write_modality(frame)
        writer.close()

        table = pa.ipc.open_file(str(tmp_path / f"{metadata.modality_key}.arrow")).read_all()
        result = ArrowBoxDetectionsSE3Reader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None
        np.testing.assert_allclose(_velocity_by_token(result, "A"), [999.0, 999.0, 999.0])
