"""Tests for the finite-difference ego-dynamics inference path."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pyarrow as pa

from py123d.api.scene.arrow.modalities.arrow_ego_state_se3 import (
    ArrowEgoStateSE3Reader,
    ArrowEgoStateSE3Writer,
)
from py123d.datatypes import Timestamp
from py123d.datatypes.vehicle_state.dynamic_state import DynamicStateSE3
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.geometry.pose import PoseSE3
from py123d.geometry.vector import Vector3D

from ..conftest import make_ego_metadata


def _make_ego(
    ts_us: int,
    pose: PoseSE3,
    dynamics: Optional[DynamicStateSE3] = None,
) -> EgoStateSE3:
    return EgoStateSE3.from_imu(
        imu_se3=pose,
        metadata=make_ego_metadata(),
        timestamp=Timestamp.from_us(ts_us),
        dynamic_state_se3=dynamics,
    )


def _write_and_read_all(log_dir: Path, ego_states: List[EgoStateSE3]) -> List[EgoStateSE3]:
    metadata = make_ego_metadata()
    writer = ArrowEgoStateSE3Writer(log_dir=log_dir, metadata=metadata, infer_ego_dynamics=True)
    for ego in ego_states:
        writer.write_modality(ego)
    writer.close()

    file_path = log_dir / f"{metadata.modality_key}.arrow"
    table: pa.Table = pa.ipc.open_file(str(file_path)).read_all()
    assert table.num_rows == len(ego_states)
    results: List[EgoStateSE3] = []
    for i in range(table.num_rows):
        result = ArrowEgoStateSE3Reader.read_at_index(i, table, metadata, "test-dataset")
        assert result is not None
        results.append(result)
    return results


class TestArrowEgoStateSE3Inference:
    def test_straight_line_constant_velocity(self, tmp_path: Path):
        """Uniform 10 m/s motion along global X, zero rotation -> body vx = 10, others zero."""
        dt_us = 100_000  # 100 ms -> 10 Hz
        speed = 10.0
        egos = [
            _make_ego(
                ts_us=i * dt_us,
                pose=PoseSE3(x=speed * i * dt_us / 1e6, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
            )
            for i in range(5)
        ]

        results = _write_and_read_all(tmp_path, egos)
        for result in results:
            assert result.dynamic_state_se3 is not None
            velocity = result.dynamic_state_se3.velocity_3d.array
            acceleration = result.dynamic_state_se3.acceleration_3d.array
            angular = result.dynamic_state_se3.angular_velocity.array
            np.testing.assert_allclose(velocity, [speed, 0.0, 0.0], atol=1e-9)
            np.testing.assert_allclose(acceleration, [0.0, 0.0, 0.0], atol=1e-9)
            np.testing.assert_allclose(angular, [0.0, 0.0, 0.0], atol=1e-9)

    def test_body_frame_rotation(self, tmp_path: Path):
        """Ego rotated 90 deg around Z, moving in global +X. Body velocity should be (0, -1, 0)."""
        dt_us = 100_000
        # 90 deg yaw rotation quaternion: qw=cos(45deg)=sqrt(2)/2, qz=sin(45deg)=sqrt(2)/2
        qw = np.sqrt(2.0) / 2.0
        qz = np.sqrt(2.0) / 2.0
        egos = [
            _make_ego(
                ts_us=i * dt_us,
                pose=PoseSE3(x=1.0 * i * dt_us / 1e6, y=0.0, z=0.0, qw=qw, qx=0.0, qy=0.0, qz=qz),
            )
            for i in range(5)
        ]

        results = _write_and_read_all(tmp_path, egos)
        for result in results:
            assert result.dynamic_state_se3 is not None
            velocity = result.dynamic_state_se3.velocity_3d.array
            # global velocity (1, 0, 0) rotated into body frame where body +x points along global +y
            # -> body frame sees it as (0, -1, 0).
            np.testing.assert_allclose(velocity, [0.0, -1.0, 0.0], atol=1e-9)

    def test_constant_acceleration(self, tmp_path: Path):
        """Uniformly accelerating along X -> body acceleration vector matches expected magnitude."""
        dt = 0.1
        accel = 2.0
        egos = []
        for i in range(5):
            t = i * dt
            x = 0.5 * accel * t * t
            egos.append(
                _make_ego(
                    ts_us=int(t * 1e6),
                    pose=PoseSE3(x=x, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
                )
            )

        results = _write_and_read_all(tmp_path, egos)
        # Centered difference on middle frames is exact for a quadratic trajectory.
        for result in results[1:-1]:
            assert result.dynamic_state_se3 is not None
            acceleration = result.dynamic_state_se3.acceleration_3d.array
            np.testing.assert_allclose(acceleration, [accel, 0.0, 0.0], atol=1e-9)

    def test_pure_yaw_rotation_angular_velocity(self, tmp_path: Path):
        """Ego spinning in place at constant yaw rate. Body omega_z should match."""
        dt = 0.1
        omega_z = 0.5  # rad/s
        egos = []
        for i in range(5):
            yaw = omega_z * i * dt
            qw = np.cos(yaw / 2.0)
            qz = np.sin(yaw / 2.0)
            egos.append(
                _make_ego(
                    ts_us=int(i * dt * 1e6),
                    pose=PoseSE3(x=0.0, y=0.0, z=0.0, qw=qw, qx=0.0, qy=0.0, qz=qz),
                )
            )

        results = _write_and_read_all(tmp_path, egos)
        for result in results[1:-1]:
            assert result.dynamic_state_se3 is not None
            angular = result.dynamic_state_se3.angular_velocity.array
            np.testing.assert_allclose(angular, [0.0, 0.0, omega_z], atol=1e-9)

    def test_single_frame_zero_dynamics(self, tmp_path: Path):
        ego = _make_ego(ts_us=1000, pose=PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0))
        results = _write_and_read_all(tmp_path, [ego])
        assert results[0].dynamic_state_se3 is not None
        np.testing.assert_allclose(results[0].dynamic_state_se3.array, [0.0] * 9)

    def test_two_frames_one_sided(self, tmp_path: Path):
        dt = 0.1
        egos = [
            _make_ego(ts_us=0, pose=PoseSE3(x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)),
            _make_ego(ts_us=int(dt * 1e6), pose=PoseSE3(x=1.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)),
        ]
        results = _write_and_read_all(tmp_path, egos)
        assert len(results) == 2
        for result in results:
            assert result.dynamic_state_se3 is not None
            np.testing.assert_allclose(result.dynamic_state_se3.velocity_3d.array, [10.0, 0.0, 0.0], atol=1e-9)

    def test_overwrites_existing_dynamics(self, tmp_path: Path):
        """Even if the parser provided dynamics, the writer replaces them when inference is on."""
        dt = 0.1
        provided = DynamicStateSE3(
            velocity=Vector3D(999.0, 999.0, 999.0),
            acceleration=Vector3D(999.0, 999.0, 999.0),
            angular_velocity=Vector3D(999.0, 999.0, 999.0),
        )
        egos = [
            _make_ego(
                ts_us=int(i * dt * 1e6),
                pose=PoseSE3(x=float(i), y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
                dynamics=provided,
            )
            for i in range(3)
        ]

        results = _write_and_read_all(tmp_path, egos)
        for result in results:
            assert result.dynamic_state_se3 is not None
            # Should now reflect actual motion (vx = 10 m/s), not the placeholder 999.
            np.testing.assert_allclose(result.dynamic_state_se3.velocity_3d.array, [10.0, 0.0, 0.0], atol=1e-9)

    def test_inference_disabled_passthrough(self, tmp_path: Path):
        """With the flag off, original dynamics survive unchanged."""
        provided = DynamicStateSE3(
            velocity=Vector3D(7.0, 0.0, 0.0),
            acceleration=Vector3D(0.0, 0.0, 0.0),
            angular_velocity=Vector3D(0.0, 0.0, 0.0),
        )
        ego = _make_ego(
            ts_us=1000,
            pose=PoseSE3(x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
            dynamics=provided,
        )

        metadata = make_ego_metadata()
        writer = ArrowEgoStateSE3Writer(log_dir=tmp_path, metadata=metadata, infer_ego_dynamics=False)
        writer.write_modality(ego)
        writer.close()

        table = pa.ipc.open_file(str(tmp_path / f"{metadata.modality_key}.arrow")).read_all()
        result = ArrowEgoStateSE3Reader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None and result.dynamic_state_se3 is not None
        np.testing.assert_allclose(result.dynamic_state_se3.velocity_3d.array, [7.0, 0.0, 0.0])
