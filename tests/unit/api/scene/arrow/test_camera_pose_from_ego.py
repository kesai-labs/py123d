"""A camera pose stored as null must be composed from the ego trajectory on read.

``camera_to_global_se3`` is ego_pose composed with the camera extrinsic, and it is the only
per-frame field a change to the ego trajectory invalidates. Storing it as a null and composing it
on read is what lets a re-estimated trajectory reach the camera poses by rewriting
``ego_state_se3.arrow`` alone instead of every image table -- megabytes rather than tens of
gigabytes. These tests pin both halves of that: the composed pose has to equal what eager
composition would have written, and a log that carries the column has to keep reading its own
stored value untouched.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa

from py123d.api.scene.arrow.modalities.arrow_camera import ArrowCameraReader, ArrowCameraWriter
from py123d.api.scene.arrow.modalities.arrow_ego_state_se3 import ArrowEgoStateSE3Writer
from py123d.datatypes import Timestamp
from py123d.datatypes.sensors.base_camera import Camera, CameraID
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraMetadata, PinholeDistortion, PinholeIntrinsics
from py123d.datatypes.vehicle_state.dynamic_state import DynamicStateSE3
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.datatypes.vehicle_state.ego_state_metadata import EgoStateSE3Metadata
from py123d.geometry.pose import PoseSE3
from py123d.geometry.transform.transform_se3 import rel_to_abs_se3
from py123d.geometry.vector import Vector3D
from py123d.parser.base_dataset_parser import ParsedCamera

_CAMERA_TO_IMU = PoseSE3(x=1.5, y=-0.2, z=1.4, qw=0.5, qx=-0.5, qy=0.5, qz=-0.5)
_RATE_US = 10_000  # 100 Hz ego states
_COUNT = 50


def _camera_metadata() -> PinholeCameraMetadata:
    return PinholeCameraMetadata(
        camera_name="front_camera",
        camera_id=CameraID.PCAM_F0,
        intrinsics=PinholeIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0),
        distortion=PinholeDistortion(k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0),
        width=64,
        height=48,
        camera_to_imu_se3=_CAMERA_TO_IMU,
    )


def _ego_metadata() -> EgoStateSE3Metadata:
    return EgoStateSE3Metadata(
        vehicle_name="test",
        width=1.8,
        length=4.5,
        height=1.5,
        wheel_base=2.7,
        center_to_imu_se3=PoseSE3.identity(),
        rear_axle_to_imu_se3=PoseSE3.identity(),
    )


def _ego_pose(index: int) -> PoseSE3:
    """A gentle turn, so interpolation of both translation and rotation is exercised."""
    yaw = 0.004 * index
    return PoseSE3(
        x=2.0 * index,
        y=0.05 * index * index,
        z=0.01 * index,
        qw=float(np.cos(yaw / 2.0)),
        qx=0.0,
        qy=0.0,
        qz=float(np.sin(yaw / 2.0)),
    )


def _write_ego(log_dir: Path) -> None:
    writer = ArrowEgoStateSE3Writer(log_dir=log_dir, metadata=_ego_metadata())
    for index in range(_COUNT):
        writer.write_modality(
            EgoStateSE3.from_imu(
                imu_se3=_ego_pose(index),
                metadata=_ego_metadata(),
                timestamp=Timestamp.from_us(index * _RATE_US),
                dynamic_state_se3=DynamicStateSE3(
                    velocity=Vector3D(0.0, 0.0, 0.0),
                    acceleration=Vector3D(0.0, 0.0, 0.0),
                    angular_velocity=Vector3D(0.0, 0.0, 0.0),
                ),
            )
        )
    writer.close()


def _write_cameras(log_dir: Path, stamps_us: list, pose: bool) -> pa.Table:
    metadata = _camera_metadata()
    writer = ArrowCameraWriter(log_dir=log_dir, metadata=metadata, camera_codec="jpeg_binary")
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    for stamp in stamps_us:
        writer.write_modality(
            Camera(
                metadata=metadata,
                image=image,
                camera_to_global_se3=rel_to_abs_se3(origin=_ego_pose(stamp // _RATE_US), pose_se3=_CAMERA_TO_IMU)
                if pose
                else None,
                timestamp=Timestamp.from_us(stamp),
            )
        )
    writer.close()
    return pa.ipc.open_file(str(log_dir / f"{metadata.modality_key}.arrow")).read_all()


def test_null_pose_is_composed_from_the_ego_trajectory(tmp_path: Path) -> None:
    """On a sample the ego states land on, the composed pose is the eager one exactly."""
    _write_ego(tmp_path)
    stamps = [index * _RATE_US for index in (0, 7, 23, _COUNT - 1)]
    table = _write_cameras(tmp_path, stamps, pose=False)

    assert table[f"{_camera_metadata().modality_key}.camera_to_global_se3"].null_count == len(stamps)
    for row, stamp in enumerate(stamps):
        camera = ArrowCameraReader.read_at_index(row, table, _camera_metadata(), "test", log_dir=tmp_path)
        assert camera is not None, "a null pose column made the frame unreadable"
        expected = rel_to_abs_se3(origin=_ego_pose(stamp // _RATE_US), pose_se3=_CAMERA_TO_IMU)
        assert np.allclose(camera.camera_to_global_se3.tolist(), expected.tolist(), atol=1e-9)


def test_a_frame_between_ego_samples_is_interpolated(tmp_path: Path) -> None:
    """Snapping to the nearest 100 Hz state would displace the camera by up to 5 ms of travel."""
    _write_ego(tmp_path)
    midpoint = 10 * _RATE_US + _RATE_US // 2
    table = _write_cameras(tmp_path, [midpoint], pose=False)

    camera = ArrowCameraReader.read_at_index(0, table, _camera_metadata(), "test", log_dir=tmp_path)
    assert camera is not None
    composed = np.array(camera.camera_to_global_se3.tolist())
    lower = np.array(rel_to_abs_se3(origin=_ego_pose(10), pose_se3=_CAMERA_TO_IMU).tolist())
    upper = np.array(rel_to_abs_se3(origin=_ego_pose(11), pose_se3=_CAMERA_TO_IMU).tolist())
    assert np.allclose(composed[:3], 0.5 * (lower[:3] + upper[:3]), atol=1e-9), "translation was not interpolated"
    assert not np.allclose(composed[:3], lower[:3]), "the pose snapped to the nearest ego state"


def test_a_stored_pose_is_never_recomputed(tmp_path: Path) -> None:
    """Backward compatibility: a log that wrote the column reads its own value, not a composed one.

    The stored pose here is deliberately inconsistent with the ego trajectory in the same
    directory, so anything recomputing it would be caught.
    """
    _write_ego(tmp_path)
    stored = PoseSE3(x=999.0, y=-42.0, z=7.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
    metadata = _camera_metadata()
    writer = ArrowCameraWriter(log_dir=tmp_path, metadata=metadata, camera_codec="jpeg_binary")
    writer.write_modality(
        Camera(
            metadata=metadata,
            image=np.zeros((48, 64, 3), dtype=np.uint8),
            camera_to_global_se3=stored,
            timestamp=Timestamp.from_us(3 * _RATE_US),
        )
    )
    writer.close()
    table = pa.ipc.open_file(str(tmp_path / f"{metadata.modality_key}.arrow")).read_all()

    camera = ArrowCameraReader.read_at_index(0, table, metadata, "test", log_dir=tmp_path)
    assert camera is not None
    assert np.allclose(camera.camera_to_global_se3.tolist(), stored.tolist())


def test_the_column_reader_composes_too(tmp_path: Path) -> None:
    """read_column_at_index is a separate path and has to give the same answer."""
    _write_ego(tmp_path)
    table = _write_cameras(tmp_path, [5 * _RATE_US], pose=False)

    pose = ArrowCameraReader.read_column_at_index(
        0, table, _camera_metadata(), "camera_to_global_se3", "test", deserialize=True, log_dir=tmp_path
    )
    assert pose is not None, "the column reader returned nothing for an implicitly stored pose"
    expected = rel_to_abs_se3(origin=_ego_pose(5), pose_se3=_CAMERA_TO_IMU)
    assert np.allclose(pose.tolist(), expected.tolist(), atol=1e-9)


def test_a_null_pose_without_an_ego_trajectory_reads_as_absent(tmp_path: Path) -> None:
    """No ego states means the pose genuinely cannot be known; the frame is reported as absent."""
    table = _write_cameras(tmp_path, [0], pose=False)
    assert ArrowCameraReader.read_at_index(0, table, _camera_metadata(), "test", log_dir=tmp_path) is None


def test_a_parsed_camera_accepts_no_pose() -> None:
    """The writer-side type has to allow it, or a parser cannot express an implicit pose."""
    camera = ParsedCamera(
        metadata=_camera_metadata(),
        timestamp=Timestamp.from_us(0),
        camera_to_global_se3=None,
        byte_string=b"not-an-image",
    )
    assert camera.camera_to_global_se3 is None
