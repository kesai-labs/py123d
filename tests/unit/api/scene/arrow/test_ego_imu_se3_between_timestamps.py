"""Tests for SceneAPI.get_ego_imu_se3_between_timestamps: the Arrow whole-column read against the generic default."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from py123d.api.scene.arrow.arrow_scene_api import ArrowSceneAPI
from py123d.api.scene.scene_api import SceneAPI
from py123d.api.utils.arrow_metadata_utils import add_metadata_to_arrow_schema
from py123d.datatypes import Timestamp
from py123d.datatypes.metadata import SceneMetadata
from py123d.datatypes.vehicle_state.dynamic_state import DynamicStateSE3Index
from py123d.geometry import PoseSE3Index

from ..conftest import make_populated_log_dir

_NUM_ROWS = 10
_TIMESTEP_US = 100_000


def _distinct_imu_se3_array(num_rows: int) -> np.ndarray:
    """Poses that differ per row, so a shifted or reordered read cannot match."""
    rows = np.arange(num_rows, dtype=np.float64)
    angles = 0.1 * rows
    imu_se3_array = np.zeros((num_rows, len(PoseSE3Index)), dtype=np.float64)
    imu_se3_array[:, PoseSE3Index.X] = 1.5 * rows
    imu_se3_array[:, PoseSE3Index.Y] = -0.5 * rows
    imu_se3_array[:, PoseSE3Index.Z] = 0.01 * rows
    imu_se3_array[:, PoseSE3Index.QW] = np.cos(angles / 2)
    imu_se3_array[:, PoseSE3Index.QZ] = np.sin(angles / 2)
    return imu_se3_array


@pytest.fixture
def scene_api(tmp_path: Path) -> ArrowSceneAPI:
    """A log whose ego poses differ per row, written in two record batches."""
    log_dir = tmp_path / "test-dataset_train" / "log_001"
    _, ego_metadata, _, _ = make_populated_log_dir(log_dir, num_rows=_NUM_ROWS, timestep_us=_TIMESTEP_US)
    key = ego_metadata.modality_key
    schema = add_metadata_to_arrow_schema(
        pa.schema(
            [
                (f"{key}.timestamp_us", pa.int64()),
                (f"{key}.imu_se3", pa.list_(pa.float64(), len(PoseSE3Index))),
                (f"{key}.dynamic_state_se3", pa.list_(pa.float64(), len(DynamicStateSE3Index))),
            ]
        ),
        ego_metadata,
    )
    table = pa.table(
        {
            f"{key}.timestamp_us": [row * _TIMESTEP_US for row in range(_NUM_ROWS)],
            f"{key}.imu_se3": _distinct_imu_se3_array(_NUM_ROWS).tolist(),
            f"{key}.dynamic_state_se3": [[0.0] * len(DynamicStateSE3Index)] * _NUM_ROWS,
        },
        schema=schema,
    )
    with pa.OSFile(str(log_dir / f"{key}.arrow"), "wb") as sink:
        writer = pa.ipc.new_file(sink, table.schema)
        for batch in table.to_batches(max_chunksize=_NUM_ROWS // 2 + 1):
            writer.write_batch(batch)
        writer.close()

    scene_metadata = SceneMetadata(
        dataset="test-dataset",
        split="test-dataset_train",
        initial_uuid="00000000-0000-0000-0000-000000000001",
        initial_idx=0,
        num_future_iterations=_NUM_ROWS - 1,
        num_history_iterations=0,
        future_duration_s=(_NUM_ROWS - 1) * _TIMESTEP_US / 1e6,
        history_duration_s=0.0,
        iteration_duration_s=_TIMESTEP_US / 1e6,
        target_iteration_stride=1,
    )
    return ArrowSceneAPI(log_dir, scene_metadata)


@pytest.mark.parametrize("inclusive", ["left", "right", "both", "neither"])
@pytest.mark.parametrize("start_us,end_us", [(0, 900_000), (100_000, 600_000), (250_000, 750_000), (-1, 10**9)])
def test_arrow_read_matches_generic_default(scene_api: ArrowSceneAPI, inclusive, start_us, end_us):
    """The whole-column read returns the same rows as building each ego state, across chunk boundaries."""
    timestamps_us, imu_se3_array = scene_api.get_ego_imu_se3_between_timestamps(start_us, end_us, inclusive)
    expected_timestamps_us, expected_imu_se3_array = SceneAPI.get_ego_imu_se3_between_timestamps(
        scene_api, start_us, end_us, inclusive
    )
    assert timestamps_us.dtype == np.int64 and imu_se3_array.dtype == np.float64
    np.testing.assert_array_equal(timestamps_us, expected_timestamps_us)
    np.testing.assert_array_equal(imu_se3_array, expected_imu_se3_array)


def test_bounds_follow_inclusive(scene_api: ArrowSceneAPI):
    """Both bounds on recorded rows: "both" keeps them, "left" drops the end, "neither" drops both."""
    expected_imu_se3_array = _distinct_imu_se3_array(_NUM_ROWS)
    timestamps_us, imu_se3_array = scene_api.get_ego_imu_se3_between_timestamps(
        Timestamp.from_us(100_000), Timestamp.from_us(300_000), inclusive="both"
    )
    np.testing.assert_array_equal(timestamps_us, [100_000, 200_000, 300_000])
    np.testing.assert_array_equal(imu_se3_array, expected_imu_se3_array[1:4])
    assert len(scene_api.get_ego_imu_se3_between_timestamps(100_000, 300_000, inclusive="left")[0]) == 2
    assert len(scene_api.get_ego_imu_se3_between_timestamps(100_000, 300_000, inclusive="neither")[0]) == 1


def test_empty_and_reversed_ranges(scene_api: ArrowSceneAPI):
    """A range without ego states, or with the end before the start, gives empty arrays of the right shape."""
    for start_us, end_us in [(10**9, 2 * 10**9), (500_000, 100_000)]:
        timestamps_us, imu_se3_array = scene_api.get_ego_imu_se3_between_timestamps(start_us, end_us, "both")
        assert timestamps_us.shape == (0,)
        assert imu_se3_array.shape == (0, len(PoseSE3Index))
