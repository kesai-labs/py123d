"""The decoder reads the frozen format-1 fixtures (fixtures/make_kedec_v1_fixtures.py) to the same values."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from py123d.common.io.lidar.kedec_lidar_io import load_point_cloud_from_kedec_binary, read_kedec_head

FIXTURES = Path(__file__).parent / "fixtures"
CASES = ["spherical", "spherical_ring_pose", "grid", "merged"]


@pytest.mark.parametrize("case", CASES)
def test_fixture_decodes_to_frozen_values(case):
    blob = (FIXTURES / f"kedec_v1_{case}.kedec").read_bytes()
    expected = np.load(FIXTURES / f"kedec_v1_{case}.npz")
    xyz, features = load_point_cloud_from_kedec_binary(blob)
    # positions: equal up to float rounding of the trigonometric functions on other platforms (0.001 mm)
    np.testing.assert_allclose(xyz, expected["xyz"], rtol=0, atol=1e-6)
    assert xyz.dtype == np.float32
    assert features is not None
    assert sorted(features) == sorted(name for name in expected.files if name != "xyz")
    for name, values in features.items():
        assert values.dtype == expected[name].dtype, name
        assert values.tobytes() == expected[name].tobytes(), name


def test_fixture_modes():
    modes = {
        case: [s["mode"] for s in read_kedec_head((FIXTURES / f"kedec_v1_{case}.kedec").read_bytes())[0]["sections"]]
        for case in CASES
    }
    assert modes == {
        "spherical": ["spherical"],
        "spherical_ring_pose": ["spherical"],
        "grid": ["grid"],
        "merged": ["spherical", "grid"],
    }
    merged, _ = read_kedec_head((FIXTURES / "kedec_v1_merged.kedec").read_bytes())
    assert merged["dropped"] == 1 and merged["split"] == "ids"
