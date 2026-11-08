"""Tests of the camera frame selection: nearest frames, the report's hysteresis rule, limits and grids."""

from __future__ import annotations

import numpy as np
import pytest

from py123d.common.frame_selection import (
    FrameSelectionConfig,
    frames_per_reference,
    kept_frames,
    match_frames,
    nearest_indices,
    select_nearest_frames,
    uniform_reference_grid,
)


def _report_select(ts, ref, hyst_us, step):
    """The rule of the report (sync_analysis.select), with the frame step as a parameter (2 in the report)."""
    j = np.clip(np.searchsorted(ts, ref), 1, len(ts) - 1)
    near = np.where(np.abs(ts[j] - ref) < np.abs(ts[j - 1] - ref), j, j - 1)
    if not hyst_us:
        return near
    out = near.copy()
    for k in range(1, len(ref)):
        cand = out[k - 1] + step
        if cand < len(ts) and near[k] != cand and abs(ts[near[k]] - ref[k]) + hyst_us >= abs(ts[cand] - ref[k]):
            out[k] = cand
    return out


def _camera(rng, rate_hz, seconds=60.0, drift_ppm=0.0, jitter_us=200.0, drop=0.0, t0=1_784_208_918_000_000):
    period = 1e6 / rate_hz * (1 + drift_ppm * 1e-6)
    n = int(seconds * rate_hz)
    ts = t0 + rng.uniform(0, period) + np.arange(n) * period + rng.normal(0, jitter_us, n)
    ts = np.sort(ts.astype(np.int64))
    if drop:
        ts = ts[rng.random(n) >= drop]
    return ts


class TestNearest:
    def test_ties_go_to_the_earlier_frame(self):
        ts = np.array([0, 100, 200])
        np.testing.assert_array_equal(nearest_indices(ts, [50, 150, 149, 151, -10, 999]), [0, 1, 1, 2, 0, 2])

    def test_single_and_empty(self):
        np.testing.assert_array_equal(nearest_indices([5], [0, 10]), [0, 0])
        with pytest.raises(ValueError):
            nearest_indices([], [0])


class TestSelection:
    @pytest.mark.parametrize("rate_hz", [20, 30, 60])
    def test_equals_the_report_rule(self, rate_hz):
        """200 random cameras with drift, jitter and dropped frames: the same frames as the report's rule."""
        changed = 0
        for seed in range(200):
            rng = np.random.default_rng(seed)
            drift = rng.uniform(-3000, 3000)  # up to 60 ms in 20 s: the selection passes several switch points
            ts = _camera(
                rng, rate_hz, seconds=20, drift_ppm=drift, jitter_us=rng.uniform(50, 2000), drop=rng.uniform(0, 0.02)
            )
            ref = uniform_reference_grid(
                int(ts[0]) + 100_000, int(ts[-1]) - 100_000, 100_000, int(rng.integers(0, 10**5))
            )
            step = rate_hz // 10
            for hyst in (0, 500):
                expected = _report_select(ts, ref, hyst, step)
                np.testing.assert_array_equal(select_nearest_frames(ts, ref, hyst, step=step), expected)
                np.testing.assert_array_equal(select_nearest_frames(ts, ref, hyst), expected)  # step estimated
            changed += int((_report_select(ts, ref, 0, step) != _report_select(ts, ref, 500, step)).any())
        assert changed > 50  # the test passes switch points: the hysteresis changes the kept frames of many cameras

    def test_kept_offset_at_most_nearest_plus_hysteresis(self):
        rng = np.random.default_rng(1)
        ts = _camera(rng, 20, drift_ppm=150, drop=0.01)
        ref = uniform_reference_grid(int(ts[0]) + 100_000, int(ts[-1]) - 100_000, 100_000, 31_000)
        kept = select_nearest_frames(ts, ref, 500)
        near = nearest_indices(ts, ref)
        assert np.all(np.abs(ts[kept] - ref) <= np.abs(ts[near] - ref) + 500)

    def test_hysteresis_removes_alternation_at_a_switch(self):
        """A camera drifting against the grid passes the midpoint between two frames once: the nearest rule
        alternates there (frame jitter), the hysteresis rule switches once."""
        rng = np.random.default_rng(2)
        ts = _camera(rng, 20, seconds=120, drift_ppm=60, jitter_us=150)
        ref = uniform_reference_grid(
            int(ts[0]) + 100_000, int(ts[-1]) - 100_000, 100_000, int(ts[0] % 100_000) + 25_000
        )
        irregular_nearest = int(np.sum(np.diff(select_nearest_frames(ts, ref, 0)) != 2))
        irregular_hysteresis = int(np.sum(np.diff(select_nearest_frames(ts, ref, 500)) != 2))
        assert irregular_hysteresis <= 1 < irregular_nearest

    def test_limit_marks_slots_without_a_frame(self):
        ts = np.array([0, 50_000, 100_000, 150_000, 400_000, 450_000])
        ref = np.array([0, 100_000, 200_000, 300_000, 400_000])
        out = select_nearest_frames(ts, ref, 500, step=2, limit_us=25_000)
        np.testing.assert_array_equal(out, [0, 2, -1, -1, 4])
        assert select_nearest_frames(ts, ref, 0, limit_us=25_000)[2] == -1

    def test_hysteresis_candidate_beyond_the_limit_is_not_kept(self):
        ts = np.array([0, 50_000, 100_000, 140_000, 190_000])
        ref = np.array([0, 100_000, 180_000])
        # the candidate after frame 2 is frame 4 (190 ms, 10 ms off); with a 5 ms limit the nearest is used instead
        np.testing.assert_array_equal(select_nearest_frames(ts, ref, 20_000, step=2, limit_us=5_000), [0, 2, -1])

    def test_edges(self):
        assert len(select_nearest_frames([1, 2], [])) == 0
        np.testing.assert_array_equal(select_nearest_frames([], [1, 2]), [-1, -1])
        np.testing.assert_array_equal(kept_frames([3, -1, 1, 3, 2]), [1, 2, 3])
        assert frames_per_reference(np.arange(0, 10**6, 33_333), np.arange(0, 10**6, 100_000)) == 3
        assert frames_per_reference([0], [0, 1]) == 1


class TestMatchAndGrid:
    def test_match_is_strict(self):
        ts = np.array([0, 50_000, 100_000])
        np.testing.assert_array_equal(match_frames(ts, [4_999, 5_000, 95_001, 300_000], 5_000), [0, -1, 2, -1])
        np.testing.assert_array_equal(match_frames([], [1], 5_000), [-1])

    def test_grid(self):
        np.testing.assert_array_equal(uniform_reference_grid(1_050, 1_400, 100, 30), [1_130, 1_230, 1_330])
        np.testing.assert_array_equal(uniform_reference_grid(1_030, 1_131, 100, 30), [1_030, 1_130])
        with pytest.raises(ValueError):
            uniform_reference_grid(0, 10, 0)

    def test_config(self):
        assert FrameSelectionConfig().mode == "all"
        with pytest.raises(AssertionError):
            FrameSelectionConfig(mode="even")
        with pytest.raises(AssertionError):
            FrameSelectionConfig(period_us=0)
