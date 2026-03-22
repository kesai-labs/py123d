"""Tests for SceneMetadata — properties, validation, repr."""

from __future__ import annotations

import logging

import pytest

from py123d.datatypes.metadata import SceneMetadata


def _make(**kwargs) -> SceneMetadata:
    defaults = dict(
        dataset="test-dataset",
        split="test-dataset_train",
        initial_uuid="00000000-0000-0000-0000-000000000001",
        initial_idx=0,
        num_future_iterations=9,
        num_history_iterations=0,
        future_duration_s=0.9,
        history_duration_s=0.0,
        iteration_duration_s=0.1,
        target_iteration_stride=1,
    )
    defaults.update(kwargs)
    return SceneMetadata(**defaults)


class TestEndIdx:
    def test_stride_1(self):
        meta = _make(initial_idx=5, num_future_iterations=10, target_iteration_stride=1)
        assert meta.end_idx == 16  # 5 + 10*1 + 1

    def test_stride_5(self):
        meta = _make(initial_idx=10, num_future_iterations=4, target_iteration_stride=5)
        assert meta.end_idx == 31  # 10 + 4*5 + 1


class TestTotalIterations:
    def test_basic(self):
        meta = _make(num_history_iterations=2, num_future_iterations=3)
        assert meta.total_iterations == 6  # 2 + 1 + 3

    def test_no_history(self):
        meta = _make(num_history_iterations=0, num_future_iterations=5)
        assert meta.total_iterations == 6  # 0 + 1 + 5


class TestPostInit:
    def test_duration_mismatch_logs_warning(self, caplog):
        """Duration/iteration mismatch should log at WARNING level."""
        with caplog.at_level(logging.WARNING):
            _make(
                num_future_iterations=5,
                future_duration_s=1.0,  # round(1.0/0.1)=10, but iterations=5
                iteration_duration_s=0.1,
            )
        assert "num_future_iterations=5" in caplog.text

    def test_zero_duration_skips_check(self, caplog):
        """iteration_duration_s=0 should skip consistency check (no division by zero)."""
        with caplog.at_level(logging.WARNING):
            _make(iteration_duration_s=0.0)
        assert "num_future_iterations" not in caplog.text

    def test_consistent_no_log(self, caplog):
        """Consistent values should produce no warning."""
        with caplog.at_level(logging.WARNING):
            _make(
                num_future_iterations=10,
                future_duration_s=1.0,
                iteration_duration_s=0.1,
            )
        scene_logs = [r for r in caplog.records if "SceneMetadata" in r.message]
        assert len(scene_logs) == 0


class TestRepr:
    def test_stride_1_omitted(self):
        meta = _make(target_iteration_stride=1)
        r = repr(meta)
        assert "target_iteration_stride" not in r

    def test_stride_gt_1_included(self):
        meta = _make(target_iteration_stride=5)
        r = repr(meta)
        assert "target_iteration_stride=5" in r


class TestFrozen:
    def test_immutable(self):
        meta = _make()
        with pytest.raises(AttributeError):
            meta.dataset = "changed"
