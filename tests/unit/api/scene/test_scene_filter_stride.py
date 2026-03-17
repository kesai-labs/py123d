import logging

import pytest

from py123d.api.scene.scene_filter import SceneFilter


class TestSceneFilterStrideValidation:
    def test_target_iteration_stride_valid(self):
        f = SceneFilter(target_iteration_stride=5)
        assert f.target_iteration_stride == 5

    def test_target_iteration_stride_one(self):
        f = SceneFilter(target_iteration_stride=1)
        assert f.target_iteration_stride == 1

    def test_target_iteration_stride_zero_raises(self):
        with pytest.raises(ValueError, match="target_iteration_stride must be >= 1"):
            SceneFilter(target_iteration_stride=0)

    def test_target_iteration_stride_negative_raises(self):
        with pytest.raises(ValueError, match="target_iteration_stride must be >= 1"):
            SceneFilter(target_iteration_stride=-3)

    def test_target_duration_valid(self):
        f = SceneFilter(target_iteration_duration_s=0.5)
        assert f.target_iteration_duration_s == 0.5

    def test_target_duration_zero_raises(self):
        with pytest.raises(ValueError, match="target_iteration_duration_s must be > 0"):
            SceneFilter(target_iteration_duration_s=0.0)

    def test_target_duration_negative_raises(self):
        with pytest.raises(ValueError, match="target_iteration_duration_s must be > 0"):
            SceneFilter(target_iteration_duration_s=-0.5)

    def test_both_set_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            f = SceneFilter(target_iteration_stride=5, target_iteration_duration_s=0.5)
        assert "target_iteration_duration_s takes priority" in caplog.text
        assert f.target_iteration_stride == 5
        assert f.target_iteration_duration_s == 0.5
