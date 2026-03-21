import time

import pandas as pd
import pytest

from py123d.common.utils.timer import Timer


class TestTimerBasicFlow:
    """Tests for the basic start/log/end flow of Timer."""

    def test_start_log_end(self):
        """Test the basic timing workflow."""
        timer = Timer()
        timer.start()
        time.sleep(0.01)
        timer.log("block_a")
        time.sleep(0.01)
        timer.log("block_b")
        timer.end()

        info = timer.info()
        assert "block_a" in info
        assert "block_b" in info
        assert "total" in info

    def test_logged_times_are_positive(self):
        """Test that all logged times are positive."""
        timer = Timer()
        timer.start()
        timer.log("block")
        timer.end()

        info = timer.info()
        assert info["block"]["mean"] > 0
        assert info["total"]["mean"] > 0

    def test_custom_end_key(self):
        """Test that a custom end_key is used instead of 'total'."""
        timer = Timer(end_key="elapsed")
        timer.start()
        timer.log("step")
        timer.end()

        info = timer.info()
        assert "elapsed" in info
        assert "total" not in info


class TestTimerLog:
    """Tests for the Timer.log method."""

    def test_same_key_accumulates(self):
        """Test that logging the same key multiple times accumulates entries."""
        timer = Timer()
        timer.start()
        timer.log("repeated")
        timer.log("repeated")
        timer.log("repeated")
        timer.end()

        info = timer.info()
        assert "repeated" in info

    def test_log_without_start_raises_assertion(self):
        """Test that log() raises AssertionError if start() was not called."""
        timer = Timer()
        with pytest.raises(AssertionError, match="Timer has not been started"):
            timer.log("block")

    def test_end_without_start_raises_assertion(self):
        """Test that end() raises AssertionError if start() was not called."""
        timer = Timer()
        with pytest.raises(AssertionError, match="Timer has not been started"):
            timer.end()


class TestTimerToPandas:
    """Tests for the Timer.to_pandas method."""

    def test_returns_dataframe(self):
        """Test that to_pandas returns a pandas DataFrame."""
        timer = Timer()
        timer.start()
        timer.log("step")
        timer.end()

        df = timer.to_pandas()
        assert isinstance(df, pd.DataFrame)

    def test_dataframe_has_expected_columns(self):
        """Test that the DataFrame contains the expected statistic columns."""
        timer = Timer()
        timer.start()
        timer.log("step")
        timer.end()

        df = timer.to_pandas()
        for col in ["mean", "min", "max", "argmax", "median"]:
            assert col in df.columns

    def test_dataframe_has_expected_rows(self):
        """Test that the DataFrame rows correspond to logged keys."""
        timer = Timer()
        timer.start()
        timer.log("alpha")
        timer.log("beta")
        timer.end()

        df = timer.to_pandas()
        assert "alpha" in df.index
        assert "beta" in df.index
        assert "total" in df.index

    def test_empty_timer_returns_empty_dataframe(self):
        """Test that to_pandas on an unused timer returns an empty DataFrame."""
        timer = Timer()
        df = timer.to_pandas()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestTimerInfo:
    """Tests for the Timer.info method."""

    def test_info_returns_dict(self):
        """Test that info returns a dictionary."""
        timer = Timer()
        timer.start()
        timer.log("step")
        timer.end()

        info = timer.info()
        assert isinstance(info, dict)

    def test_info_keys_match_logged_keys(self):
        """Test that info keys match the logged block names."""
        timer = Timer()
        timer.start()
        timer.log("x")
        timer.log("y")
        timer.end()

        info = timer.info()
        assert set(info.keys()) == {"x", "y", "total"}

    def test_info_contains_statistics(self):
        """Test that each info entry has the expected statistic keys."""
        timer = Timer()
        timer.start()
        timer.log("block")
        timer.end()

        block_info = timer.info()["block"]
        for stat in ["mean", "min", "max", "argmax", "median"]:
            assert stat in block_info


class TestTimerFlush:
    """Tests for the Timer.flush method."""

    def test_flush_clears_state(self):
        """Test that flush clears all logged times."""
        timer = Timer()
        timer.start()
        timer.log("step")
        timer.end()

        timer.flush()
        info = timer.info()
        assert len(info) == 0

    def test_flush_resets_start_time(self):
        """Test that flush resets the start time, requiring a new start()."""
        timer = Timer()
        timer.start()
        timer.log("step")
        timer.flush()

        with pytest.raises(AssertionError, match="Timer has not been started"):
            timer.log("after_flush")


class TestTimerStr:
    """Tests for the Timer string representation."""

    def test_str_with_data(self):
        """Test that __str__ returns a string when data is logged."""
        timer = Timer()
        timer.start()
        timer.log("block")
        timer.end()

        result = str(timer)
        assert "block" in result
        assert "total" in result

    def test_str_without_data(self):
        """Test that __str__ returns a non-empty string even when no data is logged."""
        timer = Timer()
        result = str(timer)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_repr_equals_str(self):
        """Test that __repr__ returns the same as __str__."""
        timer = Timer()
        assert repr(timer) == str(timer)
