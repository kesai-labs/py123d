from __future__ import annotations

import numpy as np
import numpy.typing as npt


class Timestamp:
    """Timestamp class representing a time point in microseconds."""

    __slots__ = ("_time_us",)
    _time_us: int  # [micro seconds] time since epoch in micro seconds

    @classmethod
    def from_ns(cls, t_ns: int) -> Timestamp:
        """Constructs a Timestamp from a value in nanoseconds.

        :param t_ns: Time in nanoseconds.
        :return: Timestamp.
        """
        assert isinstance(t_ns, (int, np.integer)), "Nanoseconds must be an integer!"
        instance = object.__new__(cls)
        setattr(instance, "_time_us", t_ns // 1000)
        return instance

    @classmethod
    def from_us(cls, t_us: int) -> Timestamp:
        """Constructs a Timestamp from a value in microseconds.

        :param t_us: Time in microseconds.
        :return: Timestamp.
        """
        assert isinstance(t_us, (int, np.integer)), f"Microseconds must be an integer, got {type(t_us)}!"
        instance = object.__new__(cls)
        setattr(instance, "_time_us", t_us)
        return instance

    @classmethod
    def from_ms(cls, t_ms: float) -> Timestamp:
        instance = object.__new__(cls)
        setattr(instance, "_time_us", int(t_ms * 1000))
        return instance

    @classmethod
    def from_s(cls, t_s: float) -> Timestamp:
        """Constructs a Timestamp from a value in seconds.

        :param t_s: Time in seconds.
        :return: Timestamp.
        """
        instance = object.__new__(cls)
        setattr(instance, "_time_us", int(t_s * 1_000_000))
        return instance

    @property
    def time_ns(self) -> int:
        """The timestamp in nanoseconds [ns]."""
        return self._time_us * 1000

    @property
    def time_us(self) -> int:
        """The timestamp in microseconds [μs]."""
        return self._time_us

    @property
    def time_ms(self) -> float:
        """The timestamp in milliseconds [ms]."""
        return self._time_us / 1e3

    @property
    def time_s(self) -> float:
        """The timestamp in seconds [s]."""
        return self._time_us / 1e6

    def __eq__(self, other: object) -> bool:
        """Compare two Timestamp objects for equality.

        :param other: The other object to compare with.
        :return: True if both timestamps represent the same time point.
        """
        if not isinstance(other, Timestamp):
            return NotImplemented
        return self._time_us == other._time_us

    def __hash__(self) -> int:
        """Hash of the Timestamp based on its microsecond value."""
        return hash(self._time_us)

    def __int__(self) -> int:
        """Return the timestamp as an integer in microseconds."""
        return self._time_us

    def __array__(self, dtype: npt.DTypeLike = np.int64, copy: bool = False) -> npt.NDArray:  # noqa: PLW3201
        """Allow numpy conversion, enabling ``np.array([ts1, ts2, ...])`` to return an int64 array."""
        return np.asarray(self._time_us, dtype=dtype)

    def __repr__(self):
        """String representation of :class:`Timestamp`."""
        return f"Timestamp(time_us={self._time_us})"
