"""Tests of the lossless integer and byte channels of kedec."""

from __future__ import annotations

import numpy as np
import pytest

from py123d.common.io.lidar.kedec.int_coding import (
    INT_CHOICES,
    decode_bytes_channel,
    decode_int_channel,
    encode_bytes_channel,
    encode_int_channel,
    planes,
    unplanes,
)

INTEGER_DTYPES = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]


def _random_values(rng, dtype, n):
    if dtype == np.bool_:
        return rng.integers(0, 2, n).astype(bool)
    info = np.iinfo(dtype)
    return rng.integers(info.min, info.max, n, dtype=dtype, endpoint=True)


@pytest.mark.parametrize("dtype", INTEGER_DTYPES + [np.bool_])
@pytest.mark.parametrize("choice", INT_CHOICES)
@pytest.mark.parametrize("n", [0, 1, 2, 1000])
def test_roundtrip_any_integer(dtype, choice, n):
    values = _random_values(np.random.default_rng(n), dtype, n)
    descriptor, compressed = encode_int_channel(values, 19, choice)
    decoded = decode_int_channel(descriptor, compressed, n, values.dtype)
    assert decoded.dtype == values.dtype
    np.testing.assert_array_equal(decoded, values)


@pytest.mark.parametrize("choice", INT_CHOICES)
def test_extremes(choice):
    for values in (
        np.array([np.iinfo(np.int64).min, np.iinfo(np.int64).max, 0, -1, np.iinfo(np.int64).min], np.int64),
        np.array([0, np.iinfo(np.uint64).max, 5, 2**63, 2**63 - 1], np.uint64),
    ):
        descriptor, compressed = encode_int_channel(values, 3, choice)
        np.testing.assert_array_equal(decode_int_channel(descriptor, compressed, len(values), values.dtype), values)


def test_constant_channel_stores_no_bytes():
    descriptor, compressed = encode_int_channel(np.full(1000, -7, np.int64), 19)
    assert descriptor["w"] == 0 and compressed == b""
    np.testing.assert_array_equal(decode_int_channel(descriptor, compressed, 1000, np.int64), np.full(1000, -7))


def test_width_is_minimal():
    values = np.array([1000, 1000 + 255], np.int64)
    descriptor, _ = encode_int_channel(values, 3, "raw")
    assert descriptor["w"] == 1
    descriptor, _ = encode_int_channel(np.array([0, 256], np.int64), 3, "raw")
    assert descriptor["w"] == 2


def test_choice_prefers_differences_for_sorted_times():
    times = 1_784_208_918_099_704 + np.cumsum(np.random.default_rng(0).integers(0, 3, 100_000))
    raw = encode_int_channel(times, 19, "raw")[1]
    estimate = encode_int_channel(times, 19, "estimate")
    exact = encode_int_channel(times, 19, "exact")
    assert estimate[0]["d"] == 1 and exact[0]["d"] == 1
    assert len(estimate[1]) < len(raw) and len(exact[1]) <= len(estimate[1])


def test_invalid_inputs():
    with pytest.raises(ValueError):
        encode_int_channel(np.zeros(3, np.float32), 3)
    with pytest.raises(ValueError):
        encode_int_channel(np.zeros((3, 2), np.int64), 3)
    with pytest.raises(ValueError):
        encode_int_channel(np.zeros(3, np.int64), 3, "zigzag")
    descriptor, compressed = encode_int_channel(np.arange(10), 3)
    with pytest.raises(ValueError):
        decode_int_channel(dict(descriptor, w=9), compressed, 10, np.int64)
    with pytest.raises(ValueError):
        decode_int_channel(descriptor, compressed, 11, np.int64)


def test_planes_roundtrip():
    values = np.random.default_rng(1).integers(0, 2**40, 100).astype(np.uint64)
    np.testing.assert_array_equal(unplanes(planes(values, 5), 100, 5), values)
    assert planes(values, 0) == b""


@pytest.mark.parametrize(
    "values",
    [
        np.random.default_rng(2).normal(size=100).astype(np.float32),
        np.random.default_rng(3).normal(size=(100, 3)).astype(np.float64),
        np.array([np.nan, np.inf, -0.0, 1e-45], np.float32),
        np.zeros((0, 2), np.float16),
    ],
)
def test_bytes_channel_is_bit_exact(values):
    descriptor, compressed = encode_bytes_channel(values, 19)
    decoded = decode_bytes_channel(descriptor, compressed, values.shape[0], values.dtype, values.shape[1:])
    assert decoded.dtype == values.dtype and decoded.shape == values.shape
    assert decoded.tobytes() == values.tobytes()


def test_bytes_channel_rejects_objects_and_bad_descriptors():
    with pytest.raises(ValueError):
        encode_bytes_channel(np.array(["a", "b"], dtype=object), 3)
    values = np.ones(10, np.float32)
    descriptor, compressed = encode_bytes_channel(values, 3)
    with pytest.raises(ValueError):
        decode_bytes_channel(descriptor, compressed, 10, np.float64, ())
