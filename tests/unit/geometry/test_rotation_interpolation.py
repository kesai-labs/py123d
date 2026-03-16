import numpy as np
from pyquaternion import Quaternion as PyQuaternion

from py123d.geometry.utils.rotation_utils import (
    get_quaternion_array_from_euler_array,
    get_rotation_matrix_from_quaternion_array,
    nlerp_quaternion_arrays,
    normalize_quaternion_array,
    slerp_quaternion_arrays,
)


class TestSlerpQuaternionArrays:
    """Tests for spherical linear interpolation of quaternions."""

    def test_endpoints(self):
        """SLERP at t=0 and t=1 returns the input quaternions."""
        q1 = np.array([[1.0, 0.0, 0.0, 0.0]])
        q2 = np.array([[0.0, 1.0, 0.0, 0.0]])  # 180 deg around x
        t0 = np.array([0.0])
        t1 = np.array([1.0])
        np.testing.assert_allclose(slerp_quaternion_arrays(q1, q2, t0), q1, atol=1e-10)
        np.testing.assert_allclose(slerp_quaternion_arrays(q1, q2, t1), q2, atol=1e-10)

    def test_midpoint_90_degrees(self):
        """SLERP midpoint of a 90-degree rotation around Z axis."""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])  # identity
        # 90 deg around z: qw=cos(45deg), qz=sin(45deg)
        q2 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
        t = np.array(0.5)
        result = slerp_quaternion_arrays(q1, q2, t)
        # Expected: 45 deg rotation around z
        expected = normalize_quaternion_array(np.array([np.cos(np.pi / 8), 0.0, 0.0, np.sin(np.pi / 8)]))
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_shortest_path(self):
        """SLERP takes the shortest path when quaternions are in opposite hemispheres."""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = np.array([-1.0, 0.0, 0.0, 0.0])  # same rotation, opposite hemisphere
        t = np.array(0.5)
        result = slerp_quaternion_arrays(q1, q2, t)
        # Should stay at identity (shortest path is zero rotation)
        np.testing.assert_allclose(np.abs(result[0]), 1.0, atol=1e-10)

    def test_nearly_identical_quaternions(self):
        """SLERP falls back to NLERP for nearly identical quaternions."""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = normalize_quaternion_array(np.array([1.0, 1e-8, 0.0, 0.0]))
        t = np.array(0.5)
        result = slerp_quaternion_arrays(q1, q2, t)
        assert np.allclose(np.linalg.norm(result), 1.0)

    def test_batch_interpolation(self):
        """SLERP works with batched inputs."""
        q1 = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )
        q2 = np.array(
            [
                [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)],
                [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0],
            ]
        )
        t = np.array([0.0, 1.0])
        result = slerp_quaternion_arrays(q1, q2, t)
        np.testing.assert_allclose(result[0], q1[0], atol=1e-10)
        np.testing.assert_allclose(result[1], q2[1], atol=1e-10)

    def test_unit_quaternion_output(self):
        """SLERP always returns unit quaternions."""
        rng = np.random.default_rng(42)
        q1 = normalize_quaternion_array(rng.standard_normal((10, 4)))
        q2 = normalize_quaternion_array(rng.standard_normal((10, 4)))
        t = rng.uniform(0, 1, size=10)
        result = slerp_quaternion_arrays(q1, q2, t)
        norms = np.linalg.norm(result, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_valid_rotation_matrices(self):
        """SLERP results convert to valid rotation matrices (orthogonal, det=1)."""
        euler1 = np.array([0.0, 0.0, 0.0])
        euler2 = np.array([np.pi / 6, np.pi / 4, np.pi / 3])
        q1 = get_quaternion_array_from_euler_array(euler1)
        q2 = get_quaternion_array_from_euler_array(euler2)
        for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            t = np.array(t_val)
            q_interp = slerp_quaternion_arrays(q1, q2, t)
            R = get_rotation_matrix_from_quaternion_array(q_interp)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
            np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)


class TestNlerpQuaternionArrays:
    """Tests for normalized linear interpolation of quaternions."""

    def test_endpoints(self):
        """NLERP at t=0 and t=1 returns the input quaternions."""
        q1 = np.array([[1.0, 0.0, 0.0, 0.0]])
        q2 = np.array([[0.0, 1.0, 0.0, 0.0]])
        t0 = np.array([0.0])
        t1 = np.array([1.0])
        np.testing.assert_allclose(nlerp_quaternion_arrays(q1, q2, t0), q1, atol=1e-10)
        np.testing.assert_allclose(nlerp_quaternion_arrays(q1, q2, t1), q2, atol=1e-10)

    def test_unit_quaternion_output(self):
        """NLERP always returns unit quaternions."""
        rng = np.random.default_rng(42)
        q1 = normalize_quaternion_array(rng.standard_normal((10, 4)))
        q2 = normalize_quaternion_array(rng.standard_normal((10, 4)))
        t = rng.uniform(0, 1, size=10)
        result = nlerp_quaternion_arrays(q1, q2, t)
        norms = np.linalg.norm(result, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_shortest_path(self):
        """NLERP takes the shortest path."""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = np.array([-1.0, 0.0, 0.0, 0.0])
        t = np.array(0.5)
        result = nlerp_quaternion_arrays(q1, q2, t)
        np.testing.assert_allclose(np.abs(result[0]), 1.0, atol=1e-10)

    def test_close_to_slerp_for_small_angles(self):
        """NLERP approximates SLERP for small angular differences."""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        # Small rotation (~5.7 degrees around z)
        q2 = normalize_quaternion_array(np.array([1.0, 0.0, 0.0, 0.05]))
        t = np.array(0.5)
        slerp_result = slerp_quaternion_arrays(q1, q2, t)
        nlerp_result = nlerp_quaternion_arrays(q1, q2, t)
        np.testing.assert_allclose(slerp_result, nlerp_result, atol=1e-4)


class TestSlerpAgainstPyquaternion:
    """Cross-validate SLERP implementation against pyquaternion as reference."""

    @staticmethod
    def _pyquat_slerp(q1_array: np.ndarray, q2_array: np.ndarray, t: float) -> np.ndarray:
        pq1 = PyQuaternion(array=q1_array)
        pq2 = PyQuaternion(array=q2_array)
        result = PyQuaternion.slerp(pq1, pq2, amount=t)
        return np.array([result.w, result.x, result.y, result.z])

    def test_90_degree_rotation_z(self):
        """Compare SLERP with pyquaternion for 90-degree Z rotation at multiple t values."""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
        for t_val in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            ours = slerp_quaternion_arrays(q1, q2, np.array(t_val))
            ref = self._pyquat_slerp(q1, q2, t_val)
            np.testing.assert_allclose(ours, ref, atol=1e-10, err_msg=f"Mismatch at t={t_val}")

    def test_arbitrary_rotations(self):
        """Compare SLERP with pyquaternion for random quaternion pairs."""
        rng = np.random.default_rng(123)
        for _ in range(20):
            q1 = normalize_quaternion_array(rng.standard_normal(4))
            q2 = normalize_quaternion_array(rng.standard_normal(4))
            t_val = rng.uniform(0, 1)
            ours = slerp_quaternion_arrays(q1, q2, np.array(t_val))
            ref = self._pyquat_slerp(q1, q2, t_val)
            # Quaternions q and -q represent the same rotation
            if np.dot(ours, ref) < 0:
                ref = -ref
            np.testing.assert_allclose(ours, ref, atol=1e-10, err_msg=f"Mismatch for q1={q1}, q2={q2}, t={t_val}")

    def test_large_angle_rotation(self):
        """Compare SLERP with pyquaternion for a 170-degree rotation."""
        angle = np.radians(170)
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
        for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            ours = slerp_quaternion_arrays(q1, q2, np.array(t_val))
            ref = self._pyquat_slerp(q1, q2, t_val)
            if np.dot(ours, ref) < 0:
                ref = -ref
            np.testing.assert_allclose(ours, ref, atol=1e-10, err_msg=f"Mismatch at t={t_val}")

    def test_batch_matches_pyquaternion(self):
        """Verify batched SLERP matches per-element pyquaternion results."""
        rng = np.random.default_rng(99)
        n = 15
        q1 = normalize_quaternion_array(rng.standard_normal((n, 4)))
        q2 = normalize_quaternion_array(rng.standard_normal((n, 4)))
        t = rng.uniform(0, 1, size=n)
        ours = slerp_quaternion_arrays(q1, q2, t)
        for i in range(n):
            ref = self._pyquat_slerp(q1[i], q2[i], t[i])
            if np.dot(ours[i], ref) < 0:
                ref = -ref
            np.testing.assert_allclose(ours[i], ref, atol=1e-10, err_msg=f"Mismatch at index {i}")
