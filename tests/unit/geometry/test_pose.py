import numpy as np
import pytest

from py123d.geometry import EulerAngles, Point2D, PoseSE2, PoseSE3, Vector2D, Vector3D
from py123d.geometry.geometry_index import PoseSE2Index
from py123d.geometry.point import Point3D


class TestPoseSE2:
    def test_init(self):
        """Test basic initialization with explicit x, y, yaw values."""
        pose = PoseSE2(x=1.0, y=2.0, yaw=0.5)
        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.yaw == 0.5

    def test_from_array(self):
        """Test creation from numpy array."""
        array = np.array([1.0, 2.0, 0.5])
        pose = PoseSE2.from_array(array)
        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.yaw == 0.5

    def test_from_array_copy(self):
        """Test that copy=True creates independent pose from array."""
        array = np.array([1.0, 2.0, 0.5])
        pose = PoseSE2.from_array(array, copy=True)
        array[0] = 99.0
        assert pose.x == 1.0

    def test_from_array_no_copy(self):
        """Test that copy=False links pose to original array."""
        array = np.array([1.0, 2.0, 0.5])
        pose = PoseSE2.from_array(array, copy=False)
        array[0] = 99.0
        assert pose.x == 99.0

    def test_from_transformation_matrix(self):
        """Test creation from a transformation matrix."""

        transformation_matrix = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])
        pose = PoseSE2.from_transformation_matrix(transformation_matrix)
        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.yaw == 0.0  # Assuming no rotation in the transformation matrix

        # Consistency check:
        cons_pose = PoseSE2(x=1.0, y=2.0, yaw=np.pi / 4)
        cons_transformation_matrix = cons_pose.transformation_matrix
        pose_from_cons_matrix = PoseSE2.from_transformation_matrix(cons_transformation_matrix)
        np.testing.assert_allclose(pose_from_cons_matrix.array, cons_pose.array, atol=1e-10)

    def test_from_R_t(self):
        """Test creation arbitrary from rotation and translation representations."""

        rotation_matrix = np.array([[0.0, -1.0], [1.0, 0.0]])
        rotation_float = np.pi / 2  # 90 degrees rotation
        rotation_numpy_0d = np.array(rotation_float)
        rotation_numpy_1d = np.array([rotation_float])

        translation_numpy = np.array([1.0, 2.0])
        translation_point2d = Point2D(x=1.0, y=2.0)
        translation_numpy_2 = np.array([1.0, 2.0])

        # Test all combinations of rotation and translation inputs
        pose1 = PoseSE2.from_R_t(rotation_matrix, translation_numpy)
        pose2 = PoseSE2.from_R_t(rotation_float, translation_numpy)
        pose3 = PoseSE2.from_R_t(rotation_numpy_0d, translation_numpy)
        pose4 = PoseSE2.from_R_t(rotation_numpy_1d, translation_numpy)
        pose5 = PoseSE2.from_R_t(rotation_float, translation_point2d)
        pose6 = PoseSE2.from_R_t(rotation_float, translation_numpy_2)

        # Verify all produce consistent results
        assert pose2.x == 1.0
        assert pose2.y == 2.0
        assert pytest.approx(pose2.yaw) == np.pi / 2

        np.testing.assert_allclose(pose1.array, pose2.array, atol=1e-10)
        np.testing.assert_allclose(pose2.array, pose3.array, atol=1e-10)
        np.testing.assert_allclose(pose3.array, pose4.array, atol=1e-10)
        np.testing.assert_allclose(pose4.array, pose5.array, atol=1e-10)
        np.testing.assert_allclose(pose5.array, pose6.array, atol=1e-10)

    def test_identity(self):
        """Test creation of identity pose."""
        pose = PoseSE2.identity()
        assert pose.x == 0.0
        assert pose.y == 0.0
        assert pose.yaw == 0.0
        transformation_matrix = pose.transformation_matrix
        np.testing.assert_allclose(transformation_matrix, np.eye(3), atol=1e-10)

    def test_properties(self):
        """Test access to individual pose component properties."""
        pose = PoseSE2(x=3.0, y=4.0, yaw=np.pi / 4)
        assert pose.x == 3.0
        assert pose.y == 4.0
        assert pytest.approx(pose.yaw) == np.pi / 4

    def test_array_property(self):
        """Test that the array property returns correct numpy array."""
        pose = PoseSE2(x=1.0, y=2.0, yaw=0.5)
        array = pose.array
        assert array.shape == (3,)
        assert array[PoseSE2Index.X] == 1.0
        assert array[PoseSE2Index.Y] == 2.0
        assert array[PoseSE2Index.YAW] == 0.5

    def test_point_2d(self):
        """Test extraction of 2D position as Point2D."""
        pose = PoseSE2(x=1.0, y=2.0, yaw=0.5)
        point = pose.point_2d
        assert isinstance(point, Point2D)
        assert point.x == 1.0
        assert point.y == 2.0

    def test_rotation_matrix(self):
        """Test extraction of 2x2 rotation matrix."""
        pose = PoseSE2(x=1.0, y=2.0, yaw=0.0)
        rot_mat = pose.rotation_matrix
        expected = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_allclose(rot_mat, expected)

    def test_rotation_matrix_pi_half(self):
        """Test extraction of 2x2 rotation matrix for 90 degree rotation."""
        pose = PoseSE2(x=0.0, y=0.0, yaw=np.pi / 2)
        rot_mat = pose.rotation_matrix
        expected = np.array([[0.0, -1.0], [1.0, 0.0]])
        np.testing.assert_allclose(rot_mat, expected, atol=1e-10)

    def test_transformation_matrix(self):
        """Test extraction of 3x3 transformation matrix."""
        pose = PoseSE2(x=1.0, y=2.0, yaw=0.0)
        trans_mat = pose.transformation_matrix
        assert trans_mat.shape == (3, 3)
        expected = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(trans_mat, expected)

    def test_shapely_point(self):
        """Test extraction of Shapely Point representation."""
        pose = PoseSE2(x=1.0, y=2.0, yaw=0.5)
        shapely_point = pose.shapely_point
        assert shapely_point.x == 1.0
        assert shapely_point.y == 2.0

    def test_pose_se2_property(self):
        """Test that pose_se2 property returns self."""
        pose = PoseSE2(x=1.0, y=2.0, yaw=0.5)
        assert pose.pose_se2 is pose

    def test_vector_2d(self):
        """Test extraction of translation as Vector2D."""
        pose = PoseSE2(x=1.0, y=2.0, yaw=0.5)
        vector = pose.vector_2d
        assert isinstance(vector, Vector2D)
        assert vector.x == 1.0
        assert vector.y == 2.0

    def test_from_R_t_with_vector2d(self):
        """Test from_R_t accepts Vector2D for translation."""
        vector = Vector2D(x=1.0, y=2.0)
        pose = PoseSE2.from_R_t(rotation=0.5, translation=vector)
        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pytest.approx(pose.yaw) == 0.5

    def test_equality(self):
        """Test equality comparison of PoseSE2 instances."""
        pose1 = PoseSE2(x=1.0, y=2.0, yaw=0.5)
        pose2 = PoseSE2(x=1.0, y=2.0, yaw=0.5)
        assert pose1 == pose2

    def test_inequality(self):
        """Test inequality comparison of PoseSE2 instances."""
        pose1 = PoseSE2(x=1.0, y=2.0, yaw=0.5)
        pose2 = PoseSE2(x=1.0, y=2.0, yaw=0.6)
        assert pose1 != pose2

    def test_repr(self):
        """Test __repr__ returns a string containing the class name."""
        pose = PoseSE2(x=1.0, y=2.0, yaw=0.5)
        r = repr(pose)
        assert "PoseSE2" in r

    def test_from_R_t_invalid_translation_type(self):
        """Test from_R_t with invalid translation type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported translation type"):
            PoseSE2.from_R_t(rotation=0.0, translation="bad")

    def test_from_R_t_invalid_rotation_ndarray_shape(self):
        """Test from_R_t with invalid rotation ndarray shape raises ValueError."""
        with pytest.raises(ValueError, match="Expected rotation"):
            PoseSE2.from_R_t(rotation=np.array([1.0, 2.0, 3.0]), translation=np.array([0.0, 0.0]))

    def test_from_R_t_invalid_rotation_type(self):
        """Test from_R_t with invalid rotation type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported rotation type"):
            PoseSE2.from_R_t(rotation="bad", translation=np.array([0.0, 0.0]))


class TestPoseSE3:
    def test_init(self):
        """Test basic initialization with explicit x, y, z, and quaternion values."""
        pose = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.z == 3.0
        assert pose.qw == 1.0
        assert pose.qx == 0.0
        assert pose.qy == 0.0
        assert pose.qz == 0.0

    def test_from_array(self):
        """Test creation from numpy array."""
        array = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
        pose = PoseSE3.from_array(array)
        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.z == 3.0
        assert pose.qw == 1.0
        assert pose.qx == 0.0
        assert pose.qy == 0.0
        assert pose.qz == 0.0

    def test_from_array_copy(self):
        """Test that copy=True creates independent pose from array."""
        array = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
        pose = PoseSE3.from_array(array, copy=True)
        array[0] = 99.0
        assert pose.x == 1.0

    def test_from_array_no_copy(self):
        """Test that copy=False links pose to original array."""
        array = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
        pose = PoseSE3.from_array(array, copy=False)
        array[0] = 99.0
        assert pose.x == 99.0

    def test_from_transformation_matrix(self):
        """Test creation from 4x4 transformation matrix."""
        trans_mat = np.eye(4)
        trans_mat[:3, 3] = [1.0, 2.0, 3.0]
        pose = PoseSE3.from_transformation_matrix(trans_mat)
        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.z == 3.0
        assert pose.qw == 1.0
        assert pose.qx == 0.0
        assert pose.qy == 0.0
        assert pose.qz == 0.0

    def test_from_R_t(self):
        """Test creation from arbitrary rotation and translation representations."""

        # Rotation representations
        rotation_euler = EulerAngles(yaw=np.pi / 2, pitch=np.pi / 3, roll=np.pi / 4)
        rotation_euler_array = rotation_euler.array
        rotation_quat = rotation_euler.quaternion
        rotation_quat_array = rotation_quat.array
        rotation_matrix = rotation_euler.rotation_matrix

        # Translation representations
        translation_point3d = Point3D(x=1.0, y=2.0, z=3.0)
        translation_vector3d = Vector3D(x=1.0, y=2.0, z=3.0)
        translation_array = np.array([1.0, 2.0, 3.0])

        # Reference pose for consistency checks
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = [1.0, 2.0, 3.0]
        reference_pose = PoseSE3.from_transformation_matrix(transformation_matrix)

        for rotation in [rotation_matrix, rotation_quat, rotation_quat_array, rotation_euler_array]:
            for translation in [translation_point3d, translation_vector3d, translation_array]:
                pose = PoseSE3.from_R_t(rotation, translation)
                np.testing.assert_allclose(pose.array, reference_pose.array, atol=1e-10)

    def test_identity(self):
        """Test creation of identity pose."""
        pose = PoseSE3.identity()
        np.testing.assert_allclose(pose.array, np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), atol=1e-10)
        np.testing.assert_allclose(pose.transformation_matrix, np.eye(4), atol=1e-10)
        assert pose.yaw == 0.0
        assert pose.pitch == 0.0
        assert pose.roll == 0.0

    def test_properties(self):
        """Test access to individual pose component properties."""
        pose = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.z == 3.0
        assert pose.qw == 1.0
        assert pose.qx == 0.0
        assert pose.qy == 0.0
        assert pose.qz == 0.0

    def test_array_property(self):
        """Test that the array property returns the correct numpy array representation."""
        pose = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        array = pose.array
        assert array.shape == (7,)
        np.testing.assert_allclose(array, [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])

    def test_pose_se2(self):
        """Test extraction of 2D pose from 3D pose."""
        pose = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        pose_2d = pose.pose_se2
        assert isinstance(pose_2d, PoseSE2)
        assert pose_2d.x == 1.0
        assert pose_2d.y == 2.0
        assert pytest.approx(pose_2d.yaw) == 0.0

    def test_point_3d(self):
        """Test extraction of 3D point from 3D pose."""
        pose = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        point = pose.point_3d
        assert point.x == 1.0
        assert point.y == 2.0
        assert point.z == 3.0

    def test_point_2d(self):
        """Test extraction of 2D point from 3D pose."""
        pose = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        point = pose.point_2d
        assert isinstance(point, Point2D)
        assert point.x == 1.0
        assert point.y == 2.0

    def test_vector_3d(self):
        """Test extraction of translation as Vector3D."""
        pose = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        vector = pose.vector_3d
        assert isinstance(vector, Vector3D)
        assert vector.x == 1.0
        assert vector.y == 2.0
        assert vector.z == 3.0

    def test_vector_2d(self):
        """Test extraction of 2D translation as Vector2D."""
        pose = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        vector = pose.vector_2d
        assert isinstance(vector, Vector2D)
        assert vector.x == 1.0
        assert vector.y == 2.0

    def test_shapely_point(self):
        """Test extraction of Shapely Point representation."""
        pose = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        shapely_point = pose.shapely_point
        assert shapely_point.x == 1.0
        assert shapely_point.y == 2.0
        assert shapely_point.z == 3.0

    def test_rotation_matrix(self):
        """Test extraction of 3x3 rotation matrix."""
        pose = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        rot_mat = pose.rotation_matrix
        expected = np.eye(3)
        np.testing.assert_allclose(rot_mat, expected)

    def test_transformation_matrix(self):
        """Test extraction of 4x4 transformation matrix."""
        pose = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        trans_mat = pose.transformation_matrix
        assert trans_mat.shape == (4, 4)
        expected = np.eye(4)
        expected[:3, 3] = [1.0, 2.0, 3.0]
        np.testing.assert_allclose(trans_mat, expected)

    def test_transformation_matrix_roundtrip(self):
        """Test round-trip conversion between pose and transformation matrix."""
        pose1 = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        trans_mat = pose1.transformation_matrix
        pose2 = PoseSE3.from_transformation_matrix(trans_mat)
        assert pose1 == pose2

    def test_euler_angles(self):
        """Test extraction of Euler angles from quaternion."""
        pose = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        assert pytest.approx(pose.roll) == 0.0
        assert pytest.approx(pose.pitch) == 0.0
        assert pytest.approx(pose.yaw) == 0.0

    def test_equality(self):
        """Test equality comparison of PoseSE3 instances."""
        pose1 = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        pose2 = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        assert pose1 == pose2

    def test_inequality(self):
        """Test inequality comparison of PoseSE3 instances."""
        pose1 = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        pose2 = PoseSE3(x=1.0, y=2.0, z=3.0, qw=0.9, qx=0.1, qy=0.0, qz=0.0)
        assert pose1 != pose2

    def test_pose_se3_property(self):
        """Test that pose_se3 property returns self."""
        pose = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        assert pose.pose_se3 is pose

    def test_quaternion_property(self):
        """Test extraction of quaternion from pose."""
        pose = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        q = pose.quaternion
        assert q.qw == 1.0
        assert q.qx == 0.0
        assert q.qy == 0.0
        assert q.qz == 0.0

    def test_inverse(self):
        """Test inverse of a pose."""
        pose = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        inv = pose.inverse
        # pose * inverse should give identity
        T = pose.transformation_matrix @ inv.transformation_matrix
        np.testing.assert_allclose(T, np.eye(4), atol=1e-10)

    def test_inverse_with_rotation(self):
        """Test inverse of a pose with non-trivial rotation (90° around Y-axis)."""
        # 90° rotation around Y-axis: qw=cos(45°), qy=sin(45°)
        c = np.cos(np.pi / 4)
        pose = PoseSE3(x=1.0, y=2.0, z=3.0, qw=c, qx=0.0, qy=c, qz=0.0)
        inv = pose.inverse
        # pose * inverse should give identity
        T = pose.transformation_matrix @ inv.transformation_matrix
        np.testing.assert_allclose(T, np.eye(4), atol=1e-10)
        # inverse * pose should also give identity
        T2 = inv.transformation_matrix @ pose.transformation_matrix
        np.testing.assert_allclose(T2, np.eye(4), atol=1e-10)

    def test_repr(self):
        """Test __repr__ returns a string containing the class name."""
        pose = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        r = repr(pose)
        assert "PoseSE3" in r

    def test_from_R_t_invalid_translation_type(self):
        """Test from_R_t with invalid translation type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported translation type"):
            PoseSE3.from_R_t(rotation=EulerAngles(0.0, 0.0, 0.0), translation="bad")

    def test_from_R_t_invalid_rotation_ndarray_shape(self):
        """Test from_R_t with invalid rotation ndarray shape raises ValueError."""
        with pytest.raises(ValueError, match="Expected rotation"):
            PoseSE3.from_R_t(rotation=np.array([1.0, 2.0]), translation=np.array([0.0, 0.0, 0.0]))

    def test_from_R_t_invalid_rotation_type(self):
        """Test from_R_t with invalid rotation type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported rotation type"):
            PoseSE3.from_R_t(rotation="bad", translation=np.array([0.0, 0.0, 0.0]))

    def test_from_R_t_with_euler_angles(self):
        """Test from_R_t with EulerAngles rotation."""
        euler = EulerAngles(0.0, 0.0, np.pi / 2)
        pose = PoseSE3.from_R_t(euler, np.array([1.0, 2.0, 3.0]))
        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.z == 3.0
        assert pytest.approx(pose.yaw) == np.pi / 2
