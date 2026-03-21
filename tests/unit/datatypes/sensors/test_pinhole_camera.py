import numpy as np
import pytest

from py123d.datatypes.metadata.base_metadata import BaseMetadata
from py123d.datatypes.sensors.base_camera import Camera, CameraID
from py123d.datatypes.sensors.pinhole_camera import (
    PinholeCameraMetadata,
    PinholeDistortion,
    PinholeDistortionIndex,
    PinholeIntrinsics,
)
from py123d.datatypes.time.timestamp import Timestamp
from py123d.geometry import PoseSE3

DUMMY_TIMESTAMP = Timestamp.from_s(0.0)


class TestCameraID:
    """Test CameraID enum functionality."""

    def test_camera_id_values(self):
        """Test that camera type enum has expected integer values."""
        assert CameraID.PCAM_F0.value == 0
        assert CameraID.PCAM_B0.value == 1
        assert CameraID.PCAM_L0.value == 2
        assert CameraID.PCAM_L1.value == 3
        assert CameraID.PCAM_L2.value == 4
        assert CameraID.PCAM_R0.value == 5
        assert CameraID.PCAM_R1.value == 6
        assert CameraID.PCAM_R2.value == 7
        assert CameraID.PCAM_STEREO_L.value == 8
        assert CameraID.PCAM_STEREO_R.value == 9
        assert CameraID.FMCAM_L.value == 10
        assert CameraID.FMCAM_R.value == 11

    def test_camera_id_from_int(self):
        """Test creating camera type from integer."""
        assert CameraID(0) == CameraID.PCAM_F0
        assert CameraID(5) == CameraID.PCAM_R0
        assert CameraID(9) == CameraID.PCAM_STEREO_R
        assert CameraID(10) == CameraID.FMCAM_L
        assert CameraID(11) == CameraID.FMCAM_R

    def test_camera_id_unique_values(self):
        """Test that all camera type values are unique."""
        values = [ct.value for ct in CameraID]
        assert len(values) == len(set(values))


class TestPinholeIntrinsics:
    """Test PinholeIntrinsics functionality."""

    def test_intrinsics_creation(self):
        """Test creating PinholeIntrinsics instance."""
        intrinsics = PinholeIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, skew=0.0)

        assert intrinsics.fx == 500.0
        assert intrinsics.fy == 500.0
        assert intrinsics.cx == 320.0
        assert intrinsics.cy == 240.0
        assert intrinsics.skew == 0.0

    def test_intrinsics_default_skew(self):
        """Test that skew defaults to 0.0 when not provided."""
        intrinsics = PinholeIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)

        assert intrinsics.skew == 0.0

    def test_intrinsics_from_array(self):
        """Test creating intrinsics from array."""
        array = np.array([500.0, 500.0, 320.0, 240.0, 0.0], dtype=np.float64)
        intrinsics = PinholeIntrinsics.from_array(array)

        assert intrinsics.fx == 500.0
        assert intrinsics.fy == 500.0
        assert intrinsics.cx == 320.0
        assert intrinsics.cy == 240.0
        assert intrinsics.skew == 0.0

    def test_intrinsics_from_array_copy(self):
        """Test that from_array creates a copy by default."""
        array = np.array([500.0, 500.0, 320.0, 240.0, 0.0], dtype=np.float64)
        intrinsics = PinholeIntrinsics.from_array(array, copy=True)

        # Modify original array
        array[0] = 1000.0

        # Intrinsics should still have original value
        assert intrinsics.fx == 500.0

    def test_intrinsics_from_array_no_copy(self):
        """Test that from_array can avoid copying."""
        array = np.array([500.0, 500.0, 320.0, 240.0, 0.0], dtype=np.float64)
        intrinsics = PinholeIntrinsics.from_array(array, copy=False)

        # Modify original array
        array[0] = 1000.0

        # Intrinsics should reflect the change
        assert intrinsics.fx == 1000.0

    def test_intrinsics_from_camera_matrix(self):
        """Test creating intrinsics from 3x3 camera matrix."""
        K = np.array([[500.0, 0.5, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)

        intrinsics = PinholeIntrinsics.from_camera_matrix(K)

        assert intrinsics.fx == 500.0
        assert intrinsics.fy == 500.0
        assert intrinsics.cx == 320.0
        assert intrinsics.cy == 240.0
        assert intrinsics.skew == 0.5

    def test_intrinsics_camera_matrix_property(self):
        """Test getting camera matrix from intrinsics."""
        intrinsics = PinholeIntrinsics(fx=500.0, fy=600.0, cx=320.0, cy=240.0, skew=0.5)
        K = intrinsics.camera_matrix

        expected_K = np.array([[500.0, 0.5, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)

        np.testing.assert_array_almost_equal(K, expected_K)

    def test_intrinsics_camera_matrix_roundtrip(self):
        """Test converting to camera matrix and back."""
        original_K = np.array([[500.0, 0.5, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)

        intrinsics = PinholeIntrinsics.from_camera_matrix(original_K)
        restored_K = intrinsics.camera_matrix

        np.testing.assert_array_almost_equal(restored_K, original_K)

    def test_intrinsics_array_property(self):
        """Test accessing the underlying array."""
        intrinsics = PinholeIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, skew=0.5)
        array = intrinsics.array

        assert isinstance(array, np.ndarray)
        assert array.shape == (5,)
        np.testing.assert_array_almost_equal(array, [500.0, 500.0, 320.0, 240.0, 0.5])

    def test_intrinsics_from_list(self):
        """Test creating intrinsics from list via from_list method."""
        intrinsics_list = [500.0, 500.0, 320.0, 240.0, 0.0]
        intrinsics = PinholeIntrinsics.from_list(intrinsics_list)

        assert intrinsics.fx == 500.0
        assert intrinsics.fy == 500.0
        assert intrinsics.cx == 320.0
        assert intrinsics.cy == 240.0
        assert intrinsics.skew == 0.0

    def test_intrinsics_tolist(self):
        """Test converting intrinsics to list."""
        intrinsics = PinholeIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, skew=0.5)
        intrinsics_list = intrinsics.tolist()

        assert isinstance(intrinsics_list, list)
        assert len(intrinsics_list) == 5
        assert intrinsics_list[0] == pytest.approx(500.0)
        assert intrinsics_list[1] == pytest.approx(500.0)
        assert intrinsics_list[2] == pytest.approx(320.0)
        assert intrinsics_list[3] == pytest.approx(240.0)
        assert intrinsics_list[4] == pytest.approx(0.5)

    def test_intrinsics_different_fx_fy(self):
        """Test intrinsics with different focal lengths."""
        intrinsics = PinholeIntrinsics(fx=500.0, fy=600.0, cx=320.0, cy=240.0)

        assert intrinsics.fx == 500.0
        assert intrinsics.fy == 600.0
        assert intrinsics.fx != intrinsics.fy

    def test_intrinsics_non_centered_principal_point(self):
        """Test intrinsics with non-centered principal point."""
        intrinsics = PinholeIntrinsics(fx=500.0, fy=500.0, cx=100.0, cy=100.0)

        assert intrinsics.cx == 100.0
        assert intrinsics.cy == 100.0


class TestPinholeDistortion:
    """Test PinholeDistortion functionality."""

    def test_distortion_creation(self):
        """Test creating PinholeDistortion instance."""
        distortion = PinholeDistortion(k1=0.1, k2=0.01, p1=0.001, p2=0.001, k3=0.001)

        assert distortion.k1 == 0.1
        assert distortion.k2 == 0.01
        assert distortion.p1 == 0.001
        assert distortion.p2 == 0.001
        assert distortion.k3 == 0.001

    def test_distortion_from_array(self):
        """Test creating distortion from array."""
        array = np.array([0.1, 0.01, 0.001, 0.001, 0.001], dtype=np.float64)
        distortion = PinholeDistortion.from_array(array)

        assert distortion.k1 == 0.1
        assert distortion.k2 == 0.01
        assert distortion.p1 == 0.001
        assert distortion.p2 == 0.001
        assert distortion.k3 == 0.001

    def test_distortion_from_array_copy(self):
        """Test that from_array creates a copy by default."""
        array = np.array([0.1, 0.01, 0.001, 0.001, 0.001], dtype=np.float64)
        distortion = PinholeDistortion.from_array(array, copy=True)

        # Modify original array
        array[0] = 0.5

        # Distortion should still have original value
        assert distortion.k1 == 0.1

    def test_distortion_from_array_no_copy(self):
        """Test that from_array can avoid copying."""
        array = np.array([0.1, 0.01, 0.001, 0.001, 0.001], dtype=np.float64)
        distortion = PinholeDistortion.from_array(array, copy=False)

        # Modify original array
        array[0] = 0.5

        # Distortion should reflect the change
        assert distortion.k1 == 0.5

    def test_distortion_array_property(self):
        """Test accessing the underlying array."""
        distortion = PinholeDistortion(k1=0.1, k2=0.01, p1=0.001, p2=0.001, k3=0.001)
        array = distortion.array

        assert isinstance(array, np.ndarray)
        assert array.shape == (len(PinholeDistortionIndex),)
        np.testing.assert_array_almost_equal(array, [0.1, 0.01, 0.001, 0.001, 0.001])

    def test_distortion_zero_values(self):
        """Test distortion with zero values."""
        distortion = PinholeDistortion(k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0)

        assert distortion.k1 == 0.0
        assert distortion.k2 == 0.0
        assert distortion.p1 == 0.0
        assert distortion.p2 == 0.0
        assert distortion.k3 == 0.0

    def test_distortion_negative_values(self):
        """Test distortion with negative values."""
        distortion = PinholeDistortion(k1=-0.1, k2=-0.01, p1=-0.001, p2=-0.001, k3=-0.001)

        assert distortion.k1 == -0.1
        assert distortion.k2 == -0.01
        assert distortion.p1 == -0.001
        assert distortion.p2 == -0.001
        assert distortion.k3 == -0.001

    def test_distortion_from_list(self):
        """Test creating distortion from list via from_list method."""
        distortion_list = [0.1, 0.01, 0.001, 0.001, 0.001]
        distortion = PinholeDistortion.from_list(distortion_list)

        assert distortion.k1 == 0.1
        assert distortion.k2 == 0.01
        assert distortion.p1 == 0.001
        assert distortion.p2 == 0.001
        assert distortion.k3 == 0.001

    def test_distortion_tolist(self):
        """Test converting distortion to list."""
        distortion = PinholeDistortion(k1=0.1, k2=0.01, p1=0.001, p2=0.001, k3=0.001)
        distortion_list = distortion.tolist()

        assert isinstance(distortion_list, list)
        assert len(distortion_list) == 5
        assert distortion_list[0] == pytest.approx(0.1)
        assert distortion_list[1] == pytest.approx(0.01)
        assert distortion_list[2] == pytest.approx(0.001)
        assert distortion_list[3] == pytest.approx(0.001)
        assert distortion_list[4] == pytest.approx(0.001)


class TestPinholeMetadata:
    """Test PinholeCameraMetadata functionality."""

    def test_metadata_from_dict_with_none_intrinsics(self):
        """Test creating metadata from dict with None intrinsics."""
        data_dict = {
            "camera_name": "TestCamera",
            "camera_id": 1,
            "intrinsics": None,
            "distortion": [0.1, 0.01, 0.001, 0.001, 0.001],
            "width": 800,
            "height": 600,
            "camera_to_imu_se3": PoseSE3.identity().to_list(),
            "is_undistorted": False,
        }

        metadata = PinholeCameraMetadata.from_dict(data_dict)

        assert metadata.camera_name == "TestCamera"
        assert metadata.camera_id == CameraID.PCAM_B0
        assert metadata.intrinsics is None
        assert metadata.distortion is not None
        assert metadata.width == 800
        assert metadata.height == 600

    def test_metadata_from_dict_with_none_distortion(self):
        """Test creating metadata from dict with None distortion."""
        data_dict = {
            "camera_name": "TestCamera",
            "camera_id": 2,
            "intrinsics": [600.0, 600.0, 400.0, 300.0, 0.0],
            "distortion": None,
            "width": 800,
            "height": 600,
            "camera_to_imu_se3": PoseSE3.identity().to_list(),
            "is_undistorted": False,
        }

        metadata = PinholeCameraMetadata.from_dict(data_dict)

        assert metadata.camera_id == CameraID.PCAM_L0
        assert metadata.intrinsics is not None
        assert metadata.distortion is None

    def test_metadata_different_aspect_ratios(self):
        """Test metadata with different aspect ratios."""
        intrinsics = PinholeIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)

        # 16:9 aspect ratio
        metadata_16_9 = PinholeCameraMetadata(
            camera_name="TestCamera",
            camera_id=CameraID.PCAM_F0,
            intrinsics=intrinsics,
            distortion=None,
            width=1920,
            height=1080,
            camera_to_imu_se3=PoseSE3.identity(),
        )
        assert metadata_16_9.aspect_ratio == pytest.approx(16 / 9)

        # 4:3 aspect ratio
        metadata_4_3 = PinholeCameraMetadata(
            camera_name="TestCamera",
            camera_id=CameraID.PCAM_F0,
            intrinsics=intrinsics,
            distortion=None,
            width=640,
            height=480,
            camera_to_imu_se3=PoseSE3.identity(),
        )
        assert metadata_4_3.aspect_ratio == pytest.approx(4 / 3)

    def test_metadata_fov_with_different_focal_lengths(self):
        """Test FOV calculation with different focal lengths."""
        intrinsics_narrow = PinholeIntrinsics(fx=1000.0, fy=1000.0, cx=320.0, cy=240.0)
        intrinsics_wide = PinholeIntrinsics(fx=250.0, fy=250.0, cx=320.0, cy=240.0)

        metadata_narrow = PinholeCameraMetadata(
            camera_name="TestCamera",
            camera_id=CameraID.PCAM_F0,
            intrinsics=intrinsics_narrow,
            distortion=None,
            width=640,
            height=480,
            camera_to_imu_se3=PoseSE3.identity(),
        )
        metadata_wide = PinholeCameraMetadata(
            camera_name="TestCamera",
            camera_id=CameraID.PCAM_F0,
            intrinsics=intrinsics_wide,
            distortion=None,
            width=640,
            height=480,
            camera_to_imu_se3=PoseSE3.identity(),
        )

        # Wider focal length should result in larger FOV
        assert metadata_narrow.fov_x is not None and metadata_narrow.fov_y is not None
        assert metadata_wide.fov_x is not None and metadata_wide.fov_y is not None
        assert metadata_wide.fov_x > metadata_narrow.fov_x
        assert metadata_wide.fov_y > metadata_narrow.fov_y

    def test_metadata_to_dict_preserves_types(self):
        """Test that to_dict preserves correct types."""
        intrinsics = PinholeIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        distortion = PinholeDistortion(k1=0.1, k2=0.01, p1=0.001, p2=0.001, k3=0.001)

        metadata = PinholeCameraMetadata(
            camera_name="TestCamera",
            camera_id=CameraID.PCAM_R1,
            intrinsics=intrinsics,
            distortion=distortion,
            width=1280,
            height=720,
            camera_to_imu_se3=PoseSE3.identity(),
        )

        data_dict = metadata.to_dict()

        assert isinstance(data_dict["camera_name"], str)
        assert isinstance(data_dict["camera_id"], int)
        assert isinstance(data_dict["width"], int)
        assert isinstance(data_dict["height"], int)
        assert isinstance(data_dict["intrinsics"], list)
        assert isinstance(data_dict["distortion"], list)

    def test_metadata_all_camera_ids(self):
        """Test metadata creation with all camera types."""
        intrinsics = PinholeIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)

        for camera_id in CameraID:
            metadata = PinholeCameraMetadata(
                camera_name="TestCamera",
                camera_id=camera_id,
                intrinsics=intrinsics,
                distortion=None,
                width=640,
                height=480,
                camera_to_imu_se3=PoseSE3.identity(),
            )
            assert metadata.camera_id == camera_id

    def test_is_instance_of_abstract_metadata(self):
        """PinholeCameraMetadata is an instance of BaseMetadata."""
        intrinsics = PinholeIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        metadata = PinholeCameraMetadata(
            camera_name="TestCamera",
            camera_id=CameraID.PCAM_F0,
            intrinsics=intrinsics,
            distortion=None,
            width=640,
            height=480,
            camera_to_imu_se3=PoseSE3.identity(),
        )
        assert isinstance(metadata, BaseMetadata)

    def test_roundtrip_serialization(self):
        """to_dict and from_dict are inverses."""
        intrinsics = PinholeIntrinsics(fx=600.0, fy=600.0, cx=400.0, cy=300.0)
        distortion = PinholeDistortion(k1=0.1, k2=0.02, p1=0.003, p2=0.004, k3=0.005)
        extrinsic = PoseSE3.from_list([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])
        original = PinholeCameraMetadata(
            camera_name="RoundtripCam",
            camera_id=CameraID.PCAM_L0,
            intrinsics=intrinsics,
            distortion=distortion,
            width=1280,
            height=720,
            camera_to_imu_se3=extrinsic,
            is_undistorted=True,
        )
        restored = PinholeCameraMetadata.from_dict(original.to_dict())

        assert restored.camera_name == original.camera_name
        assert restored.camera_id == original.camera_id
        assert restored.width == original.width
        assert restored.height == original.height
        assert restored.is_undistorted == original.is_undistorted
        assert restored.intrinsics.fx == pytest.approx(original.intrinsics.fx)
        assert restored.distortion.k1 == pytest.approx(original.distortion.k1)
        assert restored.camera_to_imu_se3 == original.camera_to_imu_se3

    def test_metadata_square_image(self):
        """Test metadata with square image dimensions."""
        intrinsics = PinholeIntrinsics(fx=500.0, fy=500.0, cx=256.0, cy=256.0)
        metadata = PinholeCameraMetadata(
            camera_name="TestCamera",
            camera_id=CameraID.PCAM_F0,
            intrinsics=intrinsics,
            distortion=None,
            width=512,
            height=512,
            camera_to_imu_se3=PoseSE3.identity(),
        )

        assert metadata.aspect_ratio == 1.0
        assert metadata.fov_x == pytest.approx(metadata.fov_y)

    def test_metadata_non_square_pixels(self):
        """Test metadata with non-square pixels (different fx and fy)."""
        intrinsics = PinholeIntrinsics(fx=500.0, fy=600.0, cx=320.0, cy=240.0)
        metadata = PinholeCameraMetadata(
            camera_name="TestCamera",
            camera_id=CameraID.PCAM_F0,
            intrinsics=intrinsics,
            distortion=None,
            width=640,
            height=480,
            camera_to_imu_se3=PoseSE3.identity(),
        )

        expected_fov_x = 2 * np.arctan(640 / (2 * 500.0))
        expected_fov_y = 2 * np.arctan(480 / (2 * 600.0))

        assert metadata.fov_x == pytest.approx(expected_fov_x)
        assert metadata.fov_y == pytest.approx(expected_fov_y)
        assert metadata.fov_x != pytest.approx(metadata.fov_y)


class TestCamera:
    def test_pinhole_camera_creation(self):
        """Test creating Camera instance."""

        intrinsics = PinholeIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        metadata = PinholeCameraMetadata(
            camera_name="TestCamera",
            camera_id=CameraID.PCAM_F0,
            intrinsics=intrinsics,
            distortion=None,
            width=640,
            height=480,
            camera_to_imu_se3=PoseSE3.identity(),
        )
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        extrinsic = PoseSE3(x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)

        camera = Camera(metadata=metadata, image=image, camera_to_global_se3=extrinsic, timestamp=DUMMY_TIMESTAMP)

        assert camera.metadata == metadata
        assert np.array_equal(camera.image, image)
        assert camera.camera_to_global_se3 == extrinsic

    def test_pinhole_camera_with_color_image(self):
        """Test Camera with color image."""

        intrinsics = PinholeIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        metadata = PinholeCameraMetadata(
            camera_name="ColorCamera",
            camera_id=CameraID.PCAM_F0,
            intrinsics=intrinsics,
            distortion=None,
            width=640,
            height=480,
            camera_to_imu_se3=PoseSE3.identity(),
        )
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        extrinsic = PoseSE3(x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)

        camera = Camera(metadata=metadata, image=image, camera_to_global_se3=extrinsic, timestamp=DUMMY_TIMESTAMP)

        assert camera.image.shape == (480, 640, 3)
        assert camera.image.dtype == np.uint8

    def test_pinhole_camera_with_grayscale_image(self):
        """Test Camera with grayscale image."""

        intrinsics = PinholeIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        metadata = PinholeCameraMetadata(
            camera_name="GrayCamera",
            camera_id=CameraID.PCAM_L0,
            intrinsics=intrinsics,
            distortion=None,
            width=640,
            height=480,
            camera_to_imu_se3=PoseSE3.identity(),
        )
        image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        extrinsic = PoseSE3(x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)

        camera = Camera(metadata=metadata, image=image, camera_to_global_se3=extrinsic, timestamp=DUMMY_TIMESTAMP)

        assert camera.image.shape == (480, 640)

    def test_pinhole_camera_with_distortion(self):
        """Test Camera with distortion parameters."""

        intrinsics = PinholeIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        distortion = PinholeDistortion(k1=0.1, k2=0.01, p1=0.001, p2=0.001, k3=0.001)
        metadata = PinholeCameraMetadata(
            camera_name="DistortedCamera",
            camera_id=CameraID.PCAM_F0,
            intrinsics=intrinsics,
            distortion=distortion,
            width=640,
            height=480,
            camera_to_imu_se3=PoseSE3.identity(),
        )
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        extrinsic = PoseSE3(x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)

        camera = Camera(metadata=metadata, image=image, camera_to_global_se3=extrinsic, timestamp=DUMMY_TIMESTAMP)

        assert camera.metadata.distortion is not None
        assert camera.metadata.distortion.k1 == 0.1

    def test_pinhole_camera_different_types(self):
        """Test Camera with different camera types."""

        intrinsics = PinholeIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        extrinsic = PoseSE3(x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)

        for camera_id in [
            CameraID.PCAM_F0,
            CameraID.PCAM_B0,
            CameraID.PCAM_STEREO_L,
            CameraID.PCAM_STEREO_R,
        ]:
            metadata = PinholeCameraMetadata(
                camera_name="TestCamera",
                camera_id=camera_id,
                intrinsics=intrinsics,
                distortion=None,
                width=640,
                height=480,
                camera_to_imu_se3=PoseSE3.identity(),
            )
            camera = Camera(metadata=metadata, image=image, camera_to_global_se3=extrinsic, timestamp=DUMMY_TIMESTAMP)
            assert camera.metadata.camera_id == camera_id

    def test_pinhole_camera_with_different_resolutions(self):
        """Test Camera with different image resolutions."""

        resolutions = [(640, 480), (1920, 1080), (1280, 720), (800, 600)]
        extrinsic = PoseSE3(x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)

        for width, height in resolutions:
            intrinsics = PinholeIntrinsics(fx=500.0, fy=500.0, cx=width / 2, cy=height / 2)
            metadata = PinholeCameraMetadata(
                camera_name="TestCamera",
                camera_id=CameraID.PCAM_F0,
                intrinsics=intrinsics,
                distortion=None,
                width=width,
                height=height,
                camera_to_imu_se3=PoseSE3.identity(),
            )
            image = np.zeros((height, width, 3), dtype=np.uint8)
            camera = Camera(metadata=metadata, image=image, camera_to_global_se3=extrinsic, timestamp=DUMMY_TIMESTAMP)

            assert camera.metadata.width == width
            assert camera.metadata.height == height
            assert camera.image.shape[:2] == (height, width)
