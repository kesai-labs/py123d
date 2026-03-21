import numpy as np
import pytest

from py123d.datatypes.metadata.base_metadata import BaseMetadata
from py123d.datatypes.sensors.base_camera import Camera, CameraID
from py123d.datatypes.sensors.fisheye_mei_camera import (
    FisheyeMEICameraMetadata,
    FisheyeMEIDistortion,
    FisheyeMEIDistortionIndex,
    FisheyeMEIProjection,
    FisheyeMEIProjectionIndex,
)
from py123d.datatypes.time.timestamp import Timestamp
from py123d.geometry import PoseSE3


class TestFisheyeMEICameraType:
    """Test FisheyeMEICameraType enum functionality."""

    def test_camera_id_values(self):
        """Test that fisheye camera IDs have expected values in the unified CameraID enum."""
        assert CameraID.FMCAM_L.value == 10
        assert CameraID.FMCAM_R.value == 11

    def test_camera_id_from_int(self):
        """Test creating fisheye camera IDs from integer values."""
        assert CameraID(10) == CameraID.FMCAM_L
        assert CameraID(11) == CameraID.FMCAM_R

    def test_camera_id_members(self):
        """Test that fisheye members exist in the unified CameraID enum."""
        members = list(CameraID)
        assert len(members) == 19
        assert CameraID.FMCAM_L in members
        assert CameraID.FMCAM_R in members

    def test_camera_id_comparison(self):
        """Test comparison between camera types."""
        assert CameraID.FMCAM_L != CameraID.FMCAM_R
        assert CameraID.FMCAM_L == CameraID.FMCAM_L


class TestFisheyeMEIDistortion:
    """Test FisheyeMEIDistortion functionality."""

    def test_distortion_initialization(self):
        """Test distortion parameter initialization."""
        distortion = FisheyeMEIDistortion(k1=0.1, k2=0.2, p1=0.3, p2=0.4)
        assert distortion.k1 == 0.1
        assert distortion.k2 == 0.2
        assert distortion.p1 == 0.3
        assert distortion.p2 == 0.4

    def test_distortion_from_array(self):
        """Test creating distortion from array."""
        array = np.array([0.1, 0.2, 0.3, 0.4])
        distortion = FisheyeMEIDistortion.from_array(array)
        assert distortion.k1 == 0.1
        assert distortion.k2 == 0.2
        assert distortion.p1 == 0.3
        assert distortion.p2 == 0.4

    def test_distortion_from_array_copy(self):
        """Test that from_array copies data by default."""
        array = np.array([0.1, 0.2, 0.3, 0.4])
        distortion = FisheyeMEIDistortion.from_array(array, copy=True)
        array[0] = 999.0
        assert distortion.k1 == 0.1

    def test_distortion_from_array_no_copy(self):
        """Test that from_array can avoid copying."""
        array = np.array([0.1, 0.2, 0.3, 0.4])
        distortion = FisheyeMEIDistortion.from_array(array, copy=False)
        array[0] = 999.0
        assert distortion.k1 == 999.0

    def test_distortion_array_property(self):
        """Test array property returns correct values."""
        distortion = FisheyeMEIDistortion(k1=0.1, k2=0.2, p1=0.3, p2=0.4)
        array = distortion.array
        assert len(array) == 4
        np.testing.assert_array_equal(array, [0.1, 0.2, 0.3, 0.4])

    def test_distortion_index_mapping(self):
        """Test that distortion indices map correctly."""
        distortion = FisheyeMEIDistortion(k1=0.1, k2=0.2, p1=0.3, p2=0.4)
        assert distortion.array[FisheyeMEIDistortionIndex.K1] == 0.1
        assert distortion.array[FisheyeMEIDistortionIndex.K2] == 0.2
        assert distortion.array[FisheyeMEIDistortionIndex.P1] == 0.3
        assert distortion.array[FisheyeMEIDistortionIndex.P2] == 0.4


class TestFisheyeMEIProjection:
    """Test FisheyeMEIProjection functionality."""

    def test_projection_initialization(self):
        """Test projection parameter initialization."""
        projection = FisheyeMEIProjection(gamma1=1.0, gamma2=2.0, u0=3.0, v0=4.0)
        assert projection.gamma1 == 1.0
        assert projection.gamma2 == 2.0
        assert projection.u0 == 3.0
        assert projection.v0 == 4.0

    def test_projection_from_array(self):
        """Test creating projection from array."""
        array = np.array([1.0, 2.0, 3.0, 4.0])
        projection = FisheyeMEIProjection.from_array(array)
        assert projection.gamma1 == 1.0
        assert projection.gamma2 == 2.0
        assert projection.u0 == 3.0
        assert projection.v0 == 4.0

    def test_projection_from_array_copy(self):
        """Test that from_array copies data by default."""
        array = np.array([1.0, 2.0, 3.0, 4.0])
        projection = FisheyeMEIProjection.from_array(array, copy=True)
        array[0] = 999.0
        assert projection.gamma1 == 1.0

    def test_projection_from_array_no_copy(self):
        """Test that from_array can avoid copying."""
        array = np.array([1.0, 2.0, 3.0, 4.0])
        projection = FisheyeMEIProjection.from_array(array, copy=False)
        array[0] = 999.0
        assert projection.gamma1 == 999.0

    def test_projection_array_property(self):
        """Test array property returns correct values."""
        projection = FisheyeMEIProjection(gamma1=1.0, gamma2=2.0, u0=3.0, v0=4.0)
        array = projection.array
        assert len(array) == 4
        np.testing.assert_array_equal(array, [1.0, 2.0, 3.0, 4.0])

    def test_projection_index_mapping(self):
        """Test that projection indices map correctly."""
        projection = FisheyeMEIProjection(gamma1=1.0, gamma2=2.0, u0=3.0, v0=4.0)
        assert projection.array[FisheyeMEIProjectionIndex.GAMMA1] == 1.0
        assert projection.array[FisheyeMEIProjectionIndex.GAMMA2] == 2.0
        assert projection.array[FisheyeMEIProjectionIndex.U0] == 3.0
        assert projection.array[FisheyeMEIProjectionIndex.V0] == 4.0


class TestFisheyeMEICameraMetadata:
    """Test FisheyeMEICameraMetadata functionality."""

    def test_metadata_initialization(self):
        """Test metadata initialization with all parameters."""
        distortion = FisheyeMEIDistortion(k1=0.1, k2=0.2, p1=0.3, p2=0.4)
        projection = FisheyeMEIProjection(gamma1=1.0, gamma2=2.0, u0=3.0, v0=4.0)
        metadata = FisheyeMEICameraMetadata(
            camera_name="TestCamera",
            camera_id=CameraID.FMCAM_L,
            mirror_parameter=0.5,
            distortion=distortion,
            projection=projection,
            width=1920,
            height=1080,
            camera_to_imu_se3=PoseSE3.identity(),
        )
        assert metadata.camera_name == "TestCamera"
        assert metadata.camera_id == CameraID.FMCAM_L
        assert metadata.mirror_parameter == 0.5
        assert metadata.distortion == distortion
        assert metadata.projection == projection
        assert metadata.aspect_ratio == 1920 / 1080
        assert metadata.camera_to_imu_se3 == PoseSE3.identity()

    def test_metadata_initialization_with_none(self):
        """Test metadata initialization with None distortion and projection."""
        metadata = FisheyeMEICameraMetadata(
            camera_name="TestCamera",
            camera_id=CameraID.FMCAM_R,
            mirror_parameter=None,
            distortion=None,
            projection=None,
            width=640,
            height=480,
            camera_to_imu_se3=None,
        )
        assert metadata.camera_name == "TestCamera"
        assert metadata.camera_id == CameraID.FMCAM_R
        assert metadata.mirror_parameter is None
        assert metadata.distortion is None
        assert metadata.projection is None
        assert metadata.camera_to_imu_se3 is None
        assert metadata.aspect_ratio == 640 / 480

    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        distortion = FisheyeMEIDistortion(k1=0.1, k2=0.2, p1=0.3, p2=0.4)
        projection = FisheyeMEIProjection(gamma1=1.0, gamma2=2.0, u0=3.0, v0=4.0)
        metadata = FisheyeMEICameraMetadata(
            camera_name="TestCamera",
            camera_id=CameraID.FMCAM_L,
            mirror_parameter=0.5,
            distortion=distortion,
            projection=projection,
            width=1920,
            height=1080,
            camera_to_imu_se3=PoseSE3.identity(),
        )
        result = metadata.to_dict()

        assert result["camera_name"] == "TestCamera"
        assert result["camera_id"] == 10
        assert result["mirror_parameter"] == 0.5
        assert result["distortion"] == [0.1, 0.2, 0.3, 0.4]
        assert result["projection"] == [1.0, 2.0, 3.0, 4.0]
        assert result["width"] == 1920
        assert result["height"] == 1080
        assert result["camera_to_imu_se3"] == PoseSE3.identity().to_list()

    def test_metadata_from_dict(self):
        """Test creating metadata from dictionary."""
        data = {
            "camera_name": "TestCamera",
            "camera_id": 10,
            "mirror_parameter": 0.5,
            "distortion": [0.1, 0.2, 0.3, 0.4],
            "projection": [1.0, 2.0, 3.0, 4.0],
            "width": 1920,
            "height": 1080,
            "camera_to_imu_se3": [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
        }
        metadata = FisheyeMEICameraMetadata.from_dict(data)
        assert metadata.camera_id == CameraID.FMCAM_L
        assert metadata.mirror_parameter == 0.5
        assert metadata.distortion is not None
        assert metadata.distortion.k1 == 0.1
        assert metadata.projection is not None
        assert metadata.projection.gamma1 == 1.0
        assert metadata.aspect_ratio == 1920 / 1080
        assert metadata.camera_to_imu_se3 == PoseSE3.from_list([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])

    def test_metadata_from_dict_with_none(self):
        """Test creating metadata from dictionary with None values."""
        data = {
            "camera_name": "TestCamera",
            "camera_id": 11,
            "mirror_parameter": None,
            "distortion": None,
            "projection": None,
            "width": 640,
            "height": 480,
            "camera_to_imu_se3": [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
        }
        metadata = FisheyeMEICameraMetadata.from_dict(data)
        assert metadata.camera_name == "TestCamera"
        assert metadata.camera_id == CameraID.FMCAM_R
        assert metadata.mirror_parameter is None
        assert metadata.distortion is None
        assert metadata.projection is None
        assert metadata.camera_to_imu_se3 == PoseSE3.from_list([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])

    def test_metadata_roundtrip(self):
        """Test that to_dict and from_dict are inverses."""
        distortion = FisheyeMEIDistortion(k1=0.1, k2=0.2, p1=0.3, p2=0.4)
        projection = FisheyeMEIProjection(gamma1=1.0, gamma2=2.0, u0=3.0, v0=4.0)
        metadata = FisheyeMEICameraMetadata(
            camera_name="TestCamera",
            camera_id=CameraID.FMCAM_L,
            mirror_parameter=0.5,
            distortion=distortion,
            projection=projection,
            width=1920,
            height=1080,
            camera_to_imu_se3=PoseSE3.identity(),
        )
        data_dict = metadata.to_dict()
        metadata_restored = FisheyeMEICameraMetadata.from_dict(data_dict)
        assert metadata.camera_name == metadata_restored.camera_name
        assert metadata.camera_id == metadata_restored.camera_id
        assert metadata.mirror_parameter == metadata_restored.mirror_parameter
        assert metadata.distortion is not None and metadata_restored.distortion is not None
        assert metadata.projection is not None and metadata_restored.projection is not None
        np.testing.assert_array_equal(metadata.distortion.array, metadata_restored.distortion.array)
        np.testing.assert_array_equal(metadata.projection.array, metadata_restored.projection.array)
        assert metadata.aspect_ratio == metadata_restored.aspect_ratio
        assert metadata.camera_to_imu_se3 == metadata_restored.camera_to_imu_se3

    def test_is_instance_of_abstract_metadata(self):
        """FisheyeMEICameraMetadata is an instance of BaseMetadata."""
        metadata = FisheyeMEICameraMetadata(
            camera_name="TestCamera",
            camera_id=CameraID.FMCAM_L,
            mirror_parameter=None,
            distortion=None,
            projection=None,
            width=640,
            height=480,
            camera_to_imu_se3=PoseSE3.identity(),
        )
        assert isinstance(metadata, BaseMetadata)

    def test_aspect_ratio_calculation(self):
        """Test aspect ratio calculation."""
        metadata = FisheyeMEICameraMetadata(
            camera_name="TestCamera",
            camera_id=CameraID.FMCAM_L,
            mirror_parameter=0.5,
            distortion=None,
            projection=None,
            width=1920,
            height=1080,
            camera_to_imu_se3=PoseSE3.identity(),
        )
        assert metadata.aspect_ratio == pytest.approx(16 / 9, abs=1e-05)


class TestFisheyeMEICamera:
    """Test Camera functionality."""

    def test_camera_initialization(self):
        """Test Camera initialization."""

        metadata = FisheyeMEICameraMetadata(
            camera_name="TestCamera",
            camera_id=CameraID.FMCAM_L,
            mirror_parameter=0.5,
            distortion=FisheyeMEIDistortion(k1=0.1, k2=0.2, p1=0.3, p2=0.4),
            projection=FisheyeMEIProjection(gamma1=1.0, gamma2=2.0, u0=3.0, v0=4.0),
            width=1920,
            height=1080,
            camera_to_imu_se3=PoseSE3.identity(),
        )
        image = np.zeros((1080, 1920), dtype=np.uint8)
        extrinsic = PoseSE3(x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)

        camera = Camera(
            metadata=metadata,
            image=image,
            camera_to_global_se3=extrinsic,
            timestamp=Timestamp.from_s(0.0),
        )

        assert camera.metadata == metadata
        np.testing.assert_array_equal(camera.image, image)
        assert camera.camera_to_global_se3 == extrinsic
        assert camera.timestamp == Timestamp.from_s(0.0)

    def test_camera_metadata_property(self):
        """Test that metadata property returns correct metadata."""

        metadata = FisheyeMEICameraMetadata(
            camera_name="TestCamera",
            camera_id=CameraID.FMCAM_R,
            mirror_parameter=0.8,
            distortion=None,
            projection=None,
            width=640,
            height=480,
            camera_to_imu_se3=PoseSE3.identity(),
        )
        image = np.ones((480, 640), dtype=np.uint8)
        extrinsic = PoseSE3(x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)

        camera = Camera(
            metadata=metadata,
            image=image,
            camera_to_global_se3=extrinsic,
            timestamp=Timestamp.from_s(0.0),
        )

        assert camera.metadata is metadata
        assert camera.metadata.camera_id == CameraID.FMCAM_R
        assert camera.timestamp == Timestamp.from_s(0.0)

    def test_camera_image_property(self):
        """Test that image property returns correct image."""

        metadata = FisheyeMEICameraMetadata(
            camera_name="TestCamera",
            camera_id=CameraID.FMCAM_L,
            mirror_parameter=0.5,
            distortion=None,
            projection=None,
            width=640,
            height=480,
            camera_to_imu_se3=PoseSE3.identity(),
        )
        image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        extrinsic = PoseSE3(x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)

        camera = Camera(
            metadata=metadata,
            image=image,
            camera_to_global_se3=extrinsic,
            timestamp=Timestamp.from_s(0.0),
        )

        np.testing.assert_array_equal(camera.image, image)
        assert camera.image.dtype == np.uint8

    def test_camera_extrinsic_property(self):
        """Test that extrinsic property returns correct pose."""

        metadata = FisheyeMEICameraMetadata(
            camera_name="TestCamera",
            camera_id=CameraID.FMCAM_L,
            mirror_parameter=0.5,
            distortion=None,
            projection=None,
            width=640,
            height=480,
            camera_to_imu_se3=PoseSE3.identity(),
        )
        image = np.zeros((480, 640), dtype=np.uint8)
        extrinsic = PoseSE3(x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)

        camera = Camera(metadata=metadata, image=image, camera_to_global_se3=extrinsic, timestamp=Timestamp.from_s(0.0))

        assert camera.camera_to_global_se3 is extrinsic

    def test_camera_with_color_image(self):
        """Test camera with color (3-channel) image."""

        metadata = FisheyeMEICameraMetadata(
            camera_name="TestCamera",
            camera_id=CameraID.FMCAM_L,
            mirror_parameter=0.5,
            distortion=None,
            projection=None,
            width=640,
            height=480,
            camera_to_imu_se3=PoseSE3.identity(),
        )
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        extrinsic = PoseSE3(x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)

        camera = Camera(metadata=metadata, image=image, camera_to_global_se3=extrinsic, timestamp=Timestamp.from_s(0.0))

        assert camera.image.shape == (480, 640, 3)
