from typing import Union

from py123d.datatypes.sensors.base_camera import BaseCameraMetadata, Camera, CameraChannelType, CameraModel, camera_metadata_from_dict
from py123d.datatypes.sensors.fisheye_mei_camera import (
    FisheyeMEICameraID,
    FisheyeMEICameraMetadata,
    FisheyeMEIDistortion,
    FisheyeMEIDistortionIndex,
    FisheyeMEIProjection,
    FisheyeMEIProjectionIndex,
)
from py123d.datatypes.sensors.lidar import (
    Lidar,
    LidarFeature,
    LidarID,
    LidarMergedMetadata,
    LidarMetadata,
)
from py123d.datatypes.sensors.pinhole_camera import (
    PinholeCameraID,
    PinholeCameraMetadata,
    PinholeDistortion,
    PinholeDistortionIndex,
    PinholeIntrinsics,
    PinholeIntrinsicsIndex,
)

CameraID = Union[PinholeCameraID, FisheyeMEICameraID]
CameraMetadata = Union[PinholeCameraMetadata, FisheyeMEICameraMetadata]
