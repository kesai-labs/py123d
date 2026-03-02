from dataclasses import dataclass
from typing import List, Optional, Union

from py123d.common.utils.enums import SerialIntEnum
from py123d.datatypes.sensors.fisheye_mei_camera import FisheyeMEICameraID
from py123d.datatypes.sensors.lidar import LidarID
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraID

# TODO: Add more filter options (e.g. scene tags, ego movement, or whatever appropriate)


@dataclass
class SceneFilter:
    """Class to filter scenes when building scenes from logs."""

    datasets: Optional[List[str]] = None
    """List of dataset names to filter scenes by."""

    split_types: Optional[List[str]] = None
    """List of split types to filter scenes by (e.g. `train`, `val`, `test`)."""

    split_names: Optional[List[str]] = None
    """List of split names to filter scenes by (in the form `{dataset-name}_{split_type}`)."""

    log_names: Optional[List[str]] = None
    """Name of logs to include scenes from."""

    locations: Optional[List[str]] = None
    """List of locations to filter scenes by."""

    scene_uuids: Optional[List[str]] = None
    """List of scene UUIDs to include."""

    timestamp_threshold_s: Optional[float] = None
    """Minimum time between the start timestamps of two consecutive scenes."""

    duration_s: Optional[float] = None
    """Duration of each scene in seconds."""

    history_s: Optional[float] = None
    """History duration of each scene in seconds."""

    pinhole_camera_ids: Optional[List[PinholeCameraID]] = None
    """List of :class:`PinholeCameraID` to include in the scenes."""

    fisheye_mei_camera_ids: Optional[List[FisheyeMEICameraID]] = None
    """List of :class:`FisheyeMEICameraType` to include in the scenes."""

    lidar_ids: Optional[List[LidarID]] = None
    """List of :class:`LidarID` to include in the scenes."""

    max_num_scenes: Optional[int] = None
    """Maximum number of scenes to return."""

    map_api_required: bool = False
    """Whether to only include scenes with an available map API."""

    shuffle: bool = False
    """Whether to shuffle the returned scenes."""

    def __post_init__(self):
        def _resolve_enum_arguments(
            serial_enum_cls: SerialIntEnum,
            input: Optional[List[Union[int, str, SerialIntEnum]]],
        ):
            if input is None:
                return None
            return [serial_enum_cls.from_arbitrary(value) for value in input]

        self.pinhole_camera_ids = _resolve_enum_arguments(PinholeCameraID, self.pinhole_camera_ids)  # type: ignore
        self.fisheye_mei_camera_ids = _resolve_enum_arguments(FisheyeMEICameraID, self.fisheye_mei_camera_ids)  # type: ignore
        self.lidar_ids = _resolve_enum_arguments(LidarID, self.lidar_ids)  # type: ignore
