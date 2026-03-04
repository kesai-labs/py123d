from pathlib import Path
from typing import Dict, List, Optional, Type, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from py123d.api.utils.arrow_schema import (
    BOX_DETECTIONS_SE3,
    EGO_STATE_SE3,
    FISHEYE_MEI,
    LIDAR,
    PINHOLE_CAMERA,
    SYNC,
    TRAFFIC_LIGHTS,
)
from py123d.common.dataset_paths import get_dataset_paths
from py123d.common.utils.mixin import ArrayMixin
from py123d.conversion.sensor_io.camera.jpeg_camera_io import (
    decode_image_from_jpeg_binary,
    is_jpeg_binary,
    load_image_from_jpeg_file,
)
from py123d.conversion.sensor_io.camera.mp4_camera_io import get_mp4_reader_from_path
from py123d.conversion.sensor_io.camera.png_camera_io import decode_image_from_png_binary, is_png_binary
from py123d.conversion.sensor_io.lidar.draco_lidar_io import is_draco_binary, load_point_cloud_3d_from_draco_binary
from py123d.conversion.sensor_io.lidar.ipc_lidar_io import (
    is_ipc_binary,
    load_point_cloud_3d_from_ipc_binary,
    load_point_cloud_features_from_ipc_binary,
)
from py123d.conversion.sensor_io.lidar.laz_lidar_io import is_laz_binary, load_point_cloud_3d_from_laz_binary
from py123d.conversion.sensor_io.lidar.path_lidar_io import load_point_cloud_data_from_path
from py123d.datatypes.detections import (
    BoxDetectionAttributes,
    BoxDetectionSE3,
    BoxDetectionsSE3,
    TrafficLightDetection,
    TrafficLightDetections,
    TrafficLightStatus,
)
from py123d.datatypes.detections.box_detection_label_metadata import BoxDetectionMetadata
from py123d.datatypes.metadata import LogMetadata
from py123d.datatypes.sensors import (
    FisheyeMEICamera,
    FisheyeMEICameraID,
    FisheyeMEICameraMetadata,
    Lidar,
    LidarID,
    LidarMetadata,
    PinholeCamera,
    PinholeCameraID,
    PinholeCameraMetadata,
)
from py123d.datatypes.time import Timestamp
from py123d.datatypes.vehicle_state import DynamicStateSE3, EgoMetadata, EgoStateSE3
from py123d.geometry import BoundingBoxSE3, PoseSE3, Vector3D


def get_timestamp_from_arrow_table(arrow_table: pa.Table, index: int) -> Timestamp:
    """Builds a :class:`~py123d.datatypes.time.Timestamp` from an Arrow table at a given index.

    :param arrow_table: The Arrow table containing the timestamp data.
    :param index: The index to extract the timestamp from.
    :return: The Timestamp at the given index.
    """
    assert SYNC.col("timestamp_us") in arrow_table.schema.names, "Timestamp column not found in Arrow table."
    return Timestamp.from_us(arrow_table[SYNC.col("timestamp_us")][index].as_py())


def get_ego_state_se3_from_arrow_table(
    arrow_table: pa.Table,
    index: int,
    vehicle_parameters: Optional[EgoMetadata],
) -> Optional[EgoStateSE3]:
    """Builds a :class:`~py123d.datatypes.vehicle_state.EgoStateSE3` from an Arrow table at a given index.

    :param arrow_table: The Arrow table containing the ego state data.
    :param index: The index to extract the ego state from.
    :param vehicle_parameters: The vehicle parameters used to build the ego state.
    :return: The ego state at the given index, or None if not available.
    """

    ego_state_se3: Optional[EgoStateSE3] = None
    if _all_columns_in_schema(arrow_table, EGO_STATE_SE3.all_columns()) and vehicle_parameters is not None:
        timestamp = Timestamp.from_us(arrow_table[EGO_STATE_SE3.col("timestamp_us")][index].as_py())
        imu_se3 = PoseSE3.from_list(arrow_table[EGO_STATE_SE3.col("imu_se3")][index].as_py())
        dynamic_state_se3 = _get_optional_array_mixin(
            arrow_table[EGO_STATE_SE3.col("dynamic_state_se3")][index].as_py(),
            DynamicStateSE3,
        )
        ego_state_se3 = EgoStateSE3.from_imu(
            imu_se3=imu_se3,
            vehicle_parameters=vehicle_parameters,
            dynamic_state_se3=dynamic_state_se3,  # type: ignore
            timestamp=timestamp,
        )
    return ego_state_se3


def get_box_detections_se3_from_arrow_table(
    arrow_table: pa.Table,
    index: int,
    box_detection_metadata: BoxDetectionMetadata,
    timestamp: Optional[Timestamp] = None,
) -> BoxDetectionsSE3:
    """Builds a :class:`~py123d.datatypes.detections.BoxDetectionsSE3` from an Arrow table at a given index.

    :param arrow_table: The Arrow table containing the box detections data.
    :param index: The index to extract the box detections from.
    :param box_detection_metadata: The box detection metadata containing the label class.
    :param timestamp: Optional timestamp for the detections. If None, attempts to read from the table.
    :return: The BoxDetectionsSE3 at the given index.
    """

    box_detections: Optional[BoxDetectionsSE3] = None
    if _all_columns_in_schema(arrow_table, BOX_DETECTIONS_SE3.all_columns()):
        if timestamp is None:
            if SYNC.col("timestamp_us") in arrow_table.schema.names:
                timestamp = get_timestamp_from_arrow_table(arrow_table, index)
            else:
                timestamp = Timestamp.from_us(0)
        box_detections_list: List[BoxDetectionSE3] = []
        box_detection_label_class = box_detection_metadata.box_detection_label_class
        for _bounding_box_se3, _token, _label, _velocity, _num_lidar_points in zip(
            arrow_table[BOX_DETECTIONS_SE3.col("bounding_box_se3")][index].as_py(),
            arrow_table[BOX_DETECTIONS_SE3.col("token")][index].as_py(),
            arrow_table[BOX_DETECTIONS_SE3.col("label")][index].as_py(),
            arrow_table[BOX_DETECTIONS_SE3.col("velocity_3d")][index].as_py(),
            arrow_table[BOX_DETECTIONS_SE3.col("num_lidar_points")][index].as_py(),
        ):
            box_detections_list.append(
                BoxDetectionSE3(
                    metadata=BoxDetectionAttributes(
                        label=box_detection_label_class(_label),
                        track_token=_token,
                        num_lidar_points=_num_lidar_points,
                    ),
                    bounding_box_se3=BoundingBoxSE3.from_list(_bounding_box_se3),
                    velocity_3d=_get_optional_array_mixin(_velocity, Vector3D),
                )
            )
        box_detections = BoxDetectionsSE3(box_detections=box_detections_list, timestamp=timestamp)

    return box_detections


def get_traffic_light_detections_from_arrow_table(
    arrow_table: pa.Table,
    index: int,
) -> Optional[TrafficLightDetections]:
    """Builds a :class:`~py123d.datatypes.detections.TrafficLightDetections` from an Arrow table at a given index.

    :param arrow_table: The Arrow table containing the traffic light detections data.
    :param index: The index to extract the traffic light detections from.
    :return: The TrafficLightDetections at the given index, or None if not available.
    """
    traffic_lights: Optional[TrafficLightDetections] = None
    if _all_columns_in_schema(arrow_table, TRAFFIC_LIGHTS.all_columns()):
        timestamp = Timestamp.from_us(arrow_table[TRAFFIC_LIGHTS.col("timestamp_us")][index].as_py())
        detections: List[TrafficLightDetection] = []
        for lane_id, status in zip(
            arrow_table[TRAFFIC_LIGHTS.col("lane_id")][index].as_py(),
            arrow_table[TRAFFIC_LIGHTS.col("status")][index].as_py(),
        ):
            detections.append(
                TrafficLightDetection(
                    lane_id=lane_id,
                    status=TrafficLightStatus(status),
                )
            )
        traffic_lights = TrafficLightDetections(detections=detections, timestamp=timestamp)
    return traffic_lights


def get_camera_from_arrow_table(
    arrow_table: pa.Table,
    index: int,
    camera_id: Union[PinholeCameraID, FisheyeMEICameraID],
    camera_metadata: Union[PinholeCameraMetadata, FisheyeMEICameraMetadata],
    log_metadata: LogMetadata,
) -> Optional[Union[PinholeCamera, FisheyeMEICamera]]:
    """Builds a camera object from an Arrow table at a given index.

    :param arrow_table: The Arrow table containing the camera data.
    :param index: The index to extract the camera data from.
    :param camera_id: The ID of the camera to build (Pinhole or FisheyeMEI).
    :param camera_metadata: The camera metadata (intrinsics, distortion, etc.).
    :param log_metadata: Metadata about the log (used for dataset path resolution and MP4 reading).
    :raises ValueError: If the camera data format is unsupported.
    :raises NotImplementedError: If the camera data type is not supported.
    :return: The constructed camera object, or None if not available.
    """

    assert isinstance(camera_id, (PinholeCameraID, FisheyeMEICameraID)), (
        f"camera_id must be PinholeCameraID or FisheyeMEICameraID, got {type(camera_id)}"
    )

    camera: Optional[Union[PinholeCamera, FisheyeMEICamera]] = None

    is_pinhole = isinstance(camera_id, PinholeCameraID)
    schema = PINHOLE_CAMERA if is_pinhole else FISHEYE_MEI

    camera_data_column = schema.col("data")
    camera_extrinsic_column = schema.col("state_se3")
    camera_timestamp_column = schema.col("timestamp_us")

    if _all_columns_in_schema(arrow_table, [camera_data_column, camera_extrinsic_column, camera_timestamp_column]):
        table_data = arrow_table[camera_data_column][index].as_py()
        extrinsic_data = arrow_table[camera_extrinsic_column][index].as_py()
        timestamp_data = arrow_table[camera_timestamp_column][index].as_py()

        if table_data is not None and extrinsic_data is not None:
            extrinsic = PoseSE3.from_list(extrinsic_data)
            image: Optional[npt.NDArray[np.uint8]] = None

            if isinstance(table_data, str):
                sensor_root = get_dataset_paths().get_sensor_root(log_metadata.dataset)
                assert sensor_root is not None, (
                    f"Dataset path for sensor loading not found for dataset: {log_metadata.dataset}"
                )
                full_image_path = Path(sensor_root) / table_data
                assert full_image_path.exists(), f"Camera file not found: {full_image_path}"

                image = load_image_from_jpeg_file(full_image_path)
            elif isinstance(table_data, bytes):
                if is_jpeg_binary(table_data):
                    image = decode_image_from_jpeg_binary(table_data)
                elif is_png_binary(table_data):
                    image = decode_image_from_png_binary(table_data)
                else:
                    raise ValueError("Camera binary data is neither in JPEG nor PNG format.")

            elif isinstance(table_data, int):
                image = _unoptimized_demo_mp4_read(log_metadata, camera_id.serialize(), table_data)
            else:
                raise NotImplementedError(
                    f"Only string file paths, bytes, or int frame indices are supported for camera data, got {type(table_data)}"
                )

            if is_pinhole:
                camera = PinholeCamera(
                    metadata=camera_metadata,  # type: ignore[arg-type]
                    image=image,
                    extrinsic=extrinsic,
                    timestamp=Timestamp.from_us(timestamp_data),
                )
            else:
                camera = FisheyeMEICamera(
                    metadata=camera_metadata,  # type: ignore[arg-type]
                    image=image,
                    extrinsic=extrinsic,
                    timestamp=Timestamp.from_us(timestamp_data),
                )

    return camera


def get_camera_timestamp_from_arrow_table(
    arrow_table: pa.Table,
    index: int,
    camera_id: Union[PinholeCameraID, FisheyeMEICameraID],
) -> Optional[Timestamp]:
    """Gets the camera timestamp from an Arrow table at a given index.

    :param arrow_table: The Arrow table containing the camera timestamp data.
    :param index: The index to extract the camera timestamp from.
    :param camera_id: The type of camera (Pinhole or FisheyeMEI).
    :return: The camera timestamp at the given index, or None if not available.
    """

    assert isinstance(camera_id, (PinholeCameraID, FisheyeMEICameraID)), (
        f"The argument 'camera_id' must be PinholeCameraID or FisheyeMEICameraID, got {type(camera_id)}"
    )

    camera_timestamp: Optional[Timestamp] = None
    schema = PINHOLE_CAMERA if isinstance(camera_id, PinholeCameraID) else FISHEYE_MEI
    camera_timestamp_column = schema.col("timestamp_us")

    if camera_timestamp_column in arrow_table.schema.names:
        timestamp_data = arrow_table[camera_timestamp_column][index].as_py()
        if timestamp_data is not None:
            camera_timestamp = Timestamp.from_us(timestamp_data)

    return camera_timestamp


def get_lidar_from_arrow_table(
    arrow_table: pa.Table,
    index: int,
    lidar_type: LidarID,
    lidar_metadatas: Dict[LidarID, LidarMetadata],
    log_metadata: LogMetadata,
) -> Optional[Lidar]:
    """Builds a Lidar object from an Arrow table at a given index.

    :param arrow_table: The Arrow table containing the Lidar data.
    :param index: The index to extract the Lidar data from.
    :param lidar_type: The type of Lidar to build.
    :param lidar_metadatas: Per-sensor lidar metadata dict.
    :param log_metadata: Metadata about the log (used for dataset path resolution).
    :raises ValueError: If the Lidar data format is unsupported.
    :raises NotImplementedError: If the Lidar data type is not supported.
    :return: The constructed Lidar object, or None if not available.
    """
    point_cloud_3d: Optional[np.ndarray] = None
    point_cloud_feature: Optional[Dict[str, np.ndarray]] = None
    if LIDAR.col("data") in arrow_table.schema.names:
        # 1. Load lidar sweep from origin dataset using a relative file path.
        lidar_data = arrow_table[LIDAR.col("data")][index].as_py()
        if lidar_data is not None:
            assert isinstance(lidar_data, str), f"Lidar path data must be a string file path, got {type(lidar_data)}"
            point_cloud_3d, point_cloud_feature = load_point_cloud_data_from_path(
                relative_path=lidar_data,
                log_metadata=log_metadata,
                index=index,
                lidar_metadatas=lidar_metadatas,
            )

    elif LIDAR.col("point_cloud_3d") in arrow_table.schema.names:
        # 2.1 Loading the lidar xyz point cloud from blob in the Arrow table.
        lidar_data = arrow_table[LIDAR.col("point_cloud_3d")][index].as_py()
        if lidar_data is not None:
            if is_draco_binary(lidar_data):
                point_cloud_3d = load_point_cloud_3d_from_draco_binary(lidar_data)
            elif is_laz_binary(lidar_data):
                point_cloud_3d = load_point_cloud_3d_from_laz_binary(lidar_data)
            elif is_ipc_binary(lidar_data):
                point_cloud_3d = load_point_cloud_3d_from_ipc_binary(lidar_data)

        # 2.2 Load lidar features from blob in the Arrow table, if available.
        if LIDAR.col("point_cloud_features") in arrow_table.schema.names:
            lidar_point_cloud_feature_data = arrow_table[LIDAR.col("point_cloud_features")][index].as_py()
            if lidar_point_cloud_feature_data is not None:
                if is_ipc_binary(lidar_point_cloud_feature_data):
                    point_cloud_feature = load_point_cloud_features_from_ipc_binary(lidar_point_cloud_feature_data)

    lidar: Optional[Lidar] = None
    if point_cloud_3d is not None:
        lidar = Lidar(
            metadata=lidar_metadatas[lidar_type],
            point_cloud_3d=point_cloud_3d,
            point_cloud_features=point_cloud_feature,
        )

    return lidar


def _unoptimized_demo_mp4_read(log_metadata: LogMetadata, camera_name: str, frame_index: int) -> Optional[np.ndarray]:
    """Reads a frame from an MP4 file for demonstration purposes. This features is not optimized for performance.

    :param log_metadata: The metadata of the log containing the MP4 file.
    :param camera_name: The name of the camera whose MP4 file is to be read.
    :param frame_index: The index of the frame to read from the MP4 file.
    :return: The image frame as a numpy array, or None if the file does not exist.
    """
    image: Optional[npt.NDArray[np.uint8]] = None

    py123d_sensor_root = get_dataset_paths().py123d_sensors_root
    assert py123d_sensor_root is not None, "PY123D_DATA_ROOT must be set for MP4 reading."
    mp4_path = py123d_sensor_root / log_metadata.split / log_metadata.log_name / f"{camera_name}.mp4"
    if mp4_path.exists():
        reader = get_mp4_reader_from_path(str(mp4_path))
        image = reader.get_frame(frame_index)

    return image


def _get_optional_array_mixin(data: Optional[Union[List, npt.NDArray]], cls: Type[ArrayMixin]) -> Optional[ArrayMixin]:
    """Builds an optional ArrayMixin if data is provided.

    :param data: The data to convert into an ArrayMixin.
    :param cls: The ArrayMixin class to instantiate.
    :raises ValueError: If the data type is unsupported.
    :return: The instantiated ArrayMixin, or None if data is None.
    """
    if data is None:
        return None
    if isinstance(data, list):
        return cls.from_list(data)
    elif isinstance(data, np.ndarray):
        return cls.from_array(data, copy=False)
    else:
        raise ValueError(f"Unsupported data type for ArrayMixin conversion: {type(data)}")


def _all_columns_in_schema(arrow_table: pa.Table, columns: List[str]) -> bool:
    """Checks if all specified columns are present in the Arrow table schema.

    :param arrow_table: The Arrow table to check.
    :param columns: The list of column names to check for.
    :return: True if all columns are present, False otherwise.
    """
    return all(column in arrow_table.schema.names for column in columns)
