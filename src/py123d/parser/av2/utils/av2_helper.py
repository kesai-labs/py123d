from pathlib import Path
from typing import Dict, List, Literal, Optional

import pandas as pd

from py123d.geometry.pose import PoseSE3
from py123d.parser.av2.utils.av2_constants import (
    AV2_CAMERA_ID_MAPPING,
    AV2_SENSOR_CAM_SHUTTER_INTERVAL_MS,
    AV2_SENSOR_LIDAR_SWEEP_INTERVAL_W_BUFFER_NS,
)


def get_dataframe_from_file(file_path: Path) -> pd.DataFrame:
    """Get a Pandas DataFrame from parquet or feather files."""
    if file_path.suffix == ".parquet":
        import pyarrow.parquet as pq

        result: pd.DataFrame = pq.read_table(file_path).to_pandas()
    elif file_path.suffix == ".feather":
        result = pd.read_feather(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
    return result


def get_slice_with_timestamp_ns(dataframe: pd.DataFrame, timestamp_ns: int) -> pd.DataFrame:
    """Get all rows matching the given nanosecond timestamp."""
    return dataframe[dataframe["timestamp_ns"] == timestamp_ns]


def build_sensor_dataframe(source_log_path: Path) -> pd.DataFrame:
    """Builds a sensor dataframe from the AV2 source log path."""

    # https://github.com/argoverse/av2-api/blob/main/src/av2/datasets/sensor/sensor_dataloader.py#L209

    split = source_log_path.parent.name
    log_id = source_log_path.name

    lidar_path = source_log_path / "sensors" / "lidar"
    cameras_path = source_log_path / "sensors" / "cameras"

    # Find all the lidar records and timestamps from file names.
    lidar_records = populate_sensor_records(lidar_path, split, log_id)

    # Find all the camera records and timestamps from file names.
    camera_records = []
    for camera_folder in cameras_path.iterdir():
        assert camera_folder.name in AV2_CAMERA_ID_MAPPING.keys()
        camera_record = populate_sensor_records(camera_folder, split, log_id)
        camera_records.append(camera_record)

    # Concatenate all the camera records into a single DataFrame.
    sensor_records = [lidar_records] + camera_records
    sensor_dataframe = pd.concat(sensor_records)

    # Set index as tuples of the form: (split, log_id, sensor_name, timestamp_ns) and sort the index.
    # sorts by split, log_id, and then by sensor name, and then by timestamp.
    sensor_dataframe.set_index(["split", "log_id", "sensor_name", "timestamp_ns"], inplace=True)
    sensor_dataframe.sort_index(inplace=True)

    return sensor_dataframe


def build_synchronization_dataframe(
    sensor_dataframe: pd.DataFrame,
    camera_camera_matching: Literal["nearest", "forward", "backward"] = "nearest",
    lidar_camera_matching: Literal["nearest", "sweep"] = "sweep",
) -> pd.DataFrame:
    """Builds a synchronization dataframe, between sensors observations in a log.

    :param sensor_dataframe: DataFrame containing sensor data.
    :param camera_camera_matching: Criterion for matching camera-to-camera timestamps, defaults to "nearest".
    :param lidar_camera_matching: Criterion for matching lidar-to-camera timestamps. "sweep" (default) uses
        forward/backward matching to find cameras captured during the lidar sweep window. "nearest" finds the
        closest camera frame regardless of temporal ordering.
    :return: DataFrame containing synchronized sensor data.
    """

    # https://github.com/argoverse/av2-api/blob/main/src/av2/datasets/sensor/sensor_dataloader.py#L382

    # Create list to store synchronized data frames.
    sync_list: List[pd.DataFrame] = []
    unique_sensor_names: List[str] = sensor_dataframe.index.unique(level=2).tolist()

    # Associate a 'source' sensor to a 'target' sensor for all available sensors.
    # For example, we associate the lidar sensor with each ring camera which
    # produces a mapping from lidar -> all-other-sensors.
    for src_sensor_name in unique_sensor_names:
        src_records = sensor_dataframe.xs(src_sensor_name, level=2, drop_level=False).reset_index()
        src_records = src_records.rename({"timestamp_ns": src_sensor_name}, axis=1).sort_values(src_sensor_name)

        # _Very_ important to convert to timedelta. Tolerance below causes precision loss otherwise.
        src_records[src_sensor_name] = pd.to_timedelta(src_records[src_sensor_name])
        for target_sensor_name in unique_sensor_names:
            if src_sensor_name == target_sensor_name:
                continue
            target_records = sensor_dataframe.xs(target_sensor_name, level=2).reset_index()
            target_records = target_records.rename({"timestamp_ns": target_sensor_name}, axis=1).sort_values(
                target_sensor_name
            )

            # Merge based on matching criterion.
            # _Very_ important to convert to timedelta. Tolerance below causes precision loss otherwise.
            target_records[target_sensor_name] = pd.to_timedelta(target_records[target_sensor_name])

            # The lidar timestamp marks the START of the sweep (~100ms accumulation window).
            # "sweep" mode: lidar→camera uses "forward" to find cameras captured DURING the sweep,
            #   camera→lidar uses "backward" to find the sweep that contains the image.
            # "nearest" mode: finds the closest match regardless of temporal ordering.
            # For camera→camera: always use the caller-provided matching_criterion.
            if src_sensor_name == "lidar" or target_sensor_name == "lidar":
                if lidar_camera_matching == "nearest":
                    direction = "nearest"
                    tolerance = pd.to_timedelta(AV2_SENSOR_LIDAR_SWEEP_INTERVAL_W_BUFFER_NS)
                elif src_sensor_name == "lidar":
                    direction = "forward"
                    tolerance = pd.to_timedelta(AV2_SENSOR_LIDAR_SWEEP_INTERVAL_W_BUFFER_NS)
                else:
                    direction = "backward"
                    tolerance = pd.to_timedelta(AV2_SENSOR_LIDAR_SWEEP_INTERVAL_W_BUFFER_NS)
            else:
                direction = camera_camera_matching
                tolerance = pd.to_timedelta(AV2_SENSOR_CAM_SHUTTER_INTERVAL_MS / 2 * 1e6)

            src_records = pd.merge_asof(
                src_records,
                target_records,
                left_on=src_sensor_name,
                right_on=target_sensor_name,
                by=["split", "log_id"],
                direction=direction,
                tolerance=tolerance,
            )
        sync_list.append(src_records)

    sync_records = pd.concat(sync_list).reset_index(drop=True)
    sync_records.set_index(keys=["split", "log_id", "sensor_name"], inplace=True)
    sync_records.sort_index(inplace=True)

    return sync_records


def populate_sensor_records(sensor_path: Path, split: str, log_id: str) -> pd.DataFrame:
    """Populate sensor records from a sensor path."""

    sensor_name = sensor_path.name
    sensor_files = list(sensor_path.iterdir())
    sensor_records = []

    for sensor_file in sensor_files:
        assert sensor_file.suffix in {
            ".feather",
            ".jpg",
        }, f"Unsupported file type: {sensor_file.suffix} for {str(sensor_file)}"
        row = {}
        row["split"] = split
        row["log_id"] = log_id
        row["sensor_name"] = sensor_name
        row["timestamp_ns"] = int(sensor_file.stem)
        sensor_records.append(row)

    return pd.DataFrame(sensor_records)


def find_closest_target_fpath(
    split: str,
    log_id: str,
    src_sensor_name: str,
    src_timestamp_ns: int,
    target_sensor_name: str,
    synchronization_df: pd.DataFrame,
) -> Optional[Path]:
    """Find the file path to the target sensor from a source sensor."""
    # https://github.com/argoverse/av2-api/blob/6b22766247eda941cb1953d6a58e8d5631c561da/src/av2/datasets/sensor/sensor_dataloader.py#L448

    src_timedelta_ns = pd.Timedelta(src_timestamp_ns)
    src_to_target_records: pd.DataFrame = synchronization_df.loc[(split, log_id, src_sensor_name)].set_index(  # type: ignore
        src_sensor_name
    )  # type: ignore[assignment]
    index = src_to_target_records.index
    if src_timedelta_ns not in index:
        # This timestamp does not correspond to any lidar sweep.
        return None

    # Grab the synchronization record.
    target_timestamp_ns = src_to_target_records.loc[src_timedelta_ns, target_sensor_name]  # type: ignore
    if pd.isna(target_timestamp_ns):
        # No match was found within tolerance.
        return None

    sensor_dir = Path(split) / log_id / "sensors"
    valid_cameras = list(AV2_CAMERA_ID_MAPPING.keys())
    timestamp_ns_str = str(target_timestamp_ns.asm8.item())
    if target_sensor_name in valid_cameras:
        target_path = sensor_dir / "cameras" / target_sensor_name / f"{timestamp_ns_str}.jpg"
    else:
        target_path = sensor_dir / target_sensor_name / f"{timestamp_ns_str}.feather"
    return target_path


def av2_row_dict_to_pose_se3(row_dict: Dict) -> PoseSE3:
    """Helper function to convert a row dictionary to a PoseSE3 object."""
    return PoseSE3(
        x=row_dict["tx_m"],
        y=row_dict["ty_m"],
        z=row_dict["tz_m"],
        qw=row_dict["qw"],
        qx=row_dict["qx"],
        qy=row_dict["qy"],
        qz=row_dict["qz"],
    )
