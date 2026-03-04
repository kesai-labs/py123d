from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from py123d.common.dataset_paths import get_dataset_paths
from py123d.datatypes.metadata.log_metadata import LogMetadata
from py123d.datatypes.metadata.sensor_metadata import LidarMetadatas


def load_point_cloud_data_from_path(
    relative_path: Union[str, Path],
    log_metadata: LogMetadata,
    index: Optional[int] = None,
    sensor_root: Optional[Union[str, Path]] = None,
    lidar_metadatas: Optional[LidarMetadatas] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    # NOTE @DanielDauner: This function is designed s.t. it can load multiple lidar types at the same time.
    # Several datasets (e.g., PandaSet, nuScenes) have multiple Lidar sensors stored in one file.
    # Returning this as a dict allows us to handle this case without unnecessary io overhead.

    assert relative_path is not None, "Relative path to Lidar file must be provided."
    if sensor_root is None:
        sensor_root = get_dataset_paths().get_sensor_root(log_metadata.dataset)
        assert sensor_root is not None, (
            f"Dataset path for sensor loading not found for dataset: {log_metadata.dataset}."
        )

    full_lidar_path = Path(sensor_root) / relative_path
    assert full_lidar_path.exists(), f"Lidar file not found: {sensor_root} / {relative_path}"

    # NOTE: We move data specific import into if-else block, to avoid data specific import errors
    if log_metadata.dataset == "nuplan":
        from py123d.conversion.datasets.nuplan.nuplan_sensor_io import load_nuplan_point_cloud_data_from_path

        lidar_pcs_dict = load_nuplan_point_cloud_data_from_path(full_lidar_path)

    elif log_metadata.dataset == "av2-sensor":
        from py123d.conversion.datasets.av2.av2_sensor_io import load_av2_sensor_point_cloud_data_from_path

        lidar_pcs_dict = load_av2_sensor_point_cloud_data_from_path(full_lidar_path)

    elif log_metadata.dataset == "wod_perception":
        from py123d.conversion.datasets.wod.wod_perception_sensor_io import (
            load_wod_perception_point_cloud_data_from_path,
        )

        assert index is not None, "Index must be provided for WOD Perception Lidar loading."
        lidar_pcs_dict = load_wod_perception_point_cloud_data_from_path(
            full_lidar_path, index, keep_polar_features=True
        )

    elif log_metadata.dataset == "pandaset":
        from py123d.conversion.datasets.pandaset.pandaset_sensor_io import load_pandaset_point_cloud_data_from_path

        lidar_pcs_dict = load_pandaset_point_cloud_data_from_path(full_lidar_path, index)

    elif log_metadata.dataset == "kitti360":
        from py123d.conversion.datasets.kitti360.kitti360_sensor_io import load_kitti360_point_cloud_data_from_path

        assert lidar_metadatas is not None, "Lidar metadatas must be provided for KITTI-360 Lidar loading."
        lidar_pcs_dict = load_kitti360_point_cloud_data_from_path(full_lidar_path, lidar_metadatas)

    elif log_metadata.dataset == "nuscenes":
        from py123d.conversion.datasets.nuscenes.nuscenes_sensor_io import load_nuscenes_point_cloud_data_from_path

        assert lidar_metadatas is not None, "Lidar metadatas must be provided for nuScenes Lidar loading."
        lidar_pcs_dict = load_nuscenes_point_cloud_data_from_path(full_lidar_path, lidar_metadatas)

    else:
        raise NotImplementedError(f"Loading Lidar data for dataset {log_metadata.dataset} is not implemented.")

    return lidar_pcs_dict
