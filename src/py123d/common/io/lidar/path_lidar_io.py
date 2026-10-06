from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from py123d.common.runtime import get_dataset_paths
from py123d.datatypes.sensors.lidar import LidarID, LidarMetadata


def load_point_cloud_data_from_path(
    relative_path: Union[str, Path],
    dataset: str,
    index: Optional[int] = None,
    sensor_root: Optional[Union[str, Path]] = None,
    lidar_metadatas: Optional[Dict[LidarID, LidarMetadata]] = None,
    load_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    # NOTE @DanielDauner: This function is designed s.t. it can load multiple lidar types at the same time.
    # Several datasets (e.g., PandaSet, nuScenes) have multiple Lidar sensors stored in one file.
    # Returning this as a dict allows us to handle this case without unnecessary io overhead.

    assert relative_path is not None, "Relative path to Lidar file must be provided."
    if sensor_root is None:
        sensor_root = get_dataset_paths().get_sensor_root(dataset)
        assert sensor_root is not None, f"Dataset path for sensor loading not found for dataset: {dataset}."

    full_lidar_path = Path(sensor_root) / relative_path
    assert full_lidar_path.exists(), f"Lidar file not found: {sensor_root} / {relative_path}"

    # NOTE: We move data specific import into if-else block, to avoid data specific import errors
    if dataset == "nuplan":
        from py123d.parser.nuplan.nuplan_sensor_io import load_nuplan_point_cloud_data_from_path

        lidar_pcs_dict = load_nuplan_point_cloud_data_from_path(full_lidar_path)

    elif dataset == "av2-sensor":
        from py123d.parser.av2.av2_sensor_io import load_av2_sensor_point_cloud_data_from_path

        lidar_pcs_dict = load_av2_sensor_point_cloud_data_from_path(full_lidar_path)

    elif dataset in ("wod-perception", "wod_perception"):  # "wod_perception" retained for legacy logs
        from py123d.parser.wod.wod_perception_sensor_io import (
            load_wod_perception_point_cloud_data_from_path,
        )

        assert index is not None, "Index must be provided for WOD Perception Lidar loading."
        lidar_pcs_dict = load_wod_perception_point_cloud_data_from_path(
            full_lidar_path, index, keep_polar_features=True
        )

    elif dataset == "pandaset":
        from py123d.parser.pandaset.pandaset_sensor_io import load_pandaset_point_cloud_data_from_path

        lidar_pcs_dict = load_pandaset_point_cloud_data_from_path(full_lidar_path)

    elif dataset == "kitti360":
        from py123d.parser.kitti360.kitti360_sensor_io import load_kitti360_point_cloud_data_from_path

        assert lidar_metadatas is not None, "Lidar metadatas must be provided for KITTI-360 Lidar loading."
        lidar_pcs_dict = load_kitti360_point_cloud_data_from_path(full_lidar_path, lidar_metadatas)

    elif dataset == "nuscenes":
        from py123d.parser.nuscenes.nuscenes_sensor_io import load_nuscenes_point_cloud_data_from_path

        assert lidar_metadatas is not None, "Lidar metadatas must be provided for nuScenes Lidar loading."
        # nuScenes segmentation labels live in separate, opaquely-named files (semantic in lidarseg,
        # instance in panoptic); the converter stashes their relative paths in load_kwargs so we can
        # re-read them here, resolved against the same sensor_root as the point cloud.
        kw = load_kwargs or {}
        lidarseg_path = Path(sensor_root) / kw["lidarseg_relative_path"] if kw.get("lidarseg_relative_path") else None
        panoptic_path = Path(sensor_root) / kw["panoptic_relative_path"] if kw.get("panoptic_relative_path") else None
        lidar_pcs_dict = load_nuscenes_point_cloud_data_from_path(
            full_lidar_path, lidar_metadatas, lidarseg_path=lidarseg_path, panoptic_path=panoptic_path
        )

    elif dataset == "physical-ai-av":
        from py123d.parser.physical_ai_av.physical_ai_av_sensor_io import (
            load_physical_ai_av_point_cloud_data_from_path,
        )

        assert index is not None, "Index (spin_index) must be provided for Physical AI AV LiDAR loading."
        assert lidar_metadatas is not None, "Lidar metadatas must be provided for Physical AI AV LiDAR loading."
        lidar_pcs_dict = load_physical_ai_av_point_cloud_data_from_path(full_lidar_path, index, lidar_metadatas)

    elif dataset == "griffin":
        from py123d.parser.griffin.griffin_sensor_io import load_griffin_point_cloud_data_from_path

        lidar_pcs_dict = load_griffin_point_cloud_data_from_path(full_lidar_path)

    elif dataset == "ncore":
        from py123d.parser.ncore.ncore_sensor_io import load_ncore_point_cloud_data_from_path

        assert index is not None, "Index (end-of-frame timestamp) must be provided for NCore LiDAR loading."
        assert lidar_metadatas is not None, "Lidar metadatas must be provided for NCore LiDAR loading."
        lidar_pcs_dict = load_ncore_point_cloud_data_from_path(full_lidar_path, index, lidar_metadatas)

    elif dataset == "nureasoning":
        from py123d.parser.nureasoning.nureasoning_sensor_io import load_nureasoning_point_cloud_data_from_path

        lidar_pcs_dict = load_nureasoning_point_cloud_data_from_path(full_lidar_path)
    elif dataset == "truckdrive":
        from py123d.parser.truckdrive.truckdrive_sensor_io import load_truckdrive_point_cloud_data_from_path

        lidar_pcs_dict = load_truckdrive_point_cloud_data_from_path(full_lidar_path, lidar_metadatas)

    elif dataset == "kitscenes":
        from py123d.parser.kitscenes.kitscenes_sensor_io import load_kitscenes_point_cloud_data_from_path

        lidar_pcs_dict = load_kitscenes_point_cloud_data_from_path(full_lidar_path)

    else:
        raise NotImplementedError(f"Loading Lidar data for dataset {dataset} is not implemented.")

    return lidar_pcs_dict
