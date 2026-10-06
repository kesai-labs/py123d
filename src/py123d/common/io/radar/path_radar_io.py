from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from py123d.common.runtime import get_dataset_paths
from py123d.datatypes.sensors.radar import RadarID, RadarMetadata


def load_radar_point_cloud_data_from_path(
    relative_path: Union[str, Path],
    dataset: str,
    index: Optional[int] = None,
    sensor_root: Optional[Union[str, Path]] = None,
    radar_metadatas: Optional[Dict[RadarID, RadarMetadata]] = None,
    load_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    # NOTE: Designed analogously to the lidar loader (see path_lidar_io.load_point_cloud_data_from_path):
    # several datasets store multiple radar sensors merged into one file, so this returns xyz + a features
    # dict (with a per-point RadarFeature.IDS sensor tag) to handle the multi-sensor case without extra IO.

    assert relative_path is not None, "Relative path to Radar file must be provided."
    if sensor_root is None:
        sensor_root = get_dataset_paths().get_sensor_root(dataset)
        assert sensor_root is not None, f"Dataset path for sensor loading not found for dataset: {dataset}."

    full_radar_path = Path(sensor_root) / relative_path
    assert full_radar_path.exists(), f"Radar file not found: {sensor_root} / {relative_path}"

    # NOTE: We move data specific import into if-else block, to avoid data specific import errors
    if dataset == "nuscenes":
        from py123d.parser.nuscenes.nuscenes_radar_io import load_nuscenes_radar_point_cloud_data_from_path

        assert radar_metadatas is not None, "Radar metadatas must be provided for nuScenes Radar loading."
        kw = load_kwargs or {}
        radar_pcs_dict = load_nuscenes_radar_point_cloud_data_from_path(
            full_radar_path, radar_metadatas, sensor_root=Path(sensor_root), load_kwargs=kw
        )

    elif dataset == "physical-ai-av":
        from py123d.parser.physical_ai_av.physical_ai_av_radar_io import (
            load_physical_ai_av_radar_point_cloud_data_from_path,
        )

        assert index is not None, "Index (reference timestamp) must be provided for Physical AI AV Radar loading."
        assert radar_metadatas is not None, "Radar metadatas must be provided for Physical AI AV Radar loading."
        radar_pcs_dict = load_physical_ai_av_radar_point_cloud_data_from_path(
            full_radar_path, index, radar_metadatas, sensor_root=Path(sensor_root), load_kwargs=load_kwargs or {}
        )

    elif dataset == "ncore":
        from py123d.parser.ncore.ncore_radar_io import load_ncore_radar_point_cloud_data_from_path

        assert index is not None, "Index (end-of-frame timestamp) must be provided for NCore Radar loading."
        assert radar_metadatas is not None, "Radar metadatas must be provided for NCore Radar loading."
        radar_pcs_dict = load_ncore_radar_point_cloud_data_from_path(
            full_radar_path, index, radar_metadatas, sensor_root=Path(sensor_root), load_kwargs=load_kwargs or {}
        )

    elif dataset == "kitscenes":
        from py123d.parser.kitscenes.kitscenes_sensor_io import load_kitscenes_radar_point_cloud_data_from_path

        radar_pcs_dict = load_kitscenes_radar_point_cloud_data_from_path(full_radar_path)

    else:
        raise NotImplementedError(f"Loading Radar data for dataset {dataset} is not implemented.")

    return radar_pcs_dict
