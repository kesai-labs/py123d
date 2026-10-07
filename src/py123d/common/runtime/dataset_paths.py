"""Dataset path management.

Provides a typed, frozen dataclass for all dataset root paths with:
- Environment variable resolution (works in any process)
- Hydra DictConfig conversion (main process entry point)
- Env var export for child process inheritance
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)

# Mapping from DatasetPaths field name to environment variable name.
# Only primary roots are mapped — derived paths are computed in __post_init__.
_ENV_VAR_MAP: Dict[str, str] = {
    "py123d_data_root": "PY123D_DATA_ROOT",
    "nuplan_data_root": "NUPLAN_DATA_ROOT",
    "nuplan_maps_root": "NUPLAN_MAPS_ROOT",
    "nuplan_sensor_root": "NUPLAN_SENSOR_ROOT",
    "av2_data_root": "AV2_DATA_ROOT",
    "wod_perception_data_root": "WOD_PERCEPTION_DATA_ROOT",
    "wod_motion_data_root": "WOD_MOTION_DATA_ROOT",
    "pandaset_data_root": "PANDASET_DATA_ROOT",
    "kitti360_data_root": "KITTI360_DATA_ROOT",
    "nuscenes_data_root": "NUSCENES_DATA_ROOT",
    "nureasoning_data_root": "NUREASONING_DATA_ROOT",
    "griffin_data_root": "GRIFFIN_DATA_ROOT",
    "truckdrive_data_root": "TRUCKDRIVE_DATA_ROOT",
    "kitscenes_data_root": "KITSCENES_DATA_ROOT",
}


@dataclass(frozen=True)
class DatasetPaths:
    """Immutable container for all dataset root paths.

    Provides type-safe access to dataset paths with IDE autocompletion.
    A ``None`` value means the corresponding dataset is not configured.

    To add a new dataset:
        1. Add primary root field(s) below.
        2. Add the env var mapping to ``_ENV_VAR_MAP``.
        3. Optionally add derived paths in ``__post_init__``.
        4. Optionally extend ``get_sensor_root()`` if the dataset has sensor data.
    """

    # 1. Primary root in 123D
    py123d_data_root: Optional[Path] = None

    # 2. Main data roots for each dataset.
    nuplan_data_root: Optional[Path] = None
    av2_data_root: Optional[Path] = None
    wod_perception_data_root: Optional[Path] = None
    wod_motion_data_root: Optional[Path] = None
    pandaset_data_root: Optional[Path] = None
    kitti360_data_root: Optional[Path] = None
    nuscenes_data_root: Optional[Path] = None
    nureasoning_data_root: Optional[Path] = None
    griffin_data_root: Optional[Path] = None
    truckdrive_data_root: Optional[Path] = None
    kitscenes_data_root: Optional[Path] = None

    # 2. Derived paths (if not explicitly set, will be derived from primary roots in __post_init__)
    py123d_logs_root: Optional[Path] = field(default=None, repr=False)
    py123d_maps_root: Optional[Path] = field(default=None, repr=False)
    py123d_sensors_root: Optional[Path] = field(default=None, repr=False)

    nuplan_maps_root: Optional[Path] = field(default=None, repr=False)
    nuplan_sensor_root: Optional[Path] = field(default=None, repr=False)
    av2_sensor_data_root: Optional[Path] = field(default=None, repr=False)
    nuscenes_map_root: Optional[Path] = field(default=None, repr=False)
    nuscenes_sensor_root: Optional[Path] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Derive child paths from parent roots when not explicitly provided."""
        _derive = object.__setattr__  # frozen dataclass workaround

        if self.py123d_logs_root is None and self.py123d_data_root is not None:
            _derive(self, "py123d_logs_root", self.py123d_data_root / "logs")
        if self.py123d_maps_root is None and self.py123d_data_root is not None:
            _derive(self, "py123d_maps_root", self.py123d_data_root / "maps")
        if self.py123d_sensors_root is None and self.py123d_data_root is not None:
            _derive(self, "py123d_sensors_root", self.py123d_data_root / "sensors")

        if self.nuplan_maps_root is None and self.nuplan_data_root is not None:
            _derive(self, "nuplan_maps_root", self.nuplan_data_root / "maps")
        if self.nuplan_sensor_root is None and self.nuplan_data_root is not None:
            _derive(self, "nuplan_sensor_root", self.nuplan_data_root / "nuplan-v1.1" / "sensor_blobs")
        if self.av2_sensor_data_root is None and self.av2_data_root is not None:
            _derive(self, "av2_sensor_data_root", self.av2_data_root / "sensor")
        if self.nuscenes_map_root is None and self.nuscenes_data_root is not None:
            _derive(self, "nuscenes_map_root", self.nuscenes_data_root)
        if self.nuscenes_sensor_root is None and self.nuscenes_data_root is not None:
            _derive(self, "nuscenes_sensor_root", self.nuscenes_data_root)

    @classmethod
    def from_env(cls) -> DatasetPaths:
        """Create DatasetPaths by reading environment variables.

        Works in any process because env vars are inherited by child processes.
        """
        kwargs = {}
        for field_name, env_var in _ENV_VAR_MAP.items():
            value = os.environ.get(env_var)
            if value is not None:
                kwargs[field_name] = Path(value)
        return cls(**kwargs)

    @classmethod
    def from_dict_config(cls, cfg: object) -> DatasetPaths:
        """Create DatasetPaths from an OmegaConf DictConfig (Hydra integration).

        :param cfg: The ``dataset_paths`` sub-config from Hydra.
        """
        kwargs = {}
        for f in fields(cls):
            value = getattr(cfg, f.name, None)
            if value is not None:
                kwargs[f.name] = Path(str(value))
        return cls(**kwargs)

    def export_to_env(self) -> None:
        """Export non-None primary paths to environment variables.

        Ensures child processes (forkserver, Ray) inherit Hydra CLI overrides.
        """
        for field_name, env_var in _ENV_VAR_MAP.items():
            value = getattr(self, field_name)
            if value is not None:
                os.environ[env_var] = str(value)

    def get_sensor_root(self, dataset: str) -> Optional[Path]:
        """Get the sensor root path for a given dataset name.

        :param dataset: Dataset name (e.g., ``"nuplan"``, ``"av2-sensor"``).
        :return: The sensor root path, or ``None`` if not configured.
        """
        mapping: Dict[str, Optional[Path]] = {
            "av2-sensor": self.av2_sensor_data_root,
            "nuplan": self.nuplan_sensor_root,
            "nuscenes": self.nuscenes_sensor_root,
            "wod-perception": self.wod_perception_data_root,
            "wod_perception": self.wod_perception_data_root,  # legacy alias for pre-rename logs
            "pandaset": self.pandaset_data_root,
            "kitti360": self.kitti360_data_root,
            # nuReasoning stores cameras (jpeg_binary) and lidar (path) under the data root.
            "nureasoning": self.nureasoning_data_root,
            "griffin": self.griffin_data_root,
            "truckdrive": self.truckdrive_data_root,
            "kitscenes": self.kitscenes_data_root,
        }
        return mapping.get(dataset)


# ---------------------------------------------------------------------------
# Global accessor
# ---------------------------------------------------------------------------

_global_dataset_paths: Optional[DatasetPaths] = None


def setup_dataset_paths(value: Union[DatasetPaths, object]) -> None:
    """Set the global DatasetPaths instance.

    Accepts either a :class:`DatasetPaths` instance or any object with matching
    attributes (e.g., a Hydra ``DictConfig``), which is converted via
    :meth:`DatasetPaths.from_dict_config`. Always exports values to env vars so
    forked / Ray child processes inherit them. Idempotent: only the first call
    takes effect.

    :param value: A :class:`DatasetPaths` or a DictConfig-shaped object.
    """
    paths = value if isinstance(value, DatasetPaths) else DatasetPaths.from_dict_config(value)
    global _global_dataset_paths  # noqa: PLW0603
    if _global_dataset_paths is None:
        _global_dataset_paths = paths
    paths.export_to_env()


def get_dataset_paths() -> DatasetPaths:
    """Get the global DatasetPaths.

    If not explicitly set via :func:`setup_dataset_paths`, creates one from
    environment variables. This works in child processes because env vars are
    inherited.

    :return: The global DatasetPaths instance.
    """
    global _global_dataset_paths  # noqa: PLW0603
    if _global_dataset_paths is None:
        _global_dataset_paths = DatasetPaths.from_env()
    return _global_dataset_paths
