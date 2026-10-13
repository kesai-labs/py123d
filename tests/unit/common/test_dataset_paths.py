import os
from pathlib import Path

import pytest

from py123d.common.runtime import DatasetPaths, get_dataset_paths, setup_dataset_paths
from py123d.common.runtime.dataset_paths import _ENV_VAR_MAP


@pytest.fixture
def without_env_vars(monkeypatch):
    """Removes the dataset path variables for a test, and restores the environment afterwards."""
    for env_var in _ENV_VAR_MAP.values():
        # delenv alone does not restore a variable that was not set before and that the test then exports
        monkeypatch.setenv(env_var, "")
        monkeypatch.delenv(env_var)


class TestDatasetPathsConstruction:
    """Tests for DatasetPaths dataclass construction and defaults."""

    def test_all_none_by_default(self):
        """All fields should be None when no arguments are provided."""
        paths = DatasetPaths()
        assert paths.py123d_data_root is None
        assert paths.py123d_logs_root is None
        assert paths.nuplan_data_root is None

    def test_primary_root_stored(self):
        """Primary roots should be stored as-is."""
        paths = DatasetPaths(py123d_data_root=Path("/data"))
        assert paths.py123d_data_root == Path("/data")

    def test_frozen(self):
        """DatasetPaths should be immutable."""
        paths = DatasetPaths()
        with pytest.raises(AttributeError):
            paths.py123d_data_root = Path("/data")  # type: ignore[misc]


class TestDerivedPaths:
    """Tests for derived path computation in __post_init__."""

    def test_py123d_derived_paths(self):
        """py123d sub-paths should be derived from py123d_data_root."""
        paths = DatasetPaths(py123d_data_root=Path("/data"))
        assert paths.py123d_logs_root == Path("/data/logs")
        assert paths.py123d_maps_root == Path("/data/maps")
        assert paths.py123d_sensors_root == Path("/data/sensors")
        assert paths.py123d_download_root == Path("/data/.download")

    def test_download_root_from_env(self, monkeypatch):
        """PY123D_DOWNLOAD_ROOT takes the place of the folder inside the data root, also in child processes."""
        monkeypatch.setenv("PY123D_DATA_ROOT", "/data")
        monkeypatch.setenv("PY123D_DOWNLOAD_ROOT", "/scratch/downloads")
        assert DatasetPaths.from_env().py123d_download_root == Path("/scratch/downloads")

    def test_av2_derived_path(self):
        """av2_sensor_data_root should be derived from av2_data_root."""
        paths = DatasetPaths(av2_data_root=Path("/av2"))
        assert paths.av2_sensor_data_root == Path("/av2/sensor")

    def test_nuscenes_derived_paths(self):
        """nuScenes derived paths should equal nuscenes_data_root."""
        paths = DatasetPaths(nuscenes_data_root=Path("/nuscenes"))
        assert paths.nuscenes_map_root == Path("/nuscenes")
        assert paths.nuscenes_sensor_root == Path("/nuscenes")

    def test_derived_paths_none_when_parent_none(self):
        """Derived paths should remain None when parent root is None."""
        paths = DatasetPaths()
        assert paths.py123d_logs_root is None
        assert paths.av2_sensor_data_root is None
        assert paths.nuscenes_map_root is None

    def test_explicit_derived_path_not_overwritten(self):
        """Explicitly provided derived paths should not be overwritten."""
        paths = DatasetPaths(
            py123d_data_root=Path("/data"),
            py123d_logs_root=Path("/custom/logs"),
        )
        assert paths.py123d_logs_root == Path("/custom/logs")
        # Other derived paths still computed from parent
        assert paths.py123d_maps_root == Path("/data/maps")


class TestFromEnv:
    """Tests for DatasetPaths.from_env()."""

    def test_reads_env_vars(self, monkeypatch):
        """from_env should read paths from environment variables."""
        monkeypatch.setenv("PY123D_DATA_ROOT", "/data")
        monkeypatch.setenv("NUPLAN_DATA_ROOT", "/nuplan")
        paths = DatasetPaths.from_env()
        assert paths.py123d_data_root == Path("/data")
        assert paths.nuplan_data_root == Path("/nuplan")
        assert paths.py123d_logs_root == Path("/data/logs")

    def test_missing_env_vars_are_none(self, monkeypatch):
        """Unset env vars should result in None fields."""
        for env_var in _ENV_VAR_MAP.values():
            monkeypatch.delenv(env_var, raising=False)
        paths = DatasetPaths.from_env()
        assert paths.py123d_data_root is None
        assert paths.nuplan_data_root is None


class TestFromDictConfig:
    """Tests for DatasetPaths.from_dict_config()."""

    def test_converts_dict_config(self):
        """from_dict_config should convert OmegaConf DictConfig to DatasetPaths."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"py123d_data_root": "/data", "nuplan_data_root": "/nuplan"})
        paths = DatasetPaths.from_dict_config(cfg)
        assert paths.py123d_data_root == Path("/data")
        assert paths.nuplan_data_root == Path("/nuplan")
        assert paths.py123d_logs_root == Path("/data/logs")

    def test_null_values_become_none(self):
        """OmegaConf null values should be treated as None."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"py123d_data_root": None})
        paths = DatasetPaths.from_dict_config(cfg)
        assert paths.py123d_data_root is None


class TestExportToEnv:
    """Tests for DatasetPaths.export_to_env()."""

    def test_exports_primary_roots(self, without_env_vars):
        """export_to_env should set env vars for non-None primary roots."""
        paths = DatasetPaths(py123d_data_root=Path("/data"), av2_data_root=Path("/av2"))
        paths.export_to_env()

        assert os.environ["PY123D_DATA_ROOT"] == "/data"
        assert os.environ["AV2_DATA_ROOT"] == "/av2"
        assert "NUPLAN_DATA_ROOT" not in os.environ

    def test_roundtrip_env(self, without_env_vars):
        """from_dict_config -> export_to_env -> from_env should produce equivalent paths."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "py123d_data_root": "/data",
                "nuplan_data_root": "/nuplan",
                "av2_data_root": "/av2",
            }
        )
        original = DatasetPaths.from_dict_config(cfg)
        original.export_to_env()
        restored = DatasetPaths.from_env()

        assert original.py123d_data_root == restored.py123d_data_root
        assert original.py123d_logs_root == restored.py123d_logs_root
        assert original.nuplan_data_root == restored.nuplan_data_root
        assert original.av2_data_root == restored.av2_data_root
        assert original.av2_sensor_data_root == restored.av2_sensor_data_root


class TestGetSensorRoot:
    """Tests for DatasetPaths.get_sensor_root()."""

    def test_known_datasets(self):
        """get_sensor_root should return correct paths for known datasets."""
        paths = DatasetPaths(
            nuplan_sensor_root=Path("/nuplan/sensor"),
            av2_data_root=Path("/av2"),
            nuscenes_data_root=Path("/nuscenes"),
            wod_perception_data_root=Path("/wod"),
            pandaset_data_root=Path("/pandaset"),
            kitti360_data_root=Path("/kitti360"),
        )
        assert paths.get_sensor_root("nuplan") == Path("/nuplan/sensor")
        assert paths.get_sensor_root("av2-sensor") == Path("/av2/sensor")
        assert paths.get_sensor_root("nuscenes") == Path("/nuscenes")
        assert paths.get_sensor_root("wod-perception") == Path("/wod")
        assert paths.get_sensor_root("wod_perception") == Path("/wod")  # legacy alias
        assert paths.get_sensor_root("pandaset") == Path("/pandaset")
        assert paths.get_sensor_root("kitti360") == Path("/kitti360")

    def test_unknown_dataset_returns_none(self):
        """get_sensor_root should return None for unknown datasets."""
        paths = DatasetPaths()
        assert paths.get_sensor_root("unknown_dataset") is None


class TestGlobalAccessor:
    """Tests for setup_dataset_paths and get_dataset_paths."""

    def test_get_dataset_paths_returns_from_env(self, monkeypatch):
        """get_dataset_paths should create from env when no global is set."""
        import py123d.common.runtime.dataset_paths as module

        monkeypatch.setattr(module, "_global_dataset_paths", None)
        monkeypatch.setenv("PY123D_DATA_ROOT", "/from_env")

        paths = get_dataset_paths()
        assert paths.py123d_data_root == Path("/from_env")

        # Clean up
        monkeypatch.setattr(module, "_global_dataset_paths", None)

    def test_setup_then_get(self, monkeypatch, without_env_vars):
        """setup_dataset_paths should store the provided instance."""
        import py123d.common.runtime.dataset_paths as module

        monkeypatch.setattr(module, "_global_dataset_paths", None)

        custom = DatasetPaths(py123d_data_root=Path("/custom"))
        setup_dataset_paths(custom)

        assert get_dataset_paths() is custom
        assert get_dataset_paths().py123d_data_root == Path("/custom")

        # Clean up
        monkeypatch.setattr(module, "_global_dataset_paths", None)

    def test_setup_only_sets_once(self, monkeypatch, without_env_vars):
        """setup_dataset_paths should not overwrite an existing global."""
        import py123d.common.runtime.dataset_paths as module

        monkeypatch.setattr(module, "_global_dataset_paths", None)

        first = DatasetPaths(py123d_data_root=Path("/first"))
        second = DatasetPaths(py123d_data_root=Path("/second"))
        setup_dataset_paths(first)
        setup_dataset_paths(second)

        assert get_dataset_paths().py123d_data_root == Path("/first")

        # Clean up
        monkeypatch.setattr(module, "_global_dataset_paths", None)
