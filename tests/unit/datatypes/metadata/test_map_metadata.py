import pytest

from py123d.datatypes.metadata.abstract_metadata import AbstractMetadata
from py123d.datatypes.metadata.map_metadata import MapMetadata


class TestMapMetadata:
    def test_map_metadata_initialization(self):
        """Test that MapMetadata can be initialized with required fields."""
        metadata = MapMetadata(
            dataset="test_dataset",
            split="train",
            log_name="log_001",
            location="test_location",
            map_has_z=True,
            map_is_per_log=False,
        )

        assert metadata.dataset == "test_dataset"
        assert metadata.split == "train"
        assert metadata.log_name == "log_001"
        assert metadata.location == "test_location"
        assert metadata.map_has_z is True
        assert metadata.map_is_per_log is False
        assert metadata.version is not None

    def test_map_metadata_to_dict(self):
        """Test conversion of MapMetadata to dictionary."""
        metadata = MapMetadata(
            dataset="test_dataset",
            split="val",
            log_name="log_002",
            location="test_location",
            map_has_z=False,
            map_is_per_log=True,
        )

        result = metadata.to_dict()

        assert isinstance(result, dict)
        assert result["dataset"] == "test_dataset"
        assert result["split"] == "val"
        assert result["log_name"] == "log_002"
        assert result["location"] == "test_location"
        assert result["map_has_z"] is False
        assert result["map_is_per_log"] is True
        assert "version" in result

    def test_map_metadata_from_dict(self):
        """Test creation of MapMetadata from dictionary."""
        data = {
            "dataset": "test_dataset",
            "split": "test",
            "log_name": "log_003",
            "location": "test_location",
            "map_has_z": True,
            "map_is_per_log": False,
            "version": "1.0.0",
        }

        metadata = MapMetadata.from_dict(data)

        assert metadata.dataset == "test_dataset"
        assert metadata.split == "test"
        assert metadata.log_name == "log_003"
        assert metadata.location == "test_location"
        assert metadata.map_has_z is True
        assert metadata.map_is_per_log is False
        assert metadata.version == "1.0.0"

    def test_map_metadata_with_none_values(self):
        """Test MapMetadata with None values for optional fields."""
        metadata = MapMetadata(
            dataset="test_dataset",
            split=None,
            log_name=None,
            location="test_location",
            map_has_z=True,
            map_is_per_log=False,
        )

        assert metadata.split is None
        assert metadata.log_name is None
        assert metadata.dataset == "test_dataset"

    def test_map_metadata_roundtrip(self):
        """Test that converting to dict and back preserves data."""
        original = MapMetadata(
            dataset="roundtrip_dataset",
            split="train",
            log_name="log_roundtrip",
            location="location_test",
            map_has_z=False,
            map_is_per_log=True,
            version="2.0.0",
        )

        data_dict = original.to_dict()
        restored = MapMetadata.from_dict(data_dict)

        assert restored.dataset == original.dataset
        assert restored.split == original.split
        assert restored.log_name == original.log_name
        assert restored.location == original.location
        assert restored.map_has_z == original.map_has_z
        assert restored.map_is_per_log == original.map_is_per_log
        assert restored.version == original.version

    def test_is_instance_of_abstract_metadata(self):
        """MapMetadata is an instance of AbstractMetadata."""
        metadata = MapMetadata(
            dataset="test_dataset",
            location="test_location",
            map_has_z=True,
            map_is_per_log=False,
        )
        assert isinstance(metadata, AbstractMetadata)

    def test_per_log_map_requires_split_and_log_name(self):
        """Test that per-log maps raise AssertionError when split or log_name is missing."""
        with pytest.raises(AssertionError, match="split must be provided"):
            MapMetadata(
                dataset="test_dataset",
                location="test_location",
                map_has_z=True,
                map_is_per_log=True,
                split=None,
                log_name="log_001",
            )

        with pytest.raises(AssertionError, match="log_name must be provided"):
            MapMetadata(
                dataset="test_dataset",
                location="test_location",
                map_has_z=True,
                map_is_per_log=True,
                split="train",
                log_name=None,
            )
