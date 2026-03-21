from py123d.datatypes.metadata.base_metadata import BaseMetadata
from py123d.datatypes.metadata.log_metadata import LogMetadata


class TestLogMetadata:
    def test_init_minimal(self):
        """Test LogMetadata initialization with minimal required fields."""
        log_metadata = LogMetadata(dataset="test_dataset", split="train", log_name="log_001", location="test_location")
        assert log_metadata.dataset == "test_dataset"
        assert log_metadata.split == "train"
        assert log_metadata.log_name == "log_001"
        assert log_metadata.location == "test_location"

    def test_init_with_none_location(self):
        """Test LogMetadata initialization with None location."""
        log_metadata = LogMetadata(dataset="test_dataset", split="train", log_name="log_001", location=None)
        assert log_metadata.location is None

    def test_to_dict(self):
        """Test to_dict returns correct fields."""
        log_metadata = LogMetadata(dataset="test_dataset", split="train", log_name="log_001", location="test_location")
        result = log_metadata.to_dict()
        assert result["dataset"] == "test_dataset"
        assert result["split"] == "train"
        assert result["log_name"] == "log_001"
        assert result["location"] == "test_location"
        assert "version" in result

    def test_from_dict(self):
        """Test from_dict creates correct LogMetadata."""
        data_dict = {
            "dataset": "test_dataset",
            "split": "train",
            "log_name": "log_001",
            "location": "test_location",
            "version": "1.0.0",
        }
        log_metadata = LogMetadata.from_dict(data_dict)
        assert log_metadata.dataset == "test_dataset"
        assert log_metadata.split == "train"
        assert log_metadata.log_name == "log_001"
        assert log_metadata.location == "test_location"
        assert log_metadata.version == "1.0.0"

    def test_from_dict_tolerates_legacy_timestep_seconds(self):
        """Test that from_dict ignores the legacy timestep_seconds key from old Arrow files."""
        data_dict = {
            "dataset": "test_dataset",
            "split": "train",
            "log_name": "log_001",
            "location": "test_location",
            "timestep_seconds": 0.1,
            "version": "1.0.0",
        }
        log_metadata = LogMetadata.from_dict(data_dict)
        assert log_metadata.dataset == "test_dataset"
        assert log_metadata.version == "1.0.0"

    def test_roundtrip_serialization(self):
        """Test that to_dict and from_dict are inverses."""
        original = LogMetadata(
            dataset="test_dataset",
            split="train",
            log_name="log_001",
            location="test_location",
        )
        data_dict = original.to_dict()
        reconstructed = LogMetadata.from_dict(data_dict)

        assert original.dataset == reconstructed.dataset
        assert original.split == reconstructed.split
        assert original.log_name == reconstructed.log_name
        assert original.location == reconstructed.location

    def test_is_instance_of_abstract_metadata(self):
        """LogMetadata is an instance of BaseMetadata."""
        log_metadata = LogMetadata(dataset="test_dataset", split="train", log_name="log_001", location="test_location")
        assert isinstance(log_metadata, BaseMetadata)

    def test_repr(self):
        """Test __repr__ returns a meaningful string."""
        log_metadata = LogMetadata(dataset="test_dataset", split="train", log_name="log_001", location="test_location")
        repr_str = repr(log_metadata)
        assert "test_dataset" in repr_str
        assert "train" in repr_str
        assert "log_001" in repr_str
        assert "test_location" in repr_str
