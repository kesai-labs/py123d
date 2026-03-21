import pytest

from py123d.datatypes.metadata.base_metadata import BaseMetadata


class TestBaseMetadata:
    """Tests for the BaseMetadata ABC contract."""

    def test_cannot_instantiate_directly(self):
        """BaseMetadata cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseMetadata()

    def test_subclass_missing_to_dict_cannot_instantiate(self):
        """A subclass that omits to_dict raises TypeError on instantiation."""

        class MissingToDict(BaseMetadata):
            @classmethod
            def from_dict(cls, data_dict):
                return cls()

        with pytest.raises(TypeError):
            MissingToDict()

    def test_subclass_missing_from_dict_cannot_instantiate(self):
        """A subclass that omits from_dict raises TypeError on instantiation."""

        class MissingFromDict(BaseMetadata):
            def to_dict(self):
                return {}

        with pytest.raises(TypeError):
            MissingFromDict()

    def test_concrete_subclass_is_instance_of_base_metadata(self):
        """A fully implemented subclass is an instance of BaseMetadata."""

        class ConcreteMetadata(BaseMetadata):
            def to_dict(self):
                return {}

            @classmethod
            def from_dict(cls, data_dict):
                return cls()

        assert isinstance(ConcreteMetadata(), BaseMetadata)

    def test_concrete_subclass_to_dict(self):
        """to_dict on a concrete subclass returns a plain dict."""

        class ConcreteMetadata(BaseMetadata):
            def __init__(self, value: int = 0):
                self._value = value

            def to_dict(self):
                return {"value": self._value}

            @classmethod
            def from_dict(cls, data_dict):
                return cls(data_dict["value"])

        assert ConcreteMetadata(42).to_dict() == {"value": 42}

    def test_concrete_subclass_from_dict(self):
        """from_dict on a concrete subclass constructs the instance."""

        class ConcreteMetadata(BaseMetadata):
            def __init__(self, value: int = 0):
                self._value = value

            def to_dict(self):
                return {"value": self._value}

            @classmethod
            def from_dict(cls, data_dict):
                return cls(data_dict["value"])

        obj = ConcreteMetadata.from_dict({"value": 99})
        assert isinstance(obj, BaseMetadata)
        assert obj._value == 99

    def test_concrete_subclass_roundtrip(self):
        """to_dict and from_dict are inverses on a concrete subclass."""

        class ConcreteMetadata(BaseMetadata):
            def __init__(self, name: str = "", score: float = 0.0):
                self._name = name
                self._score = score

            def to_dict(self):
                return {"name": self._name, "score": self._score}

            @classmethod
            def from_dict(cls, data_dict):
                return cls(data_dict["name"], data_dict["score"])

        original = ConcreteMetadata("test", 3.14)
        restored = ConcreteMetadata.from_dict(original.to_dict())
        assert restored._name == original._name
        assert restored._score == original._score
