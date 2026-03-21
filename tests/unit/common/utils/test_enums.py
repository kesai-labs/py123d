import pytest

from py123d.common.utils.enums import SerialIntEnum, classproperty, resolve_enum_arguments


class SampleEnum(SerialIntEnum):
    """Sample enum for testing."""

    FIRST = 0
    SECOND = 1
    THIRD = 2


class TestClassproperty:
    """Tests for the classproperty descriptor."""

    def test_access_on_class(self):
        """Test that classproperty is accessible on the class itself."""

        class MyClass:
            @classproperty
            def value(cls):
                return 42

        assert MyClass.value == 42

    def test_access_on_instance(self):
        """Test that classproperty is also accessible on instances."""

        class MyClass:
            @classproperty
            def value(cls):
                return "hello"

        assert MyClass().value == "hello"


class TestSerialIntEnumInt:
    """Tests for SerialIntEnum __int__ conversion."""

    def test_int_conversion(self):
        """Test that int(enum) returns the integer value."""
        assert int(SampleEnum.FIRST) == 0
        assert int(SampleEnum.SECOND) == 1
        assert int(SampleEnum.THIRD) == 2


class TestSerialIntEnumSerialize:
    """Tests for SerialIntEnum.serialize."""

    def test_serialize_lowercase(self):
        """Test that serialize() returns lowercase name by default."""
        assert SampleEnum.FIRST.serialize() == "first"
        assert SampleEnum.SECOND.serialize() == "second"

    def test_serialize_uppercase(self):
        """Test that serialize(lower=False) returns the original name."""
        assert SampleEnum.FIRST.serialize(lower=False) == "FIRST"
        assert SampleEnum.SECOND.serialize(lower=False) == "SECOND"


class TestSerialIntEnumDeserialize:
    """Tests for SerialIntEnum.deserialize."""

    def test_deserialize_lowercase(self):
        """Test that lowercase strings are deserialized correctly."""
        assert SampleEnum.deserialize("first") == SampleEnum.FIRST
        assert SampleEnum.deserialize("second") == SampleEnum.SECOND

    def test_deserialize_uppercase(self):
        """Test that uppercase strings are deserialized correctly."""
        assert SampleEnum.deserialize("FIRST") == SampleEnum.FIRST

    def test_deserialize_invalid_raises_key_error(self):
        """Test that an invalid key raises KeyError."""
        with pytest.raises(KeyError):
            SampleEnum.deserialize("nonexistent")

    def test_serialize_deserialize_roundtrip(self):
        """Test that serialize followed by deserialize returns the same enum."""
        for member in SampleEnum:
            assert SampleEnum.deserialize(member.serialize()) == member


class TestSerialIntEnumFromInt:
    """Tests for SerialIntEnum.from_int."""

    def test_valid_int(self):
        """Test that valid integers return the correct enum."""
        assert SampleEnum.from_int(0) == SampleEnum.FIRST
        assert SampleEnum.from_int(1) == SampleEnum.SECOND

    def test_invalid_int_raises_value_error(self):
        """Test that an invalid integer raises ValueError."""
        with pytest.raises(ValueError):
            SampleEnum.from_int(99)


class TestSerialIntEnumFromArbitrary:
    """Tests for SerialIntEnum.from_arbitrary."""

    def test_from_enum_instance(self):
        """Test that passing an enum instance returns it directly."""
        result = SampleEnum.from_arbitrary(SampleEnum.FIRST)
        assert result is SampleEnum.FIRST

    def test_from_int(self):
        """Test that an integer is converted to the correct enum."""
        assert SampleEnum.from_arbitrary(0) == SampleEnum.FIRST

    def test_from_string(self):
        """Test that a lowercase string is converted to the correct enum."""
        assert SampleEnum.from_arbitrary("first") == SampleEnum.FIRST

    def test_invalid_type_raises_value_error(self):
        """Test that an unsupported type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid value for SampleEnum"):
            SampleEnum.from_arbitrary(3.14)  # type: ignore[arg-type]


class TestResolveEnumArguments:
    """Tests for the resolve_enum_arguments function."""

    def test_none_returns_none(self):
        """Test that None input returns None."""
        result = resolve_enum_arguments(SampleEnum, None)
        assert result is None

    def test_list_of_ints(self):
        """Test resolving a list of integers."""
        result = resolve_enum_arguments(SampleEnum, [0, 1, 2])
        assert result == [SampleEnum.FIRST, SampleEnum.SECOND, SampleEnum.THIRD]

    def test_list_of_strings(self):
        """Test resolving a list of strings."""
        result = resolve_enum_arguments(SampleEnum, ["first", "second"])
        assert result == [SampleEnum.FIRST, SampleEnum.SECOND]

    def test_list_of_enums(self):
        """Test resolving a list of enum instances."""
        result = resolve_enum_arguments(SampleEnum, [SampleEnum.FIRST, SampleEnum.THIRD])
        assert result == [SampleEnum.FIRST, SampleEnum.THIRD]

    def test_mixed_input(self):
        """Test resolving a mixed list of ints, strings, and enums."""
        result = resolve_enum_arguments(SampleEnum, [0, "second", SampleEnum.THIRD])
        assert result == [SampleEnum.FIRST, SampleEnum.SECOND, SampleEnum.THIRD]

    def test_tuple_input(self):
        """Test that tuple input is accepted."""
        result = resolve_enum_arguments(SampleEnum, (0, 1))
        assert result == [SampleEnum.FIRST, SampleEnum.SECOND]

    def test_invalid_type_raises_type_error(self):
        """Test that a non-sequence input raises TypeError."""
        with pytest.raises(TypeError, match="input must be a list"):
            resolve_enum_arguments(SampleEnum, 42)  # type: ignore[arg-type]

    def test_empty_list(self):
        """Test that an empty list returns an empty list."""
        result = resolve_enum_arguments(SampleEnum, [])
        assert result == []
