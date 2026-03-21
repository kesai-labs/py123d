import uuid

import pytest

from py123d.common.utils.uuid_utils import (
    UUID_NAMESPACE_123D,
    convert_to_bytes_uuid,
    convert_to_str_uuid,
    convert_to_uuid_object,
    create_deterministic_uuid,
)


class TestCreateDeterministicUuid:
    """Tests for the create_deterministic_uuid function."""

    def test_returns_uuid_object(self):
        """Test that the function returns a uuid.UUID instance."""
        result = create_deterministic_uuid("nuscenes_train", "log001", 1000000)
        assert isinstance(result, uuid.UUID)

    def test_is_uuid_v5(self):
        """Test that the generated UUID is version 5."""
        result = create_deterministic_uuid("nuscenes_train", "log001", 1000000)
        assert result.version == 5

    def test_deterministic(self):
        """Test that the same inputs always produce the same UUID."""
        uuid_a = create_deterministic_uuid("av2_val", "log_abc", 500000)
        uuid_b = create_deterministic_uuid("av2_val", "log_abc", 500000)
        assert uuid_a == uuid_b

    def test_different_split_produces_different_uuid(self):
        """Test that different split values produce different UUIDs."""
        uuid_a = create_deterministic_uuid("nuscenes_train", "log001", 1000000)
        uuid_b = create_deterministic_uuid("nuscenes_val", "log001", 1000000)
        assert uuid_a != uuid_b

    def test_different_log_name_produces_different_uuid(self):
        """Test that different log names produce different UUIDs."""
        uuid_a = create_deterministic_uuid("nuscenes_train", "log001", 1000000)
        uuid_b = create_deterministic_uuid("nuscenes_train", "log002", 1000000)
        assert uuid_a != uuid_b

    def test_different_timestamp_produces_different_uuid(self):
        """Test that different timestamps produce different UUIDs."""
        uuid_a = create_deterministic_uuid("nuscenes_train", "log001", 1000000)
        uuid_b = create_deterministic_uuid("nuscenes_train", "log001", 2000000)
        assert uuid_a != uuid_b

    def test_with_misc(self):
        """Test that providing misc changes the UUID."""
        uuid_without = create_deterministic_uuid("av2_train", "log001", 1000000)
        uuid_with = create_deterministic_uuid("av2_train", "log001", 1000000, misc="camera_front")
        assert uuid_without != uuid_with

    def test_misc_is_deterministic(self):
        """Test that the same misc value produces the same UUID."""
        uuid_a = create_deterministic_uuid("av2_train", "log001", 1000000, misc="lidar_top")
        uuid_b = create_deterministic_uuid("av2_train", "log001", 1000000, misc="lidar_top")
        assert uuid_a == uuid_b

    def test_different_misc_produces_different_uuid(self):
        """Test that different misc values produce different UUIDs."""
        uuid_a = create_deterministic_uuid("av2_train", "log001", 1000000, misc="camera_front")
        uuid_b = create_deterministic_uuid("av2_train", "log001", 1000000, misc="camera_rear")
        assert uuid_a != uuid_b

    def test_uses_123d_namespace(self):
        """Test that the UUID is generated with the 123D namespace."""
        result = create_deterministic_uuid("nuscenes_train", "log001", 1000000)
        expected = uuid.uuid5(UUID_NAMESPACE_123D, "nuscenes_train:log001:1000000")
        assert result == expected

    def test_none_misc_same_as_no_misc(self):
        """Test that misc=None produces the same result as omitting misc."""
        uuid_a = create_deterministic_uuid("av2_train", "log001", 1000000, misc=None)
        uuid_b = create_deterministic_uuid("av2_train", "log001", 1000000)
        assert uuid_a == uuid_b

    def test_pinned_uuid_nuscenes_train(self):
        """Test that a known input always produces the exact same UUID (regression guard)."""
        result = create_deterministic_uuid("nuscenes_train", "log001", 1000000)
        assert str(result) == "5014e601-ea36-508f-9077-8556ff31a2d4"

    def test_pinned_uuid_av2_val(self):
        """Test pinned UUID for av2_val split."""
        result = create_deterministic_uuid("av2_val", "log_abc", 500000)
        assert str(result) == "c771b2f0-06d6-59f4-a093-3dfb09eb68b2"

    def test_pinned_uuid_waymo_with_misc(self):
        """Test pinned UUID for waymo_test with misc field."""
        result = create_deterministic_uuid("waymo_test", "segment_123", 999999, misc="lidar")
        assert str(result) == "a5fca8e0-2674-5659-b6fc-b0f71d6b103d"

    def test_pinned_uuid_kitti360_zero_timestamp(self):
        """Test pinned UUID for kitti360_train with zero timestamp."""
        result = create_deterministic_uuid("kitti360_train", "drive_0000", 0)
        assert str(result) == "004ef9ad-3af0-51bd-8582-0afe207c2ff5"

    def test_pinned_uuid_nuscenes_with_camera(self):
        """Test pinned UUID for nuscenes_train with camera_front misc."""
        result = create_deterministic_uuid("nuscenes_train", "log001", 1000000, misc="camera_front")
        assert str(result) == "8173cf33-e15f-5d26-9e46-d9e62fb71008"


class TestConvertToStrUuid:
    """Tests for the convert_to_str_uuid function."""

    def test_from_uuid_object(self):
        """Test conversion from uuid.UUID to string."""
        uid = uuid.uuid4()
        result = convert_to_str_uuid(uid)
        assert result == str(uid)

    def test_from_bytes(self):
        """Test conversion from 16-byte binary to string."""
        uid = uuid.uuid4()
        result = convert_to_str_uuid(uid.bytes)
        assert result == str(uid)

    def test_from_hyphenated_string(self):
        """Test conversion from a hyphenated UUID string."""
        uid = uuid.uuid4()
        result = convert_to_str_uuid(str(uid))
        assert result == str(uid)

    def test_from_hex_string(self):
        """Test conversion from a hex string without hyphens."""
        uid = uuid.uuid4()
        result = convert_to_str_uuid(uid.hex)
        assert result == str(uid)

    def test_output_is_lowercase_hyphenated(self):
        """Test that the output is lowercase and hyphenated."""
        upper_str = "A1B2C3D4-E5F6-7890-ABCD-EF1234567890"
        result = convert_to_str_uuid(upper_str)
        assert result == result.lower()
        assert "-" in result

    def test_invalid_string_raises_value_error(self):
        """Test that an invalid UUID string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid UUID"):
            convert_to_str_uuid("not-a-uuid")

    def test_invalid_type_raises_value_error(self):
        """Test that an unsupported type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid UUID input type"):
            convert_to_str_uuid(12345)  # type: ignore[arg-type]

    def test_roundtrip_with_bytes(self):
        """Test that str -> bytes -> str roundtrip preserves the UUID."""
        uid = uuid.uuid4()
        as_str = convert_to_str_uuid(uid)
        as_bytes = convert_to_bytes_uuid(as_str)
        back_to_str = convert_to_str_uuid(as_bytes)
        assert back_to_str == as_str


class TestConvertToBytesUuid:
    """Tests for the convert_to_bytes_uuid function."""

    def test_from_uuid_object(self):
        """Test conversion from uuid.UUID to bytes."""
        uid = uuid.uuid4()
        result = convert_to_bytes_uuid(uid)
        assert result == uid.bytes

    def test_from_valid_bytes(self):
        """Test that valid 16-byte input is returned as-is."""
        uid = uuid.uuid4()
        result = convert_to_bytes_uuid(uid.bytes)
        assert result == uid.bytes

    def test_from_string(self):
        """Test conversion from string to bytes."""
        uid = uuid.uuid4()
        result = convert_to_bytes_uuid(str(uid))
        assert result == uid.bytes

    def test_from_hex_string(self):
        """Test conversion from hex string without hyphens."""
        uid = uuid.uuid4()
        result = convert_to_bytes_uuid(uid.hex)
        assert result == uid.bytes

    def test_output_is_16_bytes(self):
        """Test that the output is always 16 bytes."""
        uid = uuid.uuid4()
        result = convert_to_bytes_uuid(uid)
        assert len(result) == 16

    def test_invalid_bytes_length_raises_value_error(self):
        """Test that bytes with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="expected 16 bytes"):
            convert_to_bytes_uuid(b"short")

    def test_invalid_string_raises_value_error(self):
        """Test that an invalid UUID string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid UUID"):
            convert_to_bytes_uuid("not-a-uuid")

    def test_invalid_type_raises_value_error(self):
        """Test that an unsupported type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid UUID input type"):
            convert_to_bytes_uuid(42)  # type: ignore[arg-type]


class TestConvertToUuidObject:
    """Tests for the convert_to_uuid_object function."""

    def test_from_uuid_object(self):
        """Test that a uuid.UUID input is returned as-is."""
        uid = uuid.uuid4()
        result = convert_to_uuid_object(uid)
        assert result is uid

    def test_from_bytes(self):
        """Test conversion from 16-byte binary to uuid.UUID."""
        uid = uuid.uuid4()
        result = convert_to_uuid_object(uid.bytes)
        assert result == uid

    def test_from_string(self):
        """Test conversion from string to uuid.UUID."""
        uid = uuid.uuid4()
        result = convert_to_uuid_object(str(uid))
        assert result == uid

    def test_from_hex_string(self):
        """Test conversion from hex string to uuid.UUID."""
        uid = uuid.uuid4()
        result = convert_to_uuid_object(uid.hex)
        assert result == uid

    def test_output_is_uuid_instance(self):
        """Test that the output is always a uuid.UUID instance."""
        uid = uuid.uuid4()
        for input_val in [uid, uid.bytes, str(uid), uid.hex]:
            result = convert_to_uuid_object(input_val)
            assert isinstance(result, uuid.UUID)

    def test_invalid_bytes_length_raises_value_error(self):
        """Test that bytes with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="expected 16 bytes"):
            convert_to_uuid_object(b"\x00" * 20)

    def test_invalid_string_raises_value_error(self):
        """Test that an invalid UUID string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid UUID"):
            convert_to_uuid_object("not-a-uuid")

    def test_invalid_type_raises_value_error(self):
        """Test that an unsupported type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid UUID input type"):
            convert_to_uuid_object(3.14)  # type: ignore[arg-type]


class TestCrossConversionConsistency:
    """Tests that all conversion functions produce consistent results from the same input."""

    def test_all_converters_agree(self):
        """Test that converting the same UUID through all paths gives consistent results."""
        uid = uuid.uuid4()

        str_result = convert_to_str_uuid(uid)
        bytes_result = convert_to_bytes_uuid(uid)
        obj_result = convert_to_uuid_object(uid)

        assert str_result == str(obj_result)
        assert bytes_result == obj_result.bytes
        assert convert_to_uuid_object(str_result) == obj_result
        assert convert_to_uuid_object(bytes_result) == obj_result

    def test_deterministic_uuid_roundtrips(self):
        """Test that a deterministic UUID roundtrips through all converters."""
        uid = create_deterministic_uuid("waymo_test", "segment_123", 999999, misc="lidar")

        as_str = convert_to_str_uuid(uid)
        as_bytes = convert_to_bytes_uuid(uid)
        as_obj = convert_to_uuid_object(uid)

        assert convert_to_uuid_object(as_str) == as_obj
        assert convert_to_uuid_object(as_bytes) == as_obj
        assert convert_to_str_uuid(as_bytes) == as_str
        assert convert_to_bytes_uuid(as_str) == as_bytes
