import pytest

from py123d.common.utils.dependencies import check_dependencies


class TestCheckDependencies:
    """Tests for the check_dependencies function."""

    def test_available_single_module(self):
        """Test that no error is raised for a module that can be imported."""
        check_dependencies("os", "dev")

    def test_available_module_list(self):
        """Test that no error is raised for a list of importable modules."""
        check_dependencies(["os", "sys", "json"], "dev")

    def test_missing_single_module_raises_import_error(self):
        """Test that a missing module raises ImportError."""
        with pytest.raises(ImportError, match="Missing 'nonexistent_module_xyz'"):
            check_dependencies("nonexistent_module_xyz", "waymo")

    def test_missing_module_in_list_raises_import_error(self):
        """Test that a missing module in a list raises ImportError."""
        with pytest.raises(ImportError, match="Missing 'nonexistent_module_xyz'"):
            check_dependencies(["os", "nonexistent_module_xyz", "sys"], "nuscenes")

    def test_error_message_contains_optional_name(self):
        """Test that the error message includes the pip install hint with optional_name."""
        with pytest.raises(ImportError, match=r"pip install py123d\[waymo\]"):
            check_dependencies("nonexistent_module_xyz", "waymo")

    def test_error_message_contains_editable_install_hint(self):
        """Test that the error message includes the editable install hint."""
        with pytest.raises(ImportError, match=r"pip install -e \.\[nuplan\]"):
            check_dependencies("nonexistent_module_xyz", "nuplan")

    def test_first_missing_module_stops_check(self):
        """Test that the first missing module is reported, not subsequent ones."""
        with pytest.raises(ImportError, match="Missing 'first_missing_abc'"):
            check_dependencies(["os", "first_missing_abc", "second_missing_def"], "dev")

    def test_single_string_treated_as_list(self):
        """Test that a single string input is handled the same as a one-element list."""
        check_dependencies("os", "dev")
        check_dependencies(["os"], "dev")
