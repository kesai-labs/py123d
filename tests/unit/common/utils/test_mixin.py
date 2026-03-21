from enum import IntEnum

import numpy as np
import numpy.testing as npt

from py123d.common.utils.mixin import ArrayMixin, indexed_array_repr


class ConcreteVector(ArrayMixin):
    """Concrete subclass of ArrayMixin for testing."""

    def __init__(self, array: np.ndarray):
        self._array = array

    @property
    def array(self) -> np.ndarray:
        return self._array

    @classmethod
    def from_array(cls, array: np.ndarray, copy: bool = True) -> "ConcreteVector":
        if copy:
            array = array.copy()
        return cls(array)


class TestArrayMixinConstruction:
    """Tests for ArrayMixin construction methods."""

    def test_from_array(self):
        """Test creating an instance from a numpy array."""
        arr = np.array([1.0, 2.0, 3.0])
        vec = ConcreteVector.from_array(arr)
        npt.assert_array_equal(vec.array, arr)

    def test_from_array_with_copy(self):
        """Test that from_array with copy=True creates an independent copy."""
        arr = np.array([1.0, 2.0, 3.0])
        vec = ConcreteVector.from_array(arr, copy=True)
        arr[0] = 99.0
        assert vec.array[0] == 1.0

    def test_from_array_without_copy(self):
        """Test that from_array with copy=False shares memory."""
        arr = np.array([1.0, 2.0, 3.0])
        vec = ConcreteVector.from_array(arr, copy=False)
        arr[0] = 99.0
        assert vec.array[0] == 99.0

    def test_from_list(self):
        """Test creating an instance from a Python list."""
        vec = ConcreteVector.from_list([1.0, 2.0, 3.0])
        npt.assert_array_equal(vec.array, [1.0, 2.0, 3.0])
        assert vec.array.dtype == np.float64


class TestArrayMixinProperties:
    """Tests for ArrayMixin property accessors."""

    def test_shape(self):
        """Test the shape property."""
        vec = ConcreteVector(np.array([1.0, 2.0, 3.0]))
        assert vec.shape == (3,)

    def test_shape_2d(self):
        """Test the shape property for 2D arrays."""
        vec = ConcreteVector(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert vec.shape == (2, 2)

    def test_len(self):
        """Test the __len__ method."""
        vec = ConcreteVector(np.array([1.0, 2.0, 3.0]))
        assert len(vec) == 3

    def test_tolist(self):
        """Test the tolist method."""
        vec = ConcreteVector(np.array([1.0, 2.0, 3.0]))
        assert vec.tolist() == [1.0, 2.0, 3.0]

    def test_to_list(self):
        """Test the to_list method returns same result as tolist."""
        vec = ConcreteVector(np.array([1.0, 2.0, 3.0]))
        assert vec.to_list() == vec.tolist()


class TestArrayMixinIndexing:
    """Tests for ArrayMixin indexing behavior."""

    def test_single_index(self):
        """Test single element access."""
        vec = ConcreteVector(np.array([10.0, 20.0, 30.0]))
        assert vec[0] == 10.0
        assert vec[2] == 30.0

    def test_slice(self):
        """Test slice access."""
        vec = ConcreteVector(np.array([10.0, 20.0, 30.0, 40.0]))
        npt.assert_array_equal(vec[1:3], [20.0, 30.0])

    def test_negative_index(self):
        """Test negative index access."""
        vec = ConcreteVector(np.array([10.0, 20.0, 30.0]))
        assert vec[-1] == 30.0


class TestArrayMixinComparison:
    """Tests for ArrayMixin equality and hashing."""

    def test_equal_arrays(self):
        """Test that equal arrays compare as equal."""
        vec_a = ConcreteVector(np.array([1.0, 2.0]))
        vec_b = ConcreteVector(np.array([1.0, 2.0]))
        assert vec_a == vec_b

    def test_unequal_arrays(self):
        """Test that different arrays compare as unequal."""
        vec_a = ConcreteVector(np.array([1.0, 2.0]))
        vec_b = ConcreteVector(np.array([3.0, 4.0]))
        assert vec_a != vec_b

    def test_eq_with_non_arraymixin(self):
        """Test that comparison with a non-ArrayMixin returns False."""
        vec = ConcreteVector(np.array([1.0, 2.0]))
        assert vec != "not an array"
        assert vec != [1.0, 2.0]

    def test_hash_equal_for_equal_arrays(self):
        """Test that equal arrays produce the same hash."""
        vec_a = ConcreteVector(np.array([1.0, 2.0]))
        vec_b = ConcreteVector(np.array([1.0, 2.0]))
        assert hash(vec_a) == hash(vec_b)

    def test_hash_different_for_different_arrays(self):
        """Test that different arrays typically produce different hashes."""
        vec_a = ConcreteVector(np.array([1.0, 2.0]))
        vec_b = ConcreteVector(np.array([3.0, 4.0]))
        assert hash(vec_a) != hash(vec_b)


class TestArrayMixinCopy:
    """Tests for ArrayMixin copy behavior."""

    def test_copy_returns_equal_instance(self):
        """Test that copy returns an equal but independent instance."""
        vec = ConcreteVector(np.array([1.0, 2.0, 3.0]))
        copied = vec.copy()
        assert vec == copied
        assert vec is not copied

    def test_copy_is_independent(self):
        """Test that modifying the copy does not affect the original."""
        vec = ConcreteVector(np.array([1.0, 2.0, 3.0]))
        copied = vec.copy()
        copied.array[0] = 99.0
        assert vec.array[0] == 1.0


class TestArrayMixinNumpyProtocol:
    """Tests for the __array__ numpy protocol."""

    def test_np_array_conversion(self):
        """Test that np.array(mixin) works."""
        vec = ConcreteVector(np.array([1.0, 2.0, 3.0]))
        result = np.array(vec)
        npt.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_np_array_with_dtype(self):
        """Test that np.array(mixin, dtype=...) converts dtype."""
        vec = ConcreteVector(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        result = np.array(vec, dtype=np.float32)
        assert result.dtype == np.float32
        npt.assert_array_almost_equal(result, [1.0, 2.0, 3.0])


class TestArrayMixinRepr:
    """Tests for ArrayMixin string representation."""

    def test_repr(self):
        """Test __repr__ includes class name and array."""
        vec = ConcreteVector(np.array([1.0, 2.0]))
        repr_str = repr(vec)
        assert "ConcreteVector" in repr_str
        assert "array=" in repr_str


class TestIndexedArrayRepr:
    """Tests for the indexed_array_repr function."""

    def test_indexed_repr(self):
        """Test repr with IntEnum indexing."""

        class XYZ(IntEnum):
            X = 0
            Y = 1
            Z = 2

        vec = ConcreteVector(np.array([1.0, 2.0, 3.0]))
        result = indexed_array_repr(vec, XYZ)
        assert "ConcreteVector(" in result
        assert "x=1.0" in result
        assert "y=2.0" in result
        assert "z=3.0" in result

    def test_indexed_repr_single_field(self):
        """Test repr with a single-member IntEnum."""

        class Single(IntEnum):
            VALUE = 0

        vec = ConcreteVector(np.array([42.0]))
        result = indexed_array_repr(vec, Single)
        assert "value=42.0" in result
