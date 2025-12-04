import unittest

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import shift as ndshift

from bloodmoon.images import _shift
from bloodmoon.images import fshift


def scipy_fshift(
    arr: NDArray,
    shifty: int | float,
    shiftx: int | float,
):
    """
    Applies fractional shift to input array.

    The fractional shifting value is eroded from the end-points
    wrt the shift verse and added to the front (for both rows/cols).
    """
    arr_ = ndshift(
        arr, (shifty, shiftx), output='float', order=1,
        mode='grid-constant', cval=0.0, prefilter=True,
    )
    return arr_


class TestArrayShift(unittest.TestCase):
    def setUp(self):
        # Common test arrays
        self.arr_2x2 = np.array([[1, 2], [3, 4]])
        self.arr_3x3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.arr_2x3 = np.array([[1, 2, 3], [4, 5, 6]])

    def test_basic_shifts(self):
        """Test basic positive shifts"""
        expected = np.array([[0, 0], [1, 2]])
        np.testing.assert_array_equal(_shift(self.arr_2x2, 1, 0), expected)

        expected = np.array([[0, 1], [0, 3]])
        np.testing.assert_array_equal(_shift(self.arr_2x2, 0, 1), expected)

    def test_negative_shifts(self):
        """Test negative shifts"""
        expected = np.array([[3, 4], [0, 0]])
        np.testing.assert_array_equal(_shift(self.arr_2x2, -1, 0), expected)

        expected = np.array([[2, 0], [4, 0]])
        np.testing.assert_array_equal(_shift(self.arr_2x2, 0, -1), expected)

    def test_both_dimensions(self):
        """Test shifting in both dimensions simultaneously"""
        expected = np.array([[0, 0, 0], [0, 1, 2], [0, 4, 5]])
        np.testing.assert_array_equal(_shift(self.arr_3x3, 1, 1), expected)

        expected = np.array([[5, 6, 0], [8, 9, 0], [0, 0, 0]])
        np.testing.assert_array_equal(_shift(self.arr_3x3, -1, -1), expected)

    def test_large_shifts(self):
        """Test shifts larger than array dimensions"""
        expected = np.zeros((2, 2))
        np.testing.assert_array_equal(_shift(self.arr_2x2, 20, 0), expected)
        np.testing.assert_array_equal(_shift(self.arr_2x2, 0, 20), expected)
        np.testing.assert_array_equal(_shift(self.arr_2x2, -20, 0), expected)
        np.testing.assert_array_equal(_shift(self.arr_2x2, 0, -20), expected)

    def test_zero_shift(self):
        """Test zero shift returns original array"""
        np.testing.assert_array_equal(_shift(self.arr_2x2, 0, 0), self.arr_2x2)

    def test_different_shapes(self):
        """Test with non-square arrays"""
        expected = np.array([[0, 1, 2], [0, 4, 5]])
        np.testing.assert_array_equal(_shift(self.arr_2x3, 0, 1), expected)

    def test_edge_cases(self):
        """Test edge cases like empty and single-element arrays"""
        # Empty array
        empty_arr = np.array([[]])
        np.testing.assert_array_equal(_shift(empty_arr, 1, 1), empty_arr)

        # Single element array
        single_arr = np.array([[1]])
        expected = np.array([[0]])
        np.testing.assert_array_equal(_shift(single_arr, 1, 0), expected)

    def test_different_dtypes(self):
        """Test with different data types"""
        # Float array
        float_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
        expected = np.array([[0.0, 0.0], [1.0, 2.0]])
        np.testing.assert_array_equal(_shift(float_arr, 1, 0), expected)

        # Boolean array
        bool_arr = np.array([[True, False], [False, True]], dtype=bool)
        expected = np.array([[True, False], [False, False]])
        np.testing.assert_array_equal(_shift(bool_arr, -1, -1), expected)
    
    def test_equal2scipyfshift(self):
        """
        Test if `fshift()` has the same output of scipy's `shift()`,
        with shifts greater than 1.
        """
        arr = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0,],
                [0, 0, 0, 0, 0, 0, 0,],
                [0, 0, 1, 1, 1, 0, 0,],
                [0, 0, 1, 1, 1, 0, 0,],
                [0, 0, 0, 0, 0, 0, 0,],
                [0, 0, 0, 0, 0, 0, 0,],
            ],
        )

        shifty, shiftx = 2.1, 3.4
        expected = scipy_fshift(arr, shifty, shiftx)
        shifted = fshift(arr, shifty, shiftx)
        np.testing.assert_array_almost_equal(expected, shifted)

        shifty, shiftx = -2.1, 3.4
        expected = scipy_fshift(arr, shifty, shiftx)
        shifted = fshift(arr, shifty, shiftx)
        np.testing.assert_array_almost_equal(expected, shifted)

        shifty, shiftx = 2.1, -3.4
        expected = scipy_fshift(arr, shifty, shiftx)
        shifted = fshift(arr, shifty, shiftx)
        np.testing.assert_array_almost_equal(expected, shifted)

        shifty, shiftx = -2.1, -3.4
        expected = scipy_fshift(arr, shifty, shiftx)
        shifted = fshift(arr, shifty, shiftx)
        np.testing.assert_array_almost_equal(expected, shifted)

    def test_equal2scipyfshift_decimal(self):
        """
        Test if `fshift()` has the same output of scipy's `shift()`,
        with shifts in (0, 1).
        """
        arr = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0,],
                [0, 0, 0, 0, 0, 0, 0,],
                [0, 0, 1, 1, 1, 0, 0,],
                [0, 0, 1, 1, 1, 0, 0,],
                [0, 0, 0, 0, 0, 0, 0,],
                [0, 0, 0, 0, 0, 0, 0,],
            ],
        )

        shifty, shiftx = 0.1, 0.4
        expected = scipy_fshift(arr, shifty, shiftx)
        shifted = fshift(arr, shifty, shiftx)
        np.testing.assert_array_almost_equal(expected, shifted)

        shifty, shiftx = -0.1, 0.4
        expected = scipy_fshift(arr, shifty, shiftx)
        shifted = fshift(arr, shifty, shiftx)
        np.testing.assert_array_almost_equal(expected, shifted)

        shifty, shiftx = 0.1, -0.4
        expected = scipy_fshift(arr, shifty, shiftx)
        shifted = fshift(arr, shifty, shiftx)
        np.testing.assert_array_almost_equal(expected, shifted)

        shifty, shiftx = -0.1, -0.4
        expected = scipy_fshift(arr, shifty, shiftx)
        shifted = fshift(arr, shifty, shiftx)
        np.testing.assert_array_almost_equal(expected, shifted)


if __name__ == "__main__":
    unittest.main()

