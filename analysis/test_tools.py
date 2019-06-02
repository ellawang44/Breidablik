from breidablik.analysis import tools
import numpy as np
import pytest

def assert_array_eq(x, y):
    assert np.array_equal(x,y)

class Test_cut_wavelength:

    def test_empty(self):
        assert_array_eq(tools.cut_wavelength([]), [])

    def test_inc_edge(self):
        assert_array_eq(tools.cut_wavelength([1, 2, 3, 4], center = 2, upper = 1, lower = 1), [1, 2, 3])

    def test_case1(self):
        assert_array_eq(tools.cut_wavelength([6600, 6700, 6800, 6900]), [6700, 6800])

    def test_case2(self):
        assert_array_eq(tools.cut_wavelength([8100, 8120, 8130, 8160, 8180, 8190], center = 8128.606, upper = 10, lower = 20), [8120, 8130])

    def test_case3(self):
        assert_array_eq(tools.cut_wavelength([6080, 6090, 6100, 6120, 6140], center = 6105.298, upper = 30, lower = 0), [6120])

class Test_cut:

    def test_empty(self):
        assert_array_eq(tools.cut([], []), np.array([[], []]))

    def test_inc_edge(self):
        assert_array_eq(tools.cut([1, 2, 3, 4], [4, 5, 6, 7], center = 2, upper = 1, lower = 1), [[1, 2, 3], [4, 5, 6]])

    def test_case1(self):
        assert_array_eq(tools.cut([6600, 6690, 6710, 6800, 6900], [1, -3, -6, 0, 2]), [[6690, 6710, 6800], [-3, -6, 0]])

    def test_case2(self):
        assert_array_eq(tools.cut([8100, 8120, 8130, 8160, 8180, 8190], [0, 1, 2, 3, 4, 5], center = 8128.606, upper = 10, lower = 20), [[8120, 8130], [1, 2]])

    def test_case3(self):
        assert_array_eq(tools.cut([6080, 6090, 6100, 6120, 6140], [-9, 1, 203, 40, 1], center = 6105.298, upper = 30, lower = 0), [[6120], [40]])

# not testing `rew` since it just calls cut() which is already tested and juse uses some numpy and scipy functions
