from breidablik.analysis import tools
import numpy as np
import pytest

def assert_array_eq(x, y):
    assert np.array_equal(x, y)

class Test_cut_wavelength:

    def test_empty(self):
        assert_array_eq(tools.cut_wavelength([]), [])

    def test_inc_edge(self):
        assert_array_eq(tools.cut_wavelength([1, 2, 3, 4], center = 2, upper = 1, lower = 1), [1, 2, 3])

    def test_case1(self):
        assert_array_eq(tools.cut_wavelength([660, 670, 680, 690]), [670, 680])

    def test_case2(self):
        assert_array_eq(tools.cut_wavelength([810, 812, 813, 816, 818, 819], center = 812.8606, upper = 1, lower = 2), [812, 813])

    def test_case3(self):
        assert_array_eq(tools.cut_wavelength([608, 609, 610, 612, 614], center = 610.5298, upper = 3, lower = 0), [612])

class Test_cut:

    def test_empty(self):
        assert_array_eq(tools.cut([], []), np.array([[], []]))

    def test_inc_edge(self):
        assert_array_eq(tools.cut([1, 2, 3, 4], [4, 5, 6, 7], center = 2, upper = 1, lower = 1), [[1, 2, 3], [4, 5, 6]])

    def test_case1(self):
        assert_array_eq(tools.cut([660, 669, 671, 680, 690], [1, -3, -6, 0, 2]), [[669, 671, 680], [-3, -6, 0]])

    def test_case2(self):
        assert_array_eq(tools.cut([810, 812, 813, 816, 818, 819], [0, 1, 2, 3, 4, 5], center = 812.8606, upper = 1, lower = 2), [[812, 813], [1, 2]])

    def test_case3(self):
        assert_array_eq(tools.cut([608, 609, 610, 612, 614], [-9, 1, 203, 40, 1], center = 610.5298, upper = 3, lower = 0), [[612], [40]])

# not testing `rew` since it just calls cut() which is already tested and juse uses some numpy and scipy functions
