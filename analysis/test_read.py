from breidablik.analysis import read
from hypothesis import given, assume
from hypothesis.strategies import floats
import math
import numpy as np
import os
import pytest
import warnings

pytestmark = pytest.mark.skipif((not os.path.isdir('Balder')) or (len(os.listdir('Balder')) == 0), reason = 'No raw data')

class Test_name_add:

    @given(floats())
    def test_num(self, x):
        assume(not math.isnan(x))
        assert x == float(read._name_add(x))

    @given(floats())
    def test_sign(self, x):
        assert read._name_add(x)[0] in "+-"

class Test_closest_temp:

    def test_warning(self):
        with warnings.catch_warnings(record = True) as w:
            warnings.simplefilter('always')
            read._closest_temp(1000, 2.5, -3)
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)

    def test_case1(self):
        assert read._closest_temp(5000, 2, -1) == '4985.01'

class Test_get_wavelengths:

    def test_monotonic(self):
        wl = read.get_wavelengths()
        assert np.all(wl[1:] > wl[:-1])
