from breidablik.interpolate import grid_check
import warnings

class Test_grid_check:

    def test_warning(self):
        with warnings.catch_warnings(record = True) as w:
            warnings.simplefilter('always')
            grid_check._grid_check(1, 1, 1)
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
