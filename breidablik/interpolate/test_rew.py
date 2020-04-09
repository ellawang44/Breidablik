from breidablik.interpolate.rew import Rew
import numpy as np
import pytest
import warnings

try:
    Rew()
    flag = False
except:
    flag = True
# skip these tests if the trained models are not present
pytestmark = pytest.mark.skipif(flag, reason = 'No trained Rew model')

class Test_find_abund:

    @classmethod
    def setup_class(cls):
        cls.models = Rew()

    def test_input_shape(self):
        with pytest.raises(ValueError):
            Test_find_abund.models.find_abund([1], 1, 1, 1)
            Test_find_abund.models.find_abund(1, [1], 1, 1)
            Test_find_abund.models.find_abund(1, 1, [1], 1)
            Test_find_abund.models.find_abund(1, 1, 1, [1])

    def test_warning_pred_abund(self):
        with warnings.catch_warnings(record = True) as w:
            warnings.simplefilter('always')
            Test_find_abund.models.find_abund(6000, 4.5, -2, -10)
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)

    def test_diff_line(self):
        abund_670 = Test_find_abund.models.find_abund(5000, 4.5, -1, 2.1, center = 670)
        abund_610 = Test_find_abund.models.find_abund(5000, 4.5, -1, 2.1, center = 610)
        assert abs(abund_670 - abund_610) > 1e-5
