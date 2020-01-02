from breidablik.interpolate.nlte import Nlte
import numpy as np
import pytest
import warnings

try:
    Nlte()
    flag = False
except:
    flag = True
# skip these tests if the trained models are not present
pytestmark = pytest.mark.skipif(flag, reason = 'No trained Nlte model')

class Test_nlte_correction:

    @classmethod
    def setup_class(cls):
        cls.models = Nlte()

    def test_input_shape(self):
        with pytest.raises(ValueError):
            Test_nlte_correction.models.nlte_correction([1], 1, 1, 1)
            Test_nlte_correction.models.nlte_correction(1, [1], 1, 1)
            Test_nlte_correction.models.nlte_correction(1, 1, [1], 1)
            Test_nlte_correction.models.nlte_correction(1, 1, 1, [1])

    def test_warning_pred_abund(self):
        with warnings.catch_warnings(record = True) as w:
            warnings.simplefilter('always')
            Test_nlte_correction.models.nlte_correction(6000, 4.5, -2, 5)
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
