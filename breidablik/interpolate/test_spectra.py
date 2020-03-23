from breidablik.interpolate.spectra import Spectra
import numpy as np
import pytest
import warnings

try:
    Spectra()
    flag = False
except:
    flag = True
# skip these tests if the trained models are not present
pytestmark = pytest.mark.skipif(flag, reason = 'No trained Spectra model')

class Test_find_abund:

    @classmethod
    def setup_class(cls):
        cls.models = Spectra()

    def test_monontonic(self):
        with pytest.raises(ValueError):
            Test_find_abund.models.find_abund([1, 3, 2], [4, 5, 6], [1, 1, 1], 4000, 1.5, 0)

    def test_shape(self):
        with pytest.raises(ValueError):
            Test_find_abund.models.find_abund([1, 2], [1], [1], 5000, 2.5, -2)
            Test_find_abund.models.find_abund([1], [1, 2], [1], 5000, 2.5, -2)
            Test_find_abund.models.find_abund([1], [1], [1, 2], 5000, 2.5, -2)

    def test_dimension(self):
        with pytest.raises(ValueError):
            Test_find_abund.models.find_abund([[1, 2], [2, 3]], [1], [1], 5000, 2.5, -2)
            Test_find_abund.models.find_abund([1], [[1, 2], [2, 3]], [1], 5000, 2.5, -2)
            Test_find_abund.models.find_abund([1], [1], [[1, 2], [2, 3]], 5000, 2.5, -2)

    def test_method(self):
        with pytest.raises(ValueError):
            Test_find_abund.models.find_abund([1, 2, 3], [4, 5, 6], [1, 1, 1], 5000, 1, 1, method = 'hi')

    def test_min_max_abund(self):
        with pytest.raises(ValueError):
            Test_find_abund.models.find_abund([1], [1], [1], 1, 1, 1, min_abund = 8, max_abund = -1)

    def test_abund_prior_warning(self):
        with warnings.catch_warnings(record = True) as w:
            warnings.simplefilter('always')
            Test_find_abund.models.find_abund([600, 700], [1, 0.5], [0.5, 0.5], 5000, 2.5, -2, method = 'chisq', prior = [1, 2, 3], abunds = [1, 2, 3])
            Test_find_abund.models.find_abund([600, 700], [1, 0.9], [0.5, 0.5], 5000, 2.5, -2, prior = [1, 2, 3])
            Test_find_abund.models.find_abund([600, 700], [1, 0.9], [0.5, 0.5], 5000, 2.5, -2, abunds = [1, 2, 3])
            assert len(w) == 3
            for i in range(len(w)):
                assert issubclass(w[i].category, UserWarning)

    def test_wl_overlap(self):
        with pytest.raises(ValueError):
            Test_find_abund.models.find_abund([1, 2], [1, 0.6], [0.1, 0.21], 5000, 4.5, -1)

    def test_abund_prior_shape(self):
        with pytest.raises(ValueError):
            Test_find_abund.models.find_abund([680, 690], [1, 1], [1, 1], 5000, 2.5, -2, abunds = [1, 2], prior = [1])
            Test_find_abund.models.find_abund([680, 690], [1, 1], [1, 1], 5000, 2.5, -2, abunds = [[1, 2]], prior = [[1, 3]])

    def test_warning_pred_abund(self):
        # define square wave
        wls = np.linspace(670.5, 671.4, 1000)
        flux = np.full(len(wls), 1)
        flux[(670.95 <= wls) & (wls < 670.99)] = 0
        flux_err = np.full(len(wls), 0.1)
        # catch warning for predicted value outside of grid
        with warnings.catch_warnings(record = True) as w:
            warnings.simplefilter('always')
            Test_find_abund.models.find_abund(wls, flux, flux_err, 6000, 4, -3, method = 'chisq', max_abund = 5)
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)

class Test_predict_flux:

    @classmethod
    def setup_class(cls):
        cls.models = Spectra()

    def test_input_shape(self):
        with pytest.raises(ValueError):
            Test_predict_flux.models.predict_flux([1], 1, 1, 1)
            Test_predict_flux.models.predict_flux(1, [1], 1, 1)
            Test_predict_flux.models.predict_flux(1, 1, [1], 1)
            Test_predict_flux.models.predict_flux(1, 1, 1, [1])

    def test_warning_abundance(self):
        with warnings.catch_warnings(record = True) as w:
            warnings.simplefilter('always')
            Test_predict_flux.models.predict_flux(5000, 2.5, -2, -1)
            Test_predict_flux.models.predict_flux(5000, 2.5, -2, 5)
            assert len(w) == 2
            for i in range(len(w)):
                assert issubclass(w[i].category, UserWarning)
