from breidablik.interpolate.scalar import Scalar
import numpy as np
import pytest
import warnings

def assert_array_eq(x, y):
    assert np.array_equal(x, y)

class Test_fit:

    @classmethod
    def setup_class(cls):
        cls.models = Scalar()

    def test_input(self):
        with pytest.raises(ValueError):
            Test_fit.models.fit(5)

    def test_dimension(self):
        with pytest.raises(ValueError):
            Test_fit.models.fit([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    def test_case1(self):
        Test_fit.models.fit([[1, 2], [3, 4]])
        assert_array_eq(Test_fit.models.mean, [2, 3])
        assert_array_eq(Test_fit.models.std, [1, 1])

class Test_transform:

    @classmethod
    def setup_class(cls):
        cls.models = Scalar()

    def test_existance(self):
        with pytest.raises(AttributeError):
            Test_transform.models.transform([1, 2])

    def test_input(self):
        Test_transform.models.fit([[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            Test_transform.models.transform(5)

    def test_dimension(self):
        with pytest.raises(ValueError):
            Test_transform.models.transform([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    def test_dimension2(self):
        with pytest.raises(ValueError):
            Test_transform.models.transform([[1, 2, 3, 4], [1, 2, 3, 4]])

    def test_case1(self):
        assert_array_eq(Test_transform.models.transform([[1, 2], [3, 4]]), [[-1, -1], [1, 1]])

class Test_save:

    @classmethod
    def setup_class(cls):
        cls.models = Scalar()

    def test_existance(self):
        with pytest.raises(AttributeError):
            Test_save.models.save('hi.npy')

class Test_load:

    @classmethod
    def setup_class(cls):
        cls.models = Scalar()

    def test_fnferror(self):
        with pytest.raises(FileNotFoundError):
            Test_load.models.load('__this_file_should_not_exist__')
