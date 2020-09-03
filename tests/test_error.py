from clobject import *
from error import MeanSquaredError
import numpy as np
from pytest import approx
import pyopencl as cl
import pyopencl.array as pycl_array
size = 10


def test_error_value():

    mse = MeanSquaredError()
    clsingle = ClSingleton.get_instance()
    np.random.seed(0)

    approx_zero = approx(0.0)
    approx_half_size = approx(size / 2.0)

    random = np.random.random(size).astype(np.float32)
    zeros = np.zeros(size).astype(np.float32)
    ones = np.ones(size).astype(np.float32)

    # testing error value with np arrays
    assert mse.error_value(zeros, zeros) == approx_zero
    assert mse.error_value(random, random) == approx_zero
    assert mse.error_value(ones, ones) == approx_zero
    assert mse.error_value(zeros, ones) == approx_half_size

    # testing error value with np array and pycl_array
    zeros = clsingle.get_array(zeros)

    assert mse.error_value(zeros, ones) == approx_half_size
    assert mse.error_value(ones, zeros) == approx_half_size

    # testing error value with two pycl_arrays
    ones = clsingle.get_array(ones)

    assert mse.error_value(ones, zeros) == approx_half_size
    assert mse.error_value(zeros, zeros) == approx_zero
    assert mse.error_value(ones, ones) == approx_zero

    random = clsingle.get_array(random)
    assert mse.error_value(random, random) == approx_zero


def test_error_derivative():

    mse = MeanSquaredError()
    clsingle = ClSingleton.get_instance()
    np.random.seed(0)

    random = np.random.random(size).astype(np.float32)
    zeros = np.zeros(size).astype(np.float32)
    ones = np.ones(size).astype(np.float32)

    approx_zero = approx(0.0)
    approx_one = approx(1.0)

    # testing with both numpy array
    edzero = mse.error_derivative(zeros, zeros).get()
    edones = mse.error_derivative(ones, zeros).get()

    # print(edzero[0], type(edzero[0]))

    for i in range(size):
        assert edzero[i] == approx_zero
        assert edones[i] == approx_one

    # testing with both pycl_array
    ones = clsingle.get_array(ones)
    zeros = clsingle.get_array(zeros)

    edzero = mse.error_derivative(ones, ones).get()
    edones = mse.error_derivative(ones, zeros).get()
    edrand = mse.error_derivative(random, zeros).get()

    for i in range(size):
        assert edzero[i] == approx_zero
        assert edones[i] == approx_one
        assert edrand[i] == approx(random[i])


