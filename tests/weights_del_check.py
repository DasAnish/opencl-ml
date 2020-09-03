from clobject import *
import numpy as np
import pyopencl.array as pycl_array
import pytest
import os
os.chdir("..")
# checking that the multiplication is correct

clsingle: ClSingleton = ClSingleton.get_instance()
code: Code = Code.get_instance()

delta_size = np.int32(255)
last_size = np.int32(512)


def test_ones_layer_ones_delta():
    delta_layer = clsingle.ones(delta_size)
    last_layer = clsingle.ones(last_size)
    weights_del = clsingle.zeros((delta_size, last_size))

    func = lambda: code.program.weights_del(
        clsingle.queue,
        weights_del.shape, None,
        last_size, delta_layer.data,
        last_layer.data, weights_del.data
    )
    func()
    weights = weights_del.get()
    approx_one = pytest.approx(1)

    for j in range(delta_size):
        for i in range(last_size):
            assert weights[j][i] == approx_one

    func()
    weights = weights_del.get()
    approx_two = pytest.approx(2)

    for j in range(delta_size):
        for i in range(last_size):
            assert weights[j][i] == approx_two


def test_arange_layer_ones_delta():
    delta_layer = clsingle.ones(delta_size)
    last_layer = pycl_array.to_device(clsingle.queue,
                                      np.arange(last_size).astype(np.float32))
    weights_del = clsingle.zeros((delta_size, last_size))

    func = lambda: code.program.weights_del(
        clsingle.queue,
        weights_del.shape, None,
        last_size, delta_layer.data,
        last_layer.data, weights_del.data
    )

    func()
    weights = weights_del.get()
    last = last_layer.get()

    for i in range(delta_size):
        for j in range(last_size):
            assert weights[i][j] == pytest.approx(last[j])

    func()
    weights = weights_del.get()

    for i in range(delta_size):
        for j in range(last_size):
            assert weights[i][j] == pytest.approx(2*last[j])
