from clobject import *
import numpy as np
import pyopencl.array as pycl_array
import pytest
import os
os.chdir("..")
# checking that the multiplication is correct

clsingle: ClSingleton = ClSingleton.get_instance()
code: Code = Code.get_instance()

inp_size = np.int32(256)
out_size = np.int32(256)
# TS = np.int32(8)
# WPT = np.int32(256/TS)


def test_ones_matrix_ones_vector():
    inp_layer = clsingle.ones(inp_size)
    matrix = clsingle.ones((out_size, inp_size))
    out_layer = clsingle.ones(out_size)

    code.program.matrix_vector_mul(
        clsingle.queue, (out_size, TS),
        (WPT, TS), inp_size,
        RESET_OUTPUT, inp_layer.data,
        matrix.data, out_layer.data
    )

    approx_val = pytest.approx(inp_size)

    out_layer = out_layer.get()

    for i in range(out_size):
        assert out_layer[i] == approx_val


def test_identity_matrix_random_vector():
    inp_layer = clsingle.random(inp_size)
    matrix = [np.arange(out_size) == i for i in range(out_size)]
    matrix = np.array(matrix).astype(np.float32)
    matrix = pycl_array.to_device(
        clsingle.queue,
        matrix
    )
    out_layer = clsingle.ones(out_size)

    code.program.matrix_vector_mul(
        clsingle.queue, (out_size, TS),
        (WPT, TS), inp_size,
        RESET_OUTPUT, inp_layer.data,
        matrix.data, out_layer.data
    )

    out_layer = out_layer.get()
    inp_layer = inp_layer.get()

    for i in range(out_size):
        assert out_layer[i] == pytest.approx(inp_layer[i])


def test_ones_matrix_arange_vector():
    inp_layer = np.arange(inp_size).astype(np.float32)
    inp_layer = pycl_array.to_device(clsingle.queue, inp_layer)
    matrix = clsingle.ones((out_size, inp_size))
    out_layer = clsingle.zeros(out_size)

    code.program.matrix_vector_mul(
        clsingle.queue, (out_size, TS),
        (WPT, TS), inp_size,
        RESET_OUTPUT, inp_layer.data,
        matrix.data, out_layer.data
    )

    approx_val = pytest.approx(np.sum(inp_layer.get()))
    out_layer = out_layer.get()

    for i in range(out_size):
        assert out_layer[i] == approx_val


def test_arange_matrix_ones_vector():
    inp_layer = clsingle.ones(inp_size)
    matrix = [[i for i in range(inp_size)] for j in range(out_size)]
    matrix = np.array(matrix).astype(np.float32)
    matrix = pycl_array.to_device(clsingle.queue, matrix)
    out_layer = clsingle.zeros(out_size)

    code.program.matrix_vector_mul(
        clsingle.queue, (out_size, TS),
        (WPT, TS), inp_size,
        RESET_OUTPUT, inp_layer.data,
        matrix.data, out_layer.data
    )

    approx_val = pytest.approx(np.sum(matrix.get()[0]))
    out_layer = out_layer.get()

    for i in range(out_size):
        assert out_layer[i] == approx_val


def test_matrix_mul_without_reset():
    inp_layer = clsingle.ones(inp_size)
    matrix = clsingle.ones((out_size, inp_size))
    out_layer = clsingle.ones(out_size)

    def func():
        code.program.matrix_vector_mul(
            clsingle.queue, (out_size, TS),
            (WPT, TS), inp_size,
            CUMULATIVE_OUTPUT, inp_layer.data,
            matrix.data, out_layer.data
        )

    func()
    func()

    out_layer = out_layer.get()
    approx_val = pytest.approx(inp_size*2 + 1)

    for i in range(out_size):
        assert out_layer[i] == approx_val






