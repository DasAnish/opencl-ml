from clobject import *
from layer import *
import numpy as np
import pyopencl.array as pycl_array
from pytest import raises, approx
import copy
import os
# os.chdir('..')

size = np.int32(512)
size_ = np.int32(511)


def test_layer_instance():
    layer = Layer(size, LINEAR)

    assert isinstance(layer.cl, ClSingleton)
    assert isinstance(layer.code, Code)
    assert isinstance(layer.layer, pycl_array.Array)
    assert layer.layer_size == size

    # checking the rest is none.
    assert layer.next_layer is None
    assert layer.weights is None
    assert layer.next_layer_size is None
    assert layer.bias is None
    assert layer.weights_del is None
    assert layer.bias_del is None
    assert layer.transposed is None

    # test Len
    assert len(layer) == size

    # test activation print
    assert layer.print_activation() == 'LINEAR'


def test_set_attr_functions():
    layer = Layer(size, LINEAR)
    next_layer_array = layer.cl.ones(size_)

    # testing if you can set weights & bias before setting next layer
    with raises(ValueError): layer.set_weights(next_layer_array)
    with raises(ValueError): layer.set_bias(next_layer_array)

    # testing set_next_layer
    with raises(TypeError): layer.set_next_layer(next_layer_array.get())

    layer.set_next_layer(next_layer_array)

    bias = layer.cl.ones(size)

    # testing wrong size
    with raises(ValueError): layer.set_bias(bias)

    # testing wrong type
    with raises(TypeError): layer.set_bias(bias.get())

    bias = layer.cl.ones(size_)
    layer.set_bias(bias)

    # testing wrong size for weights
    with raises(ValueError): layer.set_weights(bias)

    weights = layer.cl.ones((size_, size))

    # testing wrong type for weights
    with raises(TypeError): layer.set_weights(weights.get())

    layer.set_weights(weights)


def test_layer_deepcopy():
    layer = Layer(size, LINEAR)

    next_layer_array = layer.cl.ones(size_)
    layer.set_next_layer(next_layer_array)

    weights_array = layer.cl.ones((size_, size))
    layer.set_weights(weights_array)

    bias_array = layer.cl.ones(size_)
    layer.set_bias(bias_array)

    copied_layer = copy.deepcopy(layer)
    copied_weights = copied_layer.weights.get()
    copied_bias = copied_layer.bias.get()
    approx_one = approx(1)

    # assert that the values are the same
    for i in range(size_):
        assert copied_bias[i] == approx_one
        for j in range(size):
            assert copied_weights[i][j] == approx_one

    assert copied_layer.weights.base_data != layer.weights.base_data
    assert copied_layer.bias.base_data != layer.bias.base_data


def test_forward():
    layer = Layer(size, LINEAR)
    layer.layer.set(np.ones(size).astype(np.float32))
    layer.next_layer_size = size_
    layer.next_layer = layer.cl.zeros(size_)
    layer.weights = layer.cl.ones((size_, size))
    layer.bias = layer.cl.ones(size_)

    layer.forward()
    approx_val = approx(size+1)

    output = layer.next_layer.get()
    for i in range(size_):
        assert output[i] == approx_val

    layer.forward()
    output = layer.next_layer.get()
    for i in range(size_):
        assert output[i] == approx_val

