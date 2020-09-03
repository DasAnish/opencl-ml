from clobject import *
from pytest import raises, approx
import numpy as np
import pyopencl as cl
import pyopencl.array as pycl_array
import os
print(os.curdir)
os.chdir("..")


def test_singleton_implementation():
    # making first instance
    clsingleton_1 = ClSingleton.get_instance()
    clsingleton_2 = ClSingleton.get_instance()
    code_1 = Code.get_instance()
    code_2 = Code.get_instance()

    # testing if it is a singleton, making another instance
    with raises(Exception):
        ClSingleton()

    with raises(Exception):
        Code()

    # test that second intance has the same reference.
    assert clsingleton_1 == clsingleton_2
    assert code_1 == code_2

    # testing that members are the same
    assert clsingleton_2.context == clsingleton_1.context
    assert clsingleton_2.queue == clsingleton_1.queue

    assert code_1.program == code_2.program


def test_array_created():
    clsingleton = ClSingleton.get_instance()

    # testing zeros
    size=10
    zeros = clsingleton.zeros(size).get()
    z = [0.0 for _ in range(size)]

    # testing random
    np.random.seed(0)
    random = clsingleton.random(size).get()
    np.random.seed(0)
    expected_random = np.random.random(size)

    # testing uniform
    np.random.seed(1)
    uniform = clsingleton.uniform(-1, 1, size).get()
    np.random.seed(1)
    expected_uniform = np.random.uniform(-1, 1, size)

    # testing get_array
    zeros_array = clsingleton.get_array(zeros)

    assert type(zeros_array) == pycl_array.Array

    for i in range(size):
        assert zeros[i] == approx(z[i])
        assert expected_random[i] == approx(random[i])
        assert expected_uniform[i] == approx(uniform[i])


