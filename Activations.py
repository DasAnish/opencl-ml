import pyopencl.array as pycl_array
import pyopencl.clmath as pycl_math
import numpy as np
from ClObject import ClSingleton

BINARY_STEP, LINEAR, SIGMOID, TANH, RELU, LEAKY_RELU, SOFTMAX = (i for i in range(7))
## Could be moved into the cl code


class Activation:

    def __init__(self, activation=RELU, linear_gradient=1, leak=0.01):
        # ClObject.__init__(self, queue, context)
        self.cl = ClSingleton.get_instance()
        self.type = activation

        # values for the funcitons may be optimized in hyperparameter tuning
        self.linear_gradient = linear_gradient
        self.leak = leak

        ## the functions in a dict
        self.activation_functions = {
            BINARY_STEP: self.binary_step,
            LINEAR:      self.linear,
            SIGMOID:     self.sigmoid,
            TANH:        self.tanh,
            RELU:        self.relu,
            LEAKY_RELU:  self.leaky_relu,
            SOFTMAX:     self.softmax
        }

    def activate(self, array):
        return self.activation_functions[self.type](array)

    def binary_step(self, array):
        return pycl_array.if_positive(
            array,
            pycl_array.to_device(self.queue, np.ones(array.shape).astype(np.float32)),
            pycl_array.zeros_like(array.shape),
            queue=self.cl.qeueue
        )

    def linear(self, array):
        return array*self.linear_gradient

    def sigmoid(self, array):
        return 1 / (1 + pycl_math.exp(-array, self.queue))

    def tanh(self, array):
        return pycl_math.tanh(array, self.queue)

    def relu(self, array):
        return pycl_array.if_positive(
            array,
            array,
            pycl_array.zeros_like(array),
            queue=self.cl.queue
        )

    def leaky_relu(self, array):
        return pycl_array.if_positive(
            array,
            array,
            self.leak*array,
            self.cl.queue
        )

    def softmax(self, array):
        ret = pycl_math.exp(array, self.queue)
        ret /= pycl_array.sum(ret)
        return ret