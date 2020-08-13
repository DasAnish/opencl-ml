import pyopencl as cl
import pyopencl.array as pycl_array
import pyopencl.clmath as pycl_math
import numpy as np
from ClObject import ClObject

BINARY_STEP, LINEAR, SIGMOID, TANH, RELU, LEAKY_RELU, SOFTMAX = (i for i in range(7))


class Activation(ClObject):

    def __init__(self, queue, context, activation, linear_gradient=1, leak=0.01):
        ClObject.__init__(queue, context)
        self.activation = activation

        # values for the funcitons may be optimized in hyperparameter tuning
        self.linear_gradient = linear_gradient
        self.leak = leak

    def activate(self, array):
        if self.activation == BINARY_STEP:
            return pycl_array.if_positive(
                array,
                pycl_array.to_device(self.queue, np.ones(array.shape).astype(np.float32)),
                pycl_array.zeros_like(array.shape),
                queue=self.qeueue
            )

        elif self.activation == LINEAR:
            return array*self.linear_gradient

        elif self.activation == SIGMOID:
            return 1 / (1 + pycl_math.exp(-array, self.queue))

        elif self.activation == TANH:
            return pycl_math.tanh(array, self.queue)

        elif self.activation == RELU:
            return pycl_array.if_positive(
                array,
                array,
                pycl_array.zeros_like(array),
                queue=self.queue
            )

        elif self.activation == LEAKY_RELU:
            return pycl_array.if_positive(
                array,
                array,
                self.leak*array,
                self.queue
            )

        elif self.activation == SOFTMAX:
            ret = pycl_math.exp(array, self.queue)
            ret /= pycl_array.sum(ret)
            return ret

        else:
            raise ValueError