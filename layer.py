import pyopencl.array as pycl_array
import pyopencl.clmath as pycl_math
import numpy as np
from clobject import Code, ClSingleton
# from Activations import *
from error import *

BINARY_STEP, LINEAR, SIGMOID, TANH, RELU, LEAKY_RELU, SOFTMAX = (np.int32(i) for i in range(7))

'''Gonna put different types of layers in this'''


class LayerBase:
    def __init__(self, size: int):
        self.cl: ClSingleton = ClSingleton.get_instance()
        self.layer_size: np.int32 = np.int32(size)
        self.layer: pycl_array.Array = pycl_array.to_device(
            self.cl.queue,
            np.random.uniform(-1, 1, size).astype(np.float32)
        )
        self.code: Code = Code.get_instance()

    def __len__(self):
        return self.layer_size


class Layer(LayerBase):

    def __init__(self, n: int):
        LayerBase.__init__(self, n)
        self.next_layer: pycl_array.Array = None      # object for next layer
        self.weights: pycl_array.Array = None         # pycl_array
        self.next_layer_size: np.int32 = None
        self.bias: pycl_array.Array = None            # for the next layer
        # self.code: Code = Code.get_instance()

        # for stochastic gradient descent
        self.weights_del: pycl_array.Array = None
        self.bias_del: pycl_array.Array = None
        self.transposed: pycl_array.Array = None

    def forward(self, _input: np.array=None) -> None:
        if self.next_layer is None:
            raise ValueError("Next Layer hasn't been defined yet. "
                             "Define it first.")
        if self.weights is None:
            raise ValueError("The weights connecting to the next layer haven't been defined yet.")

        if self.code is None:
            raise ValueError("Code from kernel.cl text file needs to be added so that it runs on"
                             "GPU")

        if _input is not None:
            if len(_input) != self.layer_size:
                raise ValueError("Incorrect input size")
            self.layer.set(_input)

        self.code.program.matrix_vector_mul(
            self.cl.queue,
            self.next_layer.shape,
            None,
            self.layer_size,
            self.layer.data,
            self.weights.data,
            self.next_layer.data
        )
        self.next_layer += self.bias

        # print(f'printing a vector from layer_forward: {self.next_layer}')

        # activating the next layer
        # self.activation.activate(self.next_layer)

    def backward(self, _del: pycl_array.Array) -> pycl_array.Array:
        # _del has size next_layer_size
        # print(1, len(self.bias), len(_del))
        self.bias_del += _del

        self.code.program.weights_del(
            self.cl.queue,
            (self.layer_size, self.next_layer_size),
            (16, 16),
            self.layer_size,
            _del.data,
            self.layer.data,
            self.weights_del.data
        )

        next_del: pycl_array.Array = pycl_array.zeros_like(self.layer)

        self.code.program.matrix_vector_mul(
            self.cl.queue,
            (self.layer_size, ),
            None,
            self.next_layer_size,
            _del.data,
            self.transposed.data,
            next_del.data
        )

        return next_del

    def set_next_layer(self, layer: pycl_array.Array) -> None:
        if isinstance(layer, pycl_array.Array):
            self.next_layer = layer
            self.next_layer_size = np.int32(len(layer))
        else:
            raise TypeError(f"Layer argument should be {pycl_array.Array};"
                            f" provided {type(layer)}")

    def set_weights(self, weights: pycl_array.Array) -> None:
        if self.next_layer is None:
            raise ValueError("Please set next_layer first")
        if not isinstance(weights, pycl_array.Array):
            raise TypeError(f"Weights should be of activation_type {pycl_array.Array}"
                            f"provided {type(weights)}")
        if len(weights) != self.next_layer_size*self.layer_size:
            raise IndexError(f"The size does't fit please make sure it has shape"
                             f" {self.layer_size}*{self.next_layer_size}")

        self.weights = weights

    def set_bias(self, bias: pycl_array.Array) -> None:
        if not isinstance(bias, pycl_array.Array):
            raise TypeError(f"Please provide a {pycl_array.Array}"
                            f" instead of {type(bias)}")
        if len(bias) != self.next_layer_size:
            raise ValueError("The size should be correct size "
                             f"is ({self.next_layer_size}); and not ({len(bias)})")

        self.bias = bias


class Output(LayerBase):

    def __init__(self, size: int):
        LayerBase.__init__(self, size)
        self.error = MeanSquaredError()

    def get_error_value(self, expected: pycl_array.Array):
        return self.error.error_value(self.layer, expected)

    def get_error_derivative(self, expected: pycl_array.Array) -> pycl_array.Array:
        return self.error.error_derivative(self.layer, expected)


class Activation(LayerBase):

    def __init__(self, size, activation_type=RELU):
        LayerBase.__init__(self, size)

        self.linear_gradient: int = 1
        self.leak: int = 0.01
        self.activation_type: np.int32 = activation_type

        self.next_layer: pycl_array.Array = None
        self.next_layer_size: np.int32 = self.layer_size

        # self._del = None

        ## the functions in a dict
        # self.activation_functions = {
        #     BINARY_STEP: self.binary_step,
        #     LINEAR: self.linear,
        #     SIGMOID: self.sigmoid,
        #     TANH: self.tanh,
        #     RELU: self.relu,
        #     LEAKY_RELU: self.leaky_relu,
        #     SOFTMAX: self.softmax
        # }

        self.activation_derivatives = {
            RELU: self.relu_derivative,
            SOFTMAX: self.softmax_derivative
        }

    def set_next_layer(self, layer) -> None:
        if not isinstance(layer, pycl_array.Array):
            raise TypeError(f"Layer argument should be {pycl_array.Array};"
                            f" provided {type(layer)}")

        if self.layer_size != len(layer):
            raise ValueError(f"Layer should be of size {self.layer_size}, "
                             f"provided: {len(layer)}.")

        self.next_layer = layer
        # self.next_layer_size = np.int32(len(layer))

    def backward(self, _del) -> pycl_array.Array:
        # print(2, len(self.layer), len(_del))
        return self.activation_derivatives[self.activation_type](_del)

    def relu_derivative(self, _del: pycl_array.Array) -> pycl_array.Array:
        return _del

    def softmax_derivative(self, _del: pycl_array.Array) -> pycl_array.Array:
        return _del * self.next_layer

    def forward(self) -> None:
        self.code.program.activate(
            self.cl.queue,
            self.next_layer.shape,
            None,
            self.layer.data,
            self.next_layer.data,
            self.activation_type
        )

        if self.activation_type == SOFTMAX:
            self.next_layer /= pycl_array.sum(self.next_layer)

        # print("printing forward from Activation: ", self.next_layer)

    # def binary_step(self) -> pycl_array.Array:
    #     return pycl_array.if_positive(
    #         self.layer,
    #         pycl_array.to_device(self.cl.queue, np.ones(self.layer.shape).astype(np.float32)),
    #         pycl_array.zeros_like(self.layer.shape),
    #         queue=self.cl.qeueue
    #     )
    #
    # def linear(self) -> pycl_array.Array:
    #     return self.layer * self.linear_gradient
    #
    # def sigmoid(self) -> pycl_array.Array:
    #     return 1 / (1 + pycl_math.exp(-self.layer, self.cl.queue))
    #
    # def tanh(self) -> pycl_array.Array:
    #     return pycl_math.tanh(self.layer, self.cl.queue)
    #
    # def relu(self) -> pycl_array.Array:
    #     return pycl_array.if_positive(
    #         self.layer,
    #         self.layer,
    #         pycl_array.zeros_like(self.layer),
    #         queue=self.cl.queue
    #     )
    #
    # def leaky_relu(self) -> pycl_array.Array:
    #     return pycl_array.if_positive(
    #         self.layer,
    #         self.layer,
    #         self.leak * self.layer,
    #         self.cl.queue
    #     )
    #
    # def softmax(self) -> pycl_array.Array:
    #     ret = pycl_math.exp(self.layer, self.cl.queue)
    #     ret /= pycl_array.sum(ret)
    #     return ret
