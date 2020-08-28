import pyopencl.array as pycl_array
import pyopencl.clmath as pycl_math
import numpy as np
from clobject import Code, ClSingleton
# from Activations import *
from error import *

BINARY_STEP, LINEAR, SIGMOID, TANH, RELU, LEAKY_RELU, SOFTMAX = (np.int32(i) for i in range(7))

TS = 16
WPT = 256 // 16
'''Gonna put different types of layers in this'''


class Layer:

    def __init__(self, size: int, activation_type: int):
        # LayerBase.__init__(self, n)

        self.cl: ClSingleton = ClSingleton.get_instance()
        self.code: Code = Code.get_instance()

        self.activation_type: np.int32 = np.int32(activation_type)
        self.linear_gradient: int = 1
        self.leak: int = 0.01

        self.layer_size: np.int32 = np.int32(size)
        self.layer: pycl_array.Array = pycl_array.to_device(
            self.cl.queue,
            np.zeros(size).astype(np.float32)
        )

        self.next_layer: pycl_array.Array = None      # object for next layer
        self.weights: pycl_array.Array = None         # pycl_array [output] X [input]
        self.next_layer_size: np.int32 = None
        self.bias: pycl_array.Array = None            # for the next layer
        # self.code: Code = Code.get_instance()

        # for stochastic gradient descent
        self.weights_del: pycl_array.Array = None
        self.bias_del: pycl_array.Array = None
        self.transposed: pycl_array.Array = None

        # self.activation_functions = {
        #     BINARY_STEP: self.binary_step,
        #     LINEAR: self.linear,
        #     SIGMOID: self.sigmoid,
        #     TANH: self.tanh,
        #     RELU: self.relu,
        #     LEAKY_RELU: self.leaky_relu,
        #     SOFTMAX: self.softmax
        # }
        #
        # self.activation_derivatives = {
        #     RELU: self.relu_derivative,
        #     SOFTMAX: self.softmax_derivative,
        #     SIGMOID: self.sigmoid_derivative,
        #     LEAKY_RELU: self.leaky_rely_derivative,
        #     LINEAR: self.linear_derivative
        # }

    def __len__(self):
        return self.layer_size

    def __str__(self):
        st = "***************" + str(__class__) + "\t" + self.print_activation() + "****************"
        st += '\nLayer: \n' + str(self.layer) + '\n'
        # st += '\nWeights: \n' + str(self.weights)
        # st += '\nBias: \n' + str(self.bias)
        return st

    def print_activation(self):
        if self.activation_type == 0:
            return ("BINARY_STEP")
        elif self.activation_type == 1:
            return ("LINEAR")
        elif self.activation_type == 2:
            return ("SIGMOID")
        elif self.activation_type == 3:
            return ("TANH")
        elif self.activation_type == 4:
            return ("RELU")
        elif self.activation_type == 5:
            return ("LEAKY_RELU")
        else:
            return ("SOFTMAX")

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
            (self.next_layer_size, TS),
            (WPT, TS),
            self.layer_size,
            self.layer.data,
            self.weights.data,
            self.next_layer.data
        )

        self.next_layer += self.bias

        self.code.program.activate(
            self.cl.queue,
            self.next_layer.shape,
            None,
            self.next_layer.data,
            self.activation_type
        )

        if self.activation_type == SOFTMAX:

            layer = np.exp(self.next_layer.get())
            s = np.sum(layer)
            if s != 0.0:
                layer /= s
                # self.next_layer.set(layer)
            else:
                layer = np.ones(self.next_layer_size).astype(np.float32)
                layer /= np.sum(layer)
            self.next_layer.set(layer)

    def backward(self, _del: pycl_array.Array) -> pycl_array.Array:
        # _del is delta

        self.code.program.activation_derivative(
            self.cl.queue,
            _del.shape,
            None,
            self.next_layer.data,
            _del.data,
            self.activation_type
        )

        self.bias_del += _del

        # self.code.program.weights_del(
        #     self.cl.queue,
        #     (self.layer_size, self.next_layer_size),
        #     (16, 16),
        #     self.layer_size,
        #     _del.data,
        #     self.layer.data,
        #     self.weights_del.data
        # )

        next_del: pycl_array.Array = pycl_array.to_device(
            self.cl.queue,
            np.zeros(self.layer_size).astype(np.float32)
        )

        self.code.program.matrix_vector_mul(
            self.cl.queue,
            self.layer.shape,
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
        if weights.shape != (self.next_layer_size,self.layer_size):
            raise IndexError(f"The size does't fit please make sure it has shape"
                             f" {self.layer_size}*{self.next_layer_size}")

        self.weights = weights
        self.transposed = pycl_array.transpose(weights)

    def set_bias(self, bias: pycl_array.Array) -> None:
        if not isinstance(bias, pycl_array.Array):
            raise TypeError(f"Please provide a {pycl_array.Array}"
                            f" instead of {type(bias)}")
        if len(bias) != self.next_layer_size:
            raise ValueError("The size should be correct size "
                             f"is ({self.next_layer_size}); and not ({len(bias)})")

        self.bias = bias

    # # def backward(self, _del) -> pycl_array.Array:
    #
    #
    # def relu_derivative(self, _del: pycl_array.Array) -> pycl_array.Array:
    #     return _del
    #
    # def softmax_derivative(self, _del: pycl_array.Array) -> pycl_array.Array:
    #     ret = _del
    #     ret = ret * self.next_layer
    #     ret = ret * (1 - self.next_layer)
    #     return ret
    #
    # def sigmoid_derivative(self, _del:pycl_array.Array) -> pycl_array.Array:
    #     ret = _del
    #     ret = ret * self.next_layer
    #     ret = ret * (1 + self.next_layer)
    #     # print(ret)
    #     return ret
    #
    # def linear_derivative(self, _del:pycl_array.Array) -> pycl_array.Array:
    #     return pycl_array.zeros_like(self.layer) + 1
    #
    # # def tanh_derivative(self, _del:pycl_array.Array) -> pycl_array.Array:
    #
    # def leaky_rely_derivative(self, _del:pycl_array.Array) -> pycl_array.Array:
    #     derivative = pycl_array.if_positive(
    #         self.layer,
    #         pycl_array.zeros_like(self.layer)+1,
    #         pycl_array.zeros_like(self.layer)+0.01,
    #         queue=self.cl.queue
    #     )
    #     return _del * derivative
    #
    # def forward(self) -> None:
    #     ret = self.activation_functions[self.activation_type]()
    #     # print(ret)
    #     self.next_layer.set(
    #         ret.get()
    #     )
    #     # raise Error()
    #     # print(self.next_layer)
    #
    # def binary_step(self) -> pycl_array.Array:
    #     return pycl_array.if_positive(
    #         self.layer,
    #         pycl_array.to_device(self.cl.queue, np.ones(self.layer.shape).astype(np.float32)),
    #         pycl_array.zeros_like(self.layer.shape),
    #         queue=self.cl.queue
    #     )
    #
    # def linear(self) -> pycl_array.Array:
    #     return self.layer * self.linear_gradient
    #
    # def sigmoid(self) -> pycl_array.Array:
    #     array = self.layer.get()
    #     array = 1 / (1 + np.exp(-array))
    #
    #     return pycl_array.to_device(
    #         self.cl.queue,
    #         array
    #     )
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
    #         queue=self.cl.queue
    #     )
    #
    # def softmax(self) -> pycl_array.Array:
    #     # ret = pycl_math.exp(self.layer, self.cl.queue)
    #     ret = self.layer.get()
    #     # ret = (np.arange(len(ret)) == np.argmax(ret)).astype(np.float32)
    #     ret = np.exp(ret)
    #     ret /= np.sum(ret)
    #     ret = pycl_array.to_device(self.cl.queue,
    #                                ret)
    #     return ret


class Output:

    def __init__(self, size: int):
        # LayerBase.__init__(self, size)
        self.cl: ClSingleton = ClSingleton.get_instance()
        self.error: Error = MeanSquaredError()
        self.layer_size: np.int32 = np.int32(size)

        self.expected: pycl_array.Array = pycl_array.to_device(
            self.cl.queue,
            np.zeros(size).astype(np.float32)
        )

        self.layer = pycl_array.to_device(
            self.cl.queue,
            np.zeros(size).astype(np.float32)
        )

        self.expected = pycl_array.to_device(
            self.cl.queue,
            np.zeros(size).astype(np.float32)
        )

    def __str__(self):
        return str(__class__) + str(self.layer)

    def get_error_value(self):
        return self.error.error_value(self.layer, self.expected)

    def get_error_derivative(self) -> pycl_array.Array:
        return self.error.error_derivative(self.layer, self.expected)