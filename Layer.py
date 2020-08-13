import pyopencl.array as pycl_array
import pyopencl.clmath as pycl_math
import numpy as np
from ClObject import Code, ClObject
from Activations import *
from Error import *

'''Gonna put different types of layers in this'''


class LayerBase(ClObject):
    def __init__(self, queue, context, size, activation=RELU):
        ClObject.__init__(self, queue, context)
        self.layer_size = np.int32(size)
        self.layer = pycl_array.to_device(
            queue,
            np.random.uniform(-1, 1, size).astype(np.float32)
        )
        self.activation = activation

    def set_activation(self, activation_type):
        self.activation.type = activation_type

    def activate(self):
        self.layer = self.activation.activate(self.layer)

    def __len__(self):
        return self.layer_size


class Layer(LayerBase):

    def __init__(self, queue, context, n, activation=RELU):
        LayerBase.__init__(self, queue, context, n, activation)
        self.next_layer = None      # object for next layer
        self.weights = None         # pycl_array
        self.next_layer_size = None
        self.bias = None            # for the next layer
        self.code = None

    def forward(self, _input=None):
        if self.next_layer is None:
            raise ValueError("Next Layer hasn't been defined yet. "
                             "Define it first.")
        if self.weights is None:
            raise ValueError("The weights connecting to the next layer haven't been defined yet.")

        if self.code is None:
            raise ValueError("Code from clForward text file needs to be added so that it runs on"
                             "GPU")

        if _input is not None:
            if len(_input) != self.layer_size:
                raise ValueError("Incorrect input size")
            self.layer.set(_input)

        # print(type(self.layer_size), type(self.next_layer.layer_size))

        self.code.program.forward(
            self.queue,
            self.next_layer.shape,
            None,
            self.layer_size,
            self.layer.data,
            self.weights.data,
            self.bias.data,
            self.next_layer.data,
            np.int32(self.activation)
        )

        # self.next_layer.activate()

    def set_next_layer(self, layer):
        if isinstance(layer, pycl_array.Array):
            self.next_layer = layer
            self.next_layer_size = np.int32(len(layer))
        else:
            raise TypeError(f"Layer argument should be {pycl_array.Array};"
                            f" provided {type(layer)}")

    def set_weights(self, weights):
        if self.next_layer is None:
            raise ValueError("Please set next_layer first")
        if len(weights) != self.next_layer_size*self.layer_size:
            raise IndexError(f"The size does't fit please make sure it has shape"
                             f" {self.layer_size}*{self.next_layer_size}")

        self.weights = weights

    def set_code(self, code):
        if not isinstance(code, Code):
            raise TypeError(f"Need to provide {Code}; provided {type(code)}")

        self.code = code

    def set_bias(self, bias):
        if not isinstance(bias, pycl_array.Array):
            raise TypeError(f"Please provide a {pycl_array.Array}"
                            f" instead of {type(bias)}")
        if len(bias) != self.next_layer_size:
            raise ValueError("The size should be correct size "
                             f"is ({self.next_layer_size}); and not ({len(bias)})")

        self.bias = bias


class Output(LayerBase):

    def __init__(self, queue, context, size, activation=SOFTMAX):
        LayerBase.__init__(self, queue, context, size, activation)
        self.error = MeanSquaredError(queue, context)

    def get_error_value(self, expected):
        return self.error.error_value(self.layer, expected)
