import pyopencl.array as pycl_array
import pyopencl.clmath as pycl_math
import numpy as np
from ClObject import Code, ClObject
from Activations import *

'''Gonna put different types of layers in this'''


class Layer(ClObject):

    def __init__(self, queue, context, n):
        ClObject.__init__(self, queue, context)
        self.layer_size = np.int32(n)
        self.next_layer = None      # object for next layer
        self.weights = None         # pycl_array
        self.next_layer_size = None
        self.bias = None            # for the next layer
        self.code = None
        self.activation = Activation()

        self.layer = pycl_array.to_device(
            queue,
            np.random.uniform(-1, 1, n).astype(np.float32)
        )

    def forward(self, queue):
        if self.next_layer is None:
            raise ValueError("Next Layer hasn't been defined yet. "
                             "Define it first.")
        if self.weights is None:
            raise ValueError("The weights connecting to the next layer haven't been defined yet.")

        if self.code is None:
            raise ValueError("Code from clForward text file needs to be added so that it runs on"
                             "GPU")

        self.code.program.forward(
            queue,
            self.next_layer.layer.shape,
            None,
            self.layer_size,
            self.layer.data,
            self.weights.data,
            self.bias.data,
            self.next_layer.layer.data
        )

    def set_next_layer(self, layer):
        if isinstance(layer, pycl_array.Array):
            self.next_layer = layer
            self.next_layer_size = np.int32(len(layer))
        else:
            raise TypeError(f"Layer argument should be {type(pycl_array.Array)};"
                            f" provided {type(layer)}")

    def set_weights(self, weights):
        if self.next_layer is None:
            raise ValueError("Please set next_layer first")
        if len(weights) != self.next_layer_size*self.layer_size:
            raise IndexError(f"The size does't fit please make sure it has shape"
                             f" {self.layer_size}*{self.next_layer_size}")

        self.weights = weights

    def set_code(self, code):
        if isinstance(code, Code):
            raise TypeError(f"Need to provide {type(Code)}; provided {type(code)}")

        self.code = code

    def set_bias(self, bias):
        if not isinstance(bias, pycl_array.Array):
            raise TypeError(f"Please provide a {type(pycl_array.Array)}"
                            f" instead of {type(bias)}")
        if len(bias) == self.next_layer_size:
            raise ValueError("The size should be current size")

        self.bias = bias

    def set_activation(self, activation_type):
        self.activation.type = activation_type

    def activate(self):
        self.layer = self.activation.activate(self.layer)
