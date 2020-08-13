import numpy as np
import pyopencl as cl
import pyopencl.array as pycl_array
from ClObject import ClObject
from Activations import Activation


class Layer(ClObject):

    def __init__(self, queue, context, input_size, output_size, bias=None, activation=None):
        ClObject.__init__(self, context=context, queue=queue)
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation if activation is not None else Activation()

        with open('clForward', 'r') as f:
            code = f.read()

        # TODO: make sure you are not compiling code again and again
        self.program = self.build(code)

        ## variables defined within class
        if bias is None:
            self.bias = pycl_array.to_device(queue,
                                             np.random.random(output_size).astype(np.float32))
        else:
            self.bias = bias

        self.weights = pycl_array.to_device(
            queue,
            np.random.random(input_size*output_size).astype(np.float32)
        )
        self.input_vec = pycl_array.to_device(
            queue,
            np.random.rand(input_size).astype(np.float32)
        )
        self.output_vec = pycl_array.to_device(
            queue,
            np.zeros(output_size).astype(np.float32)
        )

    def forward(self, _input = None):
        if _input is not None:
            self.input_vec.set(_input)

        self.program.forward(
            self.queue,
            self.output_size,       # for get_global_id
            None,
            self.input_size,        # for size
            self.input_vec.data,    # for *input
            self.weights.data,      # for *weights
            self.bias.data,         # for *bias
            self.output_vec.data)   # for *output

        self.output_vec = self.activation.activate(self.output_vec)

    # def set_default_input_vec(self):
    #     self.input_vec = pycl_array.to_device(
    #         self.queue,
    #         np.zeros(self.input_size).astype(np.float32)
    #     )
    #     return self.input_vec

    def set_input_vec(self, input_vec):
        self.input_vec.set(input_vec)
