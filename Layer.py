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
        if activation is None:
            self.activation = Activation(self.queue, self.context)
        else:
            self.activation = activation


        with open('clForward', 'r') as f:
            code = f.read()
        # TODO: make sure you are not compiling code again and again
        self.program = self.build(code)


        ## variables defined within class
        if bias is None:
            self.bias = pycl_array.to_device(
                queue,
                np.random.uniform(-1, 1, output_size).astype(np.float32))
        else:
            self.bias = bias

        self.weights = pycl_array.to_device(
            queue,
            np.random.uniform(-1, 1, input_size*output_size).astype(np.float32)
        )
        self.input_vec = pycl_array.to_device(
            queue,
            np.random.uniform(-1, 1, input_size).astype(np.float32)
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
            self.output_vec.shape,
            None,
            np.int32(self.input_size),
            self.input_vec.data,
            self.weights.data,
            self.bias.data,
            self.output_vec.data)

        self.output_vec = self.activation.activate(self.output_vec)

    # def set_default_input_vec(self):
    #     self.input_vec = pycl_array.to_device(
    #         self.queue,
    #         np.zeros(self.input_size).astype(np.float32)
    #     )
    #     return self.input_vec

    def set_input_vec(self, input_vec):
        if input_vec.shape[0] == self.input_size:
            self.input_vec = input_vec
        else:
            print(input_vec.shape)
            raise ValueError
