import numpy as np
import pyopencl as cl
import pyopencl.array as pycl_array
from ClObject import ClObject


class Layer(ClObject):

    def __init__(self, queue, context, input_size, output_size, bias=None, activation=None):
        ClObject.__init__(self, context=context, queue=queue)
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        ## variables defined within class
        if bias is None:
            self.bias = pycl_array.to_device(queue,
                                             np.random.random(output_size).astype(np.float32))
        else:
            self.bias = bias

        self.weights = pycl_array.to_device(queue,
                                            np.random.random(size=input_size*output_size).astype(np.float32)
                                            )
        # self.input_vec = pycl_array.to_device(queue,
        #                                       np.zeros(input_size).astype(np.float32))
        self.output_vec = pycl_array.to_device(queue,
                                               np.zeros(output_size).astype(np.float32))

    def forward(self, _input, context):
        # expect input is a pycl_array
        with open('clForward', 'r') as f:
            code = f.read()

        program = self.build(code)
        program.forward(self.queue, self.output_size,
                        None, self.input_size,
                        _input.data, self.weights.data,
                        self.bias.data)
