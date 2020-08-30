from clobject import *
import numpy as np
import pyopencl.array as pycl_array
from layer import *

# **INCOMPLETE**


class SimpleRNN:
    """Simple Recurrent Nerual Net implementation"""
    def __init__(self, input_size, output_size):
        self.cl: ClSingleton = ClSingleton.get_instance()
        self.code: Code = Code.get_instance()

        self.input_size: np.int32 = input_size
        self.output_size: np.int32 = output_size

        ## Defining the Input, Output and Hidden Layer
        self.input_layer: Layer = Layer(input_size, LINEAR)
        self.output_layer: pycl_array.Array = self.cl.zeros(output_size)
        self.hidden_layer: pycl_array.Array = self.cl.zeros(output_size)

        self.intermediate_layer: pycl_array.Array = self.cl.zeros(output_size)

        ## Defining the weights and biases to connnect thing
        self.weights_input_hidden: pycl_array.Array = self.cl.uniform(
            low=-1, high=1, shape=(output_size,input_size)
        ) # U

        self.weights_hidden_hidden: pycl_array.Array = self.cl.uniform(
            low=-1, high=1, shape=(output_size, output_size)
        ) # W

        self.weights_hidden_output: pycl_array.Array = self.cl.uniform(
            low=-1, high=-1, shape=(output_size, output_size)
        ) # V

        self.bias_input_hidden: pycl_array.Array = self.cl.uniform(
            -1, 1, shape=output_size
        ) # b

        self.bias_hidden_output: pycl_array.Array = self.cl.uniform(
            -1, 1, shape=output_size
        ) # c

    def forward(self, _input: np.array) -> None:

        # a = Ux
        self.code.program.matrix_vector_mul(
            self.cl.queue,
            (self.output_size, TS),
            (WPT, TS),
            self.input_size,
            RESET_OUTPUT,
            self.input_layer,
            self.weights_input_hidden.data,
            self.intermediate_layer.data
        )

        # a += Wh
        self.code.program.matrix_vector_mul(
            self.cl.queue,
            (self.output_size, TS),
            (WPT, TS),
            self.output_size,
            CUMULATIVE_OUTPUT,
            self.hidden_layer.data,
            self.weights_hidden_hidden.data,
            self.intermediate_layer.data
        )

        # a += b
        self.intermediate_layer += self.bias_input_hidden

        # h = tanh(a)
        self.code.program.activate(
            self.cl.queue,
            (self.output_size, ),
            None,
            self.intermediate_layer.data,
            self.hidden_layer.data,
            TANH
        )

        # o = Vh
        self.code.program.matrix_vector_mul(
            self.cl.queue,
            (self.output_size, TS),
            (WPT, TS),
            self.output_size,
            RESET_OUTPUT,
            self.hidden_layer.data,
            self.weights_hidden_output,
            self.output_layer.data
        )

        # o += c
        self.output_layer += self.bias_hidden_output

        # y = softmax(o)
        # *TODO: gpu softmax maybe??