import pyopencl as cl
import pyopencl.array as pycl_array
import pyopencl.clmath as pycl_math
import numpy as np
from ClObject import ClSingleton, Code


class NeuralNet:

    def __init__(self, layers, code=None):
        self.cl = ClSingleton.get_instance()

        # Input layer
        self.input_size = layers[0].layer_size

        if code is None:
            self.code = Code()
            self.code.set_code(path_to_file="kernel.cl")
        else:
            self.code = code

        self.code.build()

        # Connecting the layer
        for i, layer in enumerate(layers[:-1]):
            layer.set_next_layer(layers[i+1].layer)

            layer.set_bias(pycl_array.to_device(
                self.cl.queue,
                np.random.uniform(-1, 1, layer.next_layer_size).astype(np.float32)
            ))

            layer.set_weights(pycl_array.to_device(
                self.cl.queue,
                np.random.uniform(-1, 1, layer.next_layer_size*layer.layer_size).astype(np.float32)
            ))

            layer.set_code(self.code)

        self.layers = layers

        print("SETUP NN COMPLETE")

    def forward(self, _input):
        if len(_input) != self.input_size:
            raise ValueError("Provided input is not of the correst size")

        self.layers[0].layer.set(_input)
        for layer in self.layers[:-1]:
            layer.forward()

    def __str__(self):
        st = ''
        for i, layer in enumerate(self.layers):
            st += (f'Layer[{i+1}]   '
                   f'Values: {layer.layer}\n'
                   f'***********************************')
        return st

