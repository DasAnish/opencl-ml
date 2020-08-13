from ClObject import ClObject
import pyopencl as cl
import pyopencl.array as pycl_array
import numpy as np


class Layer:
    def __init__(self, n):



class NeuralNet(ClObject):

    def __init__(self, queue, context, layers):
        ClObject.__init__(self, queue, context)
        self.layers = layers

        # self.input_vec = layers[0].set_default_input_vec()

        for i in range(1, len(layers)): # connecting the layers
            layers[i].set_input_vec(layers[i-1].output_vec)

    def fit(self, X, y):
        pass

    def predict(self, X_test):
        pass

    def forward(self, _input):
        self.layers[0].forward(_input)
        for layer in self.layers[1:]:
            layer.forward()

    def __str__(self):
        for i, layer in enumerate(self.layers):
            print(f'Layer{i+1}: ')
            # input_np = np.empty(layer.input_vec.shape)
            # cl.enqueue_copy(self.queue, input_np, layer.input_vec)
            print(f'Input: {layer.input_vec}')
            # output_np = np.empty(layer.output_vec.shape)
            # cl.enqueue_copy(self.queue, output_np, layer.output_vec)
            print(f'Output: {layer.output_vec}')
            print("*"*30)
        return ''
