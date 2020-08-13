from ClObject import ClObject
import pyopencl as cl
import pyopencl.array as pycl_array
import numpy as np


class NeuralNet(ClObject):

    def __init__(self, queue, context, layers):
        ClObject.__init__(self, queue, context)
        self.layers = layers

        # self.input_vec = layers[0].set_default_input_vec()

        for i in range(1, len(layers)): # connecting the layers
            layers[i].set_input_vec(layers[i-1].output_vec)

    def fit(self, X, y):
        pass

    def predict(self, y_test):
        pass

    def forward(self, _input):
        self.layers[0].set_input_vec(_input)
        for layer in self.layers:
            layer.forward()
