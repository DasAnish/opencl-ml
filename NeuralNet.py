import pyopencl as cl
import pyopencl.array as pycl_array
import pyopencl.clmath as pycl_math
import numpy as np
from Layer import *
from ClObject import ClSingleton, Code
from typing import List, Type


class NeuralNet:

    def __init__(self, layers):
        self.cl = ClSingleton.get_instance()

        if len(layers)<3:
            raise ValueError('NEED atleast 3 layers 1 Layer, 1 Activation and 1 Output')

        # Input layer
        self.input_size: np.int32 = layers[0].layer_size

        self.code: Code = Code.get_instance()

        # Connecting the layer
        for i, layer in enumerate(layers[:-1]):
            layer.set_next_layer(layers[i+1].layer)

            if type(layer) == Layer:
                layer.set_bias(pycl_array.to_device(
                    self.cl.queue,
                    np.random.uniform(-1, 1, layer.next_layer_size).astype(np.float32)
                ))

                layer.set_weights(pycl_array.to_device(
                    self.cl.queue,
                    np.random.uniform(-1, 1, layer.next_layer_size*layer.layer_size).astype(np.float32)
                ))

            # layer.set_activation(layers[i+1].layer.activation.type)

        if not isinstance(layers[-1], Output):
            raise TypeError(f"Final layer should be of type {Output}"
                            f", but provided {type(layer[-1])}")

        self.layers: List[Layer] = layers
        self.output_layer: Output = layers[-1]
        self.output_size: np.int32 = layers[-1].layer_size

        print("SETUP NN COMPLETE")

    def forward(self, _input: np.array) -> None:
        if len(_input) != self.input_size:
            raise ValueError("Provided input is not of the correct size")

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

    def train(self, xys, batch_size, epoch) -> None:
        # both Xs and ys are vectors
        xys = np.shuffle(xys)

        for i in range(epoch):
            i %= batch_size
            current_batch = xys[i*batch_size: (i+1)*batch_size]

            self.train_batch(current_batch)

    def train_batch(self, xys) -> None:
        total_error = 0
        total_error_derivative = []
        for x, y in xys:
            self.forward(x)
            total_error += self.output_layer.get_error_value(y)
            # Now we have del^l
            total_error_derivative.append(self.output_layer.get_error_derivative(y))

        self.update_weights(total_error_derivative)

    def update_weights(self, total_error_derivative) -> None:
        for layer in self.layers[:-1]:
            if isinstance(layer, Layer):
                layer.weights_del: pycl_array.Array = pycl_array.zeros_like(layer.weights)
                layer.transposed: pycl_array.Array = pycl_array.zeros_like(layer.weights)
                self.code.program.transpose(
                    self.cl.queue,
                    (layer.layer_size, layer.next_layer_size),
                    (16, 16),
                    layer.weights.data,
                    layer.transposed.data,
                    layer.layer_size,
                    layer.next_layer_size
                )
                layer.bias_del: pycl_array.Array = pycl_array.zeros_like(layer.bias)

        for error in total_error_derivative:
            for layer_index in range(-2, -len(self.layers)-1, -1):
                layer = self.layers[layer_index]
                error = layer.backward(error)






