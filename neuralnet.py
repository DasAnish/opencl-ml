import pyopencl as cl
import pyopencl.array as pycl_array
import pyopencl.clmath as pycl_math
import numpy as np
from layer import *
from clobject import ClSingleton, Code
from typing import List, Type


class NeuralNet:

    def __init__(self, *layers):
        self.cl = ClSingleton.get_instance()

        # Input layer
        self.input_size: np.int32 = layers[0].layer_size

        self.code: Code = Code.get_instance()

        # Connecting the layer
        for i, layer in enumerate(layers[:-1]):
            layer.set_next_layer(layers[i+1].layer)
            print(type(layer), layer.layer_size)

            layer.set_bias(pycl_array.to_device(
                self.cl.queue,
                np.random.uniform(-1, 1, layer.next_layer_size).astype(np.float32)
            ))

            layer.set_weights(pycl_array.to_device(
                self.cl.queue,
                np.random.uniform(-1, 1, (layer.next_layer_size,
                                          layer.layer_size)).astype(np.float32)
            ))

        self.layers: List[Layer] = layers[:-1]
        self.output_layer: Output = layers[-1]
        self.output_size: np.int32 = layers[-1].layer_size

        print("SETUP NN COMPLETE")

    def forward(self, _input: np.array) -> None:
        if len(_input) != self.input_size:
            raise ValueError(f"Provided input size: {len(_input)} is "
                             f"not of the correct size: {self.input_size}")

        self.layers[0].layer.set(_input)
        for layer in self.layers:
            layer.forward()

    def __str__(self):
        st = ''
        for i, layer in enumerate(self.layers):
            st += f"{i} {layer}"
        st += (f"\n####################################################\n"
               f"OUTPUT: {self.output_layer}"
               f"\n####################################################\n")
        return st

    def fit(self, x, y, batch_size, num_epochs, len_dataset):
        shuffled_range = np.arange(len_dataset)
        np.random.shuffle(shuffled_range)

        for epoch in range(num_epochs):
            current_batch = []

            for j in range(batch_size):
                i = (batch_size*epoch + j) % len_dataset
                y_i = y[shuffled_range[i]]
                x_i = x[shuffled_range[i]]
                current_batch.append((x_i, y_i))

            epoch_error = self.train_batch(current_batch)
            print(f'{epoch_error}/{epoch+1}|', end=" ")

    def train(self, xys, batch_size, num_epochs, print_every=100) -> None:
        total_error = 0

        for epoch in range(num_epochs):
            current_batch = []
            i = epoch % batch_size

            for j in range(batch_size):
                current_batch.append(xys[(batch_size*i + j) % len(xys)])

            epoch_error = self.train_batch(current_batch)
            total_error += epoch_error
            if epoch % print_every==(print_every-1):
                print(f'{total_error/print_every}/{epoch+1}') # , self.output_layer)#
                total_error=0
                # self.predict_values(xys)

    def train_batch(self, xys) -> float:
        for layer in self.layers:
            layer.weights_del: pycl_array.Array = pycl_array.to_device(
                self.cl.queue,
                np.zeros((layer.next_layer_size, layer.layer_size)).astype(np.float32)
            )
            layer.transposed: pycl_array.Array = pycl_array.transpose(layer.weights)
            layer.bias_del: pycl_array.Array = pycl_array.zeros_like(layer.bias)

        total_error = 0

        for x, y in xys:
            self.forward(x)
            self.output_layer.expected.set(y)
            total_error += self.output_layer.get_error_value()
            error_vec = self.output_layer.get_error_derivative()
            # if total_error == float('nan')

            for layer_index in range(-1, -len(self.layers) - 1, -1):
                layer = self.layers[layer_index]
                error_vec = layer.backward(error_vec)

        self.update_weights(0.025)
        return total_error

    def update_weights(self, lr) -> None:
        for layer in self.layers:
            self.code.program.weights_del(
                self.cl.queue,
                (layer.layer_size, layer.next_layer_size),
                (16, 16),
                layer.layer_size,
                layer.bias_del.data,
                layer.layer.data,
                layer.weights_del.data
            )
            layer.weights -= lr*layer.weights_del
            layer.bias -= lr*layer.bias_del

    def predict(self, X):
        self.forward(X)
        return self.output_layer.layer.get()

    def predict_values(self, xys):
        for x, y in xys:
            print(x,
                  ['%.3f' % i for i in self.predict(x)],
                  y)






