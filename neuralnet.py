from layer import *
from clobject import ClSingleton, Code
from typing import List
import copy


class NeuralNet:
    """Basic artificial neural net implementation"""
    def __init__(self, *layers, learning_rate = 0.01):
        self.cl = ClSingleton.get_instance()
        self.code: Code = Code.get_instance()
        self.learning_rate = learning_rate

        # Input layer && Output layer
        if len(layers):
            self.input_size: np.int32 = layers[0].layer_size
            self.input_layer = layers[0]
            self.__layers: List[Layer] = layers[:-1]
            self.output_layer: Output = layers[-1]
            self.output_size: np.int32 = layers[-1].layer_size

            print(type(layers[-1]), layers[-1].layer_size,
                  self.output_layer.error.__class__)

        # Connecting the layer
        for i, layer in enumerate(layers[:-1]):
            layer.set_next_layer(layers[i+1].layer)
            print(type(layer), layer.layer_size, layer.print_activation())

            layer.set_bias(self.cl.uniform(-1, 1, layer.next_layer_size))

            layer.set_weights(self.cl.uniform(-1, 1, (layer.next_layer_size,
                                                      layer.layer_size)))
        # print("SETUP NN COMPLETE")

    def forward(self, _input: np.array) -> None:
        """
        Forward propagation implementation of a feed forward implementation.
        :param _input: the input vector to run the array on.
        :return:
        """
        if len(_input) != self.input_size:
            raise ValueError(f"Provided input size: {len(_input)} is "
                             f"not of the correct size: {self.input_size}")

        self.__layers[0].layer.set(_input)
        for layer in self.__layers:
            layer.forward()

    def __repr__(self):
        st = ''
        for i, layer in enumerate(self.__layers):
            st += f"{i} {layer}"
        st += (f"\n####################################################\n"
               f"OUTPUT: {self.output_layer}"
               f"\n####################################################\n")
        return st

    def fit(self, x, y, batch_size, num_epochs, len_dataset, print_every=100) -> None:
        """Function to train on a dataset where you don't have a spliced set."""
        shuffled_range = np.arange(len_dataset)
        np.random.shuffle(shuffled_range)
        total_error = 0

        for epoch in range(num_epochs):
            current_batch = []

            for j in range(batch_size):
                i = (batch_size*epoch + j) % len_dataset
                y_i = y[shuffled_range[i]]
                x_i = x[shuffled_range[i]]
                current_batch.append((x_i, y_i))

            epoch_error = self.train_batch(current_batch)
            total_error += epoch_error
            if epoch % print_every == (print_every - 1):
                print(f'({epoch+1}) {total_error}|')  # , self.output_layer)#
                total_error = 0

    def train(self, xys, batch_size: int,
              num_epochs: int, print_every=100,
              predict_after_every_batch=False):
        """Runs the training routine"""
        total_error = 0
        min_error = float('inf')

        for epoch in range(num_epochs):
            current_batch = []
            i = epoch % batch_size

            for j in range(batch_size):
                current_batch.append(xys[np.random.randint(len(xys))])

            epoch_error = self.train_batch(current_batch)
            total_error += epoch_error
            min_error = min(min_error, epoch_error)

            if epoch % print_every==(print_every-1):
                print(f'({epoch+1}) avg: {total_error/print_every} | min: {min_error} ||', end='\t')
                total_error = 0
                min_error = float('inf')

                if predict_after_every_batch:
                    print()
                    self.predict_values(xys)

    def train_batch(self, xys) -> float:
        for layer in self.__layers:
            layer.weights_del: pycl_array.Array = self.cl.zeros((layer.next_layer_size, layer.layer_size))
            layer.transposed: pycl_array.Array = pycl_array.transpose(layer.weights)
            layer.bias_del: pycl_array.Array = pycl_array.zeros_like(layer.bias)

        total_error = 0

        for x, y in xys:
            self.forward(x)
            self.output_layer.expected.set(y)
            total_error += self.output_layer.get_error_value()
            error_vec = self.output_layer.get_error_derivative()
            # if total_error == float('nan')

            for layer_index in range(-1, -len(self.__layers) - 1, -1):
                layer = self.__layers[layer_index]
                error_vec = layer.backward(error_vec)

        # self.update_weights(0.01)
        # TODO: add adam optmizer

        for layer in self.__layers:
            layer.weights -= self.learning_rate*layer.weights_del
            layer.bias -= self.learning_rate*layer.bias_del

        return total_error

    def predict(self, X):
        self.forward(X)
        return self.output_layer.layer.get()

    def predict_values(self, xys):
        for x, y in xys:
            print(x,
                  ['%.3f' % i for i in self.predict(x)],
                  y)

    def __copy__(self):
        ret = NeuralNet(learning_rate=self.learning_rate)

        # Copying in the input layer info
        ret.input_size = self.input_size
        ret.input_layer = self.cl.zeros(ret.input_size)

        # Copying in the output layer info
        ret.output_size = self.output_size
        ret.output_layer = Output(ret.output_size)

        # Copying the layers
        ret.__layers = [
            copy.deepcopy(layer) for layer in self.__layers
        ]

        # Connecting the layers
        for i, layer in enumerate(ret.__layers[:-1]):
            layer.set_next_layer(ret.__layers[i+1].layer)

        ret.__layers[-1].set_next_layer(ret.output_layer.layer)

        return ret

    @property
    def shape(self):
        ret = []
        for layer in self.__layers:
            ret.append(layer.layer_size)
        ret.append(self.output_size)
        return tuple(ret)

    # def update_self(self, nn):
    #     if not isinstance(nn, NeuralNet):
    #         raise ValueError(f"Please provide a {NeuralNet} not a {type(nn)}")
    #
    #     if nn.shape != self.shape:
    #         raise ValueError("The shapes do not match.")
    #
    #     self.output_size = nn.output_size
    #     self.input_size = nn.input_size
    #     self.output_layer = pycl_array.to_device(
    #         self.cl.queue,
    #         np.zeros(nn.output_size).astype(np.float32)
    #     )
    #
    #     self.__layers = [
    #         copy.deepcopy(layer) for layer in nn.__layers
    #     ]
    #     self.input_layer = self.__layers[0]










