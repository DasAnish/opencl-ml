from ClObject import ClSingleton
import pyopencl.array as pycl_array
import abc


class Error:

    def __init__(self):
        self.cl = ClSingleton.get_instance()

    @abc.abstractmethod
    def error_value(self, predicted, expected):
        pass

    @abc.abstractmethod
    def error_derivative(self, predicted, expected):
        pass

    def convert_to_arrays(self, array):
        if not isinstance(array, pycl_array.Array):
            array = pycl_array.to_device(
                self.cl.queue,
                array
            )
        return array


class MeanSquaredError(Error):

    def error_value(self, predicted, expected):
        predicted = self.convert_to_arrays(predicted)
        expected = self.convert_to_arrays(expected)

        out = predicted - expected
        return pycl_array.dot(out, out) / 2

    def error_derivative(self, predicted, expected):
        predicted = self.convert_to_arrays(predicted)
        expected = self.convert_to_arrays(expected)

        return predicted - expected
