from ClObject import ClObject
import pyopencl.array as pycl_array
import pyopencl as cl
import pyopencl.clmath as pycl_math
import abc


class Error(ClObject):

    @abc.abstractmethod
    def error(self, predicted, expected):
        pass

    @abc.abstractmethod
    def error_derivative(self, predicted, expected):
        pass

    def convert_to_arrays(self, array):
        if not isinstance(array, pycl_array.Array):
            array = pycl_array.to_device(
                self.queue,
                array
            )
        return array


class MeanSquaredError(Error):

    def error(self, predicted, expected):
        predicted = self.convert_to_arrays(predicted)
        expected = self.convert_to_arrays(expected)
        Error.error(self, predicted, expected)
        out = predicted - expected
        return pycl_array.dot(out, out) / 2

    def error_derivative(self, predicted, expected):
        predicted = self.convert_to_arrays(predicted)
        expected = self.convert_to_arrays(expected)
        Error.error_derivative(self, predicted, expected)
        return predicted - expected
