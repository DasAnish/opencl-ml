from clobject import ClSingleton
import pyopencl.array as pycl_array
import abc


class Loss:
    """An abstract class that defines the template for all loss classes."""

    def __init__(self):
        self.cl = ClSingleton.get_instance()

    @abc.abstractmethod
    def error_value(self, predicted, expected):
        """Should return the absolute error value."""
        pass

    @abc.abstractmethod
    def error_derivative(self, predicted, expected):
        """Should return vector of derivative values of error_value w.r.t. each index."""
        pass

    def convert_to_arrays(self, array):
        """Converts ndarrays to pyopencl.Array for faster processing."""
        if not isinstance(array, pycl_array.Array):
            array = pycl_array.to_device(
                self.cl.queue,
                array
            )
        return array


class MeanSquaredError(Loss):

    def error_value(self, predicted, expected):
        """Returns the absolute values of 1/2 * || expected - predicted ||."""
        predicted = self.convert_to_arrays(predicted)
        expected = self.convert_to_arrays(expected)

        out = predicted - expected
        return pycl_array.dot(out, out) / 2

    def error_derivative(self, predicted, expected) -> pycl_array.Array:
        """Returns the vector (expected - predicted)."""
        predicted = self.convert_to_arrays(predicted)
        expected = self.convert_to_arrays(expected)

        return predicted - expected
