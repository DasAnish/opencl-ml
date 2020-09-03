import pyopencl as cl
import pyopencl.array as pycl_array
import numpy as np
import os

RESET_OUTPUT, CUMULATIVE_OUTPUT = (np.int32(i) for i in (1, 0))
TS = 8
WPT = 256 // TS


class ClSingleton:
    """A singleton class that contains the context and queue objects needed to run code on the gpu"""

    __instance = None

    @staticmethod
    def get_instance():
        """It will create an instance if there isn't one, else it will just return the instance."""
        if ClSingleton.__instance is None:
            ClSingleton()
        return ClSingleton.__instance

    def __init__(self):
        if ClSingleton.__instance is not None:
            raise Exception("There is already an instance please use the get_instance method.")
        else:
            ClSingleton.__instance = self

        # The variables needed.
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)

    def zeros(self, shape) -> pycl_array.Array:
        return pycl_array.to_device(
            self.queue,
            np.zeros(shape).astype(np.float32)
        )

    def ones(self, shape) -> pycl_array.Array:
        return pycl_array.to_device(
            self.queue,
            np.ones(shape).astype(np.float32)
        )

    def random(self, shape) -> pycl_array.Array:
        return pycl_array.to_device(
            self.queue,
            np.random.random(size=shape).astype(np.float32)
        )

    def uniform(self, low, high, shape) -> pycl_array.Array:
        return pycl_array.to_device(
            self.queue,
            np.random.uniform(low=low, high=high, size=shape).astype(np.float32)
        )

    def get_array(self, array: np.array) -> pycl_array.Array:
        return pycl_array.to_device(self.queue, array)


class Code:
    """A singleton class that contains the program object to run code."""

    __instance = None

    @staticmethod
    def get_instance():
        if Code.__instance is None:
            Code()
        return Code.__instance

    def __init__(self):
        if Code.__instance is not None:
            raise Exception("This is a singleton class")
        else:
            Code.__instance = self

        self.cl = ClSingleton.get_instance()
        # path_to_file = os.path.join('D:', 'Desktop', 'Diss', 'opencl-ml', 'kernel.cl')

        with open('kernel.cl', 'r') as f:
            self.code = f.read()

        self.program = cl.Program(self.cl.context, self.code).build()
