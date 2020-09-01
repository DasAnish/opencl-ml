import pyopencl as cl
import pyopencl.array as pycl_array
import os


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
