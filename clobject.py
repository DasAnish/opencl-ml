import pyopencl as cl
import pyopencl.array as pycl_array
import os

# TODO: make this a Singleton class


class ClSingleton:

    __instance = None

    @staticmethod
    def get_instance():
        if ClSingleton.__instance is None:
            ClSingleton()
        return ClSingleton.__instance

    def __init__(self):
        if ClSingleton.__instance is not None:
            raise Exception("There is already an instance please use the get_instance method.")
        else:
            ClSingleton.__instance = self

        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)


class Code:

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
        path_to_file = os.path.join('D:', 'Desktop', 'Diss', 'opencl-ml', 'kernel.cl')

        with open('D:\\Desktop\\Diss\\opencl-ml\\kernel.cl', 'r') as f:
            self.code = f.read()

        self.program = cl.Program(self.cl.context, self.code).build()
